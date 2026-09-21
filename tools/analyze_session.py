"""Quantify EEG signal quality from a NeuroAnalytics LiveSession folder.

Usage: python analyze_session.py <session_dir>
Reads UnfilteredData.csv (PseudoTimeSec, PseudoTimestamp, UF-<ch> x8).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal

session = Path(sys.argv[1])
csv = session / "UnfilteredData.csv"
raw = np.genfromtxt(csv, delimiter=",", skip_header=1, usecols=[0] + list(range(2, 10)))
t = raw[:, 0]
x = raw[:, 1:]
with open(csv) as f:
    names = [c.replace("UF-", "") for c in f.readline().strip().split(",")[2:]]

fs = 1.0 / np.median(np.diff(t))
dur = t[-1] - t[0]
print(f"session: {session.name}   samples: {len(t)}   fs: {fs:.1f} Hz   duration: {dur:.0f} s")
gaps = np.sum(np.diff(t) > 2.5 / fs)
print(f"timestamp gaps > 2.5 samples: {gaps}")

# Preprocess: 1-40 Hz bandpass + 50 Hz notch, skip first 5 s of settling
sos = signal.butter(4, [1, 40], btype="bandpass", fs=fs, output="sos")
b_n, a_n = signal.iirnotch(50, 30, fs)
start = int(5 * fs)
xf = signal.sosfiltfilt(sos, x[start:], axis=0)
xf = signal.filtfilt(b_n, a_n, xf, axis=0)
xr = x[start:]

# Per-channel metrics
print()
print(f"{'ch':>4} {'raw_mean':>12} {'raw_std':>11} {'flat%':>6} {'bp_rms':>10} {'p50Hz%':>7} {'alpha/tot':>9} {'clip%':>6}")
f_w, psd = signal.welch(xr - xr.mean(axis=0), fs=fs, nperseg=int(2 * fs), axis=0)
band = lambda lo, hi: (f_w >= lo) & (f_w < hi)
for i, n in enumerate(names):
    r = xr[:, i]
    d = np.diff(r)
    flat = np.mean(d == 0) * 100  # repeated identical samples = stuck ADC / dropout
    rng = np.percentile(r, [0.5, 99.5])
    clip = np.mean((r <= rng[0]) | (r >= rng[1])) * 100  # time spent at extremes
    tot = np.trapezoid(psd[band(1, 45), i], f_w[band(1, 45)])
    p50 = np.trapezoid(psd[band(48, 52), i], f_w[band(48, 52)]) / max(tot, 1e-12) * 100
    alpha = np.trapezoid(psd[band(8, 13), i], f_w[band(8, 13)]) / max(tot, 1e-12)
    print(f"{n:>4} {r.mean():12.1f} {r.std():11.1f} {flat:6.1f} {xf[:, i].std():10.1f} {p50:7.1f} {alpha:9.3f} {clip:6.2f}")

# Alpha peak check on occipital channels
print()
for n in ("O1", "O2"):
    i = names.index(n)
    seg = band(4, 30)
    pk = f_w[seg][np.argmax(psd[seg, i])]
    print(f"{n}: spectral peak in 4-30 Hz at {pk:.1f} Hz")

# Cross-channel correlation (common-mode / reference issue if all ~1)
c = np.corrcoef(xf.T)
iu = np.triu_indices(len(names), 1)
print(f"\nmean |corr| between bandpassed channels: {np.mean(np.abs(c[iu])):.2f}  (clean EEG ~0.2-0.5; >0.8 = common-mode/ref problem)")
