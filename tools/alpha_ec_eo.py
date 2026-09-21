"""Compare alpha power between an eyes-closed and an eyes-open segment.

Usage: python alpha_ec_eo.py <session_dir> <ec_start> <ec_end> <eo_start> <eo_end>
Times in seconds from stream start. Prints per-channel alpha (8-13 Hz) power,
alpha/total ratio, and EC/EO alpha ratio (Berger effect, expect >1.5 at O1/O2).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal

session = Path(sys.argv[1])
ec0, ec1, eo0, eo1 = map(float, sys.argv[2:6])

csv = session / "UnfilteredData.csv"
raw = np.genfromtxt(csv, delimiter=",", skip_header=1, usecols=[0] + list(range(2, 10)))
with open(csv) as f:
    names = [c.replace("UF-", "") for c in f.readline().strip().split(",")[2:]]
t, x = raw[:, 0], raw[:, 1:]
fs = 1.0 / np.median(np.diff(t))

sos = signal.butter(4, [1, 40], btype="bandpass", fs=fs, output="sos")
b_n, a_n = signal.iirnotch(50, 30, fs)


def spectrum(seg):
    seg = signal.sosfiltfilt(sos, seg, axis=0)
    seg = signal.filtfilt(b_n, a_n, seg, axis=0)
    f, p = signal.welch(seg, fs=fs, nperseg=int(2 * fs), axis=0)
    return f, p


def cut(a, b):
    m = (t >= a) & (t < b)
    return x[m]


f, p_ec = spectrum(cut(ec0, ec1))
_, p_eo = spectrum(cut(eo0, eo1))
band = lambda lo, hi: (f >= lo) & (f < hi)
al, tot = band(8, 13), band(1, 40)

print(f"fs={fs:.0f} Hz  EC {ec0:.0f}-{ec1:.0f}s ({np.sum((t>=ec0)&(t<ec1))} samples)  EO {eo0:.0f}-{eo1:.0f}s ({np.sum((t>=eo0)&(t<eo1))} samples)\n")
print(f"{'ch':>4} {'EC_alpha':>10} {'EO_alpha':>10} {'EC/EO':>6} {'EC_a/tot':>9} {'EO_a/tot':>9} {'EC_peak':>8} {'EO_peak':>8} {'EC_rms':>8} {'EO_rms':>8}")
for i, n in enumerate(names):
    a_ec = np.trapezoid(p_ec[al, i], f[al])
    a_eo = np.trapezoid(p_eo[al, i], f[al])
    t_ec = np.trapezoid(p_ec[tot, i], f[tot])
    t_eo = np.trapezoid(p_eo[tot, i], f[tot])
    m = band(4, 30)
    pk_ec = f[m][np.argmax(p_ec[m, i])]
    pk_eo = f[m][np.argmax(p_eo[m, i])]
    print(f"{n:>4} {a_ec:10.1f} {a_eo:10.1f} {a_ec / max(a_eo, 1e-9):6.2f} {a_ec / t_ec:9.3f} {a_eo / t_eo:9.3f} {pk_ec:8.1f} {pk_eo:8.1f} {np.sqrt(t_ec):8.1f} {np.sqrt(t_eo):8.1f}")

# Coarse PSD table for O1/O2 so peak shape is visible in text
print("\nPSD (EC) O1/O2, 2 Hz bins:")
for lo in range(2, 40, 2):
    m = band(lo, lo + 2)
    print(f"{lo:>3}-{lo + 2:<3} Hz  O1 {np.mean(p_ec[m, names.index('O1')]):12.1f}   O2 {np.mean(p_ec[m, names.index('O2')]):12.1f}")
