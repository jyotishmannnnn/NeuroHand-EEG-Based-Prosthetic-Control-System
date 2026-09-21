"""Contact-quality gate: refuse to arm the hand on electrode noise.

Replaying session 09-57-47 -- five floating electrodes and heavy mains hum --
through the Tier-1 chain produced 10 motor commands in 181 s. Nothing downstream
was wrong; noise simply has large excursions, and a gesture detector cannot tell
"blink" from "pad lifted off the scalp" by amplitude alone.

So contact is checked before the detector arms, on the same rest window used for
calibration. The thresholds mirror the acceptance targets already recorded in
PROJECT_NOTES.md for this cap.
"""
import numpy as np
from scipy import signal

from . import FS_EEG

MAINS_HZ = 50.0
MAX_MAINS_RATIO = 0.05    # 50 Hz power vs 1-45 Hz signal power; >0.05 = high impedance
MIN_RMS = 1e-6            # a dead or shorted channel
MAX_FLAT_FRAC = 0.20      # fraction of samples with zero first difference
MAX_TWIN_CORR = 0.98      # two pads reading one signal are not both on the scalp


def channel_metrics(x, fs=FS_EEG):
    """Per-channel quality metrics for a rest window. x is (n, n_ch)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    sos = signal.butter(4, [1.0, min(45.0, fs / 2 * 0.95)], btype="bandpass",
                        fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos, x, axis=0)   # offline window, so zero-phase is fine

    f, p = signal.welch(bp, fs=fs, nperseg=min(len(bp), int(2 * fs)), axis=0)
    band = (f >= 1) & (f < 45)
    near = (f >= MAINS_HZ - 2) & (f <= MAINS_HZ + 2)
    total = np.trapezoid(p[band], f[band], axis=0)
    mains = np.trapezoid(p[near], f[near], axis=0) if near.any() else np.zeros(x.shape[1])

    d = np.diff(x, axis=0)
    return {
        "rms": np.sqrt(np.mean(bp**2, axis=0)),
        "mains_ratio": mains / np.maximum(total, 1e-12),
        "flat_frac": np.mean(np.abs(d) < 1e-12, axis=0),
    }


def assess(x, names, fs=FS_EEG, required=("FP1", "FP2")):
    """Judge whether `required` channels are fit to drive the hand.

    Returns (ok, reasons, metrics). Only the channels Tier 1 actually reads are
    blocking: occipital and central pads may still be fought with while blink and
    jaw control already works off the frontal pair.
    """
    m = channel_metrics(x, fs)
    reasons = []
    idx = {n: i for i, n in enumerate(names)}

    for ch in required:
        if ch not in idx:
            reasons.append(f"{ch} missing from stream")
            continue
        i = idx[ch]
        if m["rms"][i] < MIN_RMS:
            reasons.append(f"{ch} flat/dead (rms {m['rms'][i]:.3g})")
        if m["flat_frac"][i] > MAX_FLAT_FRAC:
            reasons.append(f"{ch} {m['flat_frac'][i]:.0%} stuck samples")
        if m["mains_ratio"][i] > MAX_MAINS_RATIO:
            reasons.append(f"{ch} 50 Hz power is {m['mains_ratio'][i]:.2f}x the 1-45 Hz "
                           f"signal (limit {MAX_MAINS_RATIO:.2f}x)")

    # Both frontal pads echoing one waveform means at least one is off the scalp,
    # which would also defeat the frontal/posterior gate in detect.py.
    have = [idx[c] for c in required if c in idx]
    if len(have) == 2:
        a, b = np.asarray(x)[:, have[0]], np.asarray(x)[:, have[1]]
        if a.std() > 0 and b.std() > 0:
            r = abs(np.corrcoef(a, b)[0, 1])
            if r > MAX_TWIN_CORR:
                reasons.append(f"{required[0]}/{required[1]} correlate r={r:.3f} "
                               f"(>{MAX_TWIN_CORR}): likely one pad not on skin")
    return (not reasons), reasons, m


def report(x, names, fs=FS_EEG):
    """Human-readable table, for the runner and for bring-up sessions."""
    m = channel_metrics(x, fs)
    lines = [f"{'ch':>4} {'rms':>10} {'50Hz/sig':>9} {'flat%':>7}"]
    for i, n in enumerate(names):
        lines.append(f"{n:>4} {m['rms'][i]:10.1f} {m['mains_ratio'][i]:9.2f} "
                     f"{m['flat_frac'][i] * 100:7.1f}")
    return "\n".join(lines)
