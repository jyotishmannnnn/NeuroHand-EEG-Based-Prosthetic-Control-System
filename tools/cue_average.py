"""Cue-locked averaging of a run_protocol.py recording.

Usage: python tools/cue_average.py reports/protocol_<stamp>.npz

For every cue, cut the raw signal from PRE s before to POST s after it, and
average the cuts per cue kind. The same cut taken at random times inside the
rest blocks gives the null: what "nothing happened" looks like. A gesture that
is really in the signal pulls the cue average outside the null band; one that
is not stays inside it however low the detector threshold is set.

This is independent of the detector on purpose. It answers "is the gesture in
the signal at all?" before asking "does the detector find it?".

Raw means: 50 Hz notch and per-epoch DC removal (mean of the pre-cue second)
only. The frontal EMG envelope (25-110 Hz, rectified, 5 Hz smoothed) is shown
alongside because jaw clenches live there, not in the raw trace.

Writes <npz stem>_cue_average.png and prints per-kind statistics.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuro import protocol as P  # noqa: E402

PRE, POST = 1.0, 2.5
N_NULL = 300
CUE_COLOR = "#2a78d6"
NULL_COLOR = "#8a8984"
KINDS = (P.BLINK, P.HOLD, P.JAW)
TITLE = {P.BLINK: "blink", P.HOLD: "held blink", P.JAW: "jaw clench"}


def epochs(sig, times, fs):
    a, b = int(PRE * fs), int(POST * fs)
    out = [sig[int(t * fs) - a:int(t * fs) + b] for t in times
           if int(t * fs) - a >= 0 and int(t * fs) + b <= len(sig)]
    return np.array(out)


def rest_times(blocks, total, fs, rng, skip_calib=5.0):
    """Random epoch centres fully inside rest blocks (calibration start skipped)."""
    pool = []
    for b in blocks:
        if b.kind != P.REST:
            continue
        lo = b.t0 + PRE + (skip_calib if b.t0 == 0 else 0.0)
        hi = min(b.t0 + b.dur, total) - POST
        if hi > lo:
            pool.append((lo, hi))
    if not pool:
        return np.array([])
    w = np.array([hi - lo for lo, hi in pool])
    pick = rng.choice(len(pool), N_NULL, p=w / w.sum())
    return np.array([rng.uniform(*pool[i]) for i in pick])


def main(path):
    path = Path(path)
    z = np.load(path)
    x, fs = z["x"], float(z["fs"])
    names = [str(n) for n in z["names"]]
    cue_t, cue_k = z["cue_t"], z["cue_kind"]
    _, blocks, total = P.build()
    rng = np.random.default_rng(0)

    i_fp = [names.index(c) for c in ("FP1", "FP2")]
    b_n, a_n = signal.iirnotch(50.0, 30.0, fs)
    raw = signal.filtfilt(b_n, a_n, x[:, i_fp], axis=0)
    em = signal.sosfiltfilt(signal.butter(3, [25, 110], btype="band", fs=fs, output="sos"),
                            raw, axis=0)
    env = signal.sosfiltfilt(signal.butter(2, 5, fs=fs, output="sos"), np.abs(em).mean(axis=1))

    feats = [("FP1 raw", raw[:, 0]), ("FP2 raw", raw[:, 1]), ("frontal EMG envelope", env)]
    null_t = rest_times(blocks, len(x) / fs, fs, rng)
    tax = np.arange(-int(PRE * fs), int(POST * fs)) / fs
    pre = tax < 0
    post = (tax >= 0) & (tax <= 1.5)

    def prep(ep, is_env):
        # raw: remove each epoch's pre-cue mean; envelope: express as a ratio to it
        base = ep[:, pre].mean(axis=1, keepdims=True)
        return ep / np.maximum(base, 1e-9) if is_env else ep - base

    print(f"{path.name}: {len(x) / fs:.1f} s, {len(cue_t)} cues, null = {len(null_t)} rest epochs")
    print(f"window {-PRE:+.1f}..{POST:+.1f} s, response measured 0..1.5 s after the cue\n")
    print(f"{'kind':>11} {'feature':>21} {'n':>3} {'cue resp':>9} {'null p95':>9} {'trials>p95':>11}  verdict")

    results = {}
    for k in KINDS:
        tk = cue_t[cue_k == k]
        for fname, sig in feats:
            is_env = "EMG" in fname
            ep = prep(epochs(sig, tk, fs), is_env)
            nu = prep(epochs(sig, null_t, fs), is_env)
            # response size per epoch: peak |deviation| (raw) or peak ratio (envelope)
            resp = lambda e: (e[:, post].max(axis=1) if is_env
                              else np.abs(e[:, post]).max(axis=1))
            r_cue, r_null = resp(ep), resp(nu)
            p95 = np.percentile(r_null, 95)
            n_above = int((r_cue > p95).sum())
            # a trial beats the null p95 by chance 5% of the time
            verdict = ("PRESENT" if n_above >= max(3, 0.5 * len(r_cue))
                       else "weak" if n_above >= 2 else "absent")
            results[(k, fname)] = (ep, nu)
            print(f"{TITLE[k]:>11} {fname:>21} {len(ep):3d} {np.median(r_cue):9.1f} "
                  f"{p95:9.1f} {n_above:6d}/{len(ep):<4d}  {verdict}")

    plot(results, feats, tax, path.with_name(path.stem + "_cue_average.png"))


def plot(results, feats, tax, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(KINDS), len(feats), figsize=(13, 8.5), sharex=True,
                             facecolor="#fcfcfb")
    for r, k in enumerate(KINDS):
        for c, (fname, _) in enumerate(feats):
            ax = axes[r, c]
            ax.set_facecolor("#fcfcfb")
            ep, nu = results[(k, fname)]
            lo, hi = np.percentile(nu, [2.5, 97.5], axis=0)
            ax.fill_between(tax, lo, hi, color=NULL_COLOR, alpha=0.18, lw=0,
                            label="rest null, 95% band")
            ax.plot(tax, nu.mean(axis=0), color=NULL_COLOR, lw=1.5, label="rest null, mean")
            for e in ep:
                ax.plot(tax, e, color=CUE_COLOR, lw=0.8, alpha=0.25)
            ax.plot(tax, ep.mean(axis=0), color=CUE_COLOR, lw=2,
                    label=f"cue average (n={len(ep)}), thin = single trials")
            ax.axvline(0, color="#52514e", lw=1, ls="--")
            ax.grid(True, color="#e6e5e0", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors="#52514e", labelsize=8)
            if r == 0:
                ax.set_title(fname, fontsize=10, color="#0b0b0b")
            if c == 0:
                ax.set_ylabel(f"{TITLE[k]}\ncounts (DC removed)", fontsize=9, color="#0b0b0b")
            if c == len(feats) - 1:
                ax.set_ylabel(f"{TITLE[k]}\nratio to pre-cue", fontsize=9, color="#0b0b0b")
                ax.yaxis.set_label_position("right")
            if r == len(KINDS) - 1:
                ax.set_xlabel("seconds from cue", fontsize=9, color="#52514e")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=110)
    print(f"\nplot: {out}")


if __name__ == "__main__":
    main(sys.argv[1])
