"""End-to-end checks for the Tier-1 chain. Run: python tests/test_chain.py

Deliberately runnable without the cap and without the ESP32, so the control path
can be trusted before electrode contact is solved.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuro.decide import DecideConfig, Decider
from neuro.detect import DetectConfig, Event, Tier1Detector
from neuro.dsp import Baseline, StreamFilter, bandpass_sos, chain_sos, notch_sos
from neuro.hand import Hand, MockHand
from neuro import quality
from neuro.source import CsvSource, SynthSource

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


# --- 1. causal filtering must be chunk-boundary invariant --------------------

def test_filter_continuity():
    print("\n[1] streaming filter continuity")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2000, 8)) * 100
    sos = chain_sos(notch_sos(50.0), bandpass_sos(0.5, 40.0))

    one = signal.sosfilt(sos, x, axis=0)
    f = StreamFilter(sos, 8)
    chunks = [f.apply(x[i : i + 37]) for i in range(0, len(x), 37)]
    streamed = np.vstack(chunks)

    err = np.abs(one - streamed).max()
    check("chunked == one-shot", err < 1e-9, f"max abs err {err:.2e}")

    f.reset()
    check("reset clears state", np.abs(f.zi).max() == 0.0)


# --- 2. detector finds injected blinks and stays quiet at rest --------------

def run_detector(src, cfg=None):
    det = Tier1Detector(src.names, src.fs, cfg or DetectConfig(calib_sec=8.0))
    evs = []
    for chunk, _ in src:
        evs.extend(det.push(chunk))
    return det, evs


def test_blink_detection():
    print("\n[2] blink detection vs ground truth")
    truth = [12.0, 15.0, 18.0, 21.5, 25.0]
    src = SynthSource(duration=30, events=[(t, "blink") for t in truth], seed=7)
    det, evs = run_detector(src)

    check("calibrated", det.calibrated)
    blinks = [e for e in evs if e.kind == "blink"]
    matched, tol = [], 0.30
    for t in truth:
        hit = [e for e in blinks if abs(e.t - t) <= tol]
        matched.append(bool(hit))
    check("all 5 blinks found", all(matched), f"{sum(matched)}/5 within {tol}s")

    spurious = [e for e in blinks if not any(abs(e.t - t) <= tol for t in truth)]
    check("no spurious blinks", len(spurious) == 0, f"{len(spurious)} extra")

    rest_only = SynthSource(duration=30, events=[], seed=3)
    _, rest_evs = run_detector(rest_only)
    check("silent on pure rest", len(rest_evs) == 0, f"{len(rest_evs)} events")
    print(f"  INFO  rejections: {det.rejected}")


def test_short_vs_long_blink():
    print("\n[2b] short blink vs held blink (CYCLE vs CONFIRM)")
    # This separation is the whole basis of select-then-confirm. If a held blink
    # reads as short, CONFIRM is unreachable; if a short blink reads as long, the
    # hand actuates on an involuntary blink.
    for seed in range(3):
        src = SynthSource(duration=26, events=[(12.0, "blink"), (18.0, "blink_long")],
                          seed=seed)
        _, evs = run_detector(src)
        kinds = [e.kind for e in evs]
        durs = {e.kind: e.dur for e in evs}
        check(f"seed {seed}: short->blink, held->blink_long",
              kinds == ["blink", "blink_long"],
              f"{kinds} durs={ {k: round(v, 3) for k, v in durs.items()} }")

    d = Decider(DecideConfig())
    cmds = d.feed([Event("blink", 10.0, 90, 0.28, 6.0)])
    check("short blink cycles immediately, no waiting",
          len(cmds) == 1 and cmds[0].kind == "cycle", f"{cmds}")
    cmds = d.feed([Event("blink_long", 12.0, 90, 0.60, 6.0)])
    check("held blink confirms",
          len(cmds) == 1 and cmds[0].kind == "confirm", f"{cmds}")
    check("confirm executes the highlighted primitive",
          cmds[0].primitive == "power", cmds[0].primitive)


def test_dead_frontal_pad():
    print("\n[2c] one dead frontal pad must not kill blink detection")
    # Observed on real session 09-18-27: the FP1 pad carried ongoing EEG but was
    # not coupled to the eye dipole, so blink excursions hit 60 MADs on FP2 and
    # under 2 on FP1. Averaging the two channels before measuring deviation lost
    # a real held blink; taking each channel's deviation and maxing recovers it.
    truth = [12.0, 16.0, 20.0]
    src = SynthSource(duration=26, events=[(t, "blink") for t in truth], seed=6)
    rng = np.random.default_rng(99)
    i_fp1 = src.names.index("FP1")
    # Replace FP1 with independent noise of the same scale: alive, but blind to blinks.
    src.x[:, i_fp1] = rng.standard_normal(len(src.x)) * src.scale

    det, evs = run_detector(src)
    blinks = [e for e in evs if e.kind.startswith("blink")]
    found = sum(any(abs(e.t - tt) <= 0.4 for e in blinks) for tt in truth)
    check("blinks still found with FP1 blind", found == 3, f"{found}/3")
    check("no spurious events from the dead pad",
          len(blinks) == 3, f"{len(blinks)} blinks total")


def test_ratio_gate_needs_usable_reference():
    print("\n[4b] ratio gate must not compare against floating pads")
    # Session 09-33-34: six posterior pads floating at ~10x the frontal amplitude.
    # The gate asks "is this frontally dominant?", so a garbage posterior reference
    # made every real blink score a ratio below 1 and 13 were rejected. Here the
    # posterior channels carry heavy mains and so must be dropped from the gate.
    truth = [12.0, 16.0, 20.0]
    src = SynthSource(duration=26, events=[(t, "blink") for t in truth], seed=8)
    fs = src.fs
    post = [src.names.index(c) for c in ("O1", "O2", "C3", "CZ", "FZ", "C4")]
    hum = np.sin(2 * np.pi * 50.0 * np.arange(len(src.x)) / fs) * 40.0 * src.scale
    src.x[:, post] += hum[:, None]

    det, evs = run_detector(src, DetectConfig(calib_sec=8.0, require_quality=False))
    check("posterior reference dropped", det.i_ref == [],
          f"kept {[det.names[i] for i in det.i_ref]}")
    check("says why the gate is off", bool(det.gate_note) and "DISABLED" in det.gate_note,
          str(det.gate_note))
    blinks = [e for e in evs if e.kind.startswith("blink")]
    found = sum(any(abs(e.t - tt) <= 0.4 for e in blinks) for tt in truth)
    check("blinks survive a floating posterior reference", found == 3, f"{found}/3")
    check("nothing rejected by ratio", det.rejected["ratio"] == 0,
          f"{det.rejected['ratio']} rejected")

    # Clean posterior pads must still be used, so the gate keeps working.
    clean = SynthSource(duration=26, events=[(t, "blink") for t in truth], seed=8)
    det2, _ = run_detector(clean, DetectConfig(calib_sec=8.0))
    check("clean posterior pads are kept", len(det2.i_ref) == 6,
          f"kept {len(det2.i_ref)}")
    check("no gate note when all pads pass", det2.gate_note is None, str(det2.gate_note))


def test_jaw_detection():
    print("\n[3] jaw clench detection")
    truth = [12.0, 17.0, 22.0]
    src = SynthSource(duration=28, events=[(t, "jaw") for t in truth], seed=11)
    _, evs = run_detector(src)
    jaws = [e for e in evs if e.kind == "jaw"]
    # Tolerance is 0.5 s because onset is read off a smoothed EMG envelope, which
    # lags the true burst start. Irrelevant for grasp selection.
    lat = [min((e.t - t for e in jaws), key=abs, default=float("nan")) for t in truth]
    found = sum(any(abs(e.t - t) <= 0.5 for e in jaws) for t in truth)
    check("all 3 clenches found", found == 3,
          f"{found}/3, onset lag {[round(v, 2) for v in lat]}s")
    check("clench not reported as blink",
          all(e.kind == "jaw" for e in evs), f"kinds={[e.kind for e in evs]}")


# --- 3. whole-head artifact must not pass as a blink ------------------------

def test_artifact_gate():
    print("\n[4] frontal/posterior artifact gate")
    # Same deflection on every channel = cable sway / electrode pop / ref drive,
    # which is the observed failure mode on this cap, not an EOG dipole.
    src = SynthSource(duration=26, events=[], seed=5)
    fs = src.fs
    for onset in (12.0, 16.0, 20.0):
        i0 = int(onset * fs)
        dur = int(0.30 * fs)
        shape = np.sin(np.pi * np.arange(dur) / dur) ** 2 * 10.0 * src.scale
        src.x[i0 : i0 + dur, :] += shape[:, None]

    _, gated = run_detector(src, DetectConfig(calib_sec=8.0, use_ratio_gate=True))
    check("gate rejects whole-head artifact",
          len([e for e in gated if e.kind == "blink"]) == 0,
          f"{len([e for e in gated if e.kind == 'blink'])} passed")

    src2 = SynthSource(duration=26, events=[], seed=5)
    for onset in (12.0, 16.0, 20.0):
        i0 = int(onset * fs)
        dur = int(0.30 * fs)
        shape = np.sin(np.pi * np.arange(dur) / dur) ** 2 * 10.0 * src2.scale
        src2.x[i0 : i0 + dur, :] += shape[:, None]
    _, ungated = run_detector(src2, DetectConfig(calib_sec=8.0, use_ratio_gate=False))
    check("same artifact passes with gate off",
          len([e for e in ungated if e.kind == "blink"]) > 0,
          "confirms the gate is what rejected it")


# --- 4. decision layer semantics -------------------------------------------

def test_decider():
    print("\n[5] decision layer")
    d = Decider(DecideConfig())
    cmds = d.feed([Event("blink", 10.0, 90, 0.2, 3.0)])
    check("blink cycles", len(cmds) == 1 and cmds[0].kind == "cycle", f"{cmds}")
    check("cycle advanced highlight", cmds[0].primitive == "power", cmds[0].primitive)
    cmds = d.feed([Event("blink", 11.0, 90, 0.2, 3.0)])
    check("cycling wraps through the primitive list",
          cmds[0].primitive == "pinch", cmds[0].primitive)

    d3 = Decider(DecideConfig())
    cmds = d3.feed([Event("jaw", 10.0, 9, 0.4, 2.0)])
    check("jaw releases immediately",
          len(cmds) == 1 and cmds[0].kind == "release", f"{cmds}")
    check("release resets highlight to open", d3.primitive == "open")

    d4 = Decider(DecideConfig(busy_sec=1.2))
    d4.feed([Event("blink_long", 10.0, 90, 0.6, 3.0)])   # confirm -> busy until 11.2
    cmds = d4.feed([Event("blink", 10.9, 90, 0.2, 3.0)])
    check("gestures during actuation are dropped",
          cmds == [] and d4.dropped == 1, f"dropped={d4.dropped}")
    cmds = d4.feed([Event("jaw", 10.95, 9, 0.4, 2.0)])
    check("release overrides busy lockout",
          len(cmds) == 1 and cmds[0].kind == "release", f"{cmds}")


# --- 5. hand link + watchdog ----------------------------------------------

def test_hand_and_watchdog():
    print("\n[6] hand link and watchdog")
    mock = MockHand(watchdog=1.5)
    t = [0.0]
    hand = Hand(mock, heartbeat=0.4, clock=lambda: t[0])

    hand.go("pinch")
    check("primitive accepted", mock.rx and mock.rx[0] == "OK G pinch", f"{mock.rx}")
    mock.read_lines()

    try:
        hand.go("banana")
        bad = False
    except ValueError:
        bad = True
    check("host rejects unknown primitive", bad)

    # Heartbeats keep it alive.
    for _ in range(10):
        t[0] += 0.4
        mock.advance(0.4)
        hand.poll_and_beat()
    check("no watchdog trip while beating", not mock.released_by_watchdog,
          f"state={mock.state}")

    # Host goes away.
    mock.advance(2.0)
    check("watchdog releases on link loss", mock.released_by_watchdog)
    check("watchdog leaves hand open and coasting",
          mock.state == "coast" and mock.primitive == "open",
          f"{mock.state}/{mock.primitive}")
    check("watchdog is announced", "EV watchdog_release" in mock.read_lines())

    mock2 = MockHand()
    h2 = Hand(mock2, clock=lambda: 0.0)
    from neuro.decide import Command
    check("cycle sends nothing to hardware",
          h2.apply(Command("cycle", 1.0, "power")) is False)
    check("confirm sends primitive",
          h2.apply(Command("confirm", 1.0, "power")) is True
          and mock2.sent[-1][1] == "G power", f"{mock2.sent[-1:]}")


# --- 6. config <-> firmware consistency -----------------------------------

def test_hand_config():
    print("\n[7] hand config integrity")
    import json
    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / "config" / "hand.json").read_text())
    roles = [m["role"] for m in cfg["motors"]]
    check("8 motors", len(cfg["motors"]) == 8, str(len(cfg["motors"])))
    check("motor ids unique and 0-7",
          sorted(m["id"] for m in cfg["motors"]) == list(range(8)))

    pins = [p for m in cfg["motors"] for p in (m["in1"], m["in2"])]
    check("16 pins, no duplicates", len(pins) == len(set(pins)) == 16,
          f"{len(set(pins))} unique")
    bad = [p for p in pins if 6 <= p <= 11 or p in (0, 2, 15) or p > 33]
    check("no flash/strapping/input-only pins", not bad, f"bad={bad}")

    from neuro.decide import PRIMITIVES
    check("firmware primitives match host PRIMITIVES",
          tuple(cfg["primitives"]) == PRIMITIVES, f"{tuple(cfg['primitives'])}")
    for name, tbl in cfg["primitives"].items():
        check(f"primitive '{name}' dirs valid",
              set(tbl) <= set(roles) and all(v in (-1, 0, 1) for v in tbl.values()))
    check("open releases every motor",
          all(v == -1 for v in cfg["primitives"]["open"].values()))

    hdr = root / "firmware" / "esp32_hand" / "config.h"
    check("config.h generated", hdr.exists())
    if hdr.exists():
        text = hdr.read_text()
        check("config.h watchdog matches json",
              f"#define WATCHDOG_MS {cfg['watchdog_ms']}" in text)
        from neuro.hand import WATCHDOG_SEC
        check("host mirrors firmware watchdog",
              abs(WATCHDOG_SEC * 1000 - cfg["watchdog_ms"]) < 1e-6,
              f"host {WATCHDOG_SEC}s vs fw {cfg['watchdog_ms']}ms")


# --- 7. real recorded session must not drive the hand ---------------------

SESSION = Path(r"C:\Users\JYOTISHMAN\AppData\LocalLow\Qneuro\QneuroNeuroAnalytics"
               r"\Jyotishman\09-20-2026\LiveSession\09-57-47")


def test_real_session_replay():
    print("\n[8] recorded session 09-57-47 replay")
    if not SESSION.exists():
        print("  SKIP  session not present")
        return
    src = CsvSource(SESSION, chunk=25)
    det = Tier1Detector(src.names, src.fs, DetectConfig(calib_sec=20.0))
    dec = Decider(DecideConfig())
    hand = Hand(MockHand(), clock=lambda: 0.0)
    evs, cmds, sent = [], [], 0
    t = 0.0
    for chunk, stamps in src:
        t = float(stamps[-1])
        e = det.push(chunk)
        evs.extend(e)
        for c in dec.feed(e, now=t):
            cmds.append(c)
            sent += bool(hand.apply(c))

    check("replay ran to completion", t > 150, f"{t:.0f}s")
    check("channel order as documented",
          src.names == ["FP1", "FP2", "O1", "O2", "C3", "CZ", "FZ", "C4"], str(src.names))
    check("calibrated on real data", det.calibrated)
    # This recording has five floating electrodes and heavy mains hum. The bar is
    # not "few commands" but zero: the quality gate must refuse to arm at all.
    check("refused to arm on known-bad contact", not det.armed)
    check("gave a reason for refusing", len(det.quality_reasons) > 0)
    check("no hand commands from bad contact",
          sent == 0 and len(cmds) == 0, f"{len(cmds)} commands, {sent} sent")
    print(f"  INFO  refusal reasons: {det.quality_reasons}")
    print("  INFO  metrics:\n" + "\n".join(
        "         " + ln for ln in quality.report(
            CsvSource(SESSION, chunk=25).x[: int(20 * src.fs)], src.names, src.fs
        ).splitlines()))


def test_quality_gate_allows_good_contact():
    print("\n[9] quality gate arms on clean data")
    src = SynthSource(duration=26, events=[(12.0, "blink")], seed=2)
    det, evs = run_detector(src)
    check("armed on clean synthetic rest", det.armed, f"{det.quality_reasons}")
    check("still detects the blink once armed",
          len([e for e in evs if e.kind == "blink"]) == 1, f"{len(evs)} events")

    # Heavy mains on an otherwise fine signal must block arming.
    noisy = SynthSource(duration=26, events=[(12.0, "blink")], seed=2, mains=3.0)
    det2, evs2 = run_detector(noisy)
    check("mains hum blocks arming", not det2.armed, f"{det2.quality_reasons}")
    check("blocked detector emits nothing", len(evs2) == 0, f"{len(evs2)} events")

    # --force must still be able to override, since bring-up needs it.
    det3, evs3 = run_detector(
        SynthSource(duration=26, events=[(12.0, "blink")], seed=2, mains=3.0),
        DetectConfig(calib_sec=8.0, require_quality=False))
    check("--force overrides the gate", det3.armed and len(evs3) > 0,
          f"armed={det3.armed}, {len(evs3)} events")


if __name__ == "__main__":
    test_filter_continuity()
    test_blink_detection()
    test_short_vs_long_blink()
    test_dead_frontal_pad()
    test_jaw_detection()
    test_artifact_gate()
    test_ratio_gate_needs_usable_reference()
    test_decider()
    test_hand_and_watchdog()
    test_hand_config()
    test_real_session_replay()
    test_quality_gate_allows_good_contact()
    print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
