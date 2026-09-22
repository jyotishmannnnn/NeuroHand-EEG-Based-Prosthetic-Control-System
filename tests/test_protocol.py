"""Checks for the cued protocol. Run: python tests/test_protocol.py

The end-to-end case feeds synthetic gestures placed exactly on the cue times, so
a correct pipeline must score near 100%. That closes the loop the three lost
recordings exposed: cue, record, detect and score in one pass.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6 import QtWidgets

import run_protocol
from neuro import protocol as P
from neuro.detect import Event
from neuro.source import SynthSource

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


def test_schedule():
    print("\n[1] schedule")
    cues, blocks, total = P.build(calib=20.0)
    kinds = [c.kind for c in cues]
    check("10 blink cues", kinds.count(P.BLINK) == 10, str(kinds.count(P.BLINK)))
    check("5 held cues", kinds.count(P.HOLD) == 5, str(kinds.count(P.HOLD)))
    check("5 jaw cues", kinds.count(P.JAW) == 5, str(kinds.count(P.JAW)))
    check("total 115 s", abs(total - 115.0) < 1e-6, f"{total}s")
    check("opens with a rest block for the baseline",
          blocks[0].kind == P.REST and blocks[0].dur == 20.0)
    check("no cue lands inside the baseline block",
          min(c.t for c in cues) >= 20.0, f"first cue {min(c.t for c in cues):.1f}s")
    check("cues are ordered in time",
          all(cues[i].t < cues[i + 1].t for i in range(len(cues) - 1)))
    spacing = [round(cues[i + 1].t - cues[i].t, 3) for i in range(9)]
    check("blink block spaced 3 s", set(spacing) == {3.0}, str(set(spacing)))


def test_scoring():
    print("\n[2] scoring")
    cues = [P.Cue(10.0, P.BLINK), P.Cue(13.0, P.BLINK), P.Cue(20.0, P.JAW)]

    perfect = [Event(P.BLINK, 10.1, 30, 0.3, 5), Event(P.BLINK, 13.2, 30, 0.3, 5),
               Event(P.JAW, 20.3, 9, 0.4, 3)]
    per, ov = P.score(cues, perfect)
    check("all hits", ov.hits == 3 and ov.missed == 0 and ov.false_alarms == 0,
          f"hits={ov.hits} missed={ov.missed} fa={ov.false_alarms}")

    per, ov = P.score(cues, [Event(P.BLINK, 10.1, 30, 0.3, 5)])
    check("missing cues counted", ov.hits == 1 and ov.missed == 2,
          f"hits={ov.hits} missed={ov.missed}")

    # Detected, but classified as the wrong gesture: distinct from a miss.
    per, ov = P.score(cues, [Event(P.HOLD, 10.1, 30, 0.9, 5)])
    check("wrong kind is not a miss", ov.wrong_kind == 1 and ov.hits == 0,
          f"wrong={ov.wrong_kind} hits={ov.hits} missed={ov.missed}")

    per, ov = P.score(cues, perfect + [Event(P.BLINK, 45.0, 30, 0.3, 5)])
    check("false alarm counted", ov.hits == 3 and ov.false_alarms == 1,
          f"fa={ov.false_alarms}")

    # 12.5 s is within the 1.0 s blink tolerance of the 13.0 s cue, so it counts.
    per, ov = P.score(cues, [Event(P.BLINK, 12.5, 30, 0.3, 5)])
    check("detection inside tolerance is credited to its cue",
          ov.hits == 1 and ov.false_alarms == 0, f"hits={ov.hits} fa={ov.false_alarms}")

    # 11.5 s is 1.5 s from both blink cues: outside tolerance, so a false alarm.
    per, ov = P.score(cues, [Event(P.BLINK, 11.5, 30, 0.3, 5)])
    check("out-of-tolerance detection is a false alarm",
          ov.hits == 0 and ov.false_alarms == 1 and ov.missed == 3,
          f"hits={ov.hits} fa={ov.false_alarms} missed={ov.missed}")

    check("report renders", "TOTAL" in P.report(cues, perfect))


def test_end_to_end():
    print("\n[3] end-to-end: synthetic gestures on the cue times")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    args = run_protocol.build_args(["--source", "synth", "--calib", "10"])
    cues, blocks, total = P.build(calib=10.0)

    # Synthesise each cued gesture at its cue time.
    src = SynthSource(duration=total + 2, realtime=False, seed=5,
                      events=[(c.t, c.kind) for c in cues])
    w = run_protocol.ProtocolWindow(args, src)
    w.show()
    w.worker.stop()
    w.timer.stop()

    feed = SynthSource(duration=total + 2, realtime=False, seed=5,
                       events=[(c.t, c.kind) for c in cues])
    while True:
        c, s = feed.pull()
        if c is None:
            break
        w.on_chunk(c, s)
    w.tick()

    check("ran the whole schedule", w.t_now >= total, f"{w.t_now:.1f}/{total:.0f}s")
    check("detector calibrated", w.det.calibrated)
    per, ov = P.score(w.cues, w.events)
    n = ov.hits + ov.missed + ov.wrong_kind
    rate = ov.hits / max(n, 1)
    print(f"  INFO  {ov.hits}/{n} hits ({rate:.0%}), wrong_kind={ov.wrong_kind}, "
          f"missed={ov.missed}, false_alarms={ov.false_alarms}")
    for k in (P.BLINK, P.HOLD, P.JAW):
        s = per.get(k)
        if s:
            print(f"  INFO  {k:10s} {s.hits}/{s.hits + s.missed + s.wrong_kind}")
    check("scores at least 80% on clean synthetic cues", rate >= 0.80, f"{rate:.0%}")
    check("few false alarms", ov.false_alarms <= 3, str(ov.false_alarms))

    w.finish()
    out = Path(__file__).resolve().parent.parent / "reports"
    saved = sorted(out.glob("protocol_*.txt"))
    check("report written", bool(saved), str([p.name for p in saved[-1:]]))
    if saved:
        body = saved[-1].read_text(encoding="utf-8")
        check("report has the score table", "TOTAL" in body)
        check("report records contact", "50Hz/sig" in body or "contact" in body)
        for p in out.glob("protocol_*"):
            p.unlink()          # test artefacts, not results
    w.close()


if __name__ == "__main__":
    test_schedule()
    test_scoring()
    test_end_to_end()
    print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
