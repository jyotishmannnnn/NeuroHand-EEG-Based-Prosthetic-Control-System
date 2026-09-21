"""Headless smoke test for run_monitor.py. Run: python tests/test_monitor.py

Renders the window offscreen and feeds it synthetic data to check the panes
actually populate. Data is pushed straight into on_chunk with the reader thread
stopped, so the test is deterministic rather than racing a realtime source.
Also writes reports/monitor_preview.png so the layout can be inspected without a
display.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6 import QtWidgets

import run_monitor
from neuro.dsp import Ring
from neuro.source import SynthSource

FAILED = []
CALIB = 6.0


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


def pts(curve):
    x, _ = curve.getData()
    return 0 if x is None else len(x)


def feed(w, src, until):
    """Push chunks into the widget until stream time reaches `until`."""
    while True:
        c, s = src.pull()
        if c is None:
            return
        w.on_chunk(c, s)
        if float(s[-1]) >= until:
            return


def reset(w):
    """Detach the reader thread and clear buffers for a deterministic start."""
    w.worker.stop()
    w.timer.stop()
    w.raw = Ring(w.raw.n, w.n_ch)
    w.disp = Ring(w.disp.n, w.n_ch)
    w.disp_t = Ring(w.disp_t.n, 1)
    w.recalibrate()
    w._last_qual = -1e9
    w.t_now = 0.0


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    args = run_monitor.build_args([
        "--source", "synth", "--mock", "--calib", str(CALIB), "--duration", "40",
    ])
    args.realtime = False

    gestures = [(CALIB + 4, "blink"), (CALIB + 8, "blink_long"), (CALIB + 12, "jaw")]
    feed_src = SynthSource(duration=40, realtime=False, alpha_amp=0.6, seed=1,
                           events=gestures)

    print("\n[1] window construction")
    w = run_monitor.Monitor(args, SynthSource(duration=1, realtime=False))
    w.show()
    reset(w)
    check("8 rows in contact table", w.table.rowCount() == 8)
    check("one curve per channel", len(w.curves) == 8)
    check("mock hand attached", w.hand is not None and w.is_mock)
    check("release button enabled", w.btn_stop.isEnabled())

    print("\n[2] contact table is live before calibration finishes")
    feed(w, feed_src, 3.0)
    w.refresh()
    check("quality computed early", w.metrics is not None, f"t={w.t_now:.1f}s")
    check("still calibrating", not w.det.calibrated,
          f"progress {w.det.calib_progress:.0%}")
    filled = sum(1 for r in range(8) if w.table.item(r, 1).text() not in ("-", ""))
    check("all 8 rows filled", filled == 8, f"{filled}/8")
    check("rms column numeric", float(w.table.item(0, 1).text()) > 0,
          w.table.item(0, 1).text())
    check("status column filled", w.table.item(0, 6).text() != "-",
          w.table.item(0, 6).text())

    print("\n[3] traces and spectrum")
    check("trace curve has samples", pts(w.curves[0]) > 100, f"{pts(w.curves[0])} pts")
    check("traces stay in their own lanes",
          all(abs(np.median(w.curves[i].getData()[1]) - i) < 0.6 for i in range(8)))
    check("O1 plotted by default", pts(w.ps_curves["O1"]) > 10,
          f"{pts(w.ps_curves['O1'])} bins")
    check("unchecked channel not plotted", pts(w.ps_curves["C3"]) == 0)
    w.ps_checks["C3"].setChecked(True)
    w._last_qual = -1e9
    feed(w, feed_src, 4.0)
    w.refresh()
    check("checking a channel plots it", pts(w.ps_curves["C3"]) > 10)

    print("\n[4] arms on clean data, then detects all three gestures")
    feed(w, feed_src, CALIB + 14.0)
    w.refresh()
    check("calibrated", w.det.calibrated, f"t={w.t_now:.1f}s")
    check("armed on clean data", w.det.armed, str(w.det.quality_reasons))
    kinds = [k for _, k in w.events]
    check("blink, held blink and clench all seen",
          {"blink", "blink_long", "jaw"} <= set(kinds), f"{kinds}")
    check("feature trace plotted", pts(w.c_blink) > 50, f"{pts(w.c_blink)} pts")
    check("event markers shown", pts(w.sc_ev) > 0, f"{pts(w.sc_ev)} markers")
    sent = [l[1] for l in w.hand.link.sent if l[1] != "P"]
    check("hand received a primitive", any(s.startswith("G ") for s in sent),
          str(sent[:5]))
    check("hand received the release", "R" in sent, str(sent[:5]))
    check("log has content", len(w.txt.toPlainText().splitlines()) > 3)

    print("\n[5] operator controls")
    w.release_hand()
    check("release button sends R", w.hand.link.sent[-1][1] == "R",
          w.hand.link.sent[-1][1])
    w.on_pause(True)
    check("pause flags the worker", w.worker.paused)
    check("pause button relabels", w.btn_pause.text() == "Resume")
    w.on_pause(False)
    w.recalibrate()
    check("recalibrate resets calibration", not w.det.calibrated)
    check("recalibrate clears markers", w.events == [])

    print("\n[6] raw/filtered toggle")
    w.chk_raw.setChecked(True)
    feed(w, feed_src, CALIB + 16.0)
    w.refresh()
    check("raw mode still renders", pts(w.curves[0]) > 100)
    w.chk_raw.setChecked(False)
    w.refresh()
    check("filtered mode still renders", pts(w.curves[0]) > 100)

    print("\n[7] render")
    app.processEvents()
    out = Path(__file__).resolve().parent.parent / "reports" / "monitor_preview.png"
    out.parent.mkdir(exist_ok=True)
    pix = w.grab()
    saved = pix.save(str(out))
    size_kb = out.stat().st_size // 1024 if out.exists() else 0
    check("screenshot saved", saved and size_kb > 15, f"{out.name} {size_kb} KB")
    img = pix.toImage()
    cols = {img.pixel(x, y) for x in range(0, img.width(), 37)
            for y in range(0, img.height(), 37)}
    check("render is not blank", len(cols) > 8, f"{len(cols)} distinct colours")

    w.close()
    print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
