"""Cued gesture session: the software tells you when to act, and scores itself.

    python run_protocol.py --stream jyotishman_EEG

It issues cues on a fixed schedule, records the raw stream alongside the exact
cue times, and on finish writes a scored report. Nothing has to be timed by
hand, which is the point: three earlier recordings could not be scored because a
300 ms blink cannot be located inside a 95 s file from memory.

Default schedule (115 s): 20 s rest, 10 blinks at 3 s, 10 s rest, 5 held blinks
at 4 s, 10 s rest, 5 jaw clenches at 4 s, 5 s rest. The opening rest block is
what the detector calibrates its baseline on, so stay still through it.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from neuro import protocol as P
from neuro import quality
from neuro.detect import DetectConfig, Tier1Detector
from neuro.source import open_source
from run_monitor import SourceWorker

FLASH = 0.7          # seconds the cue banner stays in its "act now" state
PLOT_SEC = 12.0


class ProtocolWindow(QtWidgets.QWidget):
    def __init__(self, args, src):
        super().__init__()
        self.args = args
        self.src = src
        self.names = list(src.names)
        self.fs = float(src.fs)

        self.cues, self.blocks, self.total = P.build(calib=args.calib)
        self.t_now = 0.0
        self.chunks = []
        self.events = []
        self.done = False
        self._next_cue = 0

        self.det = Tier1Detector(self.names, self.fs, DetectConfig(
            blink_z=args.blink_z, jaw_z=args.jaw_z, calib_sec=args.calib,
            # Always produce detections so the run can be scored; contact is
            # reported instead of being allowed to block the session.
            require_quality=False, trace_sec=PLOT_SEC + 2))

        self._build_ui()
        self.worker = SourceWorker(src)
        self.worker.chunk.connect(self.on_chunk)
        self.worker.failed.connect(lambda m: self.lbl_contact.setText(f"SOURCE ERROR: {m}"))
        self.worker.start()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)

    def _build_ui(self):
        self.setWindowTitle("NeuroHand cued protocol")
        self.resize(1100, 720)
        pg.setConfigOptions(antialias=False, background="w", foreground="k")
        v = QtWidgets.QVBoxLayout(self)

        self.lbl_contact = QtWidgets.QLabel("connecting...")
        self.lbl_contact.setStyleSheet("font-size:13px;")
        v.addWidget(self.lbl_contact)

        self.lbl_cue = QtWidgets.QLabel("get ready")
        self.lbl_cue.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_cue.setStyleSheet("font-size:64px;font-weight:bold;")
        self.lbl_cue.setMinimumHeight(150)
        v.addWidget(self.lbl_cue, 2)

        self.lbl_sub = QtWidgets.QLabel("")
        self.lbl_sub.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_sub.setStyleSheet("font-size:24px;color:#555;")
        v.addWidget(self.lbl_sub)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setMaximum(1000)
        v.addWidget(self.bar)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "z (baseline MADs)")
        self.plot.setLabel("bottom", "time", units="s")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend(offset=(-10, 10))
        self.c_blink = self.plot.plot(pen=pg.mkPen("#4363d8", width=2), name="blink z")
        self.c_emg = self.plot.plot(pen=pg.mkPen("#f58231", width=2), name="EMG z")
        self.plot.addItem(pg.InfiniteLine(self.args.blink_z, angle=0,
                                          pen=pg.mkPen("#4363d8", style=QtCore.Qt.DashLine)))
        self.plot.addItem(pg.InfiniteLine(self.args.jaw_z, angle=0,
                                          pen=pg.mkPen("#f58231", style=QtCore.Qt.DashLine)))
        self.sc_cue = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(40, 160, 60, 180),
                                         symbol="t1")
        self.sc_det = pg.ScatterPlotItem(size=11, brush=pg.mkBrush(30, 30, 30, 200))
        self.plot.addItem(self.sc_cue)
        self.plot.addItem(self.sc_det)
        v.addWidget(self.plot, 3)

        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumBlockCount(400)
        self.txt.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        self.txt.setFixedHeight(120)
        v.addWidget(self.txt)

    # --- data -----------------------------------------------------------

    def on_chunk(self, chunk, stamps):
        if self.done:
            return
        self.t_now = float(stamps[-1]) - self._t0(stamps)
        self.chunks.append(np.asarray(chunk, dtype=np.float64))
        evs = self.det.push(chunk)
        for e in evs:
            e.t = e.t  # detector time already starts at 0 with the stream
            self.events.append(e)
            self.txt.appendPlainText(f"[{self.t_now:6.2f}s] {e}")

    def _t0(self, stamps):
        if not hasattr(self, "_base"):
            self._base = float(stamps[0])
        return self._base

    # --- ui -------------------------------------------------------------

    def tick(self):
        if self.done:
            return
        t = self.t_now
        self.bar.setValue(int(1000 * min(t / self.total, 1.0)))

        # advance the cue pointer and mark cues as issued
        while (self._next_cue < len(self.cues)
               and t >= self.cues[self._next_cue].t + FLASH):
            self._next_cue += 1

        blk = P.block_at(self.blocks, t)
        cue = self.cues[self._next_cue] if self._next_cue < len(self.cues) else None
        acting = cue is not None and 0 <= t - cue.t < FLASH

        if acting:
            self.lbl_cue.setText(P.LABEL[cue.kind].split(" ")[0].upper() + "  NOW")
            self.lbl_cue.setStyleSheet(
                "font-size:64px;font-weight:bold;color:white;background:#0a7a20;")
            self.lbl_sub.setText(P.LABEL[cue.kind])
        elif blk is not None and blk.kind == P.REST:
            left = blk.t0 + blk.dur - t
            first = "  (baseline - stay still, do not blink)" if blk.t0 == 0 else ""
            self.lbl_cue.setText("REST")
            self.lbl_cue.setStyleSheet("font-size:64px;font-weight:bold;color:#777;")
            self.lbl_sub.setText(f"{left:.0f}s{first}")
        elif cue is not None:
            self.lbl_cue.setText(f"{P.LABEL[cue.kind]}")
            self.lbl_cue.setStyleSheet("font-size:64px;font-weight:bold;color:#b06000;")
            n_kind = sum(1 for c in self.cues[:self._next_cue + 1] if c.kind == cue.kind)
            tot_kind = sum(1 for c in self.cues if c.kind == cue.kind)
            self.lbl_sub.setText(f"in {cue.t - t:.1f}s   ({n_kind} of {tot_kind})")
        else:
            self.lbl_cue.setText("done")
            self.lbl_sub.setText("")

        self._contact()
        self._plot()

        if t >= self.total:
            self.finish()

    def _contact(self):
        m = self.det.quality
        state = ("calibrating %.0f%%" % (self.det.calib_progress * 100)
                 if not self.det.calibrated else
                 ("contact OK" if not self.det.quality_reasons
                  else "contact: " + "; ".join(self.det.quality_reasons)))
        extra = ""
        if m is not None:
            i1, i2 = self.names.index("FP1"), self.names.index("FP2")
            extra = (f"   FP1 50Hz/sig {m['mains_ratio'][i1]:.2f}  "
                     f"FP2 {m['mains_ratio'][i2]:.2f}")
        note = f"   [{self.det.gate_note}]" if self.det.gate_note else ""
        self.lbl_contact.setText(f"t={self.t_now:6.1f}/{self.total:.0f}s   {state}{extra}{note}")

    def _plot(self):
        tr = self.det.trace()
        lo, hi = max(0.0, self.t_now - PLOT_SEC), max(PLOT_SEC, self.t_now)
        if len(tr):
            keep = tr[:, 0] >= lo
            self.c_blink.setData(tr[keep, 0], tr[keep, 1])
            self.c_emg.setData(tr[keep, 0], tr[keep, 2])
        vis = [c.t for c in self.cues if lo <= c.t <= hi]
        self.sc_cue.setData(vis, [self.args.blink_z * 1.25] * len(vis))
        dets = [e.t for e in self.events if e.t >= lo]
        self.sc_det.setData(dets, [self.args.blink_z * 1.05] * len(dets))
        self.plot.setXRange(lo, hi, padding=0)

    # --- finish ---------------------------------------------------------

    def finish(self):
        self.done = True
        self.timer.stop()
        self.worker.stop()
        x = np.vstack(self.chunks) if self.chunks else np.zeros((0, len(self.names)))

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path(__file__).resolve().parent / "reports"
        out.mkdir(exist_ok=True)
        npz = out / f"protocol_{stamp}.npz"
        np.savez_compressed(npz, x=x, fs=self.fs, names=np.array(self.names),
                            cue_t=np.array([c.t for c in self.cues]),
                            cue_kind=np.array([c.kind for c in self.cues]),
                            det_t=np.array([e.t for e in self.events]),
                            det_kind=np.array([e.kind for e in self.events]))

        lines = [f"Cued protocol {stamp}",
                 f"{len(x)} samples ({len(x) / self.fs:.1f}s) at {self.fs:.0f} Hz",
                 f"stream: {getattr(self.src, 'stream_name', self.args.source)}",
                 ""]
        if len(x) > int(self.args.calib * self.fs):
            lines.append("contact over the baseline block:")
            lines.append(quality.report(x[: int(self.args.calib * self.fs)],
                                        self.names, self.fs))
            lines.append("")
        if self.det.gate_note:
            lines += [self.det.gate_note, ""]
        lines.append(P.report(self.cues, self.events, self.blocks))
        txt = "\n".join(lines)
        (out / f"protocol_{stamp}.txt").write_text(txt, encoding="utf-8")

        self.lbl_cue.setText("done")
        self.lbl_cue.setStyleSheet("font-size:64px;font-weight:bold;color:#0a7a20;")
        per, overall = P.score(self.cues, self.events)
        n = overall.hits + overall.missed + overall.wrong_kind
        self.lbl_sub.setText(f"{overall.hits}/{n} cues detected, "
                             f"{overall.false_alarms} false alarms")
        self.txt.appendPlainText("\n" + txt)
        print(txt)
        print(f"\nsaved {npz.name} and protocol_{stamp}.txt in reports/")

    def closeEvent(self, ev):
        self.timer.stop()
        self.worker.stop()
        if not self.done and self.chunks:
            self.finish()
        self.src.close()
        super().closeEvent(ev)


def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="lsl",
                   help="'lsl', 'synth', or a LiveSession directory (for dry runs)")
    p.add_argument("--stream", help="exact LSL stream name")
    p.add_argument("--user", default="Jyotishman")
    p.add_argument("--calib", type=float, default=20.0,
                   help="opening rest block, used for the baseline")
    p.add_argument("--blink-z", type=float, default=12.0)
    p.add_argument("--jaw-z", type=float, default=5.0)
    return p.parse_args(argv)


def main(argv=None):
    args = build_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    kw = dict(name=args.stream, user=args.user,
              realtime=args.source != "lsl", speed=1.0)
    try:
        src = open_source(args.source, **kw)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "Cannot open source", str(exc))
        return 1
    w = ProtocolWindow(args, src)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
