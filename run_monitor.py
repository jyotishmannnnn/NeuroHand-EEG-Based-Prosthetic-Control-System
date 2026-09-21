"""Live monitor and electrode-contact tool for the NeuroState cap.

Runs standalone against the cap with no hand attached, which is the intended use
while adjusting electrode pads:

    python run_monitor.py --source lsl                 # contact tool, no hand
    python run_monitor.py --source synth                # demo with scripted gestures
    python run_monitor.py --source "<...>/09-57-47" --speed 4   # replay a session
    python run_monitor.py --source lsl --port COM5      # full chain, hand attached

Panes:
  contact      per-channel rms / 50 Hz ratio / stuck samples / alpha share / peak
  traces       8 channels, per-channel normalised so all are visible at once
  spectrum     PSD with the alpha band and 50 Hz marked (the eyes-closed check)
  features     blink and EMG z-scores against their thresholds, with fired events

The features pane is the one that explains a missed gesture: it shows whether the
blink fell short of threshold, was rejected by a gate, or was never detected.
Quality is computed on a rolling window independently of the detector, so the
contact table is live from the first second and does not wait for calibration.
"""
import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from neuro import quality
from neuro.decide import DecideConfig, Decider
from neuro.detect import DetectConfig, Tier1Detector
from neuro.dsp import Ring, StreamFilter, bandpass_sos
from neuro.hand import Hand, MockHand, SerialHand
from neuro.source import open_source

TRACE_SEC = 5.0
FEAT_SEC = 12.0
QUAL_SEC = 4.0
QUAL_EVERY = 0.5
OK_BG = QtGui.QColor(226, 246, 228)
BAD_BG = QtGui.QColor(252, 226, 226)
# Foreground is set explicitly alongside every background: the host may be in a
# dark theme, whose light default text is unreadable on these pale fills.
OK_FG = QtGui.QColor(16, 48, 20)
BAD_FG = QtGui.QColor(110, 8, 16)
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
          "#42d4f4", "#bfef45", "#a9a9a9"]


class SourceWorker(QtCore.QThread):
    """Pulls chunks off the source so the Qt event loop never blocks on I/O."""

    chunk = QtCore.Signal(object, object)
    finished_stream = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, src):
        super().__init__()
        self.src = src
        self._stop = False
        self.paused = False

    def run(self):
        try:
            while not self._stop:
                if self.paused:
                    self.msleep(30)
                    continue
                c, s = self.src.pull()
                if c is None:
                    self.finished_stream.emit()
                    return
                if len(c):
                    self.chunk.emit(c, s)
                else:
                    self.msleep(5)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def stop(self):
        self._stop = True
        self.wait(2000)


class Monitor(QtWidgets.QWidget):
    def __init__(self, args, src):
        super().__init__()
        self.args = args
        self.src = src
        self.names = list(src.names)
        self.fs = float(src.fs)
        self.n_ch = len(self.names)

        self.t_now = 0.0
        self.n_samples = 0
        self._last_qual = -1e9
        self._rate_mark = (time.monotonic(), 0)
        self.rate = 0.0
        self.events = []          # (t, kind) for markers
        self.metrics = None

        # One raw ring serves both panes, so it must span the longer of the two
        # windows; quality then reads its last QUAL_SEC and traces its last TRACE_SEC.
        self.raw = Ring(int(max(TRACE_SEC, QUAL_SEC) * self.fs), self.n_ch)
        self.disp = Ring(int(TRACE_SEC * self.fs), self.n_ch)
        self.disp_t = Ring(int(TRACE_SEC * self.fs), 1)
        self.fi_disp = StreamFilter(bandpass_sos(1.0, 45.0, fs=self.fs, order=3), self.n_ch)

        self._build_detector()
        self.dec = Decider(DecideConfig())
        self.hand, self.is_mock = self._make_hand()

        self._build_ui()

        self.worker = SourceWorker(src)
        self.worker.chunk.connect(self.on_chunk)
        self.worker.finished_stream.connect(lambda: self.log("stream ended"))
        self.worker.failed.connect(lambda m: self.log(f"SOURCE ERROR: {m}"))
        self.worker.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)

    # --- setup ----------------------------------------------------------

    def _build_detector(self):
        self.det = Tier1Detector(self.names, self.fs, DetectConfig(
            blink_z=self.args.blink_z, jaw_z=self.args.jaw_z,
            calib_sec=self.args.calib,
            use_ratio_gate=not self.args.no_ratio_gate,
            require_quality=not self.args.force))

    def _make_hand(self):
        if self.args.port:
            return Hand(SerialHand(self.args.port)), False
        if self.args.mock:
            clock = time.monotonic if self.args.realtime else (lambda: self.t_now)
            return Hand(MockHand(), clock=clock), True
        return None, False

    def _build_ui(self):
        mode = ("contact tool (no hand)" if self.hand is None
                else f"mock hand" if self.is_mock else f"hand on {self.args.port}")
        self.setWindowTitle(f"NeuroHand monitor - {self.args.source} - {mode}")
        self.resize(1450, 880)
        pg.setConfigOptions(antialias=False, background="w", foreground="k")

        grid = QtWidgets.QGridLayout(self)

        # --- controls row
        bar = QtWidgets.QHBoxLayout()
        self.lbl_state = QtWidgets.QLabel()
        self.lbl_state.setTextFormat(QtCore.Qt.RichText)
        bar.addWidget(self.lbl_state, 3)

        self.btn_recal = QtWidgets.QPushButton("Recalibrate")
        self.btn_recal.setToolTip("Re-learn the rest baseline after fixing a pad")
        self.btn_recal.clicked.connect(self.recalibrate)
        bar.addWidget(self.btn_recal)

        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self.on_pause)
        bar.addWidget(self.btn_pause)

        self.chk_raw = QtWidgets.QCheckBox("raw traces")
        self.chk_raw.setToolTip("Off: 1-45 Hz bandpass for shape. On: raw counts, "
                                "showing drift and mains as the amplifier sees them.")
        bar.addWidget(self.chk_raw)

        self.btn_stop = QtWidgets.QPushButton("RELEASE HAND")
        self.btn_stop.setStyleSheet(
            "QPushButton{background:#c0142b;color:white;font-weight:bold;padding:6px;}")
        self.btn_stop.clicked.connect(self.release_hand)
        self.btn_stop.setEnabled(self.hand is not None)
        bar.addWidget(self.btn_stop)
        grid.addLayout(bar, 0, 0, 1, 2)

        # --- contact table
        cols = ["ch", "rms", "50Hz/sig", "stuck%", "alpha", "peak Hz", "status"]
        self.table = QtWidgets.QTableWidget(self.n_ch, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        hh = self.table.horizontalHeader()
        # Status carries the actual complaint, so it takes the slack; the numeric
        # columns only need to fit their digits.
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(len(cols) - 1, QtWidgets.QHeaderView.Stretch)
        for r, n in enumerate(self.names):
            it = QtWidgets.QTableWidgetItem(n)
            if n in ("FP1", "FP2"):
                f = it.font()
                f.setBold(True)
                it.setFont(f)
                it.setToolTip("Required for Tier-1 gestures")
            self.table.setItem(r, 0, it)
            for c in range(1, len(cols)):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem("-"))
        self.table.setMinimumWidth(560)
        vh = self.table.verticalHeader()
        vh.setDefaultSectionSize(26)
        # Size to show every channel: a scrolled-off row is a channel you forget
        # to check while adjusting pads.
        self.table.setMinimumHeight(26 * self.n_ch + self.table.horizontalHeader()
                                    .sizeHint().height() + 8)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        box_t = QtWidgets.QGroupBox("contact quality  (rolling 4 s; bold = required "
                                    "for Tier 1)")
        lay = QtWidgets.QVBoxLayout(box_t)
        lay.addWidget(self.table)
        self.lbl_reasons = QtWidgets.QLabel("")
        self.lbl_reasons.setWordWrap(True)
        self.lbl_reasons.setStyleSheet("color:#ff6b6b;font-weight:bold;")
        lay.addWidget(self.lbl_reasons)
        grid.addWidget(box_t, 1, 0)

        # --- traces
        self.p_tr = pg.PlotWidget(title=None)
        self.p_tr.setLabel("bottom", "time", units="s")
        self.p_tr.showGrid(x=True, y=True, alpha=0.15)
        self.p_tr.setMouseEnabled(x=False, y=False)
        self.p_tr.setYRange(-1, self.n_ch)
        ax = self.p_tr.getAxis("left")
        ax.setTicks([[(i, n) for i, n in enumerate(self.names)]])
        self.curves = [self.p_tr.plot(pen=pg.mkPen(COLORS[i % len(COLORS)], width=1))
                       for i in range(self.n_ch)]
        box_tr = QtWidgets.QGroupBox("traces  (each channel normalised to its own "
                                     "spread, so all 8 stay visible)")
        QtWidgets.QVBoxLayout(box_tr).addWidget(self.p_tr)
        grid.addWidget(box_tr, 1, 1)

        # --- spectrum
        self.p_ps = pg.PlotWidget()
        self.p_ps.setLogMode(x=False, y=True)
        self.p_ps.setLabel("bottom", "frequency", units="Hz")
        self.p_ps.setLabel("left", "PSD (log)")
        self.p_ps.showGrid(x=True, y=True, alpha=0.15)
        self.p_ps.setXRange(0, 60)
        band = pg.LinearRegionItem([8, 13], movable=False,
                                   brush=pg.mkBrush(60, 160, 60, 40))
        self.p_ps.addItem(band)
        self.p_ps.addItem(pg.InfiniteLine(50, pen=pg.mkPen("#c0142b", style=QtCore.Qt.DashLine)))
        self.ps_curves = {}
        self.ps_checks = {}
        chk_row = QtWidgets.QHBoxLayout()
        for i, n in enumerate(self.names):
            cb = QtWidgets.QCheckBox(n)
            cb.setChecked(n in ("O1", "O2"))
            cb.setStyleSheet(f"color:{COLORS[i % len(COLORS)]};font-weight:bold;")
            chk_row.addWidget(cb)
            self.ps_checks[n] = cb
            self.ps_curves[n] = self.p_ps.plot(
                pen=pg.mkPen(COLORS[i % len(COLORS)], width=1))
        box_ps = QtWidgets.QGroupBox("spectrum  (green = alpha 8-13 Hz, red = 50 Hz "
                                     "mains; expect a ~10 Hz bump at O1/O2 eyes closed)")
        lps = QtWidgets.QVBoxLayout(box_ps)
        lps.addWidget(self.p_ps)
        lps.addLayout(chk_row)
        grid.addWidget(box_ps, 2, 0)

        # --- features
        self.p_ft = pg.PlotWidget()
        self.p_ft.setLabel("bottom", "time", units="s")
        self.p_ft.setLabel("left", "z (baseline MADs)")
        self.p_ft.showGrid(x=True, y=True, alpha=0.15)
        self.p_ft.addLegend(offset=(-10, 10))
        self.c_blink = self.p_ft.plot(pen=pg.mkPen("#4363d8", width=2), name="blink z")
        self.c_emg = self.p_ft.plot(pen=pg.mkPen("#f58231", width=2), name="EMG z")
        self.l_blink = pg.InfiniteLine(self.args.blink_z, angle=0,
                                       pen=pg.mkPen("#4363d8", style=QtCore.Qt.DashLine))
        self.l_jaw = pg.InfiniteLine(self.args.jaw_z, angle=0,
                                     pen=pg.mkPen("#f58231", style=QtCore.Qt.DashLine))
        self.p_ft.addItem(self.l_blink)
        self.p_ft.addItem(self.l_jaw)
        self.sc_ev = pg.ScatterPlotItem(size=11, brush=pg.mkBrush(30, 30, 30, 200))
        self.p_ft.addItem(self.sc_ev)
        box_ft = QtWidgets.QGroupBox("features  (dashed = thresholds, dots = fired "
                                     "events)")
        QtWidgets.QVBoxLayout(box_ft).addWidget(self.p_ft)
        grid.addWidget(box_ft, 2, 1)

        # --- log
        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumBlockCount(500)
        self.txt.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        self.txt.setFixedHeight(120)
        box_lg = QtWidgets.QGroupBox("events, commands, hand replies")
        QtWidgets.QVBoxLayout(box_lg).addWidget(self.txt)
        grid.addWidget(box_lg, 3, 0, 1, 2)

        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 3)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    # --- data flow ------------------------------------------------------

    def log(self, msg):
        self.txt.appendPlainText(f"[{self.t_now:7.2f}s] {msg}")

    def on_chunk(self, chunk, stamps):
        self.t_now = float(stamps[-1])
        self.n_samples += len(chunk)
        self.raw.push(chunk)
        self.disp.push(self.fi_disp.apply(chunk))
        self.disp_t.push(np.asarray(stamps, dtype=np.float64)[:, None])

        was_cal = self.det.calibrated
        evs = self.det.push(chunk)
        if self.det.calibrated and not was_cal:
            if self.det.armed:
                extra = f" (warnings: {self.det.quality_reasons})" if \
                    self.det.quality_reasons else ""
                self.log(f"baseline locked, ARMED{extra}")
            else:
                self.log("baseline locked, NOT ARMED: "
                         + "; ".join(self.det.quality_reasons))

        for ev in evs:
            self.events.append((ev.t, ev.kind))
            self.log(str(ev))
        if evs:
            for cmd in self.dec.feed(evs, now=self.t_now):
                acted = self.hand.apply(cmd) if self.hand else False
                self.log(f"{cmd}{'  [sent]' if acted else '  [highlight only]'}"
                         + ("" if self.hand else "  [no hand attached]"))
        if self.hand:
            for line in self.hand.poll_and_beat():
                if not line.startswith("OK P"):
                    self.log(f"<- {line}")
            if self.is_mock:
                self.hand.link.advance(len(chunk) / self.fs)

    # --- UI refresh -----------------------------------------------------

    def refresh(self):
        now = time.monotonic()
        dt = now - self._rate_mark[0]
        if dt >= 1.0:
            self.rate = (self.n_samples - self._rate_mark[1]) / dt
            self._rate_mark = (now, self.n_samples)

        self._refresh_traces()
        if self.t_now - self._last_qual >= QUAL_EVERY and self.raw.filled > self.fs:
            self._last_qual = self.t_now
            self._refresh_quality()
        self._refresh_features()
        self._refresh_state()

    def _refresh_traces(self):
        n = min(self.disp.filled, self.raw.filled, self.disp_t.filled)
        if n < 2:
            return
        t = self.disp_t.window()[-n:, 0]
        src = self.raw if self.chk_raw.isChecked() else self.disp
        x = src.window()[-n:]
        for i, cur in enumerate(self.curves):
            col = x[:, i]
            med = np.median(col)
            spread = 1.4826 * np.median(np.abs(col - med))
            # Per-channel normalisation: FP2 has run 11x FP1's amplitude on this
            # cap, so a shared scale would flatten most channels to a line.
            y = (col - med) / (4.0 * max(spread, 1e-9))
            cur.setData(t, np.clip(y, -0.48, 0.48) + i)
        if len(t) > 1:
            self.p_tr.setXRange(t[0], t[-1], padding=0)

    def _refresh_quality(self):
        n = min(self.raw.filled, int(QUAL_SEC * self.fs))
        x = self.raw.window()[-n:]
        m = quality.channel_metrics(x, self.fs)
        self.metrics = m
        for i, name in enumerate(self.names):
            issues = quality.channel_issues(m, i)
            vals = [f"{m['rms'][i]:.1f}", f"{m['mains_ratio'][i]:.2f}",
                    f"{m['flat_frac'][i] * 100:.1f}", f"{m['alpha_ratio'][i]:.3f}",
                    f"{m['peak_hz'][i]:.1f}",
                    "ok" if not issues else "; ".join(issues)]
            bg, fg = (OK_BG, OK_FG) if not issues else (BAD_BG, BAD_FG)
            for c, v in enumerate(vals, start=1):
                it = self.table.item(i, c)
                it.setText(v)
                it.setBackground(bg)
                it.setForeground(fg)
            head = self.table.item(i, 0)
            head.setBackground(bg)
            head.setForeground(fg)

        req = [n_ for n_ in ("FP1", "FP2") if n_ in self.names]
        bad = [f"{n_}: {', '.join(quality.channel_issues(m, self.names.index(n_)))}"
               for n_ in req if quality.channel_issues(m, self.names.index(n_))]
        self.lbl_reasons.setText(
            "" if not bad else "Tier-1 channels failing:  " + "   |   ".join(bad))

        f, p = m["freqs"], m["psd"]
        keep = f <= 62
        for i, name in enumerate(self.names):
            if self.ps_checks[name].isChecked():
                self.ps_curves[name].setData(f[keep], np.maximum(p[keep, i], 1e-12))
            else:
                self.ps_curves[name].setData([], [])

    def _refresh_features(self):
        tr = self.det.trace()
        if not len(tr):
            # Nothing to show until the detector arms; keep a sane axis rather
            # than letting pyqtgraph autoscale to an empty 0-1 s range.
            self.c_blink.setData([], [])
            self.c_emg.setData([], [])
            self.sc_ev.setData([], [])
            self.p_ft.setXRange(max(0.0, self.t_now - FEAT_SEC),
                                max(FEAT_SEC, self.t_now), padding=0)
            self.p_ft.setYRange(0, max(self.args.blink_z, self.args.jaw_z) * 1.3)
            return
        t, zb, ze = tr[:, 0], tr[:, 1], tr[:, 2]
        keep = t >= self.t_now - FEAT_SEC
        self.c_blink.setData(t[keep], zb[keep])
        self.c_emg.setData(t[keep], ze[keep])
        vis = [(et, 0.0) for et, _ in self.events if et >= self.t_now - FEAT_SEC]
        if vis:
            top = max(float(np.max(zb[keep])) if keep.any() else 1.0,
                      self.args.blink_z) * 1.05
            self.sc_ev.setData([v[0] for v in vis], [top] * len(vis))
        else:
            self.sc_ev.setData([], [])
        self.p_ft.setXRange(max(0.0, self.t_now - FEAT_SEC), self.t_now, padding=0)

    def _refresh_state(self):
        if not self.det.calibrated:
            pct = self.det.calib_progress * 100
            state = (f"<b style='color:#b06000'>CALIBRATING {pct:.0f}%</b> "
                     f"- hold still, no blinking")
        elif self.det.armed:
            state = "<b style='color:#0a7a20'>ARMED</b>"
        else:
            state = "<b style='color:#c0142b'>NOT ARMED</b>"

        hand_txt = "no hand" if self.hand is None else (
            f"hand: {self.hand.primitive}")
        self.lbl_state.setText(
            f"{state} &nbsp;|&nbsp; highlighted: <b>{self.dec.primitive}</b> "
            f"&nbsp;|&nbsp; {hand_txt} &nbsp;|&nbsp; t={self.t_now:.1f}s "
            f"&nbsp;|&nbsp; {self.rate:.0f} Sa/s &nbsp;|&nbsp; "
            f"dropped {self.dec.dropped}")

    # --- actions --------------------------------------------------------

    def recalibrate(self):
        self._build_detector()
        self.events.clear()
        self.log(f"recalibrating on {self.args.calib:.0f}s of rest")

    def on_pause(self, on):
        self.worker.paused = on
        self.btn_pause.setText("Resume" if on else "Pause")

    def release_hand(self):
        if not self.hand:
            return
        self.hand.release()
        self.log("RELEASE sent by operator")

    def closeEvent(self, ev):
        self.timer.stop()
        self.worker.stop()
        if self.hand:
            try:
                self.hand.stop()
                self.hand.close()
            except Exception:
                pass
        self.src.close()
        super().closeEvent(ev)


def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="lsl",
                   help="'lsl', 'synth', or a LiveSession directory to replay")
    p.add_argument("--port", help="ESP32 serial port; omit for a hand-free contact run")
    p.add_argument("--mock", action="store_true", help="attach the in-process mock hand")
    p.add_argument("--calib", type=float, default=20.0)
    p.add_argument("--blink-z", type=float, default=12.0)
    p.add_argument("--jaw-z", type=float, default=5.0)
    p.add_argument("--no-ratio-gate", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="arm even if frontal contact fails the quality check")
    p.add_argument("--duration", type=float, default=300.0, help="synth length, seconds")
    p.add_argument("--speed", type=float, default=1.0,
                   help="replay speed multiplier for synth/CSV sources")
    return p.parse_args(argv)


def main(argv=None):
    args = build_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])

    kw = dict(realtime=args.source != "lsl", speed=args.speed)
    if args.source == "synth":
        c = args.calib
        kw.update(duration=args.duration, alpha_amp=0.6, events=[
            (c + 4, "blink"), (c + 7, "blink"), (c + 10, "blink_long"),
            (c + 14, "jaw"), (c + 20, "blink"), (c + 23, "blink_long"),
        ])
    try:
        src = open_source(args.source, **kw)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "Cannot open source", str(exc))
        return 1

    # realtime pacing is what makes a monitor readable; args.realtime mirrors it
    # so the hand clock in _make_hand picks the matching time domain.
    args.realtime = bool(kw.get("realtime"))
    w = Monitor(args, src)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
