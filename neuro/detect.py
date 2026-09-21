"""Tier-1 gesture detection: eye blink and jaw clench from frontal channels.

Tier 1 is deliberately first because it survives poor electrode contact. A blink
is an EOG dipole roughly 10x larger than cortical EEG, and a jaw clench is EMG
larger still, so both are detectable on FP1/FP2 alone while occipital and central
pads are still being fought with.

Two causal feature streams are derived per chunk:

    blink : 6 Hz low-pass, deviation from a running median baseline
    emg   : 25-110 Hz band, rectified + smoothed -> muscle burst envelope

The blink path deliberately avoids a high-pass. A blink is a monophasic
deflection, so high-passing it produces a decaying undershoot which rectification
turns into a second bump roughly 0.85 s later, at ~15% of the primary peak but
still several times the detection threshold. Measured directly: 5 blinks yielded
11 detections. Removing DC with a running median instead of a high-pass fixes the
cause rather than the symptom, because a median has no impulse response to ring
with -- a 300 ms blink barely moves the median of a 4 s window.

Both are z-scored against a rest baseline (see dsp.Baseline) because absolute
amplitude is meaningless at unknown device gain. A frontal/posterior amplitude
ratio gate rejects whole-head artifacts: a real blink is frontally dominant,
whereas cable sway, electrode pops, and reference-drive noise move every channel
together. That gate matters on this cap specifically, where floating posterior
pads have been observed echoing a near-identical waveform.
"""
from dataclasses import dataclass, field

import numpy as np

from . import CHANNELS, FS_EEG, quality
from .dsp import (Baseline, Ring, StreamFilter, bandpass_sos, chain_sos, lowpass_sos,
                  notch_sos)

FRONTAL = ("FP1", "FP2")
POSTERIOR = ("O1", "O2", "C3", "CZ", "FZ", "C4")


@dataclass
class Event:
    """A detected gesture primitive."""

    kind: str          # 'blink' (short) | 'blink_long' (deliberate hold) | 'jaw'
    t: float           # stream time of onset, seconds
    peak_z: float      # peak feature z-score
    dur: float         # seconds above threshold
    ratio: float       # frontal/posterior amplitude ratio at peak
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"Event({self.kind} t={self.t:.2f}s z={self.peak_z:.1f} "
                f"dur={self.dur * 1000:.0f}ms ratio={self.ratio:.2f})")


@dataclass
class DetectConfig:
    blink_z: float = 12.0         # blink threshold in baseline MADs
    jaw_z: float = 5.0            # jaw EMG envelope threshold
    blink_min_dur: float = 0.06   # reject single-sample spikes / electrode pops
    blink_max_dur: float = 1.50   # longer than this is drift or motion, not a blink
    long_dur: float = 0.42        # at/above this a blink counts as a deliberate hold
    jaw_min_dur: float = 0.15     # clench is sustained; short bursts are chewing/noise
    min_ratio: float = 1.5        # frontal must exceed posterior by this factor
    use_ratio_gate: bool = True
    refractory: float = 0.25      # per-kind dead time after an event closes
    blink_emg_max_z: float = 12.0  # a blink with huge EMG is really a clench
    emg_smooth: float = 0.05      # envelope smoothing cutoff period, seconds
    median_sec: float = 4.0       # running-median window used as the DC baseline
    calib_sec: float = 20.0       # rest seconds used to learn the baseline
    warmup_sec: float = 1.5       # discard filter startup transient before calibrating
    require_quality: bool = True  # refuse to arm if frontal contact fails quality.assess


class _Channel:
    """Schmitt-trigger state machine over one feature stream.

    Entry and exit thresholds differ deliberately. A rectified envelope dips
    mid-gesture, and a single threshold fragments one sustained hold into several
    events (measured: 12 long blinks produced 24 detections without hysteresis).
    """

    def __init__(self, thr, min_dur, max_dur, refractory, fs, exit_frac=0.5):
        self.thr, self.min_dur, self.max_dur = thr, min_dur, max_dur
        self.exit_thr = thr * exit_frac
        self.refractory, self.fs = refractory, fs
        self.active = False
        self.onset = 0.0
        self.peak = 0.0
        self.peak_t = 0.0
        self.n = 0
        self.last_close = -1e9

    def step(self, t, v):
        """Feed one sample; return (onset, peak, dur) when an event closes."""
        if self.active:
            self.n += 1
            if v > self.peak:
                self.peak, self.peak_t = v, t
            if v < self.exit_thr:
                dur = self.n / self.fs
                self.active = False
                self.last_close = t
                if self.min_dur <= dur <= self.max_dur:
                    return self.onset, self.peak, dur, self.peak_t
            elif self.n / self.fs > self.max_dur:
                # Ran too long: abandon without emitting, and hold off.
                self.active = False
                self.last_close = t
            return None
        if v >= self.thr and (t - self.last_close) >= self.refractory:
            self.active = True
            self.onset, self.peak, self.peak_t, self.n = t, v, t, 1
        return None


class Tier1Detector:
    """Streaming blink/jaw detector. Feed chunks, collect Events.

    Lifecycle: instances start in calibration. Push rest data until .calibrated
    is True (cfg.calib_sec of it). Calibration then judges frontal contact via
    quality.assess; the detector only starts returning Events if it .armed.
    Bad contact therefore yields no events at all rather than fewer events.
    """

    def __init__(self, names=CHANNELS, fs=FS_EEG, cfg=None):
        self.cfg = cfg or DetectConfig()
        self.names = list(names)
        self.fs = float(fs)
        n_ch = len(self.names)

        # 6 Hz low-pass also removes mains, so the blink path needs no notch.
        self.fi_slow = StreamFilter(lowpass_sos(6.0, fs=fs, order=2), n_ch)
        self.fi_emg = StreamFilter(
            chain_sos(notch_sos(50.0, fs=fs), bandpass_sos(25.0, 110.0, fs=fs, order=3)), n_ch)
        # Envelope: rectify then low-pass. Rectification is correct here because EMG
        # is genuinely oscillatory, unlike the monophasic blink.
        self.fi_env = StreamFilter(lowpass_sos(1.0 / self.cfg.emg_smooth, fs=fs, order=2), 1)
        self.med = Ring(max(int(self.cfg.median_sec * self.fs), 8), 2)
        self._baseline_lvl = np.zeros(2)

        self.base = Baseline(2)   # [blink_frontal, emg_frontal]
        self.t = 0.0
        self._warm = 0.0
        self.i_front = [self.names.index(c) for c in FRONTAL if c in self.names]
        self.i_post = [self.names.index(c) for c in POSTERIOR if c in self.names]
        if not self.i_front:
            raise ValueError(f"need at least one of {FRONTAL} in {self.names}")

        self.sm = {
            "blink": _Channel(self.cfg.blink_z, self.cfg.blink_min_dur,
                              self.cfg.blink_max_dur, self.cfg.refractory, self.fs),
            "jaw": _Channel(self.cfg.jaw_z, self.cfg.jaw_min_dur,
                            2.5, self.cfg.refractory, self.fs),
        }
        self._trace = []   # recent (t, blink_z, emg_z, ratio) for gating + debug
        self.last_z = (0.0, 0.0)
        self.last_ratio = 0.0
        self.rejected = {"ratio": 0, "emg": 0, "slow": 0}
        self._raw_calib = []
        self.armed = False
        self.quality_reasons = []
        self.quality = None

    @property
    def calibrated(self):
        return self.base.locked

    def _features(self, chunk):
        """Return (t, blink_feat, emg_feat, ratio) arrays for this chunk."""
        s = self.fi_slow.apply(chunk)
        e = self.fi_emg.apply(chunk)
        n = len(chunk)
        t = self.t + np.arange(1, n + 1) / self.fs

        s_front = s[:, self.i_front].mean(axis=1)
        s_post = s[:, self.i_post].mean(axis=1) if self.i_post else np.zeros(n)

        # Baseline is taken from the buffer as it stood before this chunk, so the
        # gesture being measured cannot pull its own reference level. Recomputed
        # once per chunk rather than per sample; at 10 chunks/s that is ample.
        if self.med.filled >= self.med.n // 4:
            self._baseline_lvl = np.median(self.med.window()[-self.med.filled :], axis=0)
        self.med.push(np.column_stack([s_front, s_post]))

        dev_front = np.abs(s_front - self._baseline_lvl[0])
        dev_post = np.abs(s_post - self._baseline_lvl[1])

        e_front = np.abs(e[:, self.i_front]).mean(axis=1)
        emg = np.abs(self.fi_env.apply(e_front[:, None])[:, 0])

        ratio = (dev_front / np.maximum(dev_post, 1e-9) if self.i_post
                 else np.full(n, np.inf))
        self.t = t[-1]
        return t, dev_front, emg, ratio

    def push(self, chunk):
        """Feed a (n, n_ch) chunk. Returns a list of Events (empty during calibration)."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        t, bf, ef, ratio = self._features(chunk)

        if not self.base.locked:
            # Skip the filter startup transient, then accumulate rest statistics.
            if self._warm < self.cfg.warmup_sec:
                self._warm += len(chunk) / self.fs
                return []
            self.base.observe(np.column_stack([bf, ef]))
            self._raw_calib.append(chunk)
            if self.base.n_observed >= self.cfg.calib_sec * self.fs:
                self._finish_calibration()
            return []

        if not self.armed:
            return []

        z = self.base.z(np.column_stack([bf, ef]))
        events = []
        for i in range(len(t)):
            zb, ze, r = z[i, 0], z[i, 1], ratio[i]
            self.last_z, self.last_ratio = (zb, ze), r
            self._trace.append((t[i], zb, ze, r))
            if len(self._trace) > int(3 * self.fs):
                del self._trace[: len(self._trace) - int(3 * self.fs)]

            hit = self.sm["blink"].step(t[i], zb)
            if hit:
                ev = self._make("blink", *hit)
                if ev is not None:
                    events.append(ev)
            hit = self.sm["jaw"].step(t[i], ze)
            if hit:
                ev = self._make("jaw", *hit)
                if ev is not None:
                    events.append(ev)
        return events

    def _finish_calibration(self):
        """Lock the baseline and decide whether contact is good enough to arm."""
        raw = np.vstack(self._raw_calib)
        self._raw_calib = []
        self.base.lock()
        ok, reasons, metrics = quality.assess(raw, self.names, self.fs)
        self.quality, self.quality_reasons = metrics, reasons
        self.armed = ok or not self.cfg.require_quality

    def _make(self, kind, onset, peak, dur, peak_t):
        """Apply cross-feature gates; return an Event or None if rejected."""
        window = [s for s in self._trace if onset - 0.05 <= s[0] <= onset + dur + 0.05]
        if not window:
            return None
        ratio_at_peak = next((s[3] for s in window if abs(s[0] - peak_t) < 1e-9),
                             float(np.median([s[3] for s in window])))
        emg_peak = max(s[2] for s in window)
        blink_peak = max(s[1] for s in window)

        if kind == "blink":
            if self.cfg.use_ratio_gate and ratio_at_peak < self.cfg.min_ratio:
                self.rejected["ratio"] += 1
                return None
            if emg_peak > self.cfg.blink_emg_max_z:
                self.rejected["emg"] += 1
                return None
            long = dur >= self.cfg.long_dur
            return Event("blink_long" if long else "blink", onset, peak, dur,
                         ratio_at_peak, {"emg_z": emg_peak})
        # Jaw clench: EMG-dominant by construction; require it to beat the slow band.
        if blink_peak > peak * 1.5:
            self.rejected["slow"] += 1
            return None
        return Event("jaw", onset, peak, dur, ratio_at_peak, {"blink_z": blink_peak})
