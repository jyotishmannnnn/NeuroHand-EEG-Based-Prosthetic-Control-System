"""EEG sample sources: live LSL, CSV session replay, and synthetic.

All sources expose the same pull interface so the control chain can run without
the cap attached:

    src = SynthSource(duration=30)
    for chunk, stamps in src:     # chunk (n, 8) float64, stamps (n,) seconds
        ...

Amplitudes are in raw device counts, not microvolts. The cap's channel gain is
configurable and undocumented, so nothing downstream may assume a scale.
"""
import time
from pathlib import Path

import numpy as np

from . import CHANNELS, FS_EEG


class Source:
    """Base: yields (chunk, stamps) until exhausted."""

    fs = FS_EEG
    names = CHANNELS

    def pull(self):
        """Return (chunk, stamps) with chunk (n, n_ch); (None, None) when done."""
        raise NotImplementedError

    def __iter__(self):
        while True:
            chunk, stamps = self.pull()
            if chunk is None:
                return
            if len(chunk):
                yield chunk, stamps

    def close(self):
        pass


class LslSource(Source):
    """Live stream from QneuroNeuroAnalytics in STREAMING-LSL mode.

    Stream name defaults to '<user>_UnFilter' (unfiltered EEG) per the manual.
    """

    def __init__(self, name=None, user="Jyotishman", timeout=10.0, max_chunk=256):
        from pylsl import StreamInlet, resolve_byprop, resolve_streams

        if name:
            infos = resolve_byprop("name", name, timeout=timeout)
            if not infos:
                raise RuntimeError(
                    f"LSL stream {name!r} not found. Run tools/lsl_probe.py to list names.")
            info = infos[0]
        else:
            info = _pick_eeg_stream(resolve_streams(wait_time=timeout), user)
        self.stream_name = info.name()
        self.inlet = StreamInlet(info, max_chunklen=max_chunk, recover=False)
        info = self.inlet.info()
        self.fs = info.nominal_srate() or FS_EEG
        self.names = _lsl_labels(info) or CHANNELS[: info.channel_count()]

    def pull(self):
        samples, stamps = self.inlet.pull_chunk(timeout=0.2)
        if not samples:
            return np.empty((0, len(self.names))), np.empty(0)
        return np.asarray(samples, dtype=np.float64), np.asarray(stamps)

    def close(self):
        self.inlet.close_stream()


def _pick_eeg_stream(infos, user):
    """Choose the 8-channel EEG stream among the app's advertised streams.

    The manual documents '<UserName>_UnFilter', but the app actually advertises
    '<username>_EEG' with the username lower-cased, plus a bare 'EEG' duplicate.
    Rather than hard-code either, prefer an unfiltered name, fall back to _EEG,
    and match the user case-insensitively.
    """
    eeg = [i for i in infos if i.channel_count() >= 8
           and "PPG" not in i.name().upper() and "IMU" not in i.name().upper()
           and "MARKER" not in i.name().upper()]
    if not eeg:
        raise RuntimeError(
            "No 8-channel LSL stream found. Start NeuroAnalytics in STREAMING-LSL "
            "mode; run tools/lsl_probe.py to list names.")

    u = (user or "").lower()

    def rank(i):
        n = i.name().lower()
        return (0 if u and n.startswith(u) else 1,          # prefer this user's stream
                0 if "unfilter" in n else 1 if "eeg" in n else 2)

    return sorted(eeg, key=rank)[0]


def _lsl_labels(info):
    try:
        ch = info.desc().child("channels").child("channel")
        out = []
        while not ch.empty():
            out.append(ch.child_value("label"))
            ch = ch.next_sibling()
        return [c for c in out if c]
    except Exception:
        return []


class CsvSource(Source):
    """Replay UnfilteredData.csv from a saved LiveSession directory.

    realtime=False replays as fast as possible (for tests); realtime=True paces
    to wall clock so timing-dependent logic sees the same conditions as live.
    """

    def __init__(self, session_dir, chunk=25, realtime=False, start=0.0, stop=None,
                 speed=1.0):
        path = Path(session_dir)
        if path.is_dir():
            path = path / "UnfilteredData.csv"
        with open(path) as f:
            header = f.readline().strip().split(",")
        self.names = [c.replace("UF-", "").strip() for c in header[2:]]
        raw = np.genfromtxt(
            path, delimiter=",", skip_header=1,
            usecols=[0] + list(range(2, 2 + len(self.names))),
        )
        t, x = raw[:, 0], raw[:, 1:]
        keep = t >= start
        if stop is not None:
            keep &= t < stop
        self.t, self.x = t[keep], x[keep]
        self.fs = 1.0 / np.median(np.diff(self.t)) if len(self.t) > 1 else FS_EEG
        self.chunk = chunk
        self.realtime = realtime
        self.speed = max(float(speed), 1e-6)
        self.i = 0
        self._t0 = None

    def pull(self):
        if self.i >= len(self.x):
            return None, None
        j = min(self.i + self.chunk, len(self.x))
        chunk, stamps = self.x[self.i : j], self.t[self.i : j]
        self.i = j
        if self.realtime:
            if self._t0 is None:
                self._t0 = time.perf_counter() - stamps[0] / self.speed
            lag = (self._t0 + stamps[-1] / self.speed) - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        return chunk, stamps


class SynthSource(Source):
    """Synthetic 8-channel EEG with scheduled blink and jaw-clench events.

    Used to validate the detector against known ground truth. `events` is a list
    of (onset_seconds, kind) with kind in {'blink', 'jaw'}; the true onsets are
    kept in .events so tests can score detections. Kinds: 'blink', 'blink_long', 'jaw'.
    """

    def __init__(self, duration=60.0, fs=FS_EEG, events=(), chunk=25, realtime=False,
                 scale=300.0, alpha_amp=0.0, seed=0, mains=0.0, speed=1.0):
        self.fs = float(fs)
        self.names = list(CHANNELS)
        self.duration = float(duration)
        self.chunk = chunk
        self.realtime = realtime
        self.speed = max(float(speed), 1e-6)
        self.scale = scale
        self.alpha_amp = alpha_amp
        self.mains = mains
        self.events = sorted(events, key=lambda e: e[0])
        self.rng = np.random.default_rng(seed)
        self.n = int(duration * self.fs)
        self.i = 0
        self._t0 = None
        self.x = self._render()
        self.t = np.arange(self.n) / self.fs

    def _render(self):
        n, fs, nch = self.n, self.fs, len(self.names)
        t = np.arange(n) / fs
        # 1/f-ish background: integrate white noise then high-pass by detrending.
        w = self.rng.standard_normal((n, nch))
        bg = np.cumsum(w, axis=0)
        bg -= np.linspace(bg[0], bg[-1], n)
        bg /= max(np.std(bg), 1e-9)
        x = (0.7 * bg + 0.7 * w) * self.scale

        if self.alpha_amp:
            phase = self.rng.uniform(0, 2 * np.pi)
            alpha = np.sin(2 * np.pi * 10.0 * t + phase) * self.alpha_amp * self.scale
            for ch in ("O1", "O2"):
                x[:, self.names.index(ch)] += alpha
        if self.mains:
            hum = np.sin(2 * np.pi * 50.0 * t) * self.mains * self.scale
            x += hum[:, None]

        frontal = [self.names.index(c) for c in ("FP1", "FP2")]
        for onset, kind in self.events:
            i0 = int(onset * fs)
            if kind in ("blink", "blink_long"):
                # Monophasic deflection, frontal-dominant, ~10x background. A held
                # blink is the same shape sustained, which is how CONFIRM is made
                # distinguishable from CYCLE by duration alone.
                dur = int((0.85 if kind == "blink_long" else 0.30) * fs)
                shape = np.sin(np.pi * np.arange(dur) / dur) ** 2
                amp = 10.0 * self.scale
                for c in range(nch):
                    w_c = 1.0 if c in frontal else 0.15
                    _add(x, i0, shape * amp * w_c, c)
            elif kind == "jaw":
                # EMG burst: broadband 20-120 Hz, ~600 ms, frontal/temporal-dominant.
                dur = int(0.60 * fs)
                env = np.hanning(dur)
                burst = self.rng.standard_normal(dur) * env * 6.0 * self.scale
                for c in range(nch):
                    w_c = 1.0 if c in frontal else 0.4
                    _add(x, i0, burst * w_c, c)
            else:
                raise ValueError(f"unknown event kind {kind!r}")
        return x

    def pull(self):
        if self.i >= self.n:
            return None, None
        j = min(self.i + self.chunk, self.n)
        chunk, stamps = self.x[self.i : j], self.t[self.i : j]
        self.i = j
        if self.realtime:
            if self._t0 is None:
                self._t0 = time.perf_counter() - stamps[0] / self.speed
            lag = (self._t0 + stamps[-1] / self.speed) - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        return chunk, stamps


def _add(x, i0, seg, ch):
    n = min(len(seg), len(x) - i0)
    if n > 0:
        x[i0 : i0 + n, ch] += seg[:n]


def open_source(spec, **kw):
    """Build a source from a CLI-ish spec: 'lsl', 'synth', or a session path."""
    if spec == "lsl":
        return LslSource(**{k: v for k, v in kw.items() if k in ("name", "user", "timeout")})
    if spec == "synth":
        return SynthSource(**{k: v for k, v in kw.items()
                              if k in ("duration", "events", "realtime", "seed", "scale",
                                       "alpha_amp", "mains", "chunk", "speed")})
    return CsvSource(spec, **{k: v for k, v in kw.items()
                              if k in ("chunk", "realtime", "start", "stop", "speed")})
