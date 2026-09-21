"""Causal streaming filters, ring buffer, and robust baseline statistics.

The offline tools in tools/ use filtfilt/sosfiltfilt, which are zero-phase but
non-causal and therefore unusable live. Everything here is single-pass sosfilt
carrying its delay state across chunks, so a chunked stream produces bit-identical
output to filtering the same signal in one shot.
"""
import numpy as np
from scipy import signal

from . import FS_EEG


class StreamFilter:
    """Cascaded SOS filter holding per-channel delay state between chunks."""

    def __init__(self, sos, n_ch):
        self.sos = np.asarray(sos, dtype=np.float64)
        self.n_ch = n_ch
        self.zi = np.zeros((self.sos.shape[0], 2, n_ch))

    def apply(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        out, self.zi = signal.sosfilt(self.sos, chunk, axis=0, zi=self.zi)
        return out

    def reset(self):
        self.zi[:] = 0.0


def bandpass_sos(lo, hi, fs=FS_EEG, order=4):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.98)
    return signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def lowpass_sos(hi, fs=FS_EEG, order=4):
    return signal.butter(order, min(hi, fs / 2 * 0.98), btype="lowpass", fs=fs, output="sos")


def notch_sos(f0=50.0, q=30.0, fs=FS_EEG, harmonics=(1, 2)):
    """Mains notch at f0 and its harmonics below Nyquist."""
    secs = []
    for h in harmonics:
        f = f0 * h
        if f < fs / 2 * 0.95:
            b, a = signal.iirnotch(f, q, fs)
            secs.append(signal.tf2sos(b, a))
    if not secs:
        return np.zeros((0, 6))
    return np.vstack(secs)


def chain_sos(*sos_list):
    parts = [s for s in sos_list if s is not None and len(s)]
    return np.vstack(parts) if parts else np.zeros((0, 6))


class Ring:
    """Fixed-length multi-channel ring buffer over the most recent samples."""

    def __init__(self, n, n_ch):
        self.buf = np.zeros((n, n_ch))
        self.n = n
        self.filled = 0

    def push(self, chunk):
        m = len(chunk)
        if m >= self.n:
            self.buf[:] = chunk[-self.n :]
            self.filled = self.n
            return
        self.buf[:-m] = self.buf[m:]
        self.buf[-m:] = chunk
        self.filled = min(self.n, self.filled + m)

    @property
    def ready(self):
        return self.filled >= self.n

    def window(self, n=None):
        """Most recent n samples (default: whole buffer)."""
        if n is None or n >= self.n:
            return self.buf
        return self.buf[-n:]


def bandpower(win, fs=FS_EEG, lo=8.0, hi=13.0):
    """Mean PSD in [lo, hi) via Hann-windowed rFFT. Returns (n_ch,)."""
    win = np.asarray(win, dtype=np.float64)
    if win.ndim == 1:
        win = win[:, None]
    n = len(win)
    w = np.hanning(n)[:, None]
    seg = (win - win.mean(axis=0)) * w
    spec = np.abs(np.fft.rfft(seg, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    m = (freqs >= lo) & (freqs < hi)
    if not m.any():
        return np.zeros(win.shape[1])
    norm = (w**2).sum() * fs
    return spec[m].mean(axis=0) / norm


def mad(x, axis=0):
    """Median absolute deviation scaled to be a std estimate for Gaussian data."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


class Baseline:
    """Robust per-channel centre and spread, learned during calibration.

    Thresholds must be expressed in these units rather than microvolts: the cap's
    channel gain is configurable and the CSV/LSL scale is undocumented, so the
    same gesture spans two orders of magnitude between sessions (observed RMS
    ranged 162 to 22000 across recordings).
    """

    def __init__(self, n_ch):
        self.n_ch = n_ch
        self.center = np.zeros(n_ch)
        self.spread = np.ones(n_ch)
        self._acc = []
        self.locked = False

    def observe(self, chunk):
        if not self.locked:
            self._acc.append(np.asarray(chunk, dtype=np.float64))

    @property
    def n_observed(self):
        return sum(len(c) for c in self._acc)

    def lock(self):
        if not self._acc:
            raise RuntimeError("no calibration samples observed")
        x = np.vstack(self._acc)
        self.center = np.median(x, axis=0)
        self.spread = np.maximum(mad(x, axis=0), 1e-9)
        self._acc = []
        self.locked = True
        return self

    def z(self, x):
        return (np.asarray(x, dtype=np.float64) - self.center) / self.spread
