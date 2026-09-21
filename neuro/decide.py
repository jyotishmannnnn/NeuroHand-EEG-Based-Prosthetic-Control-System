"""Turn detected gesture events into hand commands.

Control model is select-then-confirm, which is what makes a noisy 3-gesture
channel safe to drive motors with:

    short blink   -> CYCLE    advance the highlighted grasp primitive
    long blink    -> CONFIRM  execute the highlighted primitive
    jaw clench    -> RELEASE  open the hand now, cancel everything

No single misdetection can actuate the hand: a stray blink only moves a
highlight, and the one gesture that always acts is the deliberate, high-amplitude
one chosen as the abort. A CONFIRM is followed by a busy lockout so gestures made
while tendons are still moving are dropped rather than queued.

CONFIRM is a held blink rather than a double blink. Double-blink was tried first
and is not reliably detectable here: the blink envelope is roughly 280 ms wide, so
two blinks 250-400 ms apart -- the natural spacing -- either merge into a single
detection or resolve too far apart to pair. Hold duration is measured directly
from one event, needs no timing window, and lets CYCLE fire immediately instead
of waiting to see whether a second blink follows.
"""
from dataclasses import dataclass

PRIMITIVES = ("open", "power", "pinch", "point", "tripod")


@dataclass
class Command:
    kind: str        # 'cycle' | 'confirm' | 'release'
    t: float
    primitive: str   # highlighted primitive after the command resolved

    def __repr__(self):
        return f"Command({self.kind} -> {self.primitive} @ {self.t:.2f}s)"


@dataclass
class DecideConfig:
    busy_sec: float = 1.20        # lockout while the hand executes a primitive
    min_peak_z: float = 0.0       # optional extra confidence floor on events
    primitives: tuple = PRIMITIVES


class Decider:
    """Event stream -> Command stream."""

    def __init__(self, cfg=None, start=0):
        self.cfg = cfg or DecideConfig()
        self.sel = int(start)
        self.busy_until = -1e9
        self.dropped = 0

    @property
    def primitive(self):
        return self.cfg.primitives[self.sel % len(self.cfg.primitives)]

    def feed(self, events, now=None):
        """Consume events; return commands ready to send."""
        out = []
        for ev in events:
            if ev.peak_z < self.cfg.min_peak_z:
                continue
            out.extend(self._on_event(ev))
        return out

    def _on_event(self, ev):
        # RELEASE always wins: it clears the lockout rather than being dropped by it.
        if ev.kind == "jaw":
            self.busy_until = ev.t + self.cfg.busy_sec
            self.sel = self.cfg.primitives.index("open") if "open" in self.cfg.primitives else 0
            return [Command("release", ev.t, self.primitive)]

        if ev.t < self.busy_until:
            self.dropped += 1
            return []

        if ev.kind == "blink":
            self.sel = (self.sel + 1) % len(self.cfg.primitives)
            return [Command("cycle", ev.t, self.primitive)]

        if ev.kind == "blink_long":
            self.busy_until = ev.t + self.cfg.busy_sec
            return [Command("confirm", ev.t, self.primitive)]
        return []
