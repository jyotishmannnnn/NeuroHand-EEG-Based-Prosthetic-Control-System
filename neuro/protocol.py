"""Cued gesture protocol: schedule, and scoring of detections against cues.

Three recordings were lost to timing uncertainty. Asking the operator to hold
timestamps by hand does not work, because a 300 ms blink has to be located
inside a 95 s file. So the software issues the cues and records when it issued
them; ground truth is then exact by construction and scoring is mechanical.

The schedule deliberately opens with a rest block. Every earlier session began
with gestures, which left the detector no clean window to learn its baseline
from and forced the analysis to go hunting for a quiet stretch afterwards.
"""
from dataclasses import dataclass, field

BLINK, HOLD, JAW, REST = "blink", "blink_long", "jaw", "rest"

LABEL = {
    BLINK: "BLINK once",
    HOLD: "HOLD blink ~1 s",
    JAW: "CLENCH jaw ~1 s",
    REST: "rest - still, eyes open",
}

# How close a detection must be to its cue to count, per cue kind. The jaw
# window is wider because onset is read off a smoothed EMG envelope that lags.
TOL = {BLINK: 1.0, HOLD: 1.5, JAW: 1.5}


@dataclass
class Cue:
    t: float          # stream time the operator is asked to act
    kind: str


@dataclass
class Block:
    kind: str
    n: int
    spacing: float
    t0: float = 0.0

    @property
    def dur(self):
        return self.spacing * self.n if self.kind != REST else self.spacing


def default_blocks(calib=20.0):
    """Rest for calibration, then blinks, held blinks and jaw clenches."""
    return [
        Block(REST, 1, calib),
        Block(BLINK, 10, 3.0),
        Block(REST, 1, 10.0),
        Block(HOLD, 5, 4.0),
        Block(REST, 1, 10.0),
        Block(JAW, 5, 4.0),
        Block(REST, 1, 5.0),
    ]


def build(blocks=None, calib=20.0):
    """Return (cues, blocks_with_t0, total_seconds)."""
    blocks = list(blocks or default_blocks(calib))
    cues, t = [], 0.0
    for b in blocks:
        b.t0 = t
        if b.kind != REST:
            # First cue sits one spacing into the block so there is a beat of
            # warning after the banner changes.
            for i in range(b.n):
                cues.append(Cue(t + i * b.spacing + b.spacing * 0.5, b.kind))
        t += b.dur
    return cues, blocks, t


def block_at(blocks, t):
    for b in blocks:
        if b.t0 <= t < b.t0 + b.dur:
            return b
    return None


@dataclass
class Score:
    hits: int = 0
    missed: int = 0
    wrong_kind: int = 0
    false_alarms: int = 0
    detail: list = field(default_factory=list)


def score(cues, events, tol=None):
    """Match detections to cues. Returns (per_kind, overall).

    A detection counts once. 'wrong_kind' means something was detected at the
    cue but classified as another gesture, which is a different failure from
    detecting nothing and is worth separating.
    """
    tol = tol or TOL
    used = set()
    per = {}
    for c in cues:
        s = per.setdefault(c.kind, Score())
        w = tol.get(c.kind, 1.0)
        near = [(i, e) for i, e in enumerate(events)
                if i not in used and abs(e.t - c.t) <= w]
        exact = [(i, e) for i, e in near if e.kind == c.kind]
        if exact:
            i, e = min(exact, key=lambda ie: abs(ie[1].t - c.t))
            used.add(i)
            s.hits += 1
            s.detail.append((c.t, c.kind, e.kind, round(e.t - c.t, 2), round(e.peak_z, 1)))
        elif near:
            i, e = min(near, key=lambda ie: abs(ie[1].t - c.t))
            used.add(i)
            s.wrong_kind += 1
            s.detail.append((c.t, c.kind, e.kind, round(e.t - c.t, 2), round(e.peak_z, 1)))
        else:
            s.missed += 1
            s.detail.append((c.t, c.kind, None, None, None))

    overall = Score()
    for s in per.values():
        overall.hits += s.hits
        overall.missed += s.missed
        overall.wrong_kind += s.wrong_kind
    overall.false_alarms = sum(1 for i in range(len(events)) if i not in used)
    return per, overall


def report(cues, events, blocks=None, tol=None):
    per, overall = score(cues, events, tol)
    L = []
    w = L.append
    w(f"{'cue kind':>12} {'cued':>5} {'hit':>5} {'wrong':>6} {'missed':>7} {'rate':>7}")
    for kind in (BLINK, HOLD, JAW):
        s = per.get(kind)
        if not s:
            continue
        n = s.hits + s.missed + s.wrong_kind
        w(f"{kind:>12} {n:5d} {s.hits:5d} {s.wrong_kind:6d} {s.missed:7d} "
          f"{s.hits / max(n, 1):6.0%}")
    n_all = overall.hits + overall.missed + overall.wrong_kind
    w(f"{'TOTAL':>12} {n_all:5d} {overall.hits:5d} {overall.wrong_kind:6d} "
      f"{overall.missed:7d} {overall.hits / max(n_all, 1):6.0%}")
    w(f"false alarms (detections matching no cue): {overall.false_alarms}")

    w("")
    w("per-cue detail  (cue_t, cued, detected, lag_s, z)")
    for kind in (BLINK, HOLD, JAW):
        s = per.get(kind)
        if not s:
            continue
        for ct, ck, dk, lag, z in s.detail:
            mark = "ok  " if dk == ck else ("KIND" if dk else "MISS")
            w(f"  {mark} t={ct:6.1f}s cued={ck:10s} got={str(dk):10s} "
              f"lag={'' if lag is None else f'{lag:+.2f}s':>7} "
              f"z={'' if z is None else f'{z:.1f}'}")
    return "\n".join(L)
