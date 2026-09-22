# Design decisions, and the measurements that forced them

Every choice below was made because a measurement demanded it, not by preference.
Each entry gives the decision, the evidence, and what it costs. If you are changing
this code, read the entry before reverting the thing it describes.

---

## 1. Thresholds are in baseline MAD units, never microvolts

**Decision.** Every detection threshold is expressed as a multiple of the median
absolute deviation of a rest baseline, learned per session.

**Why.** The cap's channel gain is configurable and the scale undocumented. Observed
band-passed RMS across sessions spans **162 to 22 000 counts** on the same hardware
and the same electrode. Any absolute threshold would be meaningless by the next
session.

**Cost.** Every run needs a rest block to learn from, and a contaminated baseline
silently raises the threshold. That happened: on session 10-25-53 the first 18 s
contained blinks, inflating the blink baseline MAD to 2491 counts.

---

## 2. Filtering is causal and carries state across chunks

**Decision.** `dsp.StreamFilter` holds `sosfilt` delay state between chunks. The test
suite asserts chunked output is bit-identical to filtering the whole signal at once.

**Why.** The original analysis tools in `tools/` use `filtfilt` / `sosfiltfilt`, which
are zero-phase but non-causal — they need the future. Fine offline, impossible live.
Reusing them in the live path would have produced results that could never be
reproduced in real time.

**Cost.** Causal filters have phase lag, so event onsets lag the true gesture. Measured
at 0.09–0.18 s on synthetic cues, which is irrelevant for grasp selection.

---

## 3. The blink feature removes DC with a running median, not a high-pass

**Decision.** Blink detection low-passes at 6 Hz and subtracts a 4 s running median.
No high-pass.

**Why.** A blink is a *monophasic* deflection. High-passing it produces a decaying
undershoot, which rectification turns into a second bump roughly 0.85 s later at
~15% of the primary peak — still several times threshold. Measured directly:
**5 injected blinks produced 11 detections.** Lowering the high-pass corner from
0.5 Hz to 0.1 Hz only improved the primary-to-echo ratio from 4.9× to 9.7×, leaving
the echo above threshold. A median has no impulse response to ring with: a 300 ms
blink barely moves the median of a 4 s window, so the echo disappears rather than
being filtered out.

**Cost.** Real, and it bit us. A median does not remove sub-Hz drift as well as a
high-pass. On drifty session 10-25-53 the blink feature's baseline MAD was **2491
counts against 589** for a plain 0.5–8 Hz bandpass — a 4× worse noise floor.
Re-adding a high-pass at 0.8 Hz cut the spread to 347 counts. That variant was tested
and did *not* recover the missing blinks, so it was not adopted; if a future session
shows drift-limited sensitivity with otherwise good contact, revisit it.

---

## 4. The threshold state machine is a Schmitt trigger

**Decision.** Entry and exit thresholds differ (`exit_frac=0.5`).

**Why.** A rectified envelope dips mid-gesture, so a single threshold fragments one
sustained hold into several events. Measured: **12 long blinks produced 24 detections**
without hysteresis.

**Cost.** Two nearby gestures are more likely to merge into one event.

---

## 5. CONFIRM is a held blink, not a double blink

**Decision.** Short blink cycles the selection; a blink held ≥0.42 s confirms it.

**Why.** Double blink was implemented first and does not work here. The blink envelope
is ~280 ms wide, so two blinks at their natural 250–400 ms spacing either merge into a
single detection or resolve too far apart to pair. Measured at three gaps: at 300 ms
only one event was detected; at 250 ms two were detected but resolved >0.5 s apart and
failed to pair. Duration is measured from a single event, needs no timing window, and
lets CYCLE fire immediately instead of waiting to see whether a second blink follows.

**Cost.** The user must hold a blink deliberately, which is less natural than
double-blinking — but it is detectable, which double-blinking was not.

---

## 6. Frontal deviation is taken per channel, then maxed

**Decision.** Each frontal channel gets its own median baseline; the feature is the
maximum of their deviations, not the deviation of their average.

**Why.** On session 09-18-27 the FP1 pad carried ongoing EEG — its alpha share dropped
normally on eye opening — but was not coupled to the eye dipole. Blink-band excursions
reached **60 MADs on FP2 and under 2 on FP1**, a ~90× difference. Averaging first
halves the gesture and doubles the noise; it lost a real held blink that per-channel
maxing recovered.

**Cost.** Slightly more sensitive to a single noisy frontal channel, since the max
takes whichever is largest.

---

## 7. The detector refuses to arm on bad contact

**Decision.** `quality.assess` runs on the calibration window; if the frontal channels
fail, the detector produces no events at all.

**Why.** Replaying session 09-57-47 — five floating electrodes, heavy mains — through
an ungated chain produced **10 motor commands in 181 s of pure noise**. Amplitude alone
cannot distinguish "blink" from "pad lifted off the scalp", so the question has to be
asked before the detector arms, not after.

**Cost.** A hard stop when contact is marginal. `--force` overrides it for bring-up,
and the refusal always prints the measured reason.

---

## 8. The ratio gate only compares against pads that pass QC

**Decision.** The frontal/posterior amplitude ratio gate narrows its posterior
reference set at calibration to channels passing contact QC. If none qualify it says so
and stops gating.

**Why.** The gate asks "is this excursion frontally dominant?", which assumes the
posterior pads are on the scalp. On session 09-33-34 six posterior pads were floating
at ~10× the frontal amplitude, so every genuine blink scored a ratio below 1 and the
gate **rejected 13 real events** — blink-type detections fell from 7 to 2.

**Cost.** Deliberate and significant: when the posterior pads are bad, artifact
rejection is weakened rather than merely degraded. The gate is the main defence against
cable sway and electrode pops being read as gestures, so a disabled gate is a reason to
fix the pads, not to proceed. It is reported loudly in both the GUI and the CLI.

---

## 9. Select-then-confirm, with the abort on the biggest gesture

**Decision.** Short blink = CYCLE (highlight only, sends nothing), held blink =
CONFIRM (executes), jaw clench = RELEASE (opens the hand, overrides the busy lockout).

**Why.** No single misdetection can actuate the hand: a stray blink only moves a
highlight. The one gesture that always acts is the most deliberate and
highest-amplitude one, and it acts in the safe direction. Gestures arriving while
tendons are still moving are dropped rather than queued.

**Cost.** Two deliberate actions to execute a grasp.

---

## 10. The firmware owns the primitives, and coasts on link loss

**Decision.** The ESP32 holds the grasp table; the host only sends a primitive name.
No serial line for 1.5 s and every channel coasts. Heartbeats alone cannot re-arm a
tripped watchdog — an explicit command is required. Every move has a per-motor
deadline, duty ramps over 80 ms, and boot state is coasted.

**Why.** These are pull-only tendons with no position feedback in the base N20 build.
A motor left energised stalls against its tendon and burns. Host crashes, USB unplugs
and Python exceptions are all routine; none of them may leave a motor driving. Eight
stalling N20s together can pull well over 10 A from a shared buck, hence the ramp.
A brownout reset mid-grip must not resume pulling, hence coasted boot.

**Cost.** The host must send heartbeats continuously, and grasp timing cannot be tuned
from the host without reflashing — `config/hand.json` regenerates the firmware header
instead, and the test suite asserts the two agree.

---

## 11. The software issues the cues and records them

**Decision.** `run_protocol.py` displays the cue schedule, records the raw stream
alongside the exact cue times, and scores itself.

**Why.** Three recordings were lost to timing uncertainty. A 300 ms blink cannot be
located inside a 95 s file from memory, and asking the operator to keep timestamps by
hand does not close the gap — session 10-25-53 still needed a 13 s lead-in recovered by
search. With cues recorded, ground truth is exact by construction and scoring is
mechanical. Validated end to end on synthetic gestures placed at the cue times:
20/20 hits, 0 false alarms, lags 0.08–0.18 s.

The schedule opens with a 20 s rest block because every earlier session began straight
into gestures, leaving the detector no clean window to calibrate from.

**Cost.** The operator is paced by the software rather than working freely.

---

## 12. Scoring separates "wrong kind" from "missed"

**Decision.** A cue matched by a detection of a different gesture counts as
`wrong_kind`, not as a miss.

**Why.** "Detected but called a jaw clench" and "saw nothing at all" have different
causes and different fixes. Collapsing them hides which one is happening. On session
10-25-53 this distinction is what showed the blink block was empty rather than
misclassified.

---

## Open tension

Items 3 and 7 pull against each other on drifty recordings. The median baseline avoids
the echo but tolerates drift; the quality gate then sees an inflated noise floor and
may refuse to arm, or the threshold rides too high to catch real gestures. The current
resolution is to fix acquisition rather than tune the detector, because every attempt
to recover gestures by tuning — lowering `blink_z` to 4, adding high-pass corners at
0.3/0.5/0.8 Hz — has failed to find gestures that good contact would have made obvious.
