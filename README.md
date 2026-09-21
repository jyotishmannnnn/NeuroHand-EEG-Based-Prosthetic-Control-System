# NeuroHand — EEG-Based Prosthetic Control System

Control a tendon-driven robotic hand from an 8-channel EEG cap. Brain/biosignal
input selects a discrete grasp primitive; the hand firmware executes it.

**Status:** the full control path is built and tested end to end offline. It has
**not** yet been driven by good EEG, because electrode contact on the cap is still
being solved (see [Signal quality](#signal-quality)). Everything below runs today
without the cap and without the hand.

---

## Hardware

| part | detail |
|---|---|
| EEG | Qneuro H100 "NeuroState" cap — 8 ch, 24-bit, 250 Hz, order `FP1 FP2 O1 O2 C3 CZ FZ C4` |
| reference | earlobe clip; PPG on the other earlobe |
| also on cap | PPG (R + IR, 100 Hz), IMU (quaternion + accel, 100 Hz) |
| link | Bluetooth (BLE serial style), device `NeuroState-XXXX` |
| hand | 8× N20 micro DC gearmotors, tendon-driven fingers |
| drivers | 4× DRV8833 (2 H-bridges each) |
| MCU | ESP32 — 16 LEDC PWM channels, exactly enough for 8 motors |
| power | stall ~0.7–1.6 A per motor → 2S LiPo + 6 V buck ≥5 A. Not USB. |

There is no vendor SDK and no BLE protocol documentation, so EEG is taken over
**LSL** (live) or **CSV** (saved sessions) rather than by reverse-engineering BLE.

## Architecture

```
cap --BLE--> NeuroAnalytics (STREAMING-LSL) --LSL--> Python --serial--> ESP32 --PWM--> 4x DRV8833 --> 8x N20
                                                        |                  ^
                                                   quality gate       watchdog: coast on link loss
```

| module | role |
|---|---|
| `neuro/source.py` | EEG input: live LSL, CSV session replay, or synthetic with scripted gestures |
| `neuro/dsp.py` | causal streaming filters (state carried across chunks), ring buffer, robust baseline |
| `neuro/quality.py` | contact check; refuses to arm on electrode noise |
| `neuro/detect.py` | Tier-1 gesture detection: blink, held blink, jaw clench |
| `neuro/decide.py` | gestures → hand commands (select-then-confirm) |
| `neuro/hand.py` | serial line protocol, heartbeat, in-process mock |
| `firmware/esp32_hand/` | ESP32 firmware; owns the grasp primitives |
| `config/hand.json` | single source of truth for pins, motor map, timings |
| `tools/` | session analysis, LSL probe, config generator |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Nothing attached: synthetic gestures through a mock hand
python run_tier1.py --source synth --mock --calib 6 --duration 30

# 2. Replay a recorded session
python run_tier1.py --source "<...>/LiveSession/09-57-47" --mock --calib 20

# 3. Live: app in STREAMING-LSL mode, ESP32 on COM5
python tools/lsl_probe.py            # confirm stream names first
python run_tier1.py --source lsl --port COM5

# validate the whole chain
python tests/test_chain.py
```

## Control model

Three gestures, deliberately select-then-confirm:

| gesture | command | effect |
|---|---|---|
| short blink | `CYCLE` | advance the highlighted grasp primitive |
| held blink (≥0.42 s) | `CONFIRM` | execute the highlighted primitive |
| jaw clench | `RELEASE` | open the hand now, cancel everything |

Primitives: `open`, `power`, `pinch`, `point`, `tripod`.

**Why this shape.** No single misdetection can actuate the hand — a stray blink
only moves a highlight. The one gesture that always acts is the most deliberate,
highest-amplitude one, and it acts in the safe direction (open).

**Why Tier 1 first.** A blink is an EOG dipole roughly 10× larger than cortical
EEG, and a jaw clench is EMG larger still, so both are detectable on `FP1`/`FP2`
alone while the occipital and central pads are still being fought with. Per-finger
intent is *not* recoverable from scalp EEG at all; that is why the firmware, not
the classifier, owns the primitives.

**Why held blink rather than double blink.** Double blink was tried first and does
not work here: the blink envelope is ~280 ms wide, so two blinks at their natural
250–400 ms spacing either merge into one detection or resolve too far apart to
pair. Hold duration is measured from a single event, needs no timing window, and
lets `CYCLE` fire immediately.

Roadmap tiers, easiest first: **1** blink/jaw (built) → **2** SSVEP (O1/O2,
flicker targets, FFT/CCA) → **3** alpha level → **4** motor imagery (C3/C4/CZ,
CSP+LDA or Riemannian) → **5** hybrid.

## Safety

These are pull-only tendons with no position feedback in the base N20 build, so a
motor left energised stalls against its tendon and burns.

1. **Watchdog.** No serial line for 1.5 s → every channel coasts, hand reported
   open. The host must keep sending heartbeats. Heartbeats alone cannot re-arm a
   tripped watchdog; an explicit command is required.
2. **Bounded actuation.** Every move carries a per-motor deadline from
   `config/hand.json`. A motor cannot physically be driven longer than its limit.
3. **Duty ramp.** Duty rises over 80 ms to limit inrush — 8 stalling N20s together
   can pull well over 10 A from a shared buck.
4. **Boot coasted.** A brownout reset mid-grip must not resume pulling.
5. **Quality gate.** The detector refuses to arm on bad electrode contact, so noise
   cannot reach the motors at all.
6. **Busy lockout.** Gestures made while tendons are moving are dropped, not queued.

## Signal quality

The gate exists because of a measured failure, not a hypothetical one. Replaying
session `09-57-47` — five floating electrodes and heavy mains hum — through an
ungated chain produced **10 motor commands in 181 s** of pure noise. Amplitude
alone cannot distinguish "blink" from "pad lifted off the scalp", so contact is
checked *before* the detector arms:

```
  ch        rms  50Hz/sig   flat%
 FP1      554.3     53.28     4.6     <- 50 Hz power 53x the 1-45 Hz signal
 FP2     6161.2     73.93     0.9     <- 74x
  O1     2438.9      0.13     0.6     \
  C3     2315.5      0.11     0.5      | near-identical RMS across five pads:
  CZ     2336.2      0.03     0.5      | floating, reading reference-drive noise
  FZ     2379.3      0.07     0.2     /
```

→ `NOT ARMED`, 0 commands. Acceptance targets for real EEG: bandpassed RMS 10–50,
50 Hz/signal < 0.05, alpha peak ~10 Hz at O1/O2 eyes-closed, EC/EO alpha ratio
> 1.5. `--force` overrides the gate for bring-up.

Before the next recording: unplug the laptop from mains and sit 1 m from
monitors/chargers; wet and tighten the ear-clip reference on cleaned skin; part
hair under each of O1/O2/C3/C4/CZ/FZ and press/wiggle with a saline drop; do not
proceed past the app's Signal Quality Index until all 8 read green.

## Hardware questions still open

`config/hand.json` holds placeholders so these stay parameters rather than
blockers: finger→motor map, encoder variant yes/no, motor voltage rating, which
driver boards are actually owned, and tendon return method (spring / antagonist /
elastic). Pin choices avoid GPIO 6–11 (flash), 34–39 (input-only) and 0/2/15
(strapping); GPIO 16/17 are consumed by PSRAM on WROVER modules, so remap motors
4/5 there.

Edit the JSON, then regenerate the firmware header — the test suite asserts the
two agree:

```bash
python tools/gen_hand_config.py
```

## Not in this repo

- **Screen recordings** (`*.mp4`, 333 MB) and extracted `frames/` — one file alone
  exceeds GitHub's 100 MB limit.
- **`NeuroAnalyticsWindows_28-05-2026/`** — Qneuro's licensed Unity application and
  manual. Vendor proprietary; obtain from Qneuro India Pvt. Ltd.
  (support@qneuro.com).
- **Recorded sessions** — they live under
  `%LOCALAPPDATA%Low/Qneuro/QneuroNeuroAnalytics/<user>/<date>/LiveSession/<time>/`.
