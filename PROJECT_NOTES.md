# NeuroState BCI → Bionic Hand: Project Notes

Last updated: 2026-09-22

Lab notebook: hardware facts, data access, session log, and what to do next.
Design rationale for the signal chain lives in [docs/DESIGN.md](docs/DESIGN.md);
per-session analysis lives in [reports/](reports/).

## Goal

Control a robotic hand (8× N20 micro DC gearmotors, tendon-driven fingers) from the
NeuroState EEG cap. Discrete grasp primitives selected by brain/biosignal, executed by
hand firmware.

## Status

The control chain is built, tested and pushed: EEG → filters → gesture detection →
command → serial → ESP32 firmware, plus a live GUI monitor and a self-scoring cued
protocol. It has not yet driven a hand, and the gesture set is not yet validated
against real intent.

As of the latest session the blocker is electrode placement, not code:

- **Jaw clench works.** Session 10-25-53 detected 5/5 cued clenches with the block
  autocorrelating at r=0.485 on a 3.98 s lag against the intended 4 s spacing.
- **Blinks reach FP1/FP2 again** (11-25-50) after moving the pads lower: a five-blink
  train on both pads. On 10-25-53 there were zero detections at any threshold. The
  frontal noise floor is now 3–5× worse, so the pads need pressing in or re-wetting.
- **Posterior pads stopped floating** on 11-25-50, but mains there is still two to
  three orders of magnitude above the signal.
- **First `run_protocol.py` runs (09-22/23) scored 0/20 twice.** Run 235229 had the
  best frontal contact recorded (FP1 rms 105, 50Hz/sig 0.03). Its cue-locked averages
  show blinks absent from the signal, and jaw present but weak (2.67× vs 1.26× rest
  null). Movement during calibration hid the jaw clenches from the detector. Run
  000722 was swamped by common-mode noise from a lost reference.
- **Detector calibration has no guard against movement.** A contaminated first 20 s
  inflates the baseline and silences detection for the whole run. Not yet fixed.
- **Occipital alpha is achievable**: session 09-18-27 produced a clean 8.5 Hz rhythm
  with an eyes-closed/eyes-open ratio of 2.65–3.49 against a >1.5 target.
- Contact regresses between sessions, so it must be verified live before every
  recording.

## Hardware: Qneuro H100 "NeuroState" cap

- Vendor: Qneuro India Pvt. Ltd. (support@qneuro.com). Manual:
  `NeuroAnalyticsWindows_28-05-2026/NeuroAnalytics manual.pdf`.
- EEG: 8 channels, 24-bit, 250 Hz. Order in all files: **FP1, FP2, O1, O2, C3, CZ, FZ, C4**.
  Reference = one earlobe clip. PPG sensor = other earlobe.
- PPG: R + IR, 100 Hz. IMU: quaternion (QX,QY,QZ,QW) + accel (AX,AY,AZ), 100 Hz.
- Link: Bluetooth (BLE serial style). USB option exists in app. Device name `NeuroState-XXXX`.
- Firmware commands seen in log: configure(11), PPG calib(235), set channel gain(8),
  start stream(170), stop stream(255). Gain is configurable.
- **Amplitudes are raw device counts, not microvolts.** Gain is configurable and the
  scale undocumented; observed RMS spans 162 to 22000 across sessions. Nothing
  downstream may assume a scale — thresholds are expressed in baseline MAD units.

## Software: QneuroNeuroAnalytics (Unity 2022.3, Mono, licensed)

- Folder: `NeuroAnalyticsWindows_28-05-2026/NeuroAnalyticsWindows/`.
- Modes: SINGLE USER, STREAMING-LSL, VIEWING-LSL.
- Bundles: lsl.dll (Lab Streaming Layer), onnxruntime, Riemannian libs
  (libcovariance, libtangentSpace), ASR, Butterworth, HR/HRV/SpO2/RR, EEG/PPG SQI.
  ONNX models: `a2_covariance_model.onnx`, `Generalized_RPA_validate_at_0_linearSVR.onnx`
  (Riemannian Procrustes Analysis + linear SVR = small classical models, not deep nets).
- App default EEG filter: Butterworth bandpass 0.5–40 Hz order 6. PPG 0.5–5 Hz order 4.
- Signal Quality Index screen before streaming: per-electrode Low/Medium/High.
- No SDK, no BLE protocol doc. Do not reverse BLE; use LSL or CSV.
- Unity player log: `%LOCALAPPDATA%Low/Qneuro/QneuroNeuroAnalytics/Player.log`
  (previous run in `Player-prev.log`). Shows the command sequence, no impedance data.

## Raw data access (verified)

### CSV (auto-saved every session)

`C:\Users\JYOTISHMAN\AppData\LocalLow\Qneuro\QneuroNeuroAnalytics\<UserName>\<MM-DD-YYYY>\LiveSession\<HH-MM-SS>\`

App username in use: `Jyotishman`.

| file | header |
|---|---|
| UnfilteredData.csv | PseudoTimeSec,PseudoTimestamp,UF-FP1,UF-FP2,UF-O1,UF-O2,UF-C3,UF-CZ,UF-FZ,UF-C4 |
| Filtered_runtime.csv | PseudoTimeSec,PseudoTimestamp,F-FP1,…,F-C4 |
| Filtered_offline_BP(...)_BS(...).csv | PseudoTimeSec,F-FP1,…,F-C4 (offline re-filter incl. 48–62 Hz bandstop) |
| PPGData.csv | PseudoTimeSec,PseudoTimestamp,R,IR |
| IMUData.csv | PseudoTimeSec,PseudoTimestamp,QX,QY,QZ,QW,AX,AY,AZ |
| BandData.csv | SystemTimestamp, Delta/Theta/Alpha/Beta/Gamma × 8 ch |
| FFTData.csv, HRData.csv, HRVTDData.csv, HRVNLData.csv, SPO2Data.csv, RRData.csv, Attention.csv, CognitiveLoad.csv | derived metrics |
| Session.json (in user folder) | session index |

PseudoTimeSec step = 0.004 s (250 Hz) for EEG, 0.01 s (100 Hz) for PPG/IMU.
Timestamp format `MM-DD-YYYY HH:mm:ss:fff`.

The offline export (`Filtered_offline_...`) applies a 48–62 Hz bandstop and is useful
for checking whether real EEG survives under mains hum. It is not a substitute for
fixing acquisition.

### LSL (live) — verified working 2026-09-22

Start the app in STREAMING-LSL mode. **The manual is wrong about the names.** It
documents `<UserName>_UnFilter`; what the app actually advertises is:

| stream | ch | rate | note |
|---|---|---|---|
| `jyotishman_EEG` | 8 | 250 Hz | EEG, user name **lower-cased**, carries FP1…C4 labels |
| `jyotishman_PPG` | 2 | 100 Hz | |
| `jyotishman_IMU` | 7 | 100 Hz | |
| `jyotishman_Marker` | 1 | 0 Hz | string markers, unused so far |
| `EEG` | 8 | 250 Hz | duplicate of the EEG stream, identical samples |

There is no `_UnFilter` stream. The `_EEG` samples carry large DC offsets
(≈ −100 000 to −230 000 counts), so it is the unfiltered signal despite the name.
`neuro.source.LslSource` therefore auto-picks the 8-channel stream, matching the user
prefix case-insensitively and preferring an unfiltered name; `--stream` overrides.

Measured throughput: 254 Sa/s against a 250 Hz nominal rate.

### Python environment

Python 3.13 (Anaconda). numpy 2.4.6, scipy 1.17.1, pyserial 3.5, pylsl 1.18.4,
PySide6 6.11.1, pyqtgraph 0.14.0, matplotlib 3.10.9, opencv.

## Tools

| script | purpose |
|---|---|
| `run_monitor.py` | live GUI: contact table, traces, spectrum, feature z-scores. Doubles as the electrode-contact tool |
| `run_protocol.py` | cued gesture session — issues cues, records them, scores itself |
| `run_tier1.py` | headless Tier-1 runner: gestures → hand commands |
| `tools/lsl_probe.py` | discover LSL streams, print labels + live sample rates |
| `tools/analyze_session.py` | per-channel quality from UnfilteredData.csv: RMS, flat%, 50 Hz ratio, alpha ratio, cross-channel corr |
| `tools/alpha_ec_eo.py` | eyes-closed vs eyes-open alpha (Berger test) |
| `tools/cue_average.py` | cue-locked raw/EMG averages of a protocol npz against a rest null; tells whether a gesture is in the signal at all, independent of the detector |
| `tools/gen_hand_config.py` | generate `firmware/esp32_hand/config.h` from `config/hand.json` |
| `tools/extract_frames.py` | sample JPEG frames from a screen-recording mp4 |

## Session log

Acceptance targets: bandpassed RMS 10–50, 50 Hz/signal < 0.05, alpha peak ~10 Hz at
O1/O2 eyes closed, EC/EO alpha ratio > 1.5.

Note on units: early entries quote "50 Hz share" as a percentage of band power; later
entries quote the 50 Hz-to-signal **ratio**, which can exceed 1. Both describe the same
measurement.

| date | session | dur | result |
|---|---|---|---|
| 09-20 | 06-33-45 | ~75 s | first test |
| 09-20 | 06-36-26 | 0 s | empty |
| 09-20 | 06-37-30 | 624 s | RMS 11k–16k, 50 Hz 4–85%, no alpha. All channels railing. |
| 09-20 | 08-32-02 | 872 s | RMS 12k–22k, no alpha. Cross-ch corr 0.47. |
| 09-20 | 09-56-36 | short | aborted |
| 09-20 | 09-57-47 | 181 s | EC/EO test. O1,O2,C3,CZ,FZ near-identical → floating. EC/EO 0.04 (artifact). Ungated, this drove **10 motor commands from pure noise** — the reason the quality gate exists. |
| 09-21 | 09-18-27 | 155 s | **First real EEG.** Cross-ch corr 0.25, occipital peak 8.5 Hz, EC/EO 2.65–3.49. FP1 carried EEG but not the eye dipole (blink-band 0.2–1.5 MAD vs FP2's 11–60). |
| 09-22 | 09-12-53 | 0 s | aborted |
| 09-22 | 09-13-19 | 105 s | **Regression.** Mains 4–5× worse in counts (O1 30×), no alpha (peak 4.2 Hz), FP1–FP2 corr −0.02 → +0.71. Gesture trains provably absent (r = −0.012 at the 3 s lag). Unusable. |
| 09-22 | 09-27-40 | 337 s | LSL session open during live probing, not a protocol run |
| 09-22 | 09-33-34 | 95 s | **Best frontal contact.** FP1 0.5%, FP2 3.8% mains; gate armed clean for the first time. Six posterior pads floating (r = 1.00). Exposed the ratio-gate bug. Three blink-like events at 84–89 s. |
| 09-22 | 10-25-53 | 126 s | **Jaw block validated**: 5/5 clenches, r=0.485 at 3.98 s lag. Blinks absent at every threshold. 13 s lead-in. |
| 09-22 | 11-25-50 | 68 s | **Blinks back on FP1/FP2** after lowering the pads: five-blink train at ~2.75 s. Frontal bp_rms 1600–2500 (3–5× worse). Posterior no longer floating (bp_rms 35–85, corr 0.27) but 50 Hz still 100–1500× signal. 15–35 s lost to cap movement. No alpha. |
| 09-22 | 11-39-58 | 101 s | Contact check, app only (no protocol ran). Frontal bp_rms 264/283, best since 09-33-34; detector arms. Posterior 50 Hz still 500–1200× signal. PPG/IMU dropped out. |
| 09-22 | 11-49-40 | 360 s | App-side CSV of protocol run 20260922_235229 (LSL = CSV t 53.9–169.1 s). **0/20.** Blinks absent in cue averages, jaw weak but present (2.67× vs 1.26× null). Calibration contaminated by early movement. |
| 09-23 | protocol 000722 | 115 s | **Unusable:** broadband common-mode noise +30–60 dB on all channels (step corr 0.74). Reference ear clip lost contact. |

PPG (HR ~70–80 bpm, SpO2 ~99%) and IMU good in every session.

## Current diagnosis

1. **Frontal pads are not coupled to the eye dipole.** Jaw clenches come through on
   FP1/FP2 while blinks do not, on the same electrodes in the same recording. Jaw is
   read from the 25–110 Hz EMG band, which couples through mediocre skin contact; a
   blink is a slow EOG dipole needing the pad near the eye. Pads riding up toward the
   hairline keep the EMG and lose the dipole. Same pattern FP1 showed on 09-18-27.
2. **Posterior pads keep floating.** O1/O2/C3/CZ/FZ repeatedly show r ≈ 1.00 with one
   another at ~10× the frontal amplitude — six channels reading one signal.
3. **Mains varies session to session**, from 0.5% to 7180% of signal on the same
   electrode within one evening. Battery power and distance from monitors matter.

## Before the next recording

1. **Position** FP1/FP2 low on the forehead, ~1–2 cm above the eyebrows, not at the
   hairline. This is the current blocker.
2. Re-seat O1/O2/C3/C4/CZ/FZ: part hair, saline drop, press and hold.
3. Laptop on battery, 1 m from monitors and chargers. Re-wet the ear-clip reference on
   cleaned skin. Check the forehead pads are not bridged by excess saline.
4. **Verify live, which takes ten seconds** — do not record blind:
   ```
   python run_monitor.py --source lsl --stream jyotishman_EEG
   ```
   - blink hard three times: the blink-z trace must tower over the dashed threshold
   - close your eyes 10 s: an 8–10 Hz bump must appear at O1/O2 in the spectrum
   - 50Hz/sig under ~0.05 on FP1/FP2
   Contact demonstrably regresses between sessions; both checks passed on 09-18-27 and
   neither passed on 09-13-19.
5. Then record with the cued runner, so cue times are recorded rather than reconstructed:
   ```
   python run_protocol.py --stream jyotishman_EEG
   ```

## Control architecture

```
cap --BLE--> NeuroAnalytics (STREAMING-LSL) --LSL--> Python --serial--> ESP32 --PWM--> 4x DRV8833 --> 8x N20
                                                        |                  ^
                                                   quality gate       watchdog: coast on link loss
```

Pipeline: pull 250 Hz EEG → causal notch + band filters → features → threshold state
machines → command with confidence, dwell and lockout → hand primitive.

### Control tiers (easiest first)

1. **Blink / jaw clench** (FP1/FP2 threshold). No ML. **Built and tested**; jaw
   validated on real data, blink pending electrode placement.
2. SSVEP (O1/O2, flicker targets on screen). FFT/CCA. 4–6 commands. No ML.
3. Alpha level (relax → open). Band-power threshold. Feasible now that alpha is real.
4. Motor imagery left/right hand (C3/C4/CZ). CSP+LDA or Riemannian+LR, per-session
   training. 2–3 classes, 65–80%. Needs the posterior pads fixed.
5. Hybrid: MI for intent, blink confirm, IMU head tilt to cycle primitives.

Per-finger decoding from scalp EEG: not feasible. Hand firmware owns grasp primitives
(open, power grip, pinch, point, tripod); the BCI only selects a primitive.

### Hand side (to decide)

- N20 = brushed DC gearmotor, no position sensing unless encoder variant. Tendon = pull only.
- Drivers: 4× DRV8833 (2 H-bridges each) or 2× TB6612FNG. MCU: ESP32 (BLE/WiFi, 16 PWM).
- Feedback: N20 encoder variant (best), or current-sense stall detect (INA219 / driver sense), or open-loop timed (fragile).
- Power: stall ~0.7–1.6 A per motor → 2S LiPo + 6 V buck ≥5 A. Not USB.
- Open questions, all parameterised in `config/hand.json` so they do not block:
  finger→motor map, encoder yes/no, motor voltage rating, boards owned, tendon return
  (spring/antagonist/elastic).

## Misc

- `EEG_Recording.mp4` (96 s, EEG tab), `EEG_Recording1.mp4` (25 s, Analysis tab) and
  `EEG_Recording2.mp4` (179 s, EC/EO test) are screen recordings only. Frames in `frames/`.
  Excluded from git: 333 MB, and one file exceeds GitHub's 100 MB limit.
- The original `EEG_Recording2.mp4` upload was a byte-identical duplicate of
  `EEG_Recording.mp4`; later replaced by the real EC/EO recording.
- `NeuroAnalyticsWindows_28-05-2026/` and `tools/assembly_userstrings.txt` are excluded
  from git: vendor-proprietary application, and a string dump taken from its DLL.
