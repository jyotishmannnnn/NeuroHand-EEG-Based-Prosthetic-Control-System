# NeuroState BCI → Bionic Hand: Project Notes

Last updated: 2026-09-20

## Goal

Control a robotic hand (8× N20 micro DC gearmotors, tendon-driven fingers) from the
NeuroState EEG cap. Discrete grasp primitives selected by brain/biosignal, executed by
hand firmware.

## Hardware: Qneuro H100 "NeuroState" cap

- Vendor: Qneuro India Pvt. Ltd. (support@qneuro.com). Manual:
  `NeuroAnalyticsWindows_28-05-2026/NeuroAnalytics manual.pdf`.
- EEG: 8 channels, 24-bit, 250 Hz. Order in all files: **FP1, FP2, O1, O2, C3, CZ, FZ, C4**.
  Reference = one earlobe clip. PPG sensor = other earlobe.
- PPG: R + IR, 100 Hz. IMU: quaternion (QX,QY,QZ,QW) + accel (AX,AY,AZ), 100 Hz.
- Link: Bluetooth (BLE serial style). USB option exists in app. Device name `NeuroState-XXXX`.
- Firmware commands seen in log: configure(11), PPG calib(235), set channel gain(8),
  start stream(170), stop stream(255). Gain is configurable.

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

### LSL (live)

Start app in STREAMING-LSL mode. Streams (case-sensitive, per manual):
`<UserName>_UnFilter` (EEG), `<UserName>_PPG`, `<UserName>_IMU`, `<UserName>_Marker`.
String dump of app DLL also contains `_EEG` suffix (possibly filtered stream). Verify with
`tools/lsl_probe.py`. Not yet tested live.

Python 3.13 system interpreter has: numpy 2.4, scipy 1.17, opencv 5.0, pyserial, pylsl 1.18.4.

## Tools written (`tools/`)

| script | purpose |
|---|---|
| extract_frames.py | sample N JPEG frames from a screen-recording mp4 (needs cv2) |
| analyze_session.py | per-channel quality metrics from UnfilteredData.csv: RMS, flat%, 50 Hz share, alpha ratio, cross-channel corr |
| alpha_ec_eo.py | eyes-closed vs eyes-open alpha comparison (Berger effect test) |
| lsl_probe.py | discover LSL streams, print labels + live sample rates |
| assembly_userstrings.txt | UTF-16 string dump of Assembly-CSharp.dll (headers, stream suffixes, class names) |

## Session log

All 2026-09-20. Quality target for real EEG: bandpassed RMS 10–50, 50 Hz share <5%,
alpha peak ~10 Hz at O1/O2 eyes closed, EC/EO alpha ratio >1.5.

| session | dur | result |
|---|---|---|
| 06-33-45 | ~75 s | first test |
| 06-36-26 | 0 s | empty |
| 06-37-30 | 624 s | RMS 11k–16k, 50 Hz 4–85%, no alpha peak. All channels railing. |
| 08-32-02 | 872 s | RMS 12k–22k, 50 Hz 4–43%, no alpha. Cross-ch corr 0.47. (matches EEG_Recording.mp4) |
| 09-56-36 | short | aborted |
| 09-57-47 | 181 s | EC/EO test. 10× better but still fails. FP1 RMS 162, others 560–1240. 50 Hz 100–270× band power on FP1/FP2/C4. **O1,O2,C3,CZ,FZ near-identical waveform → not touching scalp.** EC/EO alpha ratio 0.04 (artifact, not Berger). (matches EEG_Recording2.mp4) |

PPG (HR ~70–80 bpm, SpO2 ~99%) and IMU good in every session.

### Diagnosis (as of 09-57-47)
1. Occipital/central electrodes floating (hair, no contact) → five channels read reference-drive noise.
2. Mains hum on skin-contact channels → high impedance + laptop charger / monitors nearby / poor ear-clip reference.

### Fix checklist before next recording
1. Unplug laptop from mains, 1 m from monitors/chargers.
2. Wet ear-clip reference, clean earlobe, tight.
3. Part hair under O1/O2/C3/C4/CZ/FZ, press + wiggle each pad, saline drop on each.
4. Do not Proceed past Signal Quality Index until all 8 green.
5. Sanity test: press FP1 pad + ear clip with fingers for 10 s; if FP1 goes small/flat, contact is the story.
6. Re-record 60 s EC then 60 s EO. Run
   `python tools/alpha_ec_eo.py <session_dir> 5 60 65 120`.

## Control architecture (planned)

```
cap --BLE--> NeuroAnalytics (STREAMING-LSL) --LSL--> Python pipeline --serial/BLE--> ESP32 --PWM--> 4x DRV8833 --> 8x N20 tendons
                                                                                              ^ encoder / current feedback
```

Pipeline: pull 250 Hz EEG → notch 50 Hz + bandpass → 1 s epochs → features → classifier →
command with confidence threshold + dwell + rest class → hand primitive.

### Control tiers (easiest first)
1. Blink / jaw clench (FP1/FP2 threshold). No ML. 2–3 commands. Build first.
2. SSVEP (O1/O2, flicker targets on screen). FFT/CCA. 4–6 commands. No ML.
3. Alpha level (relax → open). Band-power threshold.
4. Motor imagery left/right hand (C3/C4/CZ). CSP+LDA or Riemannian+LR, scikit-learn, per-session training. 2–3 classes, 65–80%.
5. Hybrid: MI for intent, blink confirm, IMU head tilt to cycle primitives.

Per-finger decoding from scalp EEG: not feasible. Hand firmware owns grasp primitives
(open, power grip, pinch, point, tripod); BCI only selects primitive.

### Hand side (to decide)
- N20 = brushed DC gearmotor, no position sensing unless encoder variant. Tendon = pull only.
- Drivers: 4× DRV8833 (2 H-bridges each) or 2× TB6612FNG. MCU: ESP32 (BLE/WiFi, 16 PWM).
- Feedback: N20 encoder variant (best), or current-sense stall detect (INA219 / driver sense), or open-loop timed (fragile).
- Power: stall ~0.7–1.6 A per motor → 2S LiPo + 6 V buck ≥5 A. Not USB.
- Open questions: finger→motor map, encoder yes/no, motor voltage rating, boards owned, tendon return (spring/antagonist/elastic).

## Misc
- `EEG_Recording.mp4` (96 s, EEG tab) and `EEG_Recording1.mp4` (25 s, Analysis tab) and
  `EEG_Recording2.mp4` (179 s, EC/EO test) are screen recordings only. Frames in `frames/`.
- The original `EEG_Recording2.mp4` upload was a byte-identical duplicate of `EEG_Recording.mp4`; later replaced by the real EC/EO recording.
