# Recorded sessions

Raw streams from every LiveSession recorded with the Qneuro H100 "NeuroState" cap
through QneuroNeuroAnalytics. One subject, 2026-09-20 to 2026-09-22. Copied unchanged
from `%LOCALAPPDATA%Low/Qneuro/QneuroNeuroAnalytics/Jyotishman/<date>/LiveSession/<time>/`.

Folder name is `<MM-DD-YYYY>_<HH-MM-SS>`, the app's session start time (12-hour
clock, no AM/PM; every session here was recorded in the evening, so 09-18-27 means
21:18:27).

| file | columns | rate |
|---|---|---|
| `UnfilteredData.csv` | PseudoTimeSec, PseudoTimestamp, UF-FP1, UF-FP2, UF-O1, UF-O2, UF-C3, UF-CZ, UF-FZ, UF-C4 | 250 Hz |
| `PPGData.csv` | PseudoTimeSec, PseudoTimestamp, R, IR | 100 Hz |
| `IMUData.csv` | PseudoTimeSec, PseudoTimestamp, QX, QY, QZ, QW, AX, AY, AZ | 100 Hz |

Note the channel order: **CZ comes before FZ**. EEG values are raw device counts,
not microvolts. The gain is configurable and the scale is undocumented, so see
`PROJECT_NOTES.md` before comparing amplitudes across sessions.

The app's derived files (Filtered_runtime, the offline re-filter, FFT, BandData,
HR/HRV/SpO2/RR, Attention, CognitiveLoad) are not included. They can all be
regenerated from these three.

Load with `neuro.source.CsvSource(path)` or
`tools/analyze_session.py data/sessions/<name>`.

## Index

Verdicts are summarized from `reports/`. Full log in `PROJECT_NOTES.md`.

| session | dur | note |
|---|---|---|
| 09-20-2026_06-33-45 | ~75 s | first test |
| 09-20-2026_06-36-26 | 0 s | empty |
| 09-20-2026_06-37-30 | 624 s | all channels railing |
| 09-20-2026_08-32-02 | 872 s | no alpha, cross-channel corr 0.47 |
| 09-20-2026_09-56-36 | short | aborted, no IMU |
| 09-20-2026_09-57-47 | 181 s | eyes-closed/eyes-open test; five pads floating |
| 09-21-2026_09-18-27 | 155 s | **first real EEG**: 8.5 Hz alpha, EC/EO 2.65–3.49 |
| 09-22-2026_09-12-53 | 0 s | aborted |
| 09-22-2026_09-13-19 | 105 s | regression, unusable |
| 09-22-2026_09-27-40 | 337 s | LSL open during live probing |
| 09-22-2026_09-33-34 | 95 s | best frontal contact at the time; posterior floating |
| 09-22-2026_10-25-53 | 126 s | jaw block 5/5; blinks absent |
| 09-22-2026_11-25-50 | 68 s | blinks back on FP1/FP2; movement at 15–35 s |
| 09-22-2026_11-39-58 | 101 s | contact check only; PPG/IMU dropped (no IMU file) |
| 09-22-2026_11-49-40 | 1198 s | app-side recording spanning both `run_protocol.py` runs. 20260922_235229 starts at CSV t = 53.9 s (sample-exact). 20260923_000722 starts near t ≈ 946 s (wall clock, not aligned). Cue times and the LSL copy are in `reports/protocol_*.npz` |
