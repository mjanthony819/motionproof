# MotionProof

An activity-recognition model bench that runs entirely in the browser. It trains three models on real free-living accelerometer data, scores them on people they never saw, stress-tests them, and lets you run them on your own recordings.

**Live app:** open `index.html` (one self-contained file, works offline, no server, no accounts, no model API).

> Not a medical device. Do not use it for diagnosis, fall detection, gait assessment or monitoring a person.

## What it is

Activity models usually look good on the people they were trained on and worse on new people. MotionProof measures that gap and the other ways they fail.

- **Data:** HARTH (UCI #779, CC BY 4.0). 22 adults, about 35.9 hours, two 3-axis accelerometers (lower back, right thigh) at 50 Hz, free living, camera-annotated. 9 activities: walking, running, shuffling, stairs up, stairs down, standing, sitting, lying, cycling (the four cycling codes are merged). 118,495 two-second windows (50% overlap, windows with mixed labels dropped).
- **Models (all trained for this project, CPU, PyTorch):**
  1. `lr`: 67 hand-made features (statistics, band energies, sensor correlations) with softmax regression.
  2. `cnn`: a small 1D CNN on raw signals (3 conv layers, about 17k parameters).
  3. `cnn_aug`: the same CNN trained with random sensor rotation, gain, noise and sensor dropout.
- **App:** analyse your own recording first (CSV import with validation of columns, units and sample rate, or a built-in held-out sample), then per-window predictions, per-model confidence, abstain control, activity-time breakdown and CSV exports; a model bench on packaged held-out recordings; an evidence tab; a stress lab (damage the data and re-run); saved sessions, model card and experiment export. The JavaScript re-implements the models from exported weights and is tested against the PyTorch outputs.

**Update, November 16, 2026:** the app was reworked so the trained models drive the main workflow end to end. The first tab is now Analyse a recording: your CSV (or the built-in sample) goes through live in-browser inference to a labelled timeline, an activity-time breakdown (counted per 1-second window step, never double-counting the 2-second windows) and downloadable predictions. The interface now uses the project's charcoal/lime design. Models, data, metrics and evaluation are unchanged. The narrated demo video, if referenced from the project page, was recorded for the original submission and shows its earlier interface.

## Results (measured, subject-held-out)

Every test window comes from a person the model did not train on.

| Model | Macro-F1, 5-fold CV, pooled over all folds | Macro-F1, mean of 5 folds (sd) | Accuracy, pooled | ECE | Macro-F1 on 4 held-out subjects, mean of 3 seeds (sd) |
|---|---|---|---|---|---|
| Feature logistic regression | 0.810 | 0.810 (0.035) | 0.940 | 0.003 | 0.830 (0.009) |
| 1D CNN | 0.792 | 0.815 (0.044) | 0.928 | 0.031 | 0.865 (0.012) |
| 1D CNN + augmentation | **0.853** | 0.855 (0.034) | 0.954 | 0.011 | 0.889 (0.014) |

Read honestly:

- The augmented CNN is best, but the plain CNN is **not** better than logistic regression on the pooled cross-validation. Gaps between models are about the size of the spread between folds.
- Shuffling (F1 0.49 to 0.54), stairs up (0.59 to 0.86) and stairs down (0.44 to 0.58) are the weak classes. They are rare in the data and look like walking.
- Stress test (4 held-out subjects, mean of 3 seeds, macro-F1): losing the thigh sensor drops LR to 0.26, CNN to 0.28 and CNN+aug to 0.51. Losing the back sensor drops LR to 0.21 and CNN to 0.13, but CNN+aug keeps 0.78. A 30 degree thigh-sensor rotation costs 0.14 to 0.27.
- Confidence stays high while accuracy falls (mean confidence 0.94 for LR and CNN with the thigh sensor lost, 0.76 for CNN+aug), so abstaining on confidence alone does not detect a broken sensor.
- Per-subject macro-F1 ranges from 0.48 to 0.96 across people, so the average hides a weak worst case.
- Different protocols (pooled vs mean of folds vs the 4-subject split) give different numbers. All are reported and labelled; none is a clinical claim.

## Run it

```bash
node build.mjs            # rebuilds index.html from src/ and data/
node tests/parity.test.js # JS vs PyTorch outputs, perturbations, CSV validation (28 checks)
npm i && node tests/browser.test.js   # headless Chrome UI test (needs puppeteer-core and Chrome)
```

## Retrain from scratch

```bash
pip install -r training/requirements.txt
# download HARTH (CC BY 4.0) from https://archive.ics.uci.edu/dataset/779 and unzip the CSVs into work/harth/
python training/prep.py                        # windows -> work/windows.npz
python training/run_jobs.py 0 & python training/run_jobs.py 1 & wait   # CV folds + showcase seeds (re-run until every job has finished; finished jobs are skipped)
python training/analyze.py                     # metrics, calibration, confusion
for k in lr cnn cnn_aug; do python training/stress.py $k; done
python training/export.py                      # weights, golden vectors -> data/
python training/streams.py                     # demo excerpts and sample CSV
python training/build_results.py               # data/results.json
```

Training is on CPU (two cores). Each CNN job takes about a minute. Seeds are fixed but CPU floating point means re-runs can differ in the last digits. The numbers in this repository come from the runs saved in `results/`.

Protocol details: 5 folds of whole subjects, balanced by window count. Inside each fold, 3 other subjects are used for early stopping and temperature scaling, and the rest train. Training uses every second window (non-overlapping 2 s windows). The shipped models come from a separate split: subjects 6, 13, 20 and 28 held out for testing, 9, 15 and 24 for validation, the other 15 for training. Class weights use the inverse square root of class frequency.

## Limits

- Healthy adults wearing sensors in this exact placement and orientation. Other people, placements and devices are untested.
- Windows with mixed activities are dropped, so transitions are not scored.
- The four cycling codes are merged into one class.
- Temperature scaling is fitted on three validation subjects only.
- Stress tests are synthetic damage, not real device faults.

## AI disclosure and what was done during the event

This project was built with AI assistance. An AI assistant wrote the code, ran the training and evaluation and wrote this documentation, working for and directed by the entrant (M J Anthony). The entrant specified the goal (real trained models with honest, measured results) and approved the scope. All numbers above were produced by the scripts in `training/` and are stored in `results/`. Everything in this repository was created during the event period (November 15 to 16, 2026); the dataset, PyTorch, NumPy and pandas are the only reused material.

## Credits

HARTH: Logacjov A., Bach K., Kongsvold A., Bardstu H.B., Mork P.J., Sensors 21(23):7853 (2021), UCI ML Repository #779, CC BY 4.0. See `NOTICE.md`. Code: MIT license.
