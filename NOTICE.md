# Third-party material

## Data: HARTH
Logacjov, A., Bach, K., Kongsvold, A., Bardstu, H.B., Mork, P.J. "HARTH: A Human Activity Recognition Dataset for Machine Learning." Sensors 21(23):7853 (2021).
UCI Machine Learning Repository, dataset 779, https://archive.ics.uci.edu/dataset/779 . The repository page states the license is Creative Commons Attribution 4.0 (CC BY 4.0).
The raw recordings are not included in this repository; `training/prep.py` expects them in `work/harth/`.
`data/streams.json` contains four short excerpts (4 minutes each, converted to int16 milli-g) from subjects 6, 13, 20 and 28, adapted from HARTH under CC BY 4.0. `sample_import.csv` is derived from one of them. Labels were merged and windows were cut by this project; the authors of HARTH did not endorse this work.

## Software
Training used PyTorch (BSD-3-Clause), NumPy and pandas (BSD-3-Clause). The app has no runtime dependencies. Test tooling: puppeteer-core (Apache-2.0).
