# Physics-guided federated failure-risk prediction for CT X-ray tubes

Source code of the paper

> Z. Chen, C. Huang, K. Xia, L. Yang, Z. Qin, Y. Liu, Q. Liu, and J. Huang, "Physics-Guided Federated Learning for Failure Risk Prediction of CT X-Ray Tubes Across Hospitals," *IEEE Internet of Things Journal* (under review, manuscript IoT-69681-2026).

The code predicts, once per day, the probability that a CT X-ray tube fails within the next 7 and 30 days. Each tube is a federated client. The model starts from a Weibull wear-out prior driven by the tube's age (cumulative scans) and recent workload, and a multi-branch network learns a penalized correction of this prior on the logit scale. The clients are aggregated with FedAvg or with DTG-v2, which modulates the FedAvg weights by a label-free degradation trend of the filament health index, compared by degradation stage. The repository contains every method and every analysis reported in the paper:

- the physics prior, the multi-branch correction network and the physics-guided loss (monotonicity of the predicted risk, penalty on the correction);
- federated training with FedAvg, DTG-v2, the original DTG weighting, FedProx, SCAFFOLD, FedBN and clustered federated learning, plus centralized training and a data-only network as references;
- client-level DP-FedAvg with Rényi-DP accounting, and the Laplace mechanism for the trend statistics of DTG-v2;
- leave-one-tube-out and leave-one-hospital-out evaluation, nested hyperparameter selection on validation tubes, ensemble selection, tube-level bootstrap confidence intervals;
- the analyses of the paper: controls for the DTG-v2 gain, physical consistency, DTG weights from truncated logs, convergence, computational and communication cost.

## Data

The tube logs belong to the participating hospitals and cannot be made public under the data-use agreements, because scan timestamps can be linked to patient encounters. De-identified daily feature matrices are available from the corresponding author on reasonable request, subject to approval by the hospitals. No data are included in this repository.

The pipeline reads the tube-history logs that Siemens CT scanners store for every X-ray tube (`tube_history_<serial>.tua` or `.tua.gz`). `scripts/convert_tua.py` converts them into one CSV per tube (`device_T<serial>.csv`, one row per scan) and writes `tube_records.csv` with the installation and de-installation dates, the coverage of the log and the end-of-life signature of each tube. Tubes still in service are written to `in_service/` and are not used, because they have no failure time.

The fields used are `datetime`, `scan_count`, `voltage_nom`, `voltage_act`, `current_nom`, `current_mean`, `current_min`, `current_max`, `scan_time_nom`, `scan_time_act`, `focus`, `filament_begin`, `dom_type`, `xc_drops`, `oil_temp_begin`, `oil_temp_end`, `oil_pressure`, `gantry_temp`, `motor_temp`, `motor_curr`, `anode_freq`, `calc_temp_anode_cooling_interface`, `arcings_one` and `arcings_both`. Optional fields that a log does not contain are replaced by constants (engineered features) or left out (failure precursors).

The leave-one-hospital-out protocol needs a site map, a CSV with the columns `tube` and `hospital` (see `site_map_template.csv`).

## Installation

Python 3.10 or later. Install PyTorch first (for a GPU, choose the CUDA build that matches the driver on [pytorch.org](https://pytorch.org)), then the other requirements:

```bash
pip install -r requirements.txt
```

Every script selects the device automatically (`--device auto`): the first CUDA GPU if PyTorch sees one, otherwise the CPU. The same command therefore runs on both. An explicit `--device cuda:0` on a machine without a GPU falls back to the CPU with a warning.

## Reproducing the results

```bash
python scripts/convert_tua.py <folder or zip with tube_history_*.tua(.gz)> --out data/tubes
python reproduce.py --raw-dir data/tubes --site-map site_map.csv --out results
```

`reproduce.py` runs every experiment with the settings of the paper (`configs/paper_settings.json`) and writes the report `results/comparison/comparison_report.txt`. Each step runs as its own process with its own log in `results/logs/`. Finished fold runs are reused, so after an interruption the same command continues where it stopped. On a GPU, three fold processes run in parallel (`--workers auto`); the results are the same as in a sequential run. The complete set of experiments comprises about 3,000 fold runs of about one minute each on an NVIDIA GeForce RTX 2060.

| Step | Script | Output | Paper |
| --- | --- | --- | --- |
| Data-quality audit and exclusion list | `scripts/data_audit.py` | `results/audit/` | Sec. V-A |
| Weibull life distribution of the failed tubes | `scripts/weibull_lives.py` | printed | Sec. IV-A, V-A |
| Input B2, leave one tube out: all methods, baselines, privacy, ablation and controls | `scripts/failrisk.py` | `results/B2/loo/` | Sec. V-C to V-G |
| Input B2, leave one hospital out | `scripts/failrisk.py --protocol loho` | `results/B2/loho/` | Sec. V-C |
| Input A2 (without failure precursors), both protocols | `scripts/failrisk.py --input daily-A2` | `results/A2/` | Sec. V-E |
| DTG weights from truncated logs and under DP | `scripts/failrisk.py --analysis-only` | `results/B2/loo/dtg_stability.csv` | Sec. V-F, V-G |
| Computational and communication cost | `scripts/failrisk_complexity.py` | `results/comparison/table_complexity_*.csv` | Sec. V-H |
| Report with every table | `scripts/make_report.py` | `results/comparison/` | Sec. V |
| Convergence and other figures | `scripts/plot_convergence.py`, `scripts/failrisk_figures.py` | `results/B2/figures/` | Sec. V-H |

Single runs can be started directly, for example the proposed models in leave-one-tube-out evaluation:

```bash
python scripts/failrisk.py --raw-dir data/tubes --input daily-B2 --out results/B2/loo \
       --variants prior,fedavg-tuned,dtg2-tuned --tune-tau 0.02,0.05,0.1 \
       --batch 32 --rounds 100 --min-rounds 20 --patience 10 --seeds 0,1,2
```

`python scripts/failrisk.py -h` lists every option and the naming of the variants (for example `dtg2-t0.02-r0.1` for DTG-v2 with τ = 0.02 and correction penalty 0.1, `cdp1` for DP-FedAvg with noise multiplier 1, `fedavg-m0` for FedAvg without the monotonicity loss).

## Settings

All settings of the reported runs are in `configs/paper_settings.json`, and every run writes its own settings to `<run>/config.json`. In brief:

- Inputs: the last 60 days of daily inputs, each an offset from the tube's own first 30 active days, scaled with robust statistics of the training tubes and clipped to ±5. Input B2 has 58 features (43 engineered features, scans per day, an active-day flag and 13 failure precursors); input A2 leaves out the precursors (45 features). Samples less than one day before the failure are left out.
- Folds: leave one tube out (19 folds) and leave one hospital out (3 folds). In every fold, 3 of the remaining tubes are drawn as validation tubes with a seed that depends only on the fold and the training seed (`val_split` in `scripts/failrisk.py`); the others are the training clients. The Weibull prior is fitted to the lives of the training and validation tubes, the input scaling to the training tubes.
- Training: Adam with learning rate 10⁻³ (SCAFFOLD: SGD with step size 0.05), batch size 32, one local epoch per round, early stopping on the validation tubes (at least 20 and at most 100 rounds, patience 10 rounds), three seeds (0, 1, 2).
- Physics-guided loss: monotonicity weight 0.5; correction penalty selected per fold on the validation tubes from {0.01, 0.1, 1}; DTG-v2 temperature τ from {0.02, 0.05, 0.1}.
- Baselines: FedProx μ = 0.01; clustered FL with one bipartition after 5 rounds; FedBN and DP-FedAvg re-estimate the batch-normalization statistics of the held-out tube from its own earlier days only.
- Privacy: DP-FedAvg with clipping norm 2.0 and noise multipliers 0.5, 1 and 2, δ = 10⁻³; trend statistics released with ε = 1 and 0.1.
- Evaluation: tube-weighted AUC and Brier score; 95% confidence intervals of AUC differences from 2,000 tube-level bootstrap resamples.

## Repository structure

```
reproduce.py                 all experiments of the paper, in order
configs/paper_settings.json  settings of the reported runs
features/tube_features.py    45 engineered features per scan
dtgexp/failrisk.py           daily inputs, physics prior, network, loss, federated training, DTG-v2, metrics
dtgexp/hi.py                 protocol-conditioned filament health index
dtgexp/trends.py             stage-aligned degradation trend and pairwise similarity
dtgexp/dp.py                 Renyi-DP accounting, Laplace mechanism for the trend
dtgexp/survival.py           Weibull maximum-likelihood fit and profile-likelihood interval
dtgexp/cli.py, ct_data.py    device selection, console and file helpers
scripts/convert_tua.py       tube-history logs -> one CSV per tube + tube_records.csv
scripts/data_audit.py        data-quality audit and exclusion list
scripts/weibull_lives.py     Weibull fit of the tube lives with the profile-likelihood interval of the shape
scripts/failrisk.py          training and evaluation of all variants (one run folder per input and protocol)
scripts/failrisk_complexity.py  model size, operations, latency, training time, communication
scripts/make_report.py       comparison report and tables
scripts/plot_convergence.py  convergence figure
scripts/failrisk_figures.py  further figures (ROC, calibration, example tubes, DTG weights, privacy, Weibull prior)
```

## License

MIT License (see `LICENSE`).
