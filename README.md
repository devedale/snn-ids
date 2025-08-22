SNN‑IDS: Reproducible Deep Learning for CSE‑CIC‑IDS2018
=======================================================

This repository implements a principled, reproducible deep‑learning pipeline for network intrusion detection on the CSE‑CIC‑IDS2018 corpus. The system emphasizes: (i) a typed, centralized configuration; (ii) a deterministic data pipeline with explicit sampling and class balancing; (iii) temporal windowing suitable for sequence models; (iv) training recipes designed for tabular, time‑windowed features; and (v) auditable evaluation, including calibration analysis.

Dataset
-------
We use the Canadian Institute for Cybersecurity CSE‑CIC‑IDS2018 dataset (Sharafaldin et al.). Please refer to the official pages for licence and provenance. The code expects CSV files placed under `data/cicids/2018/` following the dataset’s conventional file names.

References:
- I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani. “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP 2018.
- Dataset page at the Canadian Institute for Cybersecurity.

Design Principles
-----------------
- Single source of truth for settings in `config.py` (no hard‑coded constants in the code path).
- Explicit, auditable preprocessing: IP anonymization support, IP‑to‑octet expansion, numerics‑only feature set, NaN/infinite handling.
- Temporal windows for sequence models; tunable window length and stride.
- Training without data leakage: scaling is fitted on the training split within each fold (not globally).
- Evaluation with task‑appropriate metrics: per‑class performance, confusion matrices, ROC where applicable, and intrusion‑detection oriented summaries (detection rate, false alarm rate, precision for attacks).
- Probability calibration analysis (Expected Calibration Error) reported for completeness; see Naeini et al. (AAAI 2015) and Guo et al. (ICML 2017) for background.

Repository Layout
-----------------
- `config.py` — Central configuration (dataset paths, features, preprocessing, training, benchmark options).
- `preprocessing/process.py` — Data loading, sampling, “security” class balancing, IP feature engineering, temporal window construction.
- `training/train.py` — Model construction (GRU/LSTM/Dense), Stratified K‑Fold training, per‑fold scaling to avoid leakage.
- `evaluation/metrics.py` — Evaluation and figures (detailed confusion matrix, cybersecurity matrix, per‑class accuracy, ROC, JSON reports; calibration utilities).
- `benchmark.py` — Unified entry‑point for smoke tests and full runs; organizes results under timestamped directories.
- `requirements.txt` — Minimal pinned versions to reproduce runs.
- `ids_pipeline_audit.ipynb` — Audit notebook documenting file‑by‑file and line‑by‑line decisions; runnable smoke and full tests.

Configuration
-------------
All behaviour is governed by `config.py`.

Key sections:
- `DATA_CONFIG`: dataset root, timestamp/label columns, canonical feature list (including IP columns that are subsequently expanded into four octets per address).
- `PREPROCESSING_CONFIG`: sampling size, balancing strategy (`security` aims at an attack/benign ratio as specified by `benign_ratio`), temporal windowing parameters (`window_size`, `step`).
- `TRAINING_CONFIG`: validation strategy (`k_fold` by default), K‑fold splits, default model type (GRU) and hyperparameter grids.
- `BENCHMARK_CONFIG`: output directory, time resolutions to explore, switches for intermediate artefacts and plots.

Preprocessing
-------------
1. CSV ingestion across the dataset directory with defensively coded parsing.
2. Efficient target‑aware sampling: benign vs non‑benign subsets are filtered and combined to match the intended benign/attack composition; this is more efficient than global sorting.
3. Feature engineering: IP addresses expanded to four octets per column (IPv4 assumed in feature engineering; Crypto‑PAn anonymization support is available in the codebase), coercion to numerics, and handling of infinities/NaNs.
4. Temporal windows: `X` becomes a 3‑tensor `[num_windows, window_size, num_features]`; `y` is aligned to the last element of each window.

Training Recipe
---------------
- Model: GRU by default (other baselines: LSTM, MLP on flattened windows). The GRU input is 3‑D: `(timesteps, features)`.
- Validation: `StratifiedKFold(n_splits=k, shuffle=True, random_state=42)`.
- Scaling: `StandardScaler` fitted on the training portion only, per fold; sequence tensors are reshaped to 2‑D for scaling, then reshaped back. This avoids target leakage, which is a common error in IDS literature.
- Loss function: `sparse_categorical_crossentropy` when `num_classes>2`, otherwise `binary_crossentropy`.
- Optimizer: Adam; the learning rate grid is configurable.

Evaluation and Reporting
------------------------
The benchmark produces both machine‑readable JSON summaries and publication‑grade figures:
- Detailed confusion matrix with class names.
- Binary (BENIGN vs ATTACK) confusion matrix for operational interpretation.
- Per‑class accuracy bar chart.
- ROC curves where the class structure allows it.
- Intrusion‑detection metrics: detection rate (recall on attack classes), false alarm rate (FPR on benign), precision for attacks.
- Calibration: Expected Calibration Error (Naeini et al., 2015; Guo et al., 2017) computed over bins of confidence.

Reproducibility
---------------
Seed control is provided at Python, NumPy, and TensorFlow levels. Beware that full determinism depends on the BLAS/GPU stack. All preprocessing choices are surfaced via configuration and are logged to JSON along with run metadata.

Quick Start
-----------
Install dependencies:

```bash
pip install -r requirements.txt
```

Run a smoke test (fast):

```bash
python3 benchmark.py --smoke-test --model gru
```

Run a full GRU benchmark with default configuration:

```bash
python3 benchmark.py --full --model gru
```

Results are saved under `benchmark_results/<timestamp>_<model>_evaluation/` with both JSON reports and PNG figures.

Colab Notebook
---------------
The notebook `ids_pipeline_audit.ipynb` contains a complete, auditable runbook: environment setup, data audit, smoke test, full grid over temporal windows (e.g., 5s/1m/5m) and learning rates, and rationale for each design decision. The notebook prints source excerpts from the repository to keep the narrative anchored to code.

Notes on IP Anonymization
-------------------------
Support for prefix‑preserving anonymization is included (Crypto‑PAn; see Xu et al., NDSS 2001). When enabled, anonymization is applied before IP‑to‑octet expansion, preserving subnet structure while protecting identities. Users are responsible for ensuring compliance with privacy policies and dataset licences.

Citations
---------
- Sharafaldin, Lashkari, Ghorbani. “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.” ICISSP, 2018.
- Xu, Fan, Ammar, Moon. “Prefix‑Preserving IP Address Anonymization: Measurement‑based Security Evaluation and a New Cryptography‑based Scheme (Crypto‑PAn).” NDSS, 2001.
- Naeini, Cooper, Hauskrecht. “Obtaining Well Calibrated Probabilities Using Bayesian Binning.” AAAI, 2015.
- Guo, Pleiss, Sun, Weinberger. “On Calibration of Modern Neural Networks.” ICML, 2017.

License
-------

Limitazioni, rischi e autocritica
---------------------------------
Questa implementazione, per quanto attenta, non elude i limiti noti della letteratura IDS:

- Distribuzioni non stazionarie: i dataset accademici non sempre rappresentano il traffico reale e la sua evoluzione. La validazione out‑of‑distribution resta necessaria.
- Class imbalance e “rare attacks”: anche con strategie di bilanciamento, i risultati su classi molto rare sono statisticamente fragili; riportiamo metriche per‑classe e suggeriamo intervalli di confidenza quando si compila un report.
- Leakage e overfitting: lo scaling è confinato ai fold per ridurre leakage; ciononostante, qualunque feature ingegnerizzata in modo target‑aware potrebbe indurre bias. Per questo il notebook include sezioni di audit ripetibili.
- Calibrazione: l’ECE fornisce un’indicazione sintetica, ma non sostituisce l’analisi di rischio operativa; vedere Naeini (2015) e Guo (2017) per limiti metodologici.
- Metriche operative: detection rate e false alarm rate sono utili ma non esaustive; in produzione si raccomanda l’uso di curve Precision‑Recall e cost‑sensitive analysis specifica al dominio.

Riproducibilità e variabilità
----------------------------
I seed sono impostati a livello Python/NumPy/TensorFlow. Alcune librerie BLAS/GPU possono introdurre variazioni numeriche. I risultati dipendono dalla versione del dataset, dall’hardware e dalle opzioni del driver. Per audit, si consiglia di salvare insieme ai risultati: commit hash, versione dei pacchetti (`requirements.txt`), configurazione completa (`config.py`) e log JSON del benchmark.

Uso responsabile
----------------
Questo codice è uno strumento di ricerca e valutazione. Qualsiasi impiego operativo richiede hardening, monitoraggio, gestione dei drift di distribuzione e una valutazione dei rischi legali/etici (privacy, data residency, explainability). Le decisioni di sicurezza non dovrebbero basarsi su un singolo modello né su un singolo dataset.
Please consult the dataset licence for usage of CSE‑CIC‑IDS2018. Code in this repository is provided under a permissive licence; see the repository’s licence file if present.


