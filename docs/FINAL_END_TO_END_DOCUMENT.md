# Comprehensive End-to-End System Documentation & Comparative Study

## Auditing Agency Performance Monitoring, Classification, Scoring, and Air-Gapped Ranking Module

**Project**: AI-Based Solution for Analysing, Benchmarking, and Quality Monitoring of Cybersecurity Audit Reports and Performance Monitoring of Auditing Organisations

**Submission Package Identifier**: `PerformanceMonitoring_Saket_20260824`

**Date of Submission**: 2 September 2026

---

## Executive Summary & Complete Deliverables Architecture (D1–D13)

This technical document serves as the authoritative, unified design and evaluation reference for the clean-room implementation of the **Auditing Agency Performance Monitoring, Scoring, and Ranking Module**, fully addressing all thirteen deliverables (**D1 through D13**).

The system evaluates the performance of empanelled cybersecurity auditing organisations on a continuous basis, computes six mandatory performance features from raw operational telemetry, trains and compares four renewal outcome classification models, synthesizes a continuous hybrid performance score (combining multi-attribute features and model-predicted renewal probabilities), executes deterministic ranking with multi-tier tie-breaking, detects decision boundary uncertainty, and provides immutable versioning and end-to-end provenance in a strictly air-gapped offline environment.

### Deliverables Mapping Matrix (D1–D13)

| Deliverable ID | Requirement / Scope | Implementation Artifact / Evidence Location | Functional Description & Document Section |
| :--- | :--- | :--- | :--- |
| **D1** | **Feature Definitions** | [`config/feature_schema.json`](config/feature_schema.json) | Mathematical formulas, units, bounds ($0–100, 1–180, 0–20$), directionality, missing/malformed data rules, and derivation classifications for all 6 features *(Document Section 1.2)*. |
| **D2** | **Synthetic Generator & Sample Dataset** | [`src/data_generator/generate_dataset.py`](src/data_generator/generate_dataset.py)<br>[`data/sample_dataset.json`](data/sample_dataset.json) | Configurable seeded generator producing 120 organisations with realistic irregularities (sparse history, missing audits, outliers, malformations, inconsistent reporting periods) and proof-of-concept labels *(Document Section 1.3)*. |
| **D3** | **Classification Pipeline** | [`src/models/`](src/models/) (`baseline.py`, `gradient_boosting.py`, `svm.py`, `neural_network.py`, `model_trainer.py`) | Source implementations of Baseline, GBDT, SVM, and Neural Network with Stratified 5-Fold Cross-Validation on training folds and automated hyperparameter tuning *(Document Section 1.4)*. |
| **D4** | **Evaluation Harness & Results** | [`src/evaluation/metrics.py`](src/evaluation/metrics.py)<br>[`output/evaluation_results.json`](output/evaluation_results.json) | Reports Macro-F1 (primary selection metric), Accuracy, Precision, Recall, Confusion Matrix, and per-class breakdowns on isolated held-out test data *(Document Section 3.1)*. |
| **D5** | **Scoring & Ranking Engine** | [`src/scoring/engine.py`](src/scoring/engine.py)<br>[`src/ranking/engine.py`](src/ranking/engine.py)<br>[`config/scoring_config.json`](config/scoring_config.json) | Implements hybrid continuous scoring combining normalized feature composite ($75\%$) with calibrated model renewal probabilities ($25\%$), deterministic tie-breaking, and boundary uncertainty detection ($57.0–63.0$) *(Document Section 1.5)*. |
| **D6** | **Versioning & Audit Registry** | [`config/version_registry.json`](config/version_registry.json)<br>[`src/versioning/registry.py`](src/versioning/registry.py) | Tracks semantic versions, SHA-256 configuration hashes, approval metadata, and preserves historical scores and rankings under their original version provenance *(Document Section 1.5 & Part 2)*. |
| **D7** | **Air-Gapped Package & Update Protocol** | [`deploy/`](deploy/) (`install_offline.py`, `verify_install.py`, `update_process.py`, `offline_package/`) | Vendored `.whl` dependencies for offline pip installation (`--no-index`); diagnostic verification script and controlled update protocol with checksum and malware scan hooks *(Document Section 2.2, 2.5)*. |
| **D8** | **Automated Test Suite** | [`tests/acceptance/test_acceptance_suite.py`](tests/acceptance/test_acceptance_suite.py)<br>[`tests/unit/`](tests/unit/) | 37 automated unit and acceptance tests, including formal acceptance tests AT-1 through AT-8, with 100% pass rate *(Document Section 2.4 & Verification Summary)*. |
| **D9** | **Sample Output Artifacts** | [`output/organisation_scores.json`](output/organisation_scores.json)<br>[`output/ranking.json`](output/ranking.json)<br>[`output/history/`](output/history/) | Complete JSON outputs for 120 organisations carrying continuous scores, model outputs, rankings, boundary uncertainty tags, and historical version archives *(Document Section 1.5, 3.2)*. |
| **D10** | **Design & Mathematical Derivation** | [`docs/FINAL_END_TO_END_DOCUMENT.md`](docs/FINAL_END_TO_END_DOCUMENT.md) (Part 1) | Comprehensive design documentation explaining system architecture, end-to-end data flow, mathematical feature formulations, scoring equations, and provenance traceability *(Part 1, Sections 1.1–1.6)*. |
| **D11** | **Operator Runbook & Guide** | [`docs/FINAL_END_TO_END_DOCUMENT.md`](docs/FINAL_END_TO_END_DOCUMENT.md) (Part 2)<br>[`README.md`](README.md) | Executable step-by-step instructions for environment setup, offline installation, pipeline execution, verification, testing, and troubleshooting *(Part 2, Sections 2.1–2.5)*. |
| **D12** | **Results & Limitations Analysis** | [`docs/FINAL_END_TO_END_DOCUMENT.md`](docs/FINAL_END_TO_END_DOCUMENT.md) (Part 3) | Transparent reporting of empirical model comparisons, confusion matrices, score distributions, boundary uncertainty triage, and synthetic proof-of-concept limitations *(Part 3, Sections 3.1–3.3)*. |
| **D13** | **Comparative Study of Alternatives** | [`docs/FINAL_END_TO_END_DOCUMENT.md`](docs/FINAL_END_TO_END_DOCUMENT.md) (Part 4) | In-depth comparative evaluation across five algorithmic components comparing $\ge 3$ viable alternatives across all seven mandatory evaluation criteria with material references *(Part 4, Sections 4.1–4.6)*. |

---

## PART 1: D10 — DESIGN AND DERIVATION

### 1.1 End-to-End System Architecture

The module architecture operates with strict separation between configuration (JSON), data pipelines, ML classification, continuous scoring, and ranking:

```
+---------------------------------------------------------------------------------------------------+
| 1. CONFIGURATION & SCHEMA REGISTRY (Strict JSON Only)                                             |
|    feature_schema.json | generation_config.json | label_rule.json                                  |
|    scoring_config.json | model_config.json      | version_registry.json                            |
+---------------------------------+-----------------------------------------------------------------+
                                  |
                                  v
+---------------------------------+-----------------------------------------------------------------+
| 2. DATA GENERATION & TELEMETRY INGESTION                                                          |
|    src/data_generator/generate_dataset.py                                                         |
|    - Configurable population distributions (Log-normal, Gamma, Beta, Binomial)                    |
|    - Operational parameters & non-standard reporting period generation                            |
|    - Controlled irregularity injection (Missing audits, sparse history, outliers, malformations)  |
|    - Synthetic proof-of-concept renewal label rule (8% stochastic noise)                          |
|    - Output: data/sample_dataset.json (N=120 organisations)                                       |
+---------------------------------+-----------------------------------------------------------------+
                                  |
                                  v
+---------------------------------+-----------------------------------------------------------------+
| 3. FEATURE EXTRACTION & VALIDATION PIPELINE                                                       |
|    src/features/validator.py & src/features/extractor.py                                          |
|    - Schema bound validation & explicit NOT_ASSESSABLE / MALFORMED status recording               |
|    - Leakage-free training-set median imputation + explicit missingness indicators                |
+---------------------------------+-----------------------------------------------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
+--------+----------------------------+   +----------------+-------------------------------+
| 4. ML CLASSIFICATION PIPELINE       |   | 5. CONTINUOUS SCORING & RANKING ENGINE         |
|    src/models/model_trainer.py      |   |    src/scoring/engine.py                       |
|    - Baseline (Dummy Classifier)    |   |    - Directional normalisation (0-100 scale)   |
|    - Gradient Boosting (GBDT)       |   |    - Dynamic reweighting for missing features  |
|    - Support Vector Machine (SVC)   |   |    - Missing feature audit penalty (-2.0 pts)  |
|    - Neural Network (MLP)           |   |    - Hybrid formula: 75% Features + 25% Model  |
|    - Stratified 5-Fold CV on Train  |   |    src/ranking/engine.py                       |
|    - Model Selection: CV Macro-F1   |   |    - Deterministic multi-tier sort             |
|    - Held-out Test Set Evaluation   |   |    - Decision boundary uncertainty [57.0, 63.0]|
|    - Output: evaluation_results.json|   |    - Outputs: organisation_scores.json,        |
|                                     |   |               ranking.json, output/history/    |
+-------------------------------------+   +------------------------------------------------+
                                  |
                                  v
+---------------------------------+-----------------------------------------------------------------+
| 6. VERSIONING, PROVENANCE & AIR-GAPPED DEPLOYMENT                                                 |
|    src/versioning/registry.py | deploy/install_offline.py | deploy/verify_install.py              |
|    - Complete provenance: Rank -> Score -> Model Prob -> Normalized Feats -> Raw Vars -> Version   |
|    - Historical archiving in output/history/v<version>/ preserves prior score immutability        |
|    - Vendored wheels in deploy/offline_package/ (100% offline pip installation)                   |
|    - Automated Acceptance Tests AT-1 through AT-8 (tests/acceptance/test_acceptance_suite.py)     |
+---------------------------------------------------------------------------------------------------+
```

---

### 1.2 Mathematical Derivation and Specification of the Six Features (D1)

Every feature is formally defined in [`config/feature_schema.json`](config/feature_schema.json).

#### Feature 1: Audit Timeliness

- **Operational Definition**: Percentage of completed audit engagements delivered on or before contractual/statutory milestone dates within the evaluation reporting period.
- **Formula**:
  $$
  \text{Audit Timeliness} = \left( \frac{\text{on\_time\_completed\_audits}}{\text{eligible\_completed\_audits}} \right) \times 100
  $$
- **Variables**: `on_time_completed_audits` (integer), `eligible_completed_audits` (integer).
- **Unit & Permitted Range**: Percentage ($0.0$ to $100.0\%$).
- **Directionality**: Higher is better.
- **Missing-Data Treatment**: If $\text{eligible\_completed\_audits} \le 0$, flagged as `NOT_ASSESSABLE`.
- **Partial-Data Treatment**: Audits in progress without sign-off timestamps are excluded from both numerator and denominator.
- **Invalid-Data Treatment**: If $\text{on\_time} > \text{completed}$ or counts $<0$, flagged as `MALFORMED` and assigned `NOT_ASSESSABLE`.
- **Derivation Classification**: `assignment_specified` (mandated in Assignment 2 Section 2.1).

#### Feature 2: Audit Efficiency

- **Operational Definition**: Mean calendar turnaround duration in days required to complete an audit from kickoff to formal report sign-off.
- **Formula**:
  $$
  \text{Audit Efficiency} = \frac{\sum (\text{actual\_completion\_date} - \text{audit\_start\_date})}{\text{eligible\_completed\_audits}}
  $$
- **Variables**: `actual_completion_date` (timestamp), `audit_start_date` (timestamp), `eligible_completed_audits` (count).
- **Unit & Permitted Range**: Calendar Days ($1.0$ to $180.0$ days).
- **Directionality**: Lower is better (inverted during scoring: 10 days $\to$ 100 score, 90 days $\to$ 0 score).
- **Missing-Data Treatment**: If $\text{eligible\_completed\_audits} \le 0$, flagged as `NOT_ASSESSABLE`.
- **Partial-Data Treatment**: Incomplete audit logs missing start or end dates are excluded from the summation.
- **Invalid-Data Treatment**: Negative turnaround times or durations exceeding 365 days trigger `MALFORMED` status.
- **Derivation Classification**: `assignment_specified` (mandated in Assignment 2 Section 2.1).

#### Feature 3: Error Rate

- **Operational Definition**: Ratio of substantive technical errors, omissions, or compliance non-conformances identified during independent QA reviews to the total number of QA-reviewed audit reports.
- **Formula**:
  $$
  \text{Error Rate} = \frac{\text{identified\_discrepancies}}{\text{reviewed\_reports}}
  $$
- **Variables**: `identified_discrepancies` (integer count), `reviewed_reports` (count of audited reports subject to peer review).
- **Unit & Permitted Range**: Discrepancies per report ($0.0$ to $20.0$).
- **Directionality**: Lower is better (inverted during scoring: 0.0 $\to$ 100 score, 6.0 $\to$ 0 score).
- **Missing-Data Treatment**: If $\text{reviewed\_reports} \le 0$, flagged as `NOT_ASSESSABLE`.
- **Partial-Data Treatment**: Reports currently undergoing QA with pending discrepancy counts are excluded.
- **Invalid-Data Treatment**: Negative error counts trigger `MALFORMED` status.
- **Derivation Classification**: `judgement` (using QA-reviewed reports as the denominator rather than total audits prevents unfair penalization of unreviewed reports).

#### Feature 4: Agency Scale

- **Operational Definition**: Composite index reflecting technical workforce capacity and institutional longevity, equally weighted across certified auditor headcount and continuous years in operation.
- **Formula**:
  $$
  \text{Agency Scale} = 50.0 \times \min\left(\frac{\text{team\_size}}{50}, 1.0\right) + 50.0 \times \min\left(\frac{\text{years\_operational}}{20}, 1.0\right)
  $$
- **Variables**: `team_size` (full-time certified cybersecurity auditors), `years_operational` (continuous operating history in years).
- **Unit & Permitted Range**: Composite Index Points ($0.0$ to $100.0$).
- **Directionality**: Higher is better.
- **Missing-Data Treatment**: If both fields missing, flagged as `NOT_ASSESSABLE`.
- **Partial-Data Treatment**: If one field is available, computed as partial with warning flag.
- **Invalid-Data Treatment**: Negative values trigger `MALFORMED` status.
- **Derivation Classification**: `judgement` (equal 50/50 weighting and benchmarks of 50 staff and 20 years are proof-of-concept design judgements).

#### Feature 5: Compliance Adherence

- **Operational Definition**: Percentage of applicable mandatory cybersecurity audit controls and procedural standards verified as fully compliant during oversight assessments.
- **Formula**:
  $$
  \text{Compliance Adherence} = \left( \frac{\text{compliant\_applicable\_controls}}{\text{applicable\_controls}} \right) \times 100
  $$
- **Variables**: `compliant_applicable_controls` (integer), `applicable_controls` (integer scope).
- **Unit & Permitted Range**: Percentage ($0.0$ to $100.0\%$).
- **Directionality**: Higher is better.
- **Missing-Data Treatment**: If $\text{applicable\_controls} \le 0$, flagged as `NOT_ASSESSABLE`.
- **Partial-Data Treatment**: Requires finalized compliance review checklist.
- **Invalid-Data Treatment**: Compliant controls exceeding applicable controls triggers `MALFORMED` status.
- **Derivation Classification**: `assignment_specified` (mandated in Assignment 2 Section 2.1).

#### Feature 6: Past Renewal Status

- **Operational Definition**: Percentage of successful renewal adjudications out of total definitive historical empanelment decisions.
- **Formula**:
  $$
  \text{Past Renewal Status} = \left( \frac{\text{historical\_renewals}}{\text{historical\_renewals} + \text{historical\_revocations}} \right) \times 100
  $$
- **Variables**: `historical_renewals` (count), `historical_revocations` (count).
- **Unit & Permitted Range**: Percentage ($0.0$ to $100.0\%$).
- **Directionality**: Higher is better.
- **Missing-Data Treatment**: If $\text{renewals} + \text{revocations} == 0$, strictly flagged as `NOT_ASSESSABLE`. New entrants with zero history are never treated as renewals or revocations.
- **Partial-Data Treatment**: Pending renewal adjudications without final ruling are omitted.
- **Invalid-Data Treatment**: Negative counts trigger `MALFORMED` status.
- **Derivation Classification**: `assignment_specified` (mandated in Assignment 2 Section 5.2).

---

### 1.3 Synthetic Dataset Generation Methodology (D2)

Configured via [`config/generation_config.json`](config/generation_config.json):

- **Population Size**: Defaults to 120 organisations (configurable via CLI `--n`).
- **Deterministic Random Seed**: `20260824` (guarantees 100% reproducible byte-identical dataset generation).
- **Configurable Statistical Distributions**:
  - `team_size`: Log-normal ($\mu=2.6, \sigma=0.6$, range 2–75).
  - `years_operational`: Gamma ($\alpha=3.0, \beta=2.5$, range 1–25).
  - `audit_timeliness_pct`: Beta ($\alpha=8.0, \beta=2.0$, mean $\approx 80\%$).
  - `audit_duration_days`: Gamma ($\alpha=4.0, \beta=8.0$, mean $\approx 32$ days).
  - `discrepancies_per_report`: Exponential ($\text{scale}=1.8$, range 0–15).
  - `compliance_adherence_pct`: Beta ($\alpha=12.0, \beta=2.0$, mean $\approx 85.7\%$).
- **Configurable Operational Parameters**:
  - `applicable_controls_choices`: `[25, 30, 40, 50, 60]`
  - `reviewed_reports_fraction`: range `[0.70, 1.00]`
  - `sparse_history_years_threshold`: `2.0` years
  - `historical_renewals_years_divisor`: `2.5`
- **Reporting Period Metadata & Inconsistent Case Generation**:
  - Each organisation record contains explicit `reporting_period` metadata (`start_date`, `end_date`, `duration_months`, `period_type`, `is_standard_cycle`).
  - Irregular reporting period cases ($5\%$ probability) generate non-standard durations (e.g. 6, 9, 18, 24 months) and tag `INCONSISTENT_REPORTING_PERIOD` in `irregularities`.
- **Realistic Irregularity Injection**:
  - Missing audit telemetry: $5\%$ probability $\to$ `NOT_ASSESSABLE` for Timeliness/Efficiency.
  - Missing compliance review: $4\%$ probability $\to$ `NOT_ASSESSABLE` for Compliance.
  - Sparse operational history: $12\%$ probability $\to$ `NOT_ASSESSABLE` for Past Renewal.
  - Outliers: $6\%$ probability (durations up to 210 days, error rates up to 22.5).
  - Malformed values: $4\%$ probability (negative headcounts, inverted counts).
- **Proof-of-Concept Synthetic Label Rule** ([`config/label_rule.json`](config/label_rule.json)):
  - **Explicit Statement**: *"A synthetic proof-of-concept label-generation rule created by the submitter; it is not a real-world empanelment policy."*
  - Applies weighted multi-attribute linear utility with non-linear missingness penalties and an $8\%$ stochastic noise rate to simulate real-world adjudication noise.

---

### 1.4 Classification & Evaluation Methodology (D3, D4)

Implemented in [`src/models/model_trainer.py`](src/models/model_trainer.py) and [`src/evaluation/metrics.py`](src/evaluation/metrics.py):

- **Models Implemented**:
  1. Baseline Classifier (`DummyClassifier`, majority/uniform strategy)
  2. Gradient Boosting Classifier (`GradientBoostingClassifier`)
  3. Support Vector Machine (`SVC` with RBF/linear kernels and calibrated probabilities)
  4. Multi-Layer Perceptron Neural Network (`MLPClassifier`)
- **Rigorous Leakage-Free Validation Discipline**:
  - Stratified 80/20 train/test split (Train: 96, Test: 24, Seed: `20260824`).
  - Feature extraction statistics (medians, missing indicators) are fit strictly on the training split.
  - **Model Selection Protocol**: 5-Fold Stratified Cross-Validation is run exclusively on training data to tune hyperparameters and select the winning model using **Validation/CV Macro-F1 (`cv_best_score`)**.
  - **Held-Out Test Set**: The held-out test split is evaluated exactly once after model selection to report unbiased test metrics. Test labels are never used for model selection or scoring.
- **Primary Comparison Metric**: **Macro-F1** (arithmetic mean of class-0 and class-1 F1 scores).

---

### 1.5 Continuous Scoring & Ranking Methodology (D5)

Implemented in [`src/scoring/engine.py`](src/scoring/engine.py) and [`src/ranking/engine.py`](src/ranking/engine.py):

- **Hybrid Continuous Score Formulation (Section 2.5)**:
  $$
  S_{\text{raw}} = w_{\text{features}} \cdot S_{\text{features}} + w_{\text{model}} \cdot (P(\text{RENEWED} \mid X) \times 100.0)
  $$

  where:- $S_{\text{features}} = \sum_{i \in \text{valid}} w_i^* S_i$ is the reweighted linear combination of valid normalized feature scores.
  - $P(\text{RENEWED} \mid X) \times 100.0$ is the continuous renewal probability emitted by the winning trained ML model.
  - $w_{\text{features}} = 0.75$ and $w_{\text{model}} = 0.25$ (both configurable in [`config/scoring_config.json`](config/scoring_config.json)).
- **Missing Feature Audit Penalisation**:
  $$
  \text{Final Score} = \max\left(0, \min\left(100, S_{\text{raw}} - (\text{count}(\text{NOT\_ASSESSABLE}) \times 2.0)\right)\right)
  $$
- **Deterministic Multi-Tier Ranking**:
  Sort key: (1) `performance_score` (Desc), (2) `compliance_adherence` (Desc), (3) `audit_timeliness` (Desc), (4) `audit_efficiency_inverted` (Desc), (5) `organisation_id` (Asc).
- **Decision Boundary Uncertainty Detection**:
  Decision threshold: $60.0$, Uncertainty margin: $\pm 3.0$ points ($[57.0, 63.0]$).
  Scores within margin are tagged `is_boundary_uncertain: true`, `boundary_flag: "UNCERTAIN_RENEWAL"`, and recommended for `RENEWAL_CONDITIONAL_PEER_REVIEW`.
- **Historical Output Preservation**:
  Every scoring and ranking execution writes output to `output/` and automatically saves an immutable snapshot in `output/history/v<version>/` ensuring previous version outputs are never silently overwritten.

---

### 1.6 Traceability Mechanism (Section 2.6)

Every output record in [`output/ranking.json`](output/ranking.json) contains full provenance:

$$
\text{Rank / Score} \longrightarrow \text{Model Output } [P(\text{RENEWED}), \text{Model Version}] \longrightarrow \text{Normalized Scores} \longrightarrow \text{Raw Variables} \longrightarrow \text{Active Registry Versions}
$$

A reviewer can take any ranked organisation, inspect its continuous score, verify the model probability contribution, inspect individual feature scores, view raw counts, and verify active configuration hashes in `config/version_registry.json`.

---

## PART 2: D11 — OPERATOR RUNBOOK

### 2.1 Prerequisites & Supported Environment

- **Operating System**: Windows (x86_64) or compatible Python environment.
- **Python Version**: CPython 3.12 (64-bit). The vendored offline package wheels in `deploy/offline_package/` are compiled for `cp312-win_amd64` and pure-Python `py3-none-any`.
- **Network Status**: Zero external internet access required (air-gapped compliant).

### 2.2 Air-Gapped Installation & Diagnostic Verification

```bash
# 1. Install all vendored scientific & ML dependencies offline from deploy/offline_package/
python deploy/install_offline.py

# 2. Execute offline diagnostic verification and pipeline smoke test
python deploy/verify_install.py
```

### 2.3 End-to-End Pipeline Execution

```bash
# Step 1: Generate synthetic population (120 organisations, seed 20260824)
python src/data_generator/generate_dataset.py --n 120 --seed 20260824 --output data/sample_dataset.json

# Step 2: Train models, execute 5-fold CV, and compute renewal probabilities
python src/models/model_trainer.py --dataset data/sample_dataset.json --config config/model_config.json --output output/evaluation_results.json --seed 20260824

# Step 3: Compute model-augmented continuous performance scores
python src/scoring/engine.py --input data/sample_dataset.json --config config/scoring_config.json --models output/evaluation_results.json --output output/organisation_scores.json

# Step 4: Generate deterministic rankings with boundary uncertainty detection
python src/ranking/engine.py --scores output/organisation_scores.json --config config/scoring_config.json --output output/ranking.json
```

### 2.4 Test Suite Execution

```bash
# Run all unit tests and formal acceptance tests (AT-1 through AT-8)
python -m unittest discover -s tests
```

### 2.5 Controlled Update Process

```bash
# Apply a verified configuration update with approval logging and hash updates
python deploy/update_process.py --version 1.1.0 --author "Chief Auditor" --approved-by "Empanelment Board" --changes "Updated compliance scoring weight"
```

---

## PART 3: D12 — RESULTS AND ANALYSIS

### 3.1 Observed Experimental Results (from `output/evaluation_results.json`)

- **Dataset Size**: 120 organisations (Train split: 96, Held-out Test split: 24).
- **Class Distribution**: 99 Renewed ($82.5\%$), 21 Not Renewed ($17.5\%$).
- **Irregular Data Count**: 44 organisations ($36.7\%$) carried at least one irregularity (missing data, sparse history, outliers, malformed inputs, inconsistent reporting periods).

#### Empirical Model Comparison Results:

| Model                                           | 5-Fold CV Macro-F1 (Selection Metric) | Test Macro-F1 (Held-Out) |  Test Accuracy  |  Test Binary F1  | Best Selected Hyperparameters                 |       Selection Status       |
| :---------------------------------------------- | :-----------------------------------: | :----------------------: | :--------------: | :--------------: | :-------------------------------------------- | :---------------------------: |
| **Simple Baseline** (`DummyClassifier`) |           **0.4654**           |          0.5636          |      0.6250      |      0.7273      | `strategy: 'uniform'`                       |           Baseline           |
| **Gradient Boosting** (`GBDT`)          |           **0.6412**           |     **0.7491**     | **0.8750** | **0.9268** | `n_estimators: 100, lr: 0.05, max_depth: 3` | **Selected Best Model** |
| **Support Vector Machine** (`SVC`)      |           **0.5523**           |          0.8500          |      0.9167      |      0.9500      | `C: 10.0, kernel: 'rbf'`                    |           Evaluated           |
| **Neural Network** (`MLPClassifier`)    |           **0.5895**           |          0.6190          |      0.8333      |      0.9048      | `hidden_layers: [16], alpha: 0.01`          |           Evaluated           |

- **Winning Model Selection**: Gradient Boosting (`gradient_boosting`) was selected based on the highest 5-Fold Stratified Cross-Validation Macro-F1 score ($0.6412$) on the training split, without using held-out test data for selection.
- **Held-Out Test Performance of Selected Model**:
  - Test Macro-F1: $0.7491$
  - Test Accuracy: $0.8750$
  - Test Binary F1: $0.9268$
  - **Confusion Matrix**:
    $$
    \begin{pmatrix} \text{True Negative}=2 & \text{False Positive}=2 \\ \text{False Negative}=1 & \text{True Positive}=19 \end{pmatrix}
    $$
  - **Per-Class Breakdown**:
    - Class 0 (NOT_RENEWED): Precision = $0.6667$, Recall = $0.5000$, F1 = $0.5714$, Support = 4.
    - Class 1 (RENEWED): Precision = $0.9048$, Recall = $0.9500$, F1 = $0.9268$, Support = 20.

### 3.2 Scoring & Ranking Observations (from `output/ranking.json`)

- **Total Ranked**: 120 organisations.
- **Score Distribution**: Mean = $78.07$, Max = $90.19$ (`ORG-0049`), Min = $38.92$ (`ORG-0004`).
- **Boundary Uncertainty**: Exactly 8 organisations ($6.67\%$) fell within the $\pm 3.0$ uncertainty band $[57.0, 63.0]$ and were tagged with `is_boundary_uncertain: true` and `boundary_flag: "UNCERTAIN_RENEWAL"`.

### 3.3 Limitations, Risks, and Honest Reporting of Limits

1. **Synthetic Dataset Limits**: Results are derived from synthetic data generated by the submitter against a synthetic proof-of-concept label rule. Such results constitute a proof of concept and not a validated production monitoring system.
2. **Transfer Risk**: Real empanelment distributions may feature different noise characteristics and non-linear interactions.
3. **Small Sample Constraints**: With 120 organisations, minority class test support is 4 samples, leading to wider confidence intervals on class-0 recall.
4. **Judgement Classifications**: Weights (e.g. 50/50 scale, compliance weight 0.20, uncertainty margin $\pm 3.0$) are expert engineering judgements and require formal empirical calibration on historical empanelment archives.

---

## PART 4: D13 — COMPARATIVE STUDY OF MODELS AND TECHNIQUES

This comparative study evaluates at least three viable alternatives for each of the five computational components across all seven mandatory criteria.

### Comparison Basis & Classification Tags:

- `[EXPERIMENTAL RESULT]`: Measured empirically on this dataset/pipeline.
- `[LITERATURE/KNOWN]`: Established mathematical or computational characteristics.
- `[ENGINEERING JUDGEMENT]`: Domain-informed design rationale.

---

### 4.1 Component 1: Renewal Outcome Classification

| Alternative                                          | Accuracy / Performance                                    | Interpretability                                    | Data Requirements                                                                     | Computational Cost        | Air-Gapped Suitability   | Ranking Stability                               | Known Failure Modes                | Selection Status |
| :--------------------------------------------------- | :-------------------------------------------------------- | :-------------------------------------------------- | :------------------------------------------------------------------------------------ | :------------------------ | :----------------------- | :---------------------------------------------- | :--------------------------------- | :--------------- |
| **1. Majority/Uniform Baseline** (Rule-based)  | Poor (Macro-F1: 0.5636)`[EXPERIMENTAL]`                 | High (Trivial logic)`[LITERATURE]`                | Minimal ($N \ge 1$) `[LITERATURE]` | Extremely Low ($<1$ ms) `[EXPERIMENTAL]` | Excellent`[LITERATURE]` | High`[LITERATURE]`     | Incapable of distinguishing agency quality.     | Rejected                           |                  |
| **2. Gradient Boosted Trees** (Learning-based) | Best CV Macro-F1 (0.6412, Test: 0.7491)`[EXPERIMENTAL]` | High (Feature importances & splits)`[LITERATURE]` | Moderate ($N \ge 50$) `[LITERATURE]` | Low ($0.32$ s) `[EXPERIMENTAL]`        | Excellent`[LITERATURE]` | High`[LITERATURE]`     | Overfitting on small noisy datasets.            | **Selected (Winning Model)** |                  |
| **3. Support Vector Machine** (Statistical)    | Moderate CV (0.5523, Test: 0.8500)`[EXPERIMENTAL]`      | Moderate (Kernel boundaries)`[LITERATURE]`        | Low-Moderate ($N \ge 30$) `[LITERATURE]` | Low ($0.18$ s) `[EXPERIMENTAL]`    | Excellent`[LITERATURE]` | High`[LITERATURE]`     | Sensitive to feature scaling and outlier noise. | Evaluated                          |                  |
| **4. Neural Network (MLP)** (Learning-based)   | Moderate CV (0.5895, Test: 0.6190)`[EXPERIMENTAL]`      | Low (Non-linear weights)`[LITERATURE]`            | High ($N \ge 100$) `[LITERATURE]` | Moderate ($0.85$ s) `[EXPERIMENTAL]`      | Excellent`[LITERATURE]` | Moderate`[LITERATURE]` | Convergence sensitivity on small samples.       | Evaluated                          |                  |

- **Selection Justification**: Gradient Boosted Trees achieved the highest 5-fold cross-validation Macro-F1 score ($0.6412$) on the training folds and robust test performance ($0.7491$), with tree-based split interpretability.
- **What would change decision**: If sample size is substantially scaled ($N > 10,000$) with deep non-linear interaction patterns, Multi-Layer Perceptrons would be re-evaluated.

---

### 4.2 Component 2: Missing-Data Handling

| Alternative                                                   | Accuracy / Performance                                        | Interpretability                              | Data Requirements                                                              | Computational Cost                        | Air-Gapped Suitability    | Ranking Stability                                           | Known Failure Modes                                          | Selection Status   |
| :------------------------------------------------------------ | :------------------------------------------------------------ | :-------------------------------------------- | :----------------------------------------------------------------------------- | :---------------------------------------- | :------------------------ | :---------------------------------------------------------- | :----------------------------------------------------------- | :----------------- |
| **1. Complete Case Analysis** (Listwise Deletion)       | Severe bias (Drops 36.7% of samples)`[EXPERIMENTAL]`        | High (Simple drop)`[LITERATURE]`            | Requires 100% complete data`[LITERATURE]`                                    | Negligible`[LITERATURE]`                | Excellent`[LITERATURE]` | Very Poor (Large rank swings)`[LITERATURE]`               | Discards valid agencies; discriminates against new entrants. | Rejected           |
| **2. Indicator + Median Imputation** (Statistical)      | High (Preserves all samples, informs model)`[EXPERIMENTAL]` | High (Explicit missing flags)`[LITERATURE]` | Low`[LITERATURE]`                                                            | Negligible ($<1$ ms) `[EXPERIMENTAL]` | Excellent`[LITERATURE]` | High`[LITERATURE]`                                        | Median may not represent extreme sub-populations.            | **Selected** |
| **3. K-Nearest Neighbours Imputation** (Learning-based) | Moderate`[LITERATURE]`                                      | Moderate (Distance-based)`[LITERATURE]`     | High ($N \ge 100$) `[LITERATURE]` | Moderate ($O(N^2)$) `[LITERATURE]` | Good`[LITERATURE]`                      | Moderate`[LITERATURE]`  | Distance metrics corrupted by missing-feature combinations. | Rejected                                                     |                    |

- **Selection Justification**: Indicator + Median Imputation guarantees zero crashes, preserves full provenance, prevents data leakage by computing medians strictly on training folds, and records explicit missing flags in outputs.
- **What would change decision**: If features exhibit strong linear correlation matrices, multivariate iterative imputation (MICE) would be evaluated.

---

### 4.3 Component 3: Feature Scaling and Normalisation

| Alternative                                           | Accuracy / Performance                                              | Interpretability                                                              | Data Requirements                               | Computational Cost           | Air-Gapped Suitability    | Ranking Stability        | Known Failure Modes                                    | Selection Status               |
| :---------------------------------------------------- | :------------------------------------------------------------------ | :---------------------------------------------------------------------------- | :---------------------------------------------- | :--------------------------- | :------------------------ | :----------------------- | :----------------------------------------------------- | :----------------------------- |
| **1. Standard Z-Score Scaling**                 | High for Gaussian features`[LITERATURE]`                          | Moderate (Mean 0, Std 1)`[LITERATURE]`                                      | Moderate`[LITERATURE]`                        | Low`[LITERATURE]`          | Excellent`[LITERATURE]` | Moderate`[LITERATURE]` | Unbounded output range; distorted by extreme outliers. | Used for ML                    |
| **2. Min-Max Normalisation**                    | High for bounded metrics`[LITERATURE]`                            | High (Direct 0–100%)`[LITERATURE]`                                         | Low`[LITERATURE]`                             | Low`[LITERATURE]`          | Excellent`[LITERATURE]` | High`[LITERATURE]`     | Outliers compress valid feature variance.              | Rejected for Raw               |
| **3. Domain-Bounded Directional Normalisation** | High (Aligns with operational semantics)`[ENGINEERING JUDGEMENT]` | Highest (0–100 intuitive score, higher is better)`[ENGINEERING JUDGEMENT]` | Minimal (Domain bounds in JSON)`[LITERATURE]` | Negligible`[EXPERIMENTAL]` | Excellent`[LITERATURE]` | High`[LITERATURE]`     | Requires expert-specified bounds in configuration.     | **Selected for Scoring** |

- **Selection Justification**: Domain-bounded directional normalization maps each metric to an intuitive 0–100 scale, correctly inverting metrics where lower is better (Efficiency, Error Rate).
- **What would change decision**: If statutory bounds are abolished in favor of empirical percentile distributions, rank-based quantile scaling would be adopted.

---

### 4.4 Component 4: Scoring Aggregation

| Alternative                                         | Accuracy / Performance                                                  | Interpretability                                                                 | Data Requirements                              | Computational Cost                  | Air-Gapped Suitability    | Ranking Stability                  | Known Failure Modes                                    | Selection Status   |
| :-------------------------------------------------- | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :--------------------------------------------- | :---------------------------------- | :------------------------ | :--------------------------------- | :----------------------------------------------------- | :----------------- |
| **1. Weighted Linear Combination**            | High (Robust & predictable)`[LITERATURE]`                             | Highest (Linear explainability)`[LITERATURE]`                                  | None (Parametric JSON weights)`[LITERATURE]` | Negligible`[EXPERIMENTAL]`        | Excellent`[LITERATURE]` | High`[LITERATURE]`               | Ignores non-linear feature synergies.                  | Baseline Component |
| **2. Pure Model-Predicted Probability**       | Moderate`[EXPERIMENTAL]`                                              | Low (Black-box probability)`[LITERATURE]`                                      | High training data`[LITERATURE]`             | Moderate`[LITERATURE]`            | Good`[LITERATURE]`      | Poor near boundary`[LITERATURE]` | Uncalibrated probabilities produce erratic ranks.      | Rejected           |
| **3. Hybrid Linear + Calibrated Model Score** | Highest (Combines explainability + non-linear signal)`[EXPERIMENTAL]` | High (Linear breakdown + model probability component)`[ENGINEERING JUDGEMENT]` | Moderate`[LITERATURE]`                       | Low ($<10$ ms) `[EXPERIMENTAL]` | Excellent`[LITERATURE]` | High`[EXPERIMENTAL]`             | Requires balancing model vs feature weights in config. | **Selected** |

- **Selection Justification**: Hybrid Combination ($75\%$ Multi-Attribute Feature Composite $+ 25\%$ Model Renewal Probability) satisfies Section 2.5 by integrating model output into a continuous performance score while preserving transparent feature accountability.
- **What would change decision**: If legal mandate requires pure linear scoring without ML influence, setting `model_output_weight: 0.0` in JSON reverts to pure linear aggregation without code changes.

---

### 4.5 Component 5: Ranking and Uncertainty Handling

| Alternative                                              | Accuracy / Performance                           | Interpretability                                                            | Data Requirements        | Computational Cost                     | Air-Gapped Suitability    | Ranking Stability                               | Known Failure Modes                             | Selection Status   |
| :------------------------------------------------------- | :----------------------------------------------- | :-------------------------------------------------------------------------- | :----------------------- | :------------------------------------- | :------------------------ | :---------------------------------------------- | :---------------------------------------------- | :----------------- |
| **1. Pure Ordinal Sort (No Ties/Uncertainty)**     | Poor for edge cases`[LITERATURE]`              | Moderate`[LITERATURE]`                                                    | None`[LITERATURE]`     | Low`[LITERATURE]`                    | Excellent`[LITERATURE]` | Poor (Arbitrary tie resolution)`[LITERATURE]` | Ignores decision boundary uncertainty.          | Rejected           |
| **2. Multi-Tier Sort + Boundary Uncertainty Flag** | Highest (Transparent & robust)`[EXPERIMENTAL]` | Highest (Clear hierarchy + uncertainty warnings)`[ENGINEERING JUDGEMENT]` | None`[LITERATURE]`     | Low ($O(N \log N)$) `[LITERATURE]` | Excellent`[LITERATURE]` | High`[EXPERIMENTAL]`                          | Requires defining uncertainty margin in config. | **Selected** |
| **3. Tournament / Pairwise Ranking**               | Moderate`[LITERATURE]`                         | Low (Complex win-loss matrix)`[LITERATURE]`                               | Moderate`[LITERATURE]` | High ($O(N^2)$) `[LITERATURE]`     | Good`[LITERATURE]`      | Moderate`[LITERATURE]`                        | Inconsistencies / non-transitive loops.         | Rejected           |

- **Selection Justification**: Multi-tier deterministic sorting ensures consistent ordering even under identical scores, while boundary uncertainty detection ($\pm 3.0$ points) flags marginal cases for mandatory expert peer review.
- **What would change decision**: If peer reviews require confidence intervals derived from bootstrap sampling, probabilistic ranking distributions would be introduced.

---

### 4.6 Material References for D13 Literature Claims

1. **Breiman, L. (2001)**. *Random Forests / Statistical Modeling: The Two Cultures*. Machine Learning, 45(1), 5–32. (Foundational reference for tree ensemble stability and non-linear interactions).
2. **Friedman, J. H. (2001)**. *Greedy Function Approximation: A Gradient Boosting Machine*. The Annals of Statistics, 29(5), 1189–1232. (Mathematical foundation for Gradient Boosted Decision Trees and loss minimization).
3. **Cortes, C., & Vapnik, V. (1995)**. *Support-Vector Networks*. Machine Learning, 20(3), 273–297. (Maximal margin hyperplanes and kernel methods).
4. **Little, R. J., & Rubin, D. B. (2019)**. *Statistical Analysis with Missing Data* (3rd ed.). John Wiley & Sons. (Theoretical analysis of missingness indicators and listwise deletion bias).
5. **Keeney, R. L., & Raiffa, H. (1993)**. *Decisions with Multiple Objectives: Preferences and Value Trade-Offs*. Cambridge University Press. (Theoretical foundation for Multi-Attribute Utility Theory and weighted linear combinations).

---

## PART 5: HUMAN REVIEW CHECKS

### Acceptance Check 9: Worked Traceability Examples (3 Actual Organisations)

#### Example 1: `ORG-0049` (Rank 1 — Top Performer with Malformed Feature Injection)

1. **Organisation ID**: `ORG-0049`
2. **Raw Input Variables** ([`data/sample_dataset.json`](data/sample_dataset.json)):
   - `team_size: -5` (injected `MALFORMED_NEGATIVE_TEAM_SIZE`), `years_operational: 5.4`, `eligible_completed_audits: 127`, `on_time_completed_audits: 97`, `mean_duration_days: 27.07`, `reviewed_reports: 122`, `identified_discrepancies: 7`, `applicable_controls: 60`, `compliant_applicable_controls: 58`, `historical_renewals: 2`, `historical_revocations: 0`, `reporting_period`: Standard 12-month annual cycle.
3. **Computed Feature Values**:
   - Audit Timeliness: `76.38%`
   - Audit Efficiency: `27.07` days
   - Error Rate: `0.057` discrepancies/report
   - Agency Scale: `None`
   - Compliance Adherence: `96.67%`
   - Past Renewal Status: `100.00%`
4. **Feature Validity / Status**:
   - Timeliness: `VALID`, Efficiency: `VALID`, Error Rate: `VALID`, Scale: `MALFORMED`, Compliance: `VALID`, Past Renewal: `VALID`.
5. **Normalized Feature Scores ($S_i \in [0, 100]$)**:
   - Timeliness: `76.38` | Efficiency: `78.66` | Error Rate: `99.05` | Scale: `None` | Compliance: `96.67` | Past Renewal: `100.00`.
6. **Feature Weights Applied (Dynamic Renormalization)**:
   - Timeliness: `0.2222`, Efficiency: `0.1667`, Error Rate: `0.1667`, Scale: `0.0000`, Compliance: `0.2222`, Past Renewal: `0.2222` (Base weights reweighted over 5 valid features).
7. **Feature Composite Score ($S_{\text{features}}$)**:
   - $S_{\text{features}} = 0.2222(76.38) + 0.1667(78.66) + 0.1667(99.05) + 0.2222(96.67) + 0.2222(100.00) = \mathbf{90.30}$.
8. **Model Name & Version**: `gradient_boosting` (`v1.0.0`).
9. **Model Predicted Renewal Probability**: $P(\text{RENEWED} \mid X) = 0.9789 \to S_{\text{model}} = 97.89$.
10. **Scoring Weights Applied**: Feature composite weight = `0.75`, Model output weight = `0.25`.
11. **Missing/Malformed Feature Penalty**: $-2.00$ (1 malformed feature: `agency_scale`).
12. **Final Continuous Performance Score**:
    - $\text{Score} = (0.75 \times 90.30) + (0.25 \times 97.89) - 2.00 = 67.725 + 24.4725 - 2.00 = 92.195 - 2.00 = \mathbf{90.19}$.
13. **Final Rank**: **Rank 1** (out of 120 organisations in [`output/ranking.json`](output/ranking.json)).
14. **Empanelment Recommendation**: `RECOMMEND_RENEWAL`.
15. **Decision Confidence**: `HIGH`.
16. **Boundary Uncertainty Status**: `False` (`boundary_flag: "CONFIDENT_RENEWAL"`).

---

#### Example 2: `ORG-0010` (Rank 61 — Median Performer with Unassessable History)

1. **Organisation ID**: `ORG-0010`
2. **Raw Input Variables** ([`data/sample_dataset.json`](data/sample_dataset.json)):
   - `team_size: 27`, `years_operational: 5.7`, `eligible_completed_audits: 107`, `on_time_completed_audits: 96`, `mean_duration_days: 6.03`, `reviewed_reports: 76`, `identified_discrepancies: 235` (injected error outlier), `applicable_controls: 60`, `compliant_applicable_controls: 50`, `historical_renewals: 0`, `historical_revocations: 0` (`SPARSE_OPERATIONAL_HISTORY_NEW_ENTRANT`), `reporting_period`: Standard 12-month annual cycle.
3. **Computed Feature Values**:
   - Audit Timeliness: `89.72%`
   - Audit Efficiency: `6.03` days
   - Error Rate: `3.092` discrepancies/report
   - Agency Scale: `41.25` points
   - Compliance Adherence: `83.33%`
   - Past Renewal Status: `None`
4. **Feature Validity / Status**:
   - Timeliness: `VALID`, Efficiency: `VALID`, Error Rate: `VALID`, Scale: `VALID`, Compliance: `VALID`, Past Renewal: `NOT_ASSESSABLE` (Zero historical renewal decisions).
5. **Normalized Feature Scores ($S_i \in [0, 100]$)**:
   - Timeliness: `89.72` | Efficiency: `100.00` | Error Rate: `48.47` | Scale: `41.25` | Compliance: `83.33` | Past Renewal: `None`.
6. **Feature Weights Applied (Dynamic Renormalization)**:
   - Timeliness: `0.2500`, Efficiency: `0.1875`, Error Rate: `0.1875`, Scale: `0.1250`, Compliance: `0.2500`, Past Renewal: `0.0000` (Base weights reweighted over 5 valid features).
7. **Feature Composite Score ($S_{\text{features}}$)**:
   - $S_{\text{features}} = 0.2500(89.72) + 0.1875(100.00) + 0.1875(48.47) + 0.1250(41.25) + 0.2500(83.33) = \mathbf{76.26}$.
8. **Model Name & Version**: `gradient_boosting` (`v1.0.0`).
9. **Model Predicted Renewal Probability**: $P(\text{RENEWED} \mid X) = 0.9763 \to S_{\text{model}} = 97.63$.
10. **Scoring Weights Applied**: Feature composite weight = `0.75`, Model output weight = `0.25`.
11. **Missing/Malformed Feature Penalty**: $-2.00$ (1 unassessable feature: `past_renewal_status`).
12. **Final Continuous Performance Score**:
    - $\text{Score} = (0.75 \times 76.26) + (0.25 \times 97.63) - 2.00 = 57.195 + 24.4075 - 2.00 = 81.6025 - 2.00 = \mathbf{79.60}$.
13. **Final Rank**: **Rank 61** (out of 120 organisations in [`output/ranking.json`](output/ranking.json)).
14. **Empanelment Recommendation**: `RECOMMEND_RENEWAL`.
15. **Decision Confidence**: `HIGH`.
16. **Boundary Uncertainty Status**: `False` (`boundary_flag: "CONFIDENT_RENEWAL"`).

---

#### Example 3: `ORG-0068` (Rank 103 — Boundary-Uncertain Agency with Missing Audits & Inconsistent Period)

1. **Organisation ID**: `ORG-0068`
2. **Raw Input Variables** ([`data/sample_dataset.json`](data/sample_dataset.json)):
   - `team_size: 5`, `years_operational: 8.4`, `eligible_completed_audits: 0` (`MISSING_AUDIT_ACTIVITY`), `on_time_completed_audits: 0`, `mean_duration_days: 20.46`, `reviewed_reports: 18`, `identified_discrepancies: 15`, `applicable_controls: 30`, `compliant_applicable_controls: 24`, `historical_renewals: 3`, `historical_revocations: 0`, `reporting_period`: 18-month non-standard cycle (`INCONSISTENT_REPORTING_PERIOD`).
3. **Computed Feature Values**:
   - Audit Timeliness: `None`
   - Audit Efficiency: `None`
   - Error Rate: `0.833` discrepancies/report
   - Agency Scale: `26.00` points
   - Compliance Adherence: `80.00%`
   - Past Renewal Status: `100.00%`
4. **Feature Validity / Status**:
   - Timeliness: `NOT_ASSESSABLE`, Efficiency: `NOT_ASSESSABLE`, Error Rate: `VALID`, Scale: `VALID`, Compliance: `VALID`, Past Renewal: `VALID`.
5. **Normalized Feature Scores ($S_i \in [0, 100]$)**:
   - Timeliness: `None` | Efficiency: `None` | Error Rate: `86.12` | Scale: `26.00` | Compliance: `80.00` | Past Renewal: `100.00`.
6. **Feature Weights Applied (Dynamic Renormalization)**:
   - Timeliness: `0.0000`, Efficiency: `0.0000`, Error Rate: `0.2308`, Scale: `0.1538`, Compliance: `0.3077`, Past Renewal: `0.3077` (Base weights reweighted over 4 valid features).
7. **Feature Composite Score ($S_{\text{features}}$)**:
   - $S_{\text{features}} = 0.2308(86.12) + 0.1538(26.00) + 0.3077(80.00) + 0.3077(100.00) = \mathbf{79.26}$.
8. **Model Name & Version**: `gradient_boosting` (`v1.0.0`).
9. **Model Predicted Renewal Probability**: $P(\text{RENEWED} \mid X) = 0.2420 \to S_{\text{model}} = 24.20$.
10. **Scoring Weights Applied**: Feature composite weight = `0.75`, Model output weight = `0.25`.
11. **Missing/Malformed Feature Penalty**: $-4.00$ ($2 \times 2.00$ for 2 unassessable features: `audit_timeliness`, `audit_efficiency`).
12. **Final Continuous Performance Score**:
    - $\text{Score} = (0.75 \times 79.26) + (0.25 \times 24.20) - 4.00 = 59.445 + 6.050 - 4.00 = 65.495 - 4.00 = \mathbf{61.49}$.
13. **Final Rank**: **Rank 103** (out of 120 organisations in [`output/ranking.json`](output/ranking.json)).
14. **Empanelment Recommendation**: `RENEWAL_CONDITIONAL_PEER_REVIEW`.
15. **Decision Confidence**: `LOW_BOUNDARY_UNCERTAIN`.
16. **Boundary Uncertainty Status**: `True` (Falls within the $\pm 3.0$ uncertainty interval $[57.0, 63.0]$ around the $60.0$ threshold; `boundary_flag: "UNCERTAIN_RENEWAL"`).

---

### Acceptance Check 10: Justification of Two Features and One Scoring Weight

1. **Feature 3 (Error Rate)**:

   - **Definition**: $\frac{\text{identified\_discrepancies}}{\text{reviewed\_reports}}$ (Range: $0.0–20.0$ discrepancies/report, lower is better).
   - **Classification**: `JUDGEMENT`
   - **Configuration Location**: [`config/feature_schema.json`](config/feature_schema.json)
   - **Derivation & Evidence**: The denominator choice (`reviewed_reports` rather than total audits) is an explicit submitter judgement. It ensures that an agency is not penalized for unreviewed audits and prevents distortions when QA audits lag operational completion.
2. **Feature 6 (Past Renewal Status)**:

   - **Definition**: $\frac{\text{renewals}}{\text{renewals} + \text{revocations}} \times 100$. If total decisions $= 0$, strictly flagged as `NOT_ASSESSABLE`.
   - **Classification**: `ASSIGNMENT_SPECIFIED`
   - **Configuration Location**: [`config/feature_schema.json`](config/feature_schema.json)
   - **Derivation & Evidence**: Explicitly mandated by Assignment 2 Section 5.2: *"If no historical decision exists: NOT_ASSESSABLE. Never treat absent history as a renewal or revocation."*
3. **Scoring Weight: Compliance Adherence ($0.20$)**:

   - **Value**: $0.20$ ($20\%$ of feature composite score).
   - **Classification**: `JUDGEMENT`
   - **Configuration Location**: [`config/scoring_config.json`](config/scoring_config.json)
   - **Derivation & Evidence**: Submitter expert judgement reflecting the statutory priority of cybersecurity control compliance in government auditing empanelment, weighted equally with Audit Timeliness ($0.20$) and Past Renewal Status ($0.20$).

---

### Acceptance Check 11: Justification of Two Selected Algorithmic Components

1. **Algorithmic Component 1: Classification Modeling**

   - **Alternatives Considered**: (1) Baseline Classifier, (2) Gradient Boosted Decision Trees (GBDT), (3) Support Vector Machine (SVC), (4) Multi-Layer Perceptron (MLP Neural Network).
   - **Selection Criteria**: 5-Fold CV Macro-F1 (on train data), Test Macro-F1, Interpretability, Data Requirements, Air-Gapped Suitability, Stability.
   - **Selected Approach**: Gradient Boosted Trees (`GBDT`).
   - **Supporting Evidence**: GBDT achieved the highest cross-validation score ($0.6412$) across training folds and strong held-out test performance ($0.7491$), while providing explainable decision splits.
   - **What Would Change Decision**: For very large dataset regimes ($N > 10,000$), deep neural network architectures would be evaluated.
2. **Algorithmic Component 2: Continuous Scoring Aggregation**

   - **Alternatives Considered**: (1) Pure Linear Feature Sum, (2) Pure Black-Box Model Probability Scoring, (3) Hybrid Multi-Attribute Feature + Calibrated Model Probability Combination.
   - **Selection Criteria**: Legal/Administrative Interpretability, Ranking Stability, Air-Gapped Feasibility, Section 2.5 Compliance.
   - **Selected Approach**: Hybrid Combination ($75\%$ Multi-Attribute Feature Composite $+ 25\%$ Model Renewal Probability).
   - **Supporting Evidence**: Pure model probability scoring exhibits high rank volatility near decision boundaries, whereas hybrid scoring provides transparent linear feature accountability while incorporating the non-linear renewal likelihood signal.
   - **What Would Change Decision**: If statutory empanelment guidelines prohibit machine learning outputs in administrative scoring, setting `model_output_weight: 0.0` in JSON reverts to pure linear aggregation without code modifications.

---

## Final Verification & Sign-Off

- **Code & Architecture**: FROZEN / VERIFIED.
- **Automated Tests**: All unit and acceptance tests passed.
- **Offline Air-Gapped Readiness**: Verified via `deploy/verify_install.py`.
- **Reproducibility**: Verified 3-run byte-identical output hashes.
- **Format Integrity**: Verified 100% strict JSON data pipeline.
