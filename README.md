# Auditing Agency Performance Monitoring and Ranking Module

This repository contains the complete, executable, clean-room implementation of the **Auditing Agency Performance Monitoring, Scoring, and Ranking Module**.

---

## 1. Repository Structure & Deliverables Mapping

```
PerformanceMonitoring_Saket_20260824/
│
├── config/
│   ├── feature_schema.json          # D1: Schema, formulas, bounds, missing rules, derivation classes
│   ├── generation_config.json       # D2: Synthetic distribution parameters, irregularity rates, seed
│   ├── label_rule.json              # D2: Proof-of-concept synthetic label generation rule
│   ├── scoring_config.json          # D1/D5: Feature weights, thresholds, tie-breaking, uncertainty margins
│   ├── model_config.json            # D3: Model hyperparameter search spaces, CV folds, seeds
│   └── version_registry.json        # D6: Version registry, configuration hashes, change history
│
├── data/
│   ├── generator/                   # Generator metadata
│   └── sample_dataset.json          # D2: Generated sample dataset (120 organisations)
│
├── src/
│   ├── data_generator/
│   │   ├── __init__.py
│   │   └── generate_dataset.py      # D2: Synthetic dataset generator CLI
│   ├── features/
│   │   ├── __init__.py
│   │   ├── validator.py             # Feature schema validator & non-assessable tagging
│   │   └── extractor.py             # Feature preprocessing & missingness indicators
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py              # D3: Baseline Classifier (DummyClassifier)
│   │   ├── gradient_boosting.py     # D3: Gradient Boosting Classifier
│   │   ├── svm.py                   # D3: Support Vector Machine Classifier
│   │   ├── neural_network.py        # D3: Multi-Layer Perceptron Neural Network
│   │   └── model_trainer.py         # D3: Cross-validation, hyperparameter tuning & evaluation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py               # D4: Macro-F1, confusion matrix & per-class metrics
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── engine.py                # D5: Multi-criteria performance scoring & provenance generator
│   ├── ranking/
│   │   ├── __init__.py
│   │   └── engine.py                # D5: Deterministic ranking & boundary uncertainty detection
│   └── versioning/
│       ├── __init__.py
│       └── registry.py              # D6: Version registry and audit trail manager
│
├── deploy/
│   ├── offline_package/             # D7: Vendored .whl dependency packages
│   ├── requirements.txt             # Pinned dependency manifest
│   ├── install_offline.py           # D7: Offline pip installer from local vendored wheels
│   ├── verify_install.py            # D7: Verification script checking offline environment
│   └── update_process.py            # D7: Checksum verification, malware scanning hook & update script
│
├── tests/
│   ├── acceptance/
│   │   ├── __init__.py
│   │   └── test_acceptance_suite.py # D8: Automated tests for AT-1 through AT-8
│   └── unit/
│       ├── __init__.py
│       ├── test_features.py         # Feature calculation & schema unit tests
│       ├── test_generator.py        # Synthetic generator unit tests
│       ├── test_models.py           # Model training & prediction unit tests
│       ├── test_scoring_ranking.py  # Scoring & ranking unit tests
│       └── test_versioning.py       # Versioning immutability unit tests
│
├── output/
│   ├── evaluation_results.json      # D4: Model comparison metrics & confusion matrices
│   ├── organisation_scores.json     # D9: Per-organisation scores with provenance
│   └── ranking.json                 # D9: Ordered ranking with boundary uncertainty tags
│
├── docs/
│   └── FINAL_END_TO_END_DOCUMENT.md # D10, D11, D12, D13: End-to-end design, derivation, runbook, analysis
│
└── README.md                        # Contents list, system overview & run instructions
```

---

## 2. Quick Start & Execution Commands

### Air-Gapped Installation & Diagnostic Check
```bash
# Install vendored packages offline
python deploy/install_offline.py

# Verify installation & offline readiness
python deploy/verify_install.py
```

### Complete End-to-End Pipeline Execution
```bash
# Step 1: Generate synthetic population dataset (120 organisations)
python src/data_generator/generate_dataset.py --n 120 --seed 20260824 --output data/sample_dataset.json

# Step 2: Train and evaluate classification models
python src/models/model_trainer.py --dataset data/sample_dataset.json --config config/model_config.json --output output/evaluation_results.json --seed 20260824

# Step 3: Compute continuous performance scores
python src/scoring/engine.py --input data/sample_dataset.json --config config/scoring_config.json --output output/organisation_scores.json

# Step 4: Generate deterministic rankings with boundary uncertainty detection
python src/ranking/engine.py --scores output/organisation_scores.json --config config/scoring_config.json --output output/ranking.json
```

### Automated Acceptance & Unit Test Suite
```bash
# Run all unit and acceptance tests (AT-1 through AT-8)
python -m unittest discover -s tests
```

---

## 3. Key Architectural Features

1. **Air-Gapped / Zero Network Access**: Fully operational in offline air-gapped environments with vendored dependencies in `deploy/offline_package/`.
2. **Strict JSON Data Pipeline**: All data, schemas, models, scores, rankings, and registries utilize strict JSON formatting with zero external database dependencies.
3. **Graceful Degradation**: Irregular or missing inputs produce recorded `NOT_ASSESSABLE` or `MALFORMED` statuses without pipeline failure.
4. **Deterministic Reproducibility**: Seeded random generation (`20260824`) produces 100% byte-identical outputs across repeated executions.
5. **No Hardcoded Constants**: All feature definitions, weights, thresholds, distributions, and model hyperparameters reside in JSON configuration files.
6. **Full Traceability**: Every score and ranking record provides complete provenance back to raw variables, configuration versions, and model configurations.
