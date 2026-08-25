#!/usr/bin/env python3
"""
Model Training and Evaluation Pipeline.
Trains and compares Baseline, Gradient Boosting, SVM, and Neural Network models
using Stratified K-Fold Cross Validation and automated hyperparameter search.
Outputs reproducible evaluation metrics and per-organisation renewal probabilities
to output/evaluation_results.json.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.evaluation.metrics import ModelEvaluator
from src.features.extractor import FeatureExtractor, get_feature_matrix


class ModelPipelineTrainer:
    """Manages cross-validation, hyperparameter tuning, model comparison, probability output, and persistence."""

    def __init__(
        self,
        model_config_path: str = "config/model_config.json",
        seed: Optional[int] = None,
    ):
        if not os.path.isabs(model_config_path):
            model_config_path = os.path.join(PROJECT_ROOT, model_config_path)

        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config not found: {model_config_path}")
        with open(model_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.seed = seed if seed is not None else self.config.get("random_seed", 20260824)
        self.models_cfg = self.config.get("models", {})
        self.best_models: Dict[str, Any] = {}
        self.winning_model_name: Optional[str] = None
        self.winning_model: Optional[Any] = None
        self.extractor: Optional[FeatureExtractor] = None

    def run(
        self,
        dataset_path: str = "data/sample_dataset.json",
        output_path: str = "output/evaluation_results.json",
    ) -> Dict[str, Any]:
        """Executes the full training, validation, cross-validation, probability calculation, and evaluation pipeline."""
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(PROJECT_ROOT, dataset_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(PROJECT_ROOT, output_path)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        organisations = dataset.get("organisations", [])
        if len(organisations) < 10:
            raise ValueError("Dataset must contain at least 10 organisations for model training.")

        # 1. Stratified Train / Test Split
        split_cfg = self.config.get("data_split", {})
        test_size = split_cfg.get("test_size", 0.20)
        labels = [1 if o.get("label") == "RENEWED" else 0 for o in organisations]

        train_orgs, test_orgs = train_test_split(
            organisations,
            test_size=test_size,
            random_state=self.seed,
            stratify=labels if split_cfg.get("stratify", True) else None,
            shuffle=split_cfg.get("shuffle", True),
        )

        # 2. Fit Feature Extractor strictly on Train Split (No data leakage)
        schema_path = os.path.join(PROJECT_ROOT, "config", "feature_schema.json")
        self.extractor = FeatureExtractor(schema_path=schema_path)
        self.extractor.fit(train_orgs)

        X_train, y_train, feature_names, train_ids = self.extractor.transform(train_orgs)
        X_test, y_test, _, test_ids = self.extractor.transform(test_orgs)

        # 3. Cross-Validation Setup
        cv_cfg = self.config.get("validation_strategy", {})
        n_splits = cv_cfg.get("n_splits", 5)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        model_results: Dict[str, Any] = {}

        # 4. Train & Hyperparameter Search for each model
        # Model 1: Baseline
        base_grid = self.models_cfg.get("baseline", {}).get("hyperparameter_grid", {"strategy": ["most_frequent"]})
        clf_base = GridSearchCV(
            DummyClassifier(random_state=self.seed),
            base_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        clf_base.fit(X_train, y_train)
        self.best_models["baseline"] = clf_base.best_estimator_
        model_results["baseline"] = self._evaluate_model(clf_base, X_train, y_train, X_test, y_test, "baseline")

        # Model 2: Gradient Boosting
        gb_grid = self.models_cfg.get("gradient_boosting", {}).get(
            "hyperparameter_grid",
            {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [2, 3]},
        )
        clf_gb = GridSearchCV(
            GradientBoostingClassifier(random_state=self.seed),
            gb_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        clf_gb.fit(X_train, y_train)
        self.best_models["gradient_boosting"] = clf_gb.best_estimator_
        model_results["gradient_boosting"] = self._evaluate_model(clf_gb, X_train, y_train, X_test, y_test, "gradient_boosting")

        # Model 3: Support Vector Machine
        svm_grid = self.models_cfg.get("svm", {}).get(
            "hyperparameter_grid",
            {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
        )
        clf_svm = GridSearchCV(
            SVC(probability=True, random_state=self.seed),
            svm_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        clf_svm.fit(X_train, y_train)
        self.best_models["svm"] = clf_svm.best_estimator_
        model_results["svm"] = self._evaluate_model(clf_svm, X_train, y_train, X_test, y_test, "svm")

        # Model 4: Multi-Layer Perceptron (Neural Network)
        nn_grid = self.models_cfg.get("neural_network", {}).get(
            "hyperparameter_grid",
            {"hidden_layer_sizes": [(16,), (32, 16)], "alpha": [0.001, 0.01]},
        )
        clf_nn = GridSearchCV(
            MLPClassifier(max_iter=500, early_stopping=True, random_state=self.seed),
            nn_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        clf_nn.fit(X_train, y_train)
        self.best_models["neural_network"] = clf_nn.best_estimator_
        model_results["neural_network"] = self._evaluate_model(clf_nn, X_train, y_train, X_test, y_test, "neural_network")

        # 5. Determine Winning Model strictly using Cross-Validation Macro-F1 (without test data leakage)
        best_model_name = max(
            model_results.keys(),
            key=lambda m: model_results[m]["cv_best_score"],
        )
        self.winning_model_name = best_model_name
        self.winning_model = self.best_models[best_model_name]

        # 6. Compute Continuous Model Output (Probabilities) for ALL organisations in the dataset
        X_all, _, _, all_org_ids = self.extractor.transform(organisations)
        if hasattr(self.winning_model, "predict_proba"):
            all_probs = self.winning_model.predict_proba(X_all)[:, 1]
        else:
            # Fallback if model has no probabilities
            all_probs = self.winning_model.predict(X_all).astype(float)

        model_predictions: Dict[str, Dict[str, Any]] = {}
        for org_id, prob in zip(all_org_ids, all_probs):
            model_predictions[org_id] = {
                "organisation_id": org_id,
                "model_name": best_model_name,
                "model_version": self.config.get("model_config_version", "1.0.0"),
                "predicted_renewal_probability": round(float(prob), 4),
                "model_score_component": round(float(prob) * 100.0, 2),
            }

        comparison_summary = {
            m_name: {
                "cv_mean_macro_f1": model_results[m_name]["cv_best_score"],
                "macro_f1": model_results[m_name]["test_metrics"]["macro_f1"],
                "accuracy": model_results[m_name]["test_metrics"]["accuracy"],
                "binary_f1": model_results[m_name]["test_metrics"]["binary_f1"],
            }
            for m_name in model_results
        }

        rel_dataset_path = (
            os.path.relpath(dataset_path, PROJECT_ROOT).replace("\\", "/")
            if os.path.isabs(dataset_path)
            else dataset_path.replace("\\", "/")
        )

        evaluation_output = {
            "evaluation_metadata": {
                "evaluation_timestamp": "2026-08-24T16:00:00Z",
                "random_seed": self.seed,
                "dataset_path": rel_dataset_path,
                "train_sample_count": len(train_orgs),
                "test_sample_count": len(test_orgs),
                "features_used": feature_names,
                "primary_metric": "macro_f1",
                "selection_criterion": "5-fold_stratified_cv_macro_f1",
                "selected_best_model": best_model_name,
                "model_config_version": self.config.get("model_config_version", "1.0.0"),
            },
            "comparison_summary": comparison_summary,
            "detailed_model_results": model_results,
            "model_predictions": model_predictions,
        }

        # 7. Save results to output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_output, f, indent=2)

        return evaluation_output

    def _evaluate_model(
        self,
        grid_search: GridSearchCV,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        best_estimator = grid_search.best_estimator_
        y_test_pred = best_estimator.predict(X_test)
        y_test_prob = (
            best_estimator.predict_proba(X_test)[:, 1]
            if hasattr(best_estimator, "predict_proba")
            else None
        )

        test_metrics = ModelEvaluator.evaluate_predictions(y_test, y_test_pred, y_test_prob)

        return {
            "model_name": model_name,
            "best_hyperparameters": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in grid_search.best_params_.items()
            },
            "cv_best_score": round(float(grid_search.best_score_), 4),
            "test_metrics": test_metrics,
        }


def run_pipeline(
    dataset_path: str = "data/sample_dataset.json",
    model_config_path: str = "config/model_config.json",
    output_path: str = "output/evaluation_results.json",
    seed: Optional[int] = 20260824,
) -> Dict[str, Any]:
    """Top-level training and evaluation run command."""
    trainer = ModelPipelineTrainer(model_config_path=model_config_path, seed=seed)
    return trainer.run(dataset_path=dataset_path, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate renewal classification models.")
    parser.add_argument("--dataset", type=str, default="data/sample_dataset.json", help="Path to input JSON dataset")
    parser.add_argument("--config", type=str, default="config/model_config.json", help="Path to model config JSON")
    parser.add_argument("--output", type=str, default="output/evaluation_results.json", help="Output path for evaluation results JSON")
    parser.add_argument("--seed", type=int, default=20260824, help="Random seed for reproducibility")

    args = parser.parse_args()
    results = run_pipeline(
        dataset_path=args.dataset,
        model_config_path=args.config,
        output_path=args.output,
        seed=args.seed,
    )
    print("=" * 60)
    print("MODEL COMPARISON RESULTS (Primary Metric: Macro-F1)")
    print("=" * 60)
    for model_name, summary in results["comparison_summary"].items():
        print(
            f"- {model_name:20s}: Macro-F1 = {summary['macro_f1']:.4f} | "
            f"Accuracy = {summary['accuracy']:.4f} | CV-Macro-F1 = {summary['cv_mean_macro_f1']:.4f}"
        )
    print("=" * 60)
    print(f"Selected Best Model: {results['evaluation_metadata']['selected_best_model']}")
    print(f"Generated model renewal probabilities for {len(results['model_predictions'])} organisations.")
    print(f"Results saved to: {args.output}")
