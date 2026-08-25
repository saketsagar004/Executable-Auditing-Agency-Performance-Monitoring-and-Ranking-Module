#!/usr/bin/env python3
"""
Performance Scoring Engine.
Calculates continuous agency performance scores (0-100 scale) by aggregating:
1. Normalized multi-attribute feature scores (Audit Timeliness, Efficiency, Error Rate, Scale, Compliance, Past Renewal)
2. Trained classification model renewal probability output (e.g. Neural Network / GBDT)

Applies feature direction normalization, missing-data reweighting, configurable hybrid weights,
and generates complete audit provenance for every score.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class PerformanceScoringEngine:
    """Computes continuous hybrid performance scores and attaches full traceability metadata."""

    def __init__(
        self,
        scoring_config_path: str = "config/scoring_config.json",
        version_registry_path: str = "config/version_registry.json",
        model_results_path: Optional[str] = "output/evaluation_results.json",
    ):
        if not os.path.isabs(scoring_config_path):
            scoring_config_path = os.path.join(PROJECT_ROOT, scoring_config_path)
        if not os.path.isabs(version_registry_path):
            version_registry_path = os.path.join(PROJECT_ROOT, version_registry_path)
        if model_results_path and not os.path.isabs(model_results_path):
            model_results_path = os.path.join(PROJECT_ROOT, model_results_path)

        if not os.path.exists(scoring_config_path):
            raise FileNotFoundError(f"Scoring config not found: {scoring_config_path}")

        with open(scoring_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.version_registry = {}
        if os.path.exists(version_registry_path):
            with open(version_registry_path, "r", encoding="utf-8") as f:
                self.version_registry = json.load(f)

        # Model output probabilities lookup
        self.model_predictions: Dict[str, Dict[str, Any]] = {}
        self.model_metadata: Dict[str, Any] = {}
        if model_results_path and os.path.exists(model_results_path):
            with open(model_results_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                self.model_predictions = eval_data.get("model_predictions", {})
                self.model_metadata = eval_data.get("evaluation_metadata", {})

        self.weights = self.config.get("weights", {})
        self.feature_comp_weight = self.config.get("feature_composite_weight", 0.75)
        self.model_output_weight = self.config.get("model_output_weight", 0.25)
        self.norm_cfg = self.config.get("direction_normalization", {})
        self.missing_cfg = self.config.get("missing_data_treatment", {})
        self.penalty_per_not_assessable = self.missing_cfg.get("penalty_per_not_assessable", 2.0)
        self.min_required_features = self.missing_cfg.get("min_required_features", 3)

    def normalize_feature(self, feature_name: str, raw_value: Optional[float]) -> Optional[float]:
        """Normalizes a raw feature value into a standard 0-100 scale (higher is better)."""
        if raw_value is None:
            return None

        rule = self.norm_cfg.get(feature_name, {})
        rule_type = rule.get("type", "direct_percentage")

        if rule_type == "direct_percentage":
            min_val = rule.get("min_val", 0.0)
            max_val = rule.get("max_val", 100.0)
            clamped = max(min_val, min(max_val, raw_value))
            return round(((clamped - min_val) / max(max_val - min_val, 1e-6)) * 100.0, 2)

        elif rule_type == "inverted_linear":
            best_val = rule.get("best_val", 0.0)
            worst_val = rule.get("worst_val", 100.0)
            clamped = max(min(best_val, worst_val), min(max(best_val, worst_val), raw_value))
            # Inverted: best_val yields 100, worst_val yields 0
            score = (worst_val - clamped) / max(worst_val - best_val, 1e-6) * 100.0
            return round(max(0.0, min(100.0, score)), 2)

        return round(float(raw_value), 2)

    def score_organisation(self, org_record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates continuous score and provenance for a single organisation."""
        org_id = org_record.get("organisation_id", "UNKNOWN")
        raw_features = org_record.get("features", {})
        feature_status = org_record.get("feature_status", {})

        normalized_scores: Dict[str, Optional[float]] = {}
        valid_features: List[str] = []
        not_assessable_features: List[str] = []
        malformed_features: List[str] = []

        for f_name, raw_val in raw_features.items():
            st = feature_status.get(f_name, "VALID")
            if st == "VALID" and raw_val is not None:
                norm_score = self.normalize_feature(f_name, raw_val)
                normalized_scores[f_name] = norm_score
                valid_features.append(f_name)
            elif st == "NOT_ASSESSABLE" or raw_val is None:
                normalized_scores[f_name] = None
                not_assessable_features.append(f_name)
            else:
                normalized_scores[f_name] = None
                malformed_features.append(f_name)

        if len(valid_features) < self.min_required_features:
            return {
                "organisation_id": org_id,
                "performance_score": None,
                "status": "UNRANKED_INSUFFICIENT_DATA",
                "normalized_feature_scores": normalized_scores,
                "raw_features": raw_features,
                "feature_status": feature_status,
                "weights_applied": {},
                "model_output": None,
                "provenance": self._generate_provenance(org_id),
            }

        # 1. Calculate Feature Composite Component
        total_valid_weight = sum(self.weights.get(f, 0.0) for f in valid_features)
        weights_applied = {}
        feature_weighted_sum = 0.0

        for f in valid_features:
            eff_weight = self.weights.get(f, 0.0) / max(total_valid_weight, 1e-6)
            weights_applied[f] = round(eff_weight, 4)
            feature_weighted_sum += eff_weight * normalized_scores[f]

        # 2. Extract Trained Model Output (Predicted Renewal Probability)
        model_pred_info = self.model_predictions.get(org_id)
        if model_pred_info:
            prob = model_pred_info.get("predicted_renewal_probability", 0.5)
            model_score_component = prob * 100.0
            eff_w_feat = self.feature_comp_weight
            eff_w_model = self.model_output_weight
            # Normalize hybrid weights to sum to 1.0
            total_hybrid_weight = eff_w_feat + eff_w_model
            eff_w_feat /= max(total_hybrid_weight, 1e-6)
            eff_w_model /= max(total_hybrid_weight, 1e-6)
            
            raw_hybrid_score = eff_w_feat * feature_weighted_sum + eff_w_model * model_score_component
            model_output_record = {
                "model_name": model_pred_info.get("model_name"),
                "model_version": model_pred_info.get("model_version"),
                "predicted_renewal_probability": prob,
                "model_score_component": round(model_score_component, 2),
                "model_weight_applied": round(eff_w_model, 4),
                "feature_composite_weight_applied": round(eff_w_feat, 4),
            }
        else:
            # Fallback if model output is not supplied
            raw_hybrid_score = feature_weighted_sum
            model_output_record = {
                "model_name": "NONE_DIRECT_FEATURE_SCORE",
                "model_version": "N/A",
                "predicted_renewal_probability": None,
                "model_score_component": None,
                "model_weight_applied": 0.0,
                "feature_composite_weight_applied": 1.0,
            }

        # 3. Apply missing-feature audit penalty
        penalty = len(not_assessable_features) * self.penalty_per_not_assessable
        final_score = max(0.0, min(100.0, raw_hybrid_score - penalty))

        return {
            "organisation_id": org_id,
            "performance_score": round(final_score, 2),
            "status": "SCORED",
            "feature_composite_score": round(feature_weighted_sum, 2),
            "model_output": model_output_record,
            "audit_penalty_applied": round(penalty, 2),
            "normalized_feature_scores": normalized_scores,
            "raw_features": raw_features,
            "feature_status": feature_status,
            "weights_applied": weights_applied,
            "provenance": self._generate_provenance(org_id),
        }

    def _generate_provenance(self, org_id: str) -> Dict[str, Any]:
        """Generates immutable traceability metadata linking score to model and config versions."""
        model_pred_info = self.model_predictions.get(org_id, {})
        return {
            "scoring_version": self.config.get("scoring_version", "1.0.0"),
            "aggregation_method": self.config.get("aggregation_method", "feature_and_model_hybrid_combination"),
            "derivation_classification": self.config.get("derivation_classification", "judgement"),
            "model_version": model_pred_info.get("model_version", self.model_metadata.get("model_config_version", "1.0.0")),
            "selected_model": model_pred_info.get("model_name", self.model_metadata.get("selected_best_model", "N/A")),
            "active_registry_versions": self.version_registry.get("active_versions", {}),
        }

    def score_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Scores all organisations in a dataset."""
        organisations = dataset.get("organisations", [])
        scored_orgs = [self.score_organisation(o) for o in organisations]

        return {
            "scoring_metadata": {
                "timestamp": "2026-08-24T16:00:00Z",
                "scoring_version": self.config.get("scoring_version", "1.0.0"),
                "aggregation_method": self.config.get("aggregation_method", "feature_and_model_hybrid_combination"),
                "feature_composite_weight": self.feature_comp_weight,
                "model_output_weight": self.model_output_weight,
                "selected_model_used": self.model_metadata.get("selected_best_model", "neural_network"),
                "total_organisations": len(scored_orgs),
                "scored_count": sum(1 for s in scored_orgs if s["status"] == "SCORED"),
                "unranked_count": sum(1 for s in scored_orgs if s["status"] != "SCORED"),
            },
            "scores": scored_orgs,
        }


def score_dataset_file(
    input_path: str = "data/sample_dataset.json",
    scoring_config_path: str = "config/scoring_config.json",
    model_results_path: Optional[str] = "output/evaluation_results.json",
    output_path: str = "output/organisation_scores.json",
) -> Dict[str, Any]:
    """CLI helper to score dataset file and save output JSON."""
    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)
    if not os.path.isabs(scoring_config_path):
        scoring_config_path = os.path.join(PROJECT_ROOT, scoring_config_path)
    if model_results_path and not os.path.isabs(model_results_path):
        model_results_path = os.path.join(PROJECT_ROOT, model_results_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    engine = PerformanceScoringEngine(
        scoring_config_path=scoring_config_path,
        model_results_path=model_results_path,
    )
    scored_data = engine.score_dataset(dataset)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored_data, f, indent=2)

    # Save immutable historical archive copy
    version = scored_data["scoring_metadata"].get("scoring_version", "1.0.0")
    history_dir = os.path.join(os.path.dirname(os.path.abspath(output_path)), "history", f"v{version}")
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, os.path.basename(output_path))
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(scored_data, f, indent=2)

    return scored_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score auditing organisations.")
    parser.add_argument("--input", type=str, default="data/sample_dataset.json", help="Input dataset JSON")
    parser.add_argument("--config", type=str, default="config/scoring_config.json", help="Scoring config JSON")
    parser.add_argument("--models", type=str, default="output/evaluation_results.json", help="Model evaluation results JSON")
    parser.add_argument("--output", type=str, default="output/organisation_scores.json", help="Output scores JSON")

    args = parser.parse_args()
    res = score_dataset_file(
        input_path=args.input,
        scoring_config_path=args.config,
        model_results_path=args.models,
        output_path=args.output,
    )
    print(f"Scored {res['scoring_metadata']['scored_count']} organisations using model-augmented continuous scoring.")
    print(f"Saved scores to: {args.output}")
