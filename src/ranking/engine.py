#!/usr/bin/env python3
"""
Ranking Engine.
Ranks auditing organisations deterministically using model-augmented continuous performance scores,
applies multi-tier tie-breaking rules, detects decision boundary uncertainty,
and outputs structured rankings to output/ranking.json.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scoring.engine import PerformanceScoringEngine


class RankingEngine:
    """Ranks scored organisations with deterministic tie-breaking and boundary uncertainty analysis."""

    def __init__(self, scoring_config_path: str = "config/scoring_config.json"):
        if not os.path.isabs(scoring_config_path):
            scoring_config_path = os.path.join(PROJECT_ROOT, scoring_config_path)

        if not os.path.exists(scoring_config_path):
            raise FileNotFoundError(f"Scoring config not found: {scoring_config_path}")

        with open(scoring_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.threshold = self.config.get("renewal_decision_threshold", 60.0)
        self.boundary_cfg = self.config.get("boundary_uncertainty", {})
        self.margin = self.boundary_cfg.get("margin", 3.0)
        self.lower_boundary = self.boundary_cfg.get("lower_boundary", self.threshold - self.margin)
        self.upper_boundary = self.boundary_cfg.get("upper_boundary", self.threshold + self.margin)

    def rank_organisations(self, scored_output: Dict[str, Any]) -> Dict[str, Any]:
        """Performs deterministic ranking and flags boundary uncertainties."""
        all_scores = scored_output.get("scores", [])

        # Filter scored vs unranked
        ranked_candidates = [
            s for s in all_scores if s.get("status") == "SCORED" and s.get("performance_score") is not None
        ]
        unranked_candidates = [
            s for s in all_scores if s.get("status") != "SCORED" or s.get("performance_score") is None
        ]

        # Multi-tier deterministic sort key:
        # 1. Performance Score (Desc)
        # 2. Compliance Adherence (Desc)
        # 3. Audit Timeliness (Desc)
        # 4. Audit Efficiency (Inverted / Ascending duration)
        # 5. Organisation ID (Ascending lexicographical tie-breaker)
        def sort_key(item: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
            score = item.get("performance_score", 0.0)
            norm_scores = item.get("normalized_feature_scores", {})

            comp = norm_scores.get("compliance_adherence") or 0.0
            time = norm_scores.get("audit_timeliness") or 0.0
            eff = norm_scores.get("audit_efficiency") or 0.0
            org_id = item.get("organisation_id", "")

            # Return tuple for descending sort (negate numerics, ascending org_id)
            return (-score, -comp, -time, -eff, org_id)

        ranked_candidates.sort(key=sort_key)

        ranked_results: List[Dict[str, Any]] = []
        for idx, item in enumerate(ranked_candidates, start=1):
            score = item["performance_score"]
            is_uncertain = self.lower_boundary <= score <= self.upper_boundary

            if is_uncertain:
                boundary_flag = self.boundary_cfg.get("uncertain_flag", "UNCERTAIN_RENEWAL")
                recommendation = "RENEWAL_CONDITIONAL_PEER_REVIEW"
                confidence = "LOW_BOUNDARY_UNCERTAIN"
            elif score > self.upper_boundary:
                boundary_flag = "CONFIDENT_RENEWAL"
                recommendation = "RECOMMEND_RENEWAL"
                confidence = "HIGH"
            else:
                boundary_flag = "CONFIDENT_NON_RENEWAL"
                recommendation = "RECOMMEND_NON_RENEWAL"
                confidence = "HIGH"

            rank_record = {
                "rank": idx,
                "organisation_id": item["organisation_id"],
                "performance_score": score,
                "feature_composite_score": item.get("feature_composite_score"),
                "model_output": item.get("model_output"),
                "recommendation": recommendation,
                "decision_confidence": confidence,
                "is_boundary_uncertain": is_uncertain,
                "boundary_flag": boundary_flag,
                "boundary_evaluation": {
                    "decision_threshold": self.threshold,
                    "uncertainty_margin": self.margin,
                    "uncertainty_range": [self.lower_boundary, self.upper_boundary],
                    "derivation_classification": self.boundary_cfg.get("derivation_classification", "judgement"),
                },
                "normalized_feature_scores": item.get("normalized_feature_scores"),
                "weights_applied": item.get("weights_applied"),
                "provenance": item.get("provenance"),
            }
            ranked_results.append(rank_record)

        # Append unranked candidates at the bottom with explicit status
        unranked_results: List[Dict[str, Any]] = []
        for item in unranked_candidates:
            unranked_results.append({
                "rank": None,
                "organisation_id": item["organisation_id"],
                "performance_score": None,
                "status": item.get("status", "UNRANKED_INSUFFICIENT_DATA"),
                "recommendation": "DEFER_PENDING_DATA_REMEDIATION",
                "provenance": item.get("provenance"),
            })

        uncertain_count = sum(1 for r in ranked_results if r["is_boundary_uncertain"])

        return {
            "ranking_metadata": {
                "timestamp": "2026-08-24T16:00:00Z",
                "total_organisations": len(all_scores),
                "ranked_count": len(ranked_results),
                "unranked_count": len(unranked_results),
                "uncertain_boundary_count": uncertain_count,
                "uncertainty_rate": round(uncertain_count / max(len(ranked_results), 1), 4),
                "scoring_config_version": self.config.get("scoring_version", "1.0.0"),
            },
            "rankings": ranked_results,
            "unranked_organisations": unranked_results,
        }


def generate_ranking(
    scores_input_path: str = "output/organisation_scores.json",
    scoring_config_path: str = "config/scoring_config.json",
    output_path: str = "output/ranking.json",
) -> Dict[str, Any]:
    """CLI helper to read scores, rank organisations, and output ranking.json."""
    if not os.path.isabs(scores_input_path):
        scores_input_path = os.path.join(PROJECT_ROOT, scores_input_path)
    if not os.path.isabs(scoring_config_path):
        scoring_config_path = os.path.join(PROJECT_ROOT, scoring_config_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    with open(scores_input_path, "r", encoding="utf-8") as f:
        scored_data = json.load(f)

    ranker = RankingEngine(scoring_config_path=scoring_config_path)
    ranking_data = ranker.rank_organisations(scored_data)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ranking_data, f, indent=2)

    # Save immutable historical archive copy
    version = ranking_data["ranking_metadata"].get("scoring_config_version", "1.0.0")
    history_dir = os.path.join(os.path.dirname(os.path.abspath(output_path)), "history", f"v{version}")
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, os.path.basename(output_path))
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(ranking_data, f, indent=2)

    return ranking_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank organisations and detect boundary uncertainty.")
    parser.add_argument("--scores", type=str, default="output/organisation_scores.json", help="Input scores JSON")
    parser.add_argument("--config", type=str, default="config/scoring_config.json", help="Scoring config JSON")
    parser.add_argument("--output", type=str, default="output/ranking.json", help="Output rankings JSON")

    args = parser.parse_args()
    res = generate_ranking(scores_input_path=args.scores, scoring_config_path=args.config, output_path=args.output)
    print(f"Ranked {res['ranking_metadata']['ranked_count']} organisations.")
    print(f"Identified {res['ranking_metadata']['uncertain_boundary_count']} boundary-uncertain organisations.")
    print(f"Saved rankings to: {args.output}")
