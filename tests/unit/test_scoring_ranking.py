"""Unit tests for performance scoring, ranking engine, boundary uncertainty, and configuration independence."""

import copy
import json
import os
import tempfile
import unittest

from src.ranking.engine import RankingEngine
from src.scoring.engine import PerformanceScoringEngine


class TestScoringAndRanking(unittest.TestCase):
    """Verifies scoring engine, ranking logic, boundary uncertainty flags, and traceability."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        cls.scoring_config_path = os.path.join(cls.base_dir, "config", "scoring_config.json")
        cls.version_registry_path = os.path.join(cls.base_dir, "config", "version_registry.json")

    def test_direction_normalization(self):
        engine = PerformanceScoringEngine(
            scoring_config_path=self.scoring_config_path,
            version_registry_path=self.version_registry_path,
        )

        # Higher is better: 85% -> 85.0
        s_time = engine.normalize_feature("audit_timeliness", 85.0)
        self.assertEqual(s_time, 85.0)

        # Lower is better: duration 10 days (best) -> 100.0, 90 days (worst) -> 0.0
        s_eff_best = engine.normalize_feature("audit_efficiency", 10.0)
        s_eff_worst = engine.normalize_feature("audit_efficiency", 90.0)
        self.assertEqual(s_eff_best, 100.0)
        self.assertEqual(s_eff_worst, 0.0)

        # Lower is better: error rate 0.0 -> 100.0, 6.0 -> 0.0
        s_err_best = engine.normalize_feature("error_rate", 0.0)
        s_err_worst = engine.normalize_feature("error_rate", 6.0)
        self.assertEqual(s_err_best, 100.0)
        self.assertEqual(s_err_worst, 0.0)

    def test_configuration_independence(self):
        """Modifying weights in JSON config must alter the output scores and ranks without code changes."""
        org_sample = {
            "organisation_id": "TEST-ORG-01",
            "features": {
                "audit_timeliness": 95.0,
                "audit_efficiency": 15.0,
                "error_rate": 0.5,
                "agency_scale": 80.0,
                "compliance_adherence": 60.0,
                "past_renewal_status": 70.0,
            },
            "feature_status": {
                "audit_timeliness": "VALID",
                "audit_efficiency": "VALID",
                "error_rate": "VALID",
                "agency_scale": "VALID",
                "compliance_adherence": "VALID",
                "past_renewal_status": "VALID",
            },
        }

        # Run 1: Default config
        engine1 = PerformanceScoringEngine(scoring_config_path=self.scoring_config_path)
        score1 = engine1.score_organisation(org_sample)["performance_score"]

        # Run 2: Heavy weight on compliance adherence (where this org has lower score 60)
        with open(self.scoring_config_path, "r", encoding="utf-8") as f:
            cfg_modified = json.load(f)

        cfg_modified["weights"] = {
            "audit_timeliness": 0.05,
            "audit_efficiency": 0.05,
            "error_rate": 0.05,
            "agency_scale": 0.05,
            "compliance_adherence": 0.70,
            "past_renewal_status": 0.10,
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(cfg_modified, tf)
            mod_path = tf.name

        try:
            engine2 = PerformanceScoringEngine(scoring_config_path=mod_path)
            score2 = engine2.score_organisation(org_sample)["performance_score"]
            self.assertNotEqual(score1, score2, "Altering JSON weights MUST alter the computed score.")
        finally:
            if os.path.exists(mod_path):
                os.remove(mod_path)

    def test_model_output_weight_configuration_independence(self):
        """Modifying model_output_weight in scoring_config.json alters score without code edits."""
        org_sample = {
            "organisation_id": "ORG-MW-01",
            "features": {
                "audit_timeliness": 80.0,
                "audit_efficiency": 30.0,
                "error_rate": 1.0,
                "agency_scale": 60.0,
                "compliance_adherence": 85.0,
                "past_renewal_status": 90.0,
            },
            "feature_status": {f: "VALID" for f in [
                "audit_timeliness", "audit_efficiency", "error_rate",
                "agency_scale", "compliance_adherence", "past_renewal_status"
            ]},
        }

        # Mock model evaluation results where model probability differs from feature score
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf_eval:
            eval_payload = {
                "evaluation_metadata": {"selected_best_model": "neural_network", "model_config_version": "1.0.0"},
                "model_predictions": {
                    "ORG-MW-01": {
                        "organisation_id": "ORG-MW-01",
                        "model_name": "neural_network",
                        "model_version": "1.0.0",
                        "predicted_renewal_probability": 0.20,  # low model score (20.0)
                        "model_score_component": 20.0,
                    }
                }
            }
            json.dump(eval_payload, tf_eval)
            eval_path = tf_eval.name

        try:
            # Run 1: default model weight (0.25)
            engine1 = PerformanceScoringEngine(
                scoring_config_path=self.scoring_config_path,
                model_results_path=eval_path,
            )
            score1 = engine1.score_organisation(org_sample)["performance_score"]

            # Run 2: high model weight (0.75) vs feature weight (0.25)
            with open(self.scoring_config_path, "r", encoding="utf-8") as f:
                cfg_mod = json.load(f)
            cfg_mod["model_output_weight"] = 0.75
            cfg_mod["feature_composite_weight"] = 0.25

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf_cfg:
                json.dump(cfg_mod, tf_cfg)
                cfg_path = tf_cfg.name

            try:
                engine2 = PerformanceScoringEngine(
                    scoring_config_path=cfg_path,
                    model_results_path=eval_path,
                )
                score2 = engine2.score_organisation(org_sample)["performance_score"]
                self.assertNotEqual(score1, score2, "Altering model_output_weight in JSON must change the final score.")
                # Because model probability is low (20%), higher model weight must pull score down
                self.assertLess(score2, score1, "Higher weight on low model probability must decrease overall score.")
            finally:
                if os.path.exists(cfg_path):
                    os.remove(cfg_path)
        finally:
            if os.path.exists(eval_path):
                os.remove(eval_path)

    def test_boundary_uncertainty_flagging(self):
        ranker = RankingEngine(scoring_config_path=self.scoring_config_path)

        dummy_scored_output = {
            "scores": [
                {
                    "organisation_id": "ORG-HIGH",
                    "performance_score": 85.0,
                    "status": "SCORED",
                    "normalized_feature_scores": {"compliance_adherence": 85.0},
                },
                {
                    "organisation_id": "ORG-UNCERTAIN",
                    "performance_score": 59.5,  # Within [57.0, 63.0]
                    "status": "SCORED",
                    "normalized_feature_scores": {"compliance_adherence": 60.0},
                },
                {
                    "organisation_id": "ORG-LOW",
                    "performance_score": 42.0,
                    "status": "SCORED",
                    "normalized_feature_scores": {"compliance_adherence": 40.0},
                },
            ]
        }

        ranking_res = ranker.rank_organisations(dummy_scored_output)
        rankings = {r["organisation_id"]: r for r in ranking_res["rankings"]}

        self.assertFalse(rankings["ORG-HIGH"]["is_boundary_uncertain"])
        self.assertEqual(rankings["ORG-HIGH"]["boundary_flag"], "CONFIDENT_RENEWAL")

        self.assertTrue(rankings["ORG-UNCERTAIN"]["is_boundary_uncertain"])
        self.assertEqual(rankings["ORG-UNCERTAIN"]["boundary_flag"], "UNCERTAIN_RENEWAL")

        self.assertFalse(rankings["ORG-LOW"]["is_boundary_uncertain"])
        self.assertEqual(rankings["ORG-LOW"]["boundary_flag"], "CONFIDENT_NON_RENEWAL")

    def test_traceability_provenance(self):
        engine = PerformanceScoringEngine(
            scoring_config_path=self.scoring_config_path,
            version_registry_path=self.version_registry_path,
        )
        org_sample = {
            "organisation_id": "TRACE-01",
            "features": {"audit_timeliness": 80.0, "compliance_adherence": 85.0, "agency_scale": 50.0},
            "feature_status": {"audit_timeliness": "VALID", "compliance_adherence": "VALID", "agency_scale": "VALID"},
        }
        res = engine.score_organisation(org_sample)

        self.assertIn("provenance", res)
        prov = res["provenance"]
        self.assertIn("scoring_version", prov)
        self.assertIn("active_registry_versions", prov)
        self.assertIn("weights_applied", res)

    def test_model_output_contribution(self):
        """Verifies that trained model probabilities contribute to continuous score and model weight is configurable."""
        org_sample = {
            "organisation_id": "ORG-MODEL-01",
            "features": {
                "audit_timeliness": 80.0,
                "audit_efficiency": 30.0,
                "error_rate": 1.0,
                "agency_scale": 60.0,
                "compliance_adherence": 85.0,
                "past_renewal_status": 90.0,
            },
            "feature_status": {f: "VALID" for f in [
                "audit_timeliness", "audit_efficiency", "error_rate",
                "agency_scale", "compliance_adherence", "past_renewal_status"
            ]},
        }

        # Mock model evaluation results
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf_eval:
            eval_payload = {
                "evaluation_metadata": {
                    "selected_best_model": "neural_network",
                    "model_config_version": "1.0.0",
                },
                "model_predictions": {
                    "ORG-MODEL-01": {
                        "organisation_id": "ORG-MODEL-01",
                        "model_name": "neural_network",
                        "model_version": "1.0.0",
                        "predicted_renewal_probability": 0.95,
                        "model_score_component": 95.0,
                    }
                }
            }
            json.dump(eval_payload, tf_eval)
            eval_path = tf_eval.name

        try:
            engine = PerformanceScoringEngine(
                scoring_config_path=self.scoring_config_path,
                model_results_path=eval_path,
            )
            scored = engine.score_organisation(org_sample)
            self.assertIn("model_output", scored)
            self.assertEqual(scored["model_output"]["model_name"], "neural_network")
            self.assertEqual(scored["model_output"]["predicted_renewal_probability"], 0.95)
            self.assertEqual(scored["provenance"]["selected_model"], "neural_network")
            self.assertEqual(scored["provenance"]["model_version"], "1.0.0")
        finally:
            if os.path.exists(eval_path):
                os.remove(eval_path)


if __name__ == "__main__":
    unittest.main()
