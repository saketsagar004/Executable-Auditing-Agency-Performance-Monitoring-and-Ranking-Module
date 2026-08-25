#!/usr/bin/env python3
"""
Automated Acceptance Test Suite (AT-1 through AT-8).
Implements the 8 formal acceptance tests mandated in Section 6 of the Assignment 2 specification:
  AT-1: Fresh-clone execution
  AT-2: Offline installation & verification
  AT-3: Feature coverage check (all 6 features)
  AT-4: Deterministic reproducibility (3-run byte-identical check)
  AT-5: Configuration independence check (weight changes reflect without code edits)
  AT-6: Irregular-input handling (graceful degradation, recorded non-assessable entries)
  AT-7: Versioning check (audit trail, immutability, change history)
  AT-8: Model comparison check (Baseline, GBDT, SVM, Neural Network with Macro-F1)
"""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deploy.verify_install import run_full_verification
from src.data_generator.generate_dataset import generate_dataset
from src.models.model_trainer import run_pipeline
from src.ranking.engine import generate_ranking
from src.scoring.engine import score_dataset_file
from src.versioning.registry import VersionRegistryManager


class AutomatedAcceptanceTests(unittest.TestCase):
    """Executes the formal Assignment 2 Acceptance Test Suite (AT-1 to AT-8)."""

    @classmethod
    def setUpClass(cls):
        cls.config_dir = os.path.join(PROJECT_ROOT, "config")
        cls.schema_path = os.path.join(cls.config_dir, "feature_schema.json")
        cls.gen_config_path = os.path.join(cls.config_dir, "generation_config.json")
        cls.scoring_config_path = os.path.join(cls.config_dir, "scoring_config.json")
        cls.model_config_path = os.path.join(cls.config_dir, "model_config.json")

    def test_AT1_fresh_clone_execution(self):
        """AT-1: Complete pipeline executes end-to-end producing all expected output JSON artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = os.path.join(tmpdir, "sample_dataset.json")
            tmp_eval = os.path.join(tmpdir, "evaluation_results.json")
            tmp_scores = os.path.join(tmpdir, "organisation_scores.json")
            tmp_rankings = os.path.join(tmpdir, "ranking.json")

            # 1. Dataset Generation
            d_res = generate_dataset(n=40, seed=20260824, output_path=tmp_data)
            self.assertTrue(os.path.exists(tmp_data))
            self.assertEqual(len(d_res["organisations"]), 40)

            # 2. Model Training & Evaluation
            e_res = run_pipeline(dataset_path=tmp_data, output_path=tmp_eval, seed=20260824)
            self.assertTrue(os.path.exists(tmp_eval))
            self.assertIn("comparison_summary", e_res)

            # 3. Scoring (consuming trained model output)
            s_res = score_dataset_file(input_path=tmp_data, model_results_path=tmp_eval, output_path=tmp_scores)
            self.assertTrue(os.path.exists(tmp_scores))
            self.assertEqual(s_res["scoring_metadata"]["total_organisations"], 40)
            self.assertIsNotNone(s_res["scores"][0]["model_output"])

            # 4. Ranking
            r_res = generate_ranking(scores_input_path=tmp_scores, output_path=tmp_rankings)
            self.assertTrue(os.path.exists(tmp_rankings))
            self.assertEqual(r_res["ranking_metadata"]["total_organisations"], 40)

    def test_AT2_offline_installation(self):
        """AT-2: Offline package verification check passes with network disabled."""
        passed = run_full_verification()
        self.assertTrue(passed, "Offline installation and pipeline smoke test must pass.")

    def test_AT3_feature_coverage(self):
        """AT-3: All six features are computed, bounded, and defined in JSON configuration."""
        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        required = [
            "audit_timeliness",
            "audit_efficiency",
            "error_rate",
            "agency_scale",
            "compliance_adherence",
            "past_renewal_status",
        ]
        features = schema.get("features", {})
        for req in required:
            self.assertIn(req, features, f"Feature {req} missing from schema.")
            f_def = features[req]
            self.assertIn("formula", f_def)
            self.assertIn("permitted_range", f_def)
            self.assertIn("missing_data_rule", f_def)
            self.assertIn("derivation_classification", f_def)

    def test_AT4_deterministic_reproducibility(self):
        """AT-4: 3 pipeline runs with fixed seed produce byte-identical output JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = []
            for run_i in range(1, 4):
                run_out = os.path.join(tmpdir, f"ranking_run_{run_i}.json")
                temp_ds = os.path.join(tmpdir, f"dataset_run_{run_i}.json")
                temp_eval = os.path.join(tmpdir, f"eval_run_{run_i}.json")
                temp_sc = os.path.join(tmpdir, f"scores_run_{run_i}.json")

                generate_dataset(n=30, seed=20260824, output_path=temp_ds)
                run_pipeline(dataset_path=temp_ds, output_path=temp_eval, seed=20260824)
                score_dataset_file(input_path=temp_ds, model_results_path=temp_eval, output_path=temp_sc)
                generate_ranking(scores_input_path=temp_sc, output_path=run_out)

                with open(run_out, "rb") as f:
                    outputs.append(f.read())

            self.assertEqual(outputs[0], outputs[1], "Run 1 and Run 2 must be byte-identical.")
            self.assertEqual(outputs[1], outputs[2], "Run 2 and Run 3 must be byte-identical.")

    def test_AT5_configuration_independence(self):
        """AT-5: Modifying scoring weights in JSON config alters scores/rankings without code edits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "sample.json")
            sc_path1 = os.path.join(tmpdir, "scores1.json")
            sc_path2 = os.path.join(tmpdir, "scores2.json")
            rk_path1 = os.path.join(tmpdir, "rank1.json")
            rk_path2 = os.path.join(tmpdir, "rank2.json")

            generate_dataset(n=30, seed=42, output_path=ds_path)

            # Run with baseline config
            score_dataset_file(input_path=ds_path, scoring_config_path=self.scoring_config_path, output_path=sc_path1)
            generate_ranking(scores_input_path=sc_path1, scoring_config_path=self.scoring_config_path, output_path=rk_path1)

            # Create modified config with altered weights
            with open(self.scoring_config_path, "r", encoding="utf-8") as f:
                mod_cfg = json.load(f)
            mod_cfg["weights"] = {
                "audit_timeliness": 0.05,
                "audit_efficiency": 0.05,
                "error_rate": 0.05,
                "agency_scale": 0.05,
                "compliance_adherence": 0.70,
                "past_renewal_status": 0.10,
            }
            mod_cfg_path = os.path.join(tmpdir, "modified_scoring_config.json")
            with open(mod_cfg_path, "w", encoding="utf-8") as f:
                json.dump(mod_cfg, f)

            # Run with modified config
            score_dataset_file(input_path=ds_path, scoring_config_path=mod_cfg_path, output_path=sc_path2)
            generate_ranking(scores_input_path=sc_path2, scoring_config_path=mod_cfg_path, output_path=rk_path2)

            with open(sc_path1, "r", encoding="utf-8") as f1, open(sc_path2, "r", encoding="utf-8") as f2:
                s1 = json.load(f1)
                s2 = json.load(f2)

            scores_1 = [x["performance_score"] for x in s1["scores"]]
            scores_2 = [x["performance_score"] for x in s2["scores"]]
            self.assertNotEqual(scores_1, scores_2, "Scores must change when configuration weights change.")

    def test_AT6_irregular_input_handling(self):
        """AT-6: Organisations with missing, sparse, or outlier data degrade gracefully without unhandled exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "irregular_sample.json")
            sc_path = os.path.join(tmpdir, "irregular_scores.json")
            rk_path = os.path.join(tmpdir, "irregular_rank.json")

            ds = generate_dataset(n=60, seed=20260824, output_path=ds_path)
            # Ensure irregular records exist
            self.assertGreater(ds["summary_statistics"]["organisations_with_irregularities"], 0)

            # Pipeline execution must complete without crashing
            s_res = score_dataset_file(input_path=ds_path, output_path=sc_path)
            r_res = generate_ranking(scores_input_path=sc_path, output_path=rk_path)

            self.assertEqual(len(s_res["scores"]), 60)
            self.assertEqual(len(r_res["rankings"]) + len(r_res["unranked_organisations"]), 60)

    def test_AT7_versioning_and_change_history(self):
        """AT-7: Version update creates new version; historical results remain retrievable with original provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "sample.json")
            sc_path = os.path.join(tmpdir, "output", "organisation_scores.json")
            rk_path = os.path.join(tmpdir, "output", "ranking.json")
            reg_path = os.path.join(tmpdir, "config", "version_registry.json")
            cfg1_path = os.path.join(tmpdir, "config", "scoring_config_v1.json")
            cfg2_path = os.path.join(tmpdir, "config", "scoring_config_v2.json")

            os.makedirs(os.path.dirname(reg_path), exist_ok=True)
            os.makedirs(os.path.dirname(sc_path), exist_ok=True)

            generate_dataset(n=20, seed=20260824, output_path=ds_path)
            mgr = VersionRegistryManager(registry_path=reg_path)

            # --- Phase 1: Initial Run under v1.0.0 ---
            with open(os.path.join(PROJECT_ROOT, "config", "scoring_config.json"), "r", encoding="utf-8") as f:
                cfg1 = json.load(f)
            cfg1["scoring_version"] = "1.0.0"
            with open(cfg1_path, "w", encoding="utf-8") as f:
                json.dump(cfg1, f)

            score_dataset_file(input_path=ds_path, scoring_config_path=cfg1_path, output_path=sc_path)
            generate_ranking(scores_input_path=sc_path, scoring_config_path=cfg1_path, output_path=rk_path)

            hist_sc_v1 = os.path.join(tmpdir, "output", "history", "v1.0.0", "organisation_scores.json")
            hist_rk_v1 = os.path.join(tmpdir, "output", "history", "v1.0.0", "ranking.json")
            self.assertTrue(os.path.exists(hist_sc_v1), "Historical v1.0.0 scores must be archived.")
            self.assertTrue(os.path.exists(hist_rk_v1), "Historical v1.0.0 ranking must be archived.")

            # --- Phase 2: Controlled Version Update to v1.1.0 ---
            entry = mgr.register_change(
                new_version="1.1.0",
                changes=["Updated compliance weight in scoring configuration"],
                author="Auditing Committee Lead",
                approved_by="Chief Quality Officer",
                component_versions={"scoring_config": "1.1.0"},
            )
            self.assertEqual(mgr.registry["system_version"], "1.1.0")
            self.assertEqual(entry["approved_by"], "Chief Quality Officer")

            cfg2 = dict(cfg1)
            cfg2["scoring_version"] = "1.1.0"
            cfg2["weights"]["compliance_adherence"] = 0.50
            cfg2["weights"]["audit_timeliness"] = 0.10
            with open(cfg2_path, "w", encoding="utf-8") as f:
                json.dump(cfg2, f)

            # --- Phase 3: Run under v1.1.0 ---
            score_dataset_file(input_path=ds_path, scoring_config_path=cfg2_path, output_path=sc_path)
            generate_ranking(scores_input_path=sc_path, scoring_config_path=cfg2_path, output_path=rk_path)

            hist_sc_v2 = os.path.join(tmpdir, "output", "history", "v1.1.0", "organisation_scores.json")
            self.assertTrue(os.path.exists(hist_sc_v2), "Historical v1.1.0 scores must be archived.")

            # --- Phase 4: Verify Old Result Retains Old Version Provenance ---
            with open(hist_sc_v1, "r", encoding="utf-8") as f:
                old_res = json.load(f)
            with open(hist_sc_v2, "r", encoding="utf-8") as f:
                new_res = json.load(f)

            self.assertEqual(old_res["scoring_metadata"]["scoring_version"], "1.0.0")
            self.assertEqual(new_res["scoring_metadata"]["scoring_version"], "1.1.0")
            self.assertEqual(old_res["scores"][0]["provenance"]["scoring_version"], "1.0.0")
            self.assertEqual(new_res["scores"][0]["provenance"]["scoring_version"], "1.1.0")
            self.assertTrue(os.path.exists(hist_sc_v1), "Original historical output remains retrievable.")

    def test_AT8_model_comparison(self):
        """AT-8: All four models (Baseline, GBDT, SVM, Neural Network) are compared using Macro-F1."""
        eval_path = os.path.join(PROJECT_ROOT, "output", "evaluation_results.json")
        self.assertTrue(os.path.exists(eval_path), "evaluation_results.json must exist.")
        
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        summary = eval_data.get("comparison_summary", {})
        required_models = ["baseline", "gradient_boosting", "svm", "neural_network"]
        for m in required_models:
            self.assertIn(m, summary, f"Model {m} missing from comparison summary.")
            self.assertIn("macro_f1", summary[m])
            self.assertIn("accuracy", summary[m])

        self.assertEqual(eval_data["evaluation_metadata"]["primary_metric"], "macro_f1")


if __name__ == "__main__":
    unittest.main()
