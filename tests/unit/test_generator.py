"""Unit tests for synthetic dataset generator, determinism, irregularity tagging, and configurability."""

import copy
import json
import os
import tempfile
import unittest

from src.data_generator.generate_dataset import SyntheticDataGenerator, generate_dataset


class TestDataGenerator(unittest.TestCase):
    """Tests synthetic dataset generator against requirements."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        cls.schema_path = os.path.join(cls.base_dir, "config", "feature_schema.json")
        cls.gen_config_path = os.path.join(cls.base_dir, "config", "generation_config.json")
        cls.label_rule_path = os.path.join(cls.base_dir, "config", "label_rule.json")

    def test_json_validity_all_configs(self):
        config_files = [
            "feature_schema.json",
            "generation_config.json",
            "label_rule.json",
            "scoring_config.json",
            "model_config.json",
            "version_registry.json",
        ]
        for cfg in config_files:
            p = os.path.join(self.base_dir, "config", cfg)
            self.assertTrue(os.path.exists(p), f"Config file missing: {p}")
            with open(p, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    self.assertIsInstance(data, dict, f"{cfg} root must be a JSON object")
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON syntax in {cfg}: {e}")

    def test_deterministic_generation(self):
        gen1 = SyntheticDataGenerator(
            schema_path=self.schema_path,
            generation_config_path=self.gen_config_path,
            label_rule_path=self.label_rule_path,
            seed=20260824,
        )
        data1 = gen1.generate(n_organisations=40)

        gen2 = SyntheticDataGenerator(
            schema_path=self.schema_path,
            generation_config_path=self.gen_config_path,
            label_rule_path=self.label_rule_path,
            seed=20260824,
        )
        data2 = gen2.generate(n_organisations=40)

        json1 = json.dumps(data1, sort_keys=True)
        json2 = json.dumps(data2, sort_keys=True)
        self.assertEqual(json1, json2, "Identical seeds must produce byte-identical datasets.")

    def test_configurable_organisation_count(self):
        counts = [10, 35, 75]
        for n in counts:
            gen = SyntheticDataGenerator(
                schema_path=self.schema_path,
                generation_config_path=self.gen_config_path,
                label_rule_path=self.label_rule_path,
                seed=42,
            )
            data = gen.generate(n_organisations=n)
            self.assertEqual(len(data["organisations"]), n)
            self.assertEqual(data["dataset_metadata"]["organisation_count"], n)

    def test_missing_and_sparse_data_handling(self):
        gen = SyntheticDataGenerator(
            schema_path=self.schema_path,
            generation_config_path=self.gen_config_path,
            label_rule_path=self.label_rule_path,
            seed=20260824,
        )
        data = gen.generate(n_organisations=150)
        
        found_not_assessable = False
        found_sparse_history = False

        for org in data["organisations"]:
            statuses = org["feature_status"]
            if any(s == "NOT_ASSESSABLE" for s in statuses.values()):
                found_not_assessable = True
            if "SPARSE_OPERATIONAL_HISTORY_NEW_ENTRANT" in org["irregularities"]:
                found_sparse_history = True
                self.assertEqual(org["feature_status"]["past_renewal_status"], "NOT_ASSESSABLE")
                self.assertIsNone(org["features"]["past_renewal_status"])

        self.assertTrue(found_not_assessable, "Generator must produce NOT_ASSESSABLE entries for missing data.")
        self.assertTrue(found_sparse_history, "Generator must produce sparse history records flagged as NOT_ASSESSABLE.")

    def test_outliers_and_malformed_records(self):
        gen = SyntheticDataGenerator(
            schema_path=self.schema_path,
            generation_config_path=self.gen_config_path,
            label_rule_path=self.label_rule_path,
            seed=20260824,
        )
        data = gen.generate(n_organisations=150)
        found_malformed = False

        for org in data["organisations"]:
            statuses = org["feature_status"]
            if any(s == "MALFORMED" for s in statuses.values()):
                found_malformed = True

        self.assertTrue(found_malformed, "Generator must record MALFORMED status for irregular records.")

    def test_configurable_label_rule(self):
        # Test that changing decision threshold changes label proportions
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            with open(self.label_rule_path, "r", encoding="utf-8") as f:
                rule_strict = json.load(f)
            rule_strict["decision_threshold"] = 0.95  # extremely strict
            json.dump(rule_strict, tf)
            strict_path = tf.name

        try:
            gen_default = SyntheticDataGenerator(
                schema_path=self.schema_path,
                generation_config_path=self.gen_config_path,
                label_rule_path=self.label_rule_path,
                seed=100,
            )
            data_default = gen_default.generate(n_organisations=60)
            renewals_default = sum(1 for o in data_default["organisations"] if o["label"] == "RENEWED")

            gen_strict = SyntheticDataGenerator(
                schema_path=self.schema_path,
                generation_config_path=self.gen_config_path,
                label_rule_path=strict_path,
                seed=100,
            )
            data_strict = gen_strict.generate(n_organisations=60)
            renewals_strict = sum(1 for o in data_strict["organisations"] if o["label"] == "RENEWED")

            self.assertLess(renewals_strict, renewals_default, "Strict threshold must produce fewer renewals.")
        finally:
            if os.path.exists(strict_path):
                os.remove(strict_path)

    def test_inconsistent_reporting_period_metadata(self):
        gen = SyntheticDataGenerator(
            schema_path=self.schema_path,
            generation_config_path=self.gen_config_path,
            label_rule_path=self.label_rule_path,
            seed=20260824,
        )
        data = gen.generate(n_organisations=120)
        found_inconsistent = False
        for org in data["organisations"]:
            self.assertIn("reporting_period", org, "Organisation record must contain reporting_period metadata.")
            rp = org["reporting_period"]
            self.assertIn("duration_months", rp)
            self.assertIn("period_type", rp)
            if "INCONSISTENT_REPORTING_PERIOD" in org["irregularities"]:
                found_inconsistent = True
                self.assertFalse(rp["is_standard_cycle"])

        self.assertTrue(found_inconsistent, "Generator must produce at least some inconsistent reporting period records.")

    def test_operational_parameters_configurability(self):
        with open(self.gen_config_path, "r", encoding="utf-8") as f:
            cfg_mod = json.load(f)

        # Restrict applicable controls choices to only [100]
        cfg_mod["operational_parameters"]["applicable_controls_choices"] = [100]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(cfg_mod, tf)
            mod_gen_path = tf.name

        try:
            gen_mod = SyntheticDataGenerator(
                schema_path=self.schema_path,
                generation_config_path=mod_gen_path,
                label_rule_path=self.label_rule_path,
                seed=20260824,
            )
            data_mod = gen_mod.generate(n_organisations=30)
            for org in data_mod["organisations"]:
                if "MISSING_COMPLIANCE_REVIEW" not in org["irregularities"]:
                    self.assertEqual(org["raw_variables"]["applicable_controls"], 100)
        finally:
            if os.path.exists(mod_gen_path):
                os.remove(mod_gen_path)


if __name__ == "__main__":
    unittest.main()
