"""Unit tests for feature schema definitions and mathematical/operational properties."""

import json
import os
import unittest


class TestFeatureSchema(unittest.TestCase):
    """Verifies that all 6 required features adhere to Assignment 2 specification."""

    @classmethod
    def setUpClass(cls):
        cls.schema_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "feature_schema.json"
        )
        with open(cls.schema_path, "r", encoding="utf-8") as f:
            cls.schema = json.load(f)

    def test_six_required_features_present(self):
        required_features = [
            "audit_timeliness",
            "audit_efficiency",
            "error_rate",
            "agency_scale",
            "compliance_adherence",
            "past_renewal_status",
        ]
        features = self.schema.get("features", {})
        for feat in required_features:
            self.assertIn(feat, features, f"Required feature '{feat}' missing from schema.")

    def test_feature_fields_completeness(self):
        mandatory_fields = [
            "description",
            "operational_definition",
            "formula",
            "formula_variables",
            "unit",
            "permitted_range",
            "directionality",
            "missing_data_rule",
            "partial_data_rule",
            "invalid_data_behaviour",
            "derivation_justification",
            "derivation_classification",
        ]
        for feat_name, feat_def in self.schema["features"].items():
            for field in mandatory_fields:
                self.assertIn(
                    field,
                    feat_def,
                    f"Feature '{feat_name}' missing mandatory schema field: {field}",
                )

    def test_derivation_classifications_valid(self):
        valid_classes = {"assignment_specified", "judgement", "synthetic_data_bound"}
        for feat_name, feat_def in self.schema["features"].items():
            d_class = feat_def.get("derivation_classification")
            self.assertIn(
                d_class,
                valid_classes,
                f"Feature '{feat_name}' has invalid derivation classification '{d_class}'",
            )

    def test_directionality_values(self):
        for feat_name, feat_def in self.schema["features"].items():
            direction = feat_def.get("directionality")
            self.assertIn(
                direction,
                ["higher_is_better", "lower_is_better"],
                f"Invalid directionality for {feat_name}: {direction}",
            )

    def test_permitted_ranges_valid(self):
        for feat_name, feat_def in self.schema["features"].items():
            p_range = feat_def.get("permitted_range", {})
            self.assertIn("min", p_range)
            self.assertIn("max", p_range)
            self.assertLess(p_range["min"], p_range["max"])

    def test_past_renewal_status_rule(self):
        past_renewal = self.schema["features"]["past_renewal_status"]
        missing_rule = past_renewal.get("missing_data_rule", {})
        self.assertEqual(missing_rule.get("status_code"), "NOT_ASSESSABLE")


if __name__ == "__main__":
    unittest.main()
