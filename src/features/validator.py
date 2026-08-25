"""Feature Validator module for verifying dataset conformity against feature schema."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class FeatureValidator:
    """Validates raw and computed organisation feature dictionaries against schema rules."""

    def __init__(self, schema_path: str = "config/feature_schema.json"):
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Feature schema not found: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.feature_defs = self.schema.get("features", {})

    def validate_organisation(self, org_record: Dict[str, Any]) -> Tuple[bool, Dict[str, str], List[str]]:
        """
        Validates a single organisation record against schema bounds and missing rules.
        Returns: (is_valid, feature_status_dict, validation_messages)
        """
        messages: List[str] = []
        status: Dict[str, str] = {}
        features = org_record.get("features", {})

        for feat_name, feat_def in self.feature_defs.items():
            val = features.get(feat_name)
            p_range = feat_def["permitted_range"]
            min_val = p_range["min"]
            max_val = p_range["max"]

            if val is None:
                status[feat_name] = "NOT_ASSESSABLE"
                messages.append(f"{feat_name}: missing value recorded as NOT_ASSESSABLE")
            elif not isinstance(val, (int, float)):
                status[feat_name] = "MALFORMED"
                messages.append(f"{feat_name}: non-numeric value '{val}' marked as MALFORMED")
            elif val < min_val or val > max_val:
                status[feat_name] = "MALFORMED"
                messages.append(f"{feat_name}: value {val} outside permitted range [{min_val}, {max_val}]")
            else:
                status[feat_name] = "VALID"

        all_valid = all(s == "VALID" for s in status.values())
        return all_valid, status, messages

    def validate_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Validates all organisations in a dataset and returns summary report."""
        organisations = dataset.get("organisations", [])
        total = len(organisations)
        valid_count = 0
        status_breakdown = {feat: {"VALID": 0, "NOT_ASSESSABLE": 0, "MALFORMED": 0} for feat in self.feature_defs}

        for org in organisations:
            _, status, _ = self.validate_organisation(org)
            if all(s == "VALID" for s in status.values()):
                valid_count += 1
            for feat, st in status.items():
                if st in status_breakdown[feat]:
                    status_breakdown[feat][st] += 1

        return {
            "total_organisations": total,
            "fully_valid_organisations": valid_count,
            "validation_rate": round(valid_count / max(total, 1), 4),
            "status_breakdown": status_breakdown,
        }
