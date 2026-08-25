"""Feature Extractor and Preprocessing Pipeline for Machine Learning Models."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class FeatureExtractor:
    """Transforms raw organisation dictionaries into machine learning feature matrices."""

    FEATURE_NAMES = [
        "audit_timeliness",
        "audit_efficiency",
        "error_rate",
        "agency_scale",
        "compliance_adherence",
        "past_renewal_status",
    ]

    def __init__(self, schema_path: str = "config/feature_schema.json"):
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Feature schema not found: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.medians_: Dict[str, float] = {}
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, organisations: List[Dict[str, Any]]) -> "FeatureExtractor":
        """Computes imputation statistics strictly on training organisations."""
        feat_values: Dict[str, List[float]] = {f: [] for f in self.FEATURE_NAMES}

        for org in organisations:
            feats = org.get("features", {})
            status = org.get("feature_status", {})
            for f_name in self.FEATURE_NAMES:
                val = feats.get(f_name)
                st = status.get(f_name, "VALID")
                if val is not None and st == "VALID" and isinstance(val, (int, float)):
                    feat_values[f_name].append(float(val))

        for f_name, vals in feat_values.items():
            if vals:
                self.medians_[f_name] = float(np.median(vals))
                self.means_[f_name] = float(np.mean(vals))
                std_val = float(np.std(vals))
                self.stds_[f_name] = std_val if std_val > 1e-6 else 1.0
            else:
                # Neutral fallback if entire feature is missing
                self.medians_[f_name] = 50.0
                self.means_[f_name] = 50.0
                self.stds_[f_name] = 1.0

        self.is_fitted = True
        return self

    def transform(
        self,
        organisations: List[Dict[str, Any]],
        include_missing_indicators: bool = True,
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Transforms organisations into feature matrix X and label array y.
        Returns: (X, y, feature_column_names, org_ids)
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transforming data.")

        feature_cols = list(self.FEATURE_NAMES)
        if include_missing_indicators:
            feature_cols += [f"missing_{f}" for f in self.FEATURE_NAMES]

        X_rows: List[List[float]] = []
        y_list: List[int] = []
        org_ids: List[str] = []

        for org in organisations:
            org_id = org.get("organisation_id", "UNKNOWN")
            org_ids.append(org_id)
            feats = org.get("features", {})
            status = org.get("feature_status", {})

            row_base: List[float] = []
            row_indicators: List[float] = []

            for f_name in self.FEATURE_NAMES:
                val = feats.get(f_name)
                st = status.get(f_name, "VALID")
                if val is None or st != "VALID" or not isinstance(val, (int, float)):
                    imputed_val = self.medians_[f_name]
                    is_miss = 1.0
                else:
                    imputed_val = float(val)
                    is_miss = 0.0

                if scale:
                    # Standard scaling (zero mean, unit variance)
                    scaled_val = (imputed_val - self.means_[f_name]) / self.stds_[f_name]
                    row_base.append(scaled_val)
                else:
                    row_base.append(imputed_val)

                row_indicators.append(is_miss)

            full_row = row_base + row_indicators if include_missing_indicators else row_base
            X_rows.append(full_row)

            # Target label: RENEWED = 1, NOT_RENEWED = 0
            lbl = org.get("label")
            if lbl == "RENEWED":
                y_list.append(1)
            elif lbl == "NOT_RENEWED":
                y_list.append(0)
            else:
                y_list.append(-1)  # Unlabeled or test case

        return np.array(X_rows, dtype=np.float64), np.array(y_list, dtype=np.int64), feature_cols, org_ids


def get_feature_matrix(
    dataset: Dict[str, Any],
    extractor: Optional[FeatureExtractor] = None,
    fit_if_needed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], FeatureExtractor]:
    """Helper function to extract feature matrix from a loaded dataset."""
    organisations = dataset.get("organisations", [])
    if extractor is None:
        extractor = FeatureExtractor()
    if fit_if_needed and not extractor.is_fitted:
        extractor.fit(organisations)

    X, y, cols, org_ids = extractor.transform(organisations)
    return X, y, cols, org_ids, extractor
