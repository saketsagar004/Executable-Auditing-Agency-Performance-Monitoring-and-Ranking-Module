#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Auditing Agency Performance Monitoring.
Generates reproducible synthetic auditing organizations carrying the six mandatory features,
realistic irregularities, quality flags, and proof-of-concept renewal labels.
"""

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SyntheticDataGenerator:
    """Generates synthetic auditing agency populations according to configuration schemas."""

    def __init__(
        self,
        schema_path: str = "config/feature_schema.json",
        generation_config_path: str = "config/generation_config.json",
        label_rule_path: str = "config/label_rule.json",
        seed: Optional[int] = None,
    ):
        self.schema = self._load_json(schema_path)
        self.gen_config = self._load_json(generation_config_path)
        self.label_rule = self._load_json(label_rule_path)

        self.seed = seed if seed is not None else self.gen_config.get("random_seed", 20260824)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        """Safely load JSON configuration file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate(self, n_organisations: Optional[int] = None) -> Dict[str, Any]:
        """Generate the complete synthetic population dataset."""
        n = (
            n_organisations
            if n_organisations is not None
            else self.gen_config.get("default_organisation_count", 120)
        )

        dists = self.gen_config["distributions"]
        irreg = self.gen_config["irregularity_probabilities"]
        outlier_cfg = self.gen_config.get("outlier_ranges", {})

        organisations: List[Dict[str, Any]] = []

        for i in range(1, n + 1):
            org_id = f"ORG-{i:04d}"
            org_data, raw_vars, feature_status, irregularities = self._generate_single_org(
                org_id=org_id,
                dists=dists,
                irreg=irreg,
                outlier_cfg=outlier_cfg,
            )

            # Assign synthetic renewal label
            label, raw_score, label_notes = self._assign_label(org_data, feature_status)

            org_record = {
                "organisation_id": org_id,
                "reporting_period": raw_vars.get("reporting_period"),
                "features": org_data,
                "feature_status": feature_status,
                "raw_variables": raw_vars,
                "irregularities": irregularities,
                "label": label,
                "label_metadata": {
                    "raw_composite_score": round(raw_score, 4),
                    "rule_version": self.label_rule.get("rule_version", "1.0.0"),
                    "disclaimer": self.label_rule.get("disclaimer"),
                    "notes": label_notes,
                },
            }
            organisations.append(org_record)

        dataset = {
            "dataset_metadata": {
                "dataset_name": "Synthetic Auditing Agency Population",
                "organisation_count": len(organisations),
                "random_seed": self.seed,
                "generation_timestamp": "2026-08-24T16:00:00Z",
                "schema_version": self.schema.get("schema_version", "1.0.0"),
                "generation_config_version": self.gen_config.get("config_version", "1.0.0"),
                "label_rule_version": self.label_rule.get("rule_version", "1.0.0"),
                "is_synthetic_proof_of_concept": True,
            },
            "summary_statistics": self._compute_summary_stats(organisations),
            "organisations": organisations,
        }
        return dataset

    def _generate_single_org(
        self,
        org_id: str,
        dists: Dict[str, Any],
        irreg: Dict[str, Any],
        outlier_cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Any], Dict[str, str], List[str]]:
        irregularities: List[str] = []
        feature_status: Dict[str, str] = {
            "audit_timeliness": "VALID",
            "audit_efficiency": "VALID",
            "error_rate": "VALID",
            "agency_scale": "VALID",
            "compliance_adherence": "VALID",
            "past_renewal_status": "VALID",
        }

        # 1. Base scale variables
        ts_cfg = dists["team_size"]
        team_size = int(
            np.clip(
                self.np_rng.lognormal(mean=ts_cfg["mean"], sigma=ts_cfg["sigma"]),
                ts_cfg["min_val"],
                ts_cfg["max_val"],
            )
        )

        yo_cfg = dists["years_operational"]
        years_op = round(
            float(
                np.clip(
                    self.np_rng.gamma(shape=yo_cfg["shape"], scale=yo_cfg["scale"]),
                    yo_cfg["min_val"],
                    yo_cfg["max_val"],
                )
            ),
            1,
        )

        # 2. Audits completed
        aud_cfg = dists["eligible_completed_audits"]
        expected_audits = team_size * aud_cfg["base_per_auditor"]
        completed_audits = int(
            np.clip(
                self.np_rng.poisson(lam=expected_audits),
                aud_cfg["min_val"],
                aud_cfg["max_val"],
            )
        )

        # 3. Timeliness
        time_cfg = dists["audit_timeliness_pct"]
        raw_timeliness_rate = float(
            self.np_rng.beta(a=time_cfg["alpha"], b=time_cfg["beta_param"])
        )
        on_time_audits = int(round(raw_timeliness_rate * completed_audits))
        on_time_audits = min(on_time_audits, completed_audits)

        # 4. Efficiency (Duration in days)
        dur_cfg = dists["audit_duration_days"]
        mean_duration = float(
            np.clip(
                self.np_rng.gamma(shape=dur_cfg["shape"], scale=dur_cfg["scale"]),
                dur_cfg["min_val"],
                dur_cfg["max_val"],
            )
        )

        # 5. Error Rate (Discrepancies per report)
        err_cfg = dists["discrepancies_per_report"]
        raw_err_rate = float(
            np.clip(
                self.np_rng.exponential(scale=err_cfg["scale"]),
                err_cfg["min_val"],
                err_cfg["max_val"],
            )
        )
        op_params = self.gen_config.get("operational_parameters", {})
        rev_min = op_params.get("reviewed_reports_fraction_min", 0.7)
        rev_max = op_params.get("reviewed_reports_fraction_max", 1.0)
        reviewed_reports = max(1, int(round(completed_audits * self.rng.uniform(rev_min, rev_max))))
        identified_discrepancies = int(round(raw_err_rate * reviewed_reports))

        # 6. Compliance Adherence
        comp_cfg = dists["compliance_adherence_pct"]
        raw_compliance_rate = float(
            self.np_rng.beta(a=comp_cfg["alpha"], b=comp_cfg["beta_param"])
        )
        ctrl_choices = op_params.get("applicable_controls_choices", [25, 30, 40, 50, 60])
        applicable_controls = int(self.rng.choice(ctrl_choices))
        compliant_controls = int(round(raw_compliance_rate * applicable_controls))
        compliant_controls = min(compliant_controls, applicable_controls)

        # 7. Past Renewal History
        sparse_thresh = op_params.get("sparse_history_years_threshold", 2.0)
        is_sparse_history = self.rng.random() < irreg["sparse_history_prob"]
        if is_sparse_history or years_op < sparse_thresh:
            hist_renewals = 0
            hist_revocations = 0
            irregularities.append("SPARSE_OPERATIONAL_HISTORY_NEW_ENTRANT")
        else:
            ren_cfg = dists["historical_renewals"]
            rev_cfg = dists["historical_revocations"]
            ren_div = op_params.get("historical_renewals_years_divisor", 2.5)
            n_ren = min(ren_cfg["n_max"], max(1, int(years_op / ren_div)))
            hist_renewals = int(self.np_rng.binomial(n=n_ren, p=ren_cfg["prob"]))
            hist_revocations = int(self.np_rng.binomial(n=rev_cfg["n_max"], p=rev_cfg["prob"]))

        # --- Reporting Period & Inconsistency ---
        period_cfg = op_params.get("reporting_period", {})
        if self.rng.random() < irreg.get("inconsistent_period_prob", 0.05):
            irreg_durations = period_cfg.get("irregular_durations_months", [6, 9, 18, 24])
            dur_m = int(self.rng.choice(irreg_durations))
            reporting_period = {
                "start_date": period_cfg.get("standard_start_date", "2025-01-01"),
                "end_date": f"2025-{dur_m:02d}-30" if dur_m <= 12 else "2026-06-30",
                "duration_months": dur_m,
                "period_type": "INCONSISTENT_NON_STANDARD_CYCLE",
                "is_standard_cycle": False,
            }
            irregularities.append("INCONSISTENT_REPORTING_PERIOD")
        else:
            reporting_period = {
                "start_date": period_cfg.get("standard_start_date", "2025-01-01"),
                "end_date": period_cfg.get("standard_end_date", "2025-12-31"),
                "duration_months": period_cfg.get("standard_duration_months", 12),
                "period_type": "STANDARD_ANNUAL_CYCLE",
                "is_standard_cycle": True,
            }

        # --- Inject Realistic Irregularities ---

        # Missing Audit Data
        if self.rng.random() < irreg["missing_audit_data_prob"]:
            completed_audits = 0
            on_time_audits = 0
            irregularities.append("MISSING_AUDIT_ACTIVITY")

        # Missing Compliance Review
        missing_comp = self.rng.random() < irreg["missing_compliance_prob"]
        if missing_comp:
            applicable_controls = 0
            compliant_controls = 0
            irregularities.append("MISSING_COMPLIANCE_REVIEW")

        # Outliers
        if self.rng.random() < irreg["outlier_prob"]:
            outlier_type = self.rng.choice(["duration", "discrepancies", "timeliness"])
            if outlier_type == "duration":
                mean_duration = outlier_cfg.get("extreme_high_duration_days", 210.0)
                irregularities.append("OUTLIER_EXTREME_AUDIT_DURATION")
            elif outlier_type == "discrepancies":
                identified_discrepancies = int(reviewed_reports * outlier_cfg.get("extreme_high_discrepancies", 22.5))
                irregularities.append("OUTLIER_EXTREME_ERROR_RATE")
            elif outlier_type == "timeliness":
                on_time_audits = int(completed_audits * 0.10)
                irregularities.append("OUTLIER_ABYSMAL_TIMELINESS")

        # Malformed Data
        if self.rng.random() < irreg["malformed_data_prob"]:
            malformed_type = self.rng.choice(["negative_team", "inverted_audit_counts", "excess_controls"])
            if malformed_type == "negative_team":
                team_size = -5
                irregularities.append("MALFORMED_NEGATIVE_TEAM_SIZE")
            elif malformed_type == "inverted_audit_counts" and completed_audits > 0:
                on_time_audits = completed_audits + 15
                irregularities.append("MALFORMED_ON_TIME_EXCEEDS_TOTAL")
            elif malformed_type == "excess_controls" and applicable_controls > 0:
                compliant_controls = applicable_controls + 10
                irregularities.append("MALFORMED_COMPLIANT_EXCEEDS_APPLICABLE")

        # --- Compute Features according to schema ---

        # F1: Audit Timeliness
        if completed_audits <= 0:
            feat_timeliness = None
            feature_status["audit_timeliness"] = "NOT_ASSESSABLE"
        elif on_time_audits > completed_audits or on_time_audits < 0:
            feat_timeliness = None
            feature_status["audit_timeliness"] = "MALFORMED"
        else:
            feat_timeliness = round((on_time_audits / completed_audits) * 100.0, 2)

        # F2: Audit Efficiency
        if completed_audits <= 0:
            feat_efficiency = None
            feature_status["audit_efficiency"] = "NOT_ASSESSABLE"
        elif mean_duration <= 0 or mean_duration > 365.0:
            feat_efficiency = None
            feature_status["audit_efficiency"] = "MALFORMED"
        else:
            feat_efficiency = round(mean_duration, 2)

        # F3: Error Rate
        if reviewed_reports <= 0:
            feat_error_rate = None
            feature_status["error_rate"] = "NOT_ASSESSABLE"
        elif identified_discrepancies < 0:
            feat_error_rate = None
            feature_status["error_rate"] = "MALFORMED"
        else:
            feat_error_rate = round(identified_discrepancies / reviewed_reports, 3)

        # F4: Agency Scale
        if team_size < 0 or years_op < 0:
            feat_agency_scale = None
            feature_status["agency_scale"] = "MALFORMED"
        else:
            norm_team = min(team_size / 50.0, 1.0)
            norm_years = min(years_op / 20.0, 1.0)
            feat_agency_scale = round(50.0 * norm_team + 50.0 * norm_years, 2)

        # F5: Compliance Adherence
        if applicable_controls <= 0:
            feat_compliance = None
            feature_status["compliance_adherence"] = "NOT_ASSESSABLE"
        elif compliant_controls > applicable_controls or compliant_controls < 0:
            feat_compliance = None
            feature_status["compliance_adherence"] = "MALFORMED"
        else:
            feat_compliance = round((compliant_controls / applicable_controls) * 100.0, 2)

        # F6: Past Renewal Status
        total_decisions = hist_renewals + hist_revocations
        if total_decisions <= 0:
            feat_past_renewal = None
            feature_status["past_renewal_status"] = "NOT_ASSESSABLE"
        elif hist_renewals < 0 or hist_revocations < 0:
            feat_past_renewal = None
            feature_status["past_renewal_status"] = "MALFORMED"
        else:
            feat_past_renewal = round((hist_renewals / total_decisions) * 100.0, 2)

        features = {
            "audit_timeliness": feat_timeliness,
            "audit_efficiency": feat_efficiency,
            "error_rate": feat_error_rate,
            "agency_scale": feat_agency_scale,
            "compliance_adherence": feat_compliance,
            "past_renewal_status": feat_past_renewal,
        }

        raw_vars = {
            "team_size": team_size,
            "years_operational": years_op,
            "eligible_completed_audits": completed_audits,
            "on_time_completed_audits": on_time_audits,
            "mean_duration_days": round(mean_duration, 2),
            "reviewed_reports": reviewed_reports,
            "identified_discrepancies": identified_discrepancies,
            "applicable_controls": applicable_controls,
            "compliant_applicable_controls": compliant_controls,
            "historical_renewals": hist_renewals,
            "historical_revocations": hist_revocations,
            "reporting_period": reporting_period,
        }

        return features, raw_vars, feature_status, irregularities

    def _assign_label(
        self,
        features: Dict[str, Optional[float]],
        feature_status: Dict[str, str],
    ) -> Tuple[str, float, str]:
        """Compute synthetic proof-of-concept label based on label rule configuration."""
        contribs = self.label_rule["feature_contributions"]
        threshold = self.label_rule.get("decision_threshold", 0.62)
        noise_rate = self.label_rule.get("noise_rate", 0.08)
        missing_cfg = self.label_rule.get("missing_feature_treatment", {})
        neutral_score = missing_cfg.get("neutral_score", 0.5)
        penalty_per_missing = missing_cfg.get("penalty_per_missing_feature", 0.03)

        composite_score = 0.0
        missing_count = 0

        for feat_name, cfg in contribs.items():
            val = features.get(feat_name)
            weight = cfg["weight"]
            direction = cfg["direction"]
            s_min = cfg["scale_min"]
            s_max = cfg["scale_max"]

            if val is None or feature_status.get(feat_name) != "VALID":
                feat_norm = neutral_score
                missing_count += 1
            else:
                if direction == "positive":
                    feat_norm = (val - s_min) / max(s_max - s_min, 1e-6)
                else:  # negative (lower is better)
                    feat_norm = 1.0 - ((val - s_min) / max(s_max - s_min, 1e-6))
                feat_norm = max(0.0, min(1.0, feat_norm))

            composite_score += weight * feat_norm

        composite_score -= missing_count * penalty_per_missing
        composite_score = max(0.0, min(1.0, composite_score))

        # Base determination
        is_renewed = composite_score >= threshold

        # Add realistic noise/boundary perturbation
        notes = "Deterministic rule application"
        if self.rng.random() < noise_rate:
            is_renewed = not is_renewed
            notes = "Synthetic noise perturbation applied"

        label = self.label_rule["positive_class"] if is_renewed else self.label_rule["negative_class"]
        return label, composite_score, notes

    def _compute_summary_stats(self, organisations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate statistics of the generated dataset."""
        total = len(organisations)
        if total == 0:
            return {}

        renewed_count = sum(1 for o in organisations if o["label"] == "RENEWED")
        not_renewed_count = total - renewed_count

        status_counts = {}
        for feat in ["audit_timeliness", "audit_efficiency", "error_rate", "agency_scale", "compliance_adherence", "past_renewal_status"]:
            status_counts[feat] = {
                "valid": sum(1 for o in organisations if o["feature_status"].get(feat) == "VALID"),
                "not_assessable": sum(1 for o in organisations if o["feature_status"].get(feat) == "NOT_ASSESSABLE"),
                "malformed": sum(1 for o in organisations if o["feature_status"].get(feat) == "MALFORMED"),
            }

        irreg_count = sum(1 for o in organisations if len(o["irregularities"]) > 0)

        return {
            "total_organisations": total,
            "class_distribution": {
                "RENEWED": renewed_count,
                "NOT_RENEWED": not_renewed_count,
                "renewal_rate": round(renewed_count / total, 4),
            },
            "organisations_with_irregularities": irreg_count,
            "irregularity_rate": round(irreg_count / total, 4),
            "feature_status_breakdown": status_counts,
        }


def generate_dataset(
    n: int = 120,
    seed: int = 20260824,
    output_path: str = "data/sample_dataset.json",
    schema_path: str = "config/feature_schema.json",
    gen_config_path: str = "config/generation_config.json",
    label_rule_path: str = "config/label_rule.json",
) -> Dict[str, Any]:
    """Top-level helper function to generate and save synthetic dataset."""
    generator = SyntheticDataGenerator(
        schema_path=schema_path,
        generation_config_path=gen_config_path,
        label_rule_path=label_rule_path,
        seed=seed,
    )
    dataset = generator.generate(n_organisations=n)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic auditing agency population dataset.")
    parser.add_argument("--n", type=int, default=120, help="Number of organisations to generate (default: 120)")
    parser.add_argument("--seed", type=int, default=20260824, help="Random seed for deterministic generation")
    parser.add_argument("--output", type=str, default="data/sample_dataset.json", help="Output path for JSON dataset")
    parser.add_argument("--schema", type=str, default="config/feature_schema.json", help="Path to feature schema JSON")
    parser.add_argument("--gen_config", type=str, default="config/generation_config.json", help="Path to generation config JSON")
    parser.add_argument("--label_rule", type=str, default="config/label_rule.json", help="Path to label rule JSON")

    args = parser.parse_args()
    data = generate_dataset(
        n=args.n,
        seed=args.seed,
        output_path=args.output,
        schema_path=args.schema,
        gen_config_path=args.gen_config,
        label_rule_path=args.label_rule,
    )
    print(f"Successfully generated {data['dataset_metadata']['organisation_count']} organisations.")
    print(f"Output saved to: {args.output}")
    print(f"Class distribution: {data['summary_statistics']['class_distribution']}")
    print(f"Organisations with irregularities: {data['summary_statistics']['organisations_with_irregularities']}")
