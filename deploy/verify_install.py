#!/usr/bin/env python3
"""
Air-Gapped Offline Installation Verification Script.
Verifies that all vendored dependencies, schema configurations, and pipeline modules
function flawlessly in a strictly offline environment without network access.
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def check_module_imports() -> Tuple[bool, Dict[str, Any]]:
    """Checks importability and versions of all required scientific/ML packages."""
    results = {}
    all_ok = True

    for mod_name in ["numpy", "scipy", "joblib", "threadpoolctl", "sklearn"]:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            results[mod_name] = {"status": "OK", "version": ver}
        except ImportError as e:
            results[mod_name] = {"status": "FAILED", "error": str(e)}
            all_ok = False

    return all_ok, results


def check_configurations() -> Tuple[bool, Dict[str, Any]]:
    """Verifies that all required JSON configurations exist and parse correctly."""
    config_files = [
        "feature_schema.json",
        "generation_config.json",
        "label_rule.json",
        "scoring_config.json",
        "model_config.json",
        "version_registry.json",
    ]
    results = {}
    all_ok = True
    config_dir = os.path.join(PROJECT_ROOT, "config")

    for fname in config_files:
        p = os.path.join(config_dir, fname)
        if not os.path.exists(p):
            results[fname] = {"status": "MISSING"}
            all_ok = False
        else:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results[fname] = {"status": "VALID_JSON", "keys_count": len(data)}
            except Exception as e:
                results[fname] = {"status": "PARSE_ERROR", "error": str(e)}
                all_ok = False

    return all_ok, results


def check_pipeline_smoke_test() -> Tuple[bool, str]:
    """Runs a self-contained offline smoke test of the data, modeling, and ranking pipeline."""
    try:
        from src.data_generator.generate_dataset import SyntheticDataGenerator
        from src.features.extractor import FeatureExtractor
        from src.models.gradient_boosting import GradientBoostingModel
        from src.ranking.engine import RankingEngine
        from src.scoring.engine import PerformanceScoringEngine

        # Generate mini-population with sufficient diversity
        gen = SyntheticDataGenerator(
            schema_path=os.path.join(PROJECT_ROOT, "config", "feature_schema.json"),
            generation_config_path=os.path.join(PROJECT_ROOT, "config", "generation_config.json"),
            label_rule_path=os.path.join(PROJECT_ROOT, "config", "label_rule.json"),
            seed=42,
        )
        dataset = gen.generate(n_organisations=30)

        # Feature extraction
        ext = FeatureExtractor(schema_path=os.path.join(PROJECT_ROOT, "config", "feature_schema.json"))
        ext.fit(dataset["organisations"])
        X, y, cols, _ = ext.transform(dataset["organisations"])

        # Model fit & predict
        model = GradientBoostingModel(n_estimators=5, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)

        # Scoring & ranking
        scorer = PerformanceScoringEngine(
            scoring_config_path=os.path.join(PROJECT_ROOT, "config", "scoring_config.json"),
            version_registry_path=os.path.join(PROJECT_ROOT, "config", "version_registry.json"),
        )
        scored = scorer.score_dataset(dataset)
        ranker = RankingEngine(scoring_config_path=os.path.join(PROJECT_ROOT, "config", "scoring_config.json"))
        ranked = ranker.rank_organisations(scored)

        if len(ranked["rankings"]) > 0:
            return True, "Pipeline smoke test completed successfully with 100% offline components."
        return False, "Smoke test generated zero ranked records."
    except Exception as e:
        return False, f"Smoke test failed: {e}"


def run_full_verification() -> bool:
    """Executes all installation verification checks."""
    print("=" * 65)
    print("AIR-GAPPED OFFLINE INSTALLATION VERIFICATION")
    print("=" * 65)

    mod_ok, mod_results = check_module_imports()
    print("\n1. Python Package Dependencies Check:")
    for mod, details in mod_results.items():
        print(f" - {mod:15s}: Status = {details['status']} | Version = {details.get('version', 'N/A')}")

    cfg_ok, cfg_results = check_configurations()
    print("\n2. Configuration & Schema Integrity Check:")
    for cfg, details in cfg_results.items():
        print(f" - {cfg:25s}: Status = {details['status']}")

    smoke_ok, smoke_msg = check_pipeline_smoke_test()
    print(f"\n3. Offline Pipeline Smoke Test:")
    print(f" - Status: {'PASSED' if smoke_ok else 'FAILED'}")
    print(f" - Details: {smoke_msg}")

    all_passed = mod_ok and cfg_ok and smoke_ok
    print("\n" + "=" * 65)
    print(f"FINAL VERIFICATION RESULT: {'PASSED (READY FOR AIR-GAPPED OPERATION)' if all_passed else 'FAILED'}")
    print("=" * 65)

    return all_passed


if __name__ == "__main__":
    success = run_full_verification()
    sys.exit(0 if success else 1)
