#!/usr/bin/env python3
"""
Versioning and Configuration Registry Module.
Manages version lifecycle, change history, cryptographic configuration hashing,
and immutable score attribution.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class VersionRegistryManager:
    """Manages version-controlled configuration, hashing, and audit trails."""

    def __init__(self, registry_path: str = "config/version_registry.json"):
        if not os.path.isabs(registry_path):
            registry_path = os.path.join(PROJECT_ROOT, registry_path)
        self.registry_path = registry_path

        if not os.path.exists(self.registry_path):
            self.registry = self._init_default_registry()
            self.save_registry()
        else:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self.registry = json.load(f)

    @staticmethod
    def _init_default_registry() -> Dict[str, Any]:
        return {
            "system_version": "1.0.0",
            "active_versions": {
                "feature_schema": "1.0.0",
                "generation_config": "1.0.0",
                "label_rule": "1.0.0",
                "scoring_config": "1.0.0",
                "model_config": "1.0.0",
            },
            "metadata": {
                "project": "Auditing Agency Performance Monitoring and Ranking Module",
                "target_air_gapped": True,
                "last_updated": "2026-08-24T16:00:00Z",
            },
            "change_history": [],
        }

    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        """Computes SHA-256 hash of a file for integrity verification."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_config_hashes(self, config_dir: str = "config") -> Dict[str, str]:
        """Calculates current SHA-256 hashes for all active configuration files."""
        if not os.path.isabs(config_dir):
            config_dir = os.path.join(PROJECT_ROOT, config_dir)

        hashes = {}
        for fname in os.listdir(config_dir):
            if fname.endswith(".json") and fname != "version_registry.json":
                p = os.path.join(config_dir, fname)
                hashes[fname] = self.compute_file_hash(p)
        return hashes

    def register_change(
        self,
        new_version: str,
        changes: List[str],
        author: str,
        approved_by: Optional[str] = None,
        component_versions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Records a new version entry and appends to immutable change history."""
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if component_versions:
            self.registry["active_versions"].update(component_versions)

        self.registry["system_version"] = new_version
        self.registry["metadata"]["last_updated"] = now_iso

        entry = {
            "version": new_version,
            "timestamp": now_iso,
            "author": author,
            "approval_status": "APPROVED" if approved_by else "PENDING_FORMAL_EVALUATION",
            "approved_by": approved_by,
            "approval_date": now_iso if approved_by else None,
            "changes": changes,
            "config_hashes": self.get_config_hashes(),
        }

        self.registry["change_history"].append(entry)
        self.save_registry()
        return entry

    def save_registry(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage version registry.")
    parser.add_argument("--registry", type=str, default="config/version_registry.json")
    parser.add_argument("--check-hashes", action="store_true", help="Print current config hashes")

    args = parser.parse_args()
    mgr = VersionRegistryManager(registry_path=args.registry)
    if args.check_hashes:
        hashes = mgr.get_config_hashes()
        print("Active Configuration Checksums (SHA-256):")
        for k, v in hashes.items():
            print(f"- {k:25s}: {v}")
    else:
        print(f"System Version: {mgr.registry['system_version']}")
        print(f"Active Versions: {json.dumps(mgr.registry['active_versions'], indent=2)}")
