"""Unit tests for version registry management and cryptographic config tracking."""

import json
import os
import tempfile
import unittest

from src.versioning.registry import VersionRegistryManager


class TestVersioning(unittest.TestCase):
    """Verifies semantic versioning, hashing, and audit change logging."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.reg_path = os.path.join(self.temp_dir, "version_registry.json")
        self.mgr = VersionRegistryManager(registry_path=self.reg_path)

    def test_initial_registry_structure(self):
        self.assertEqual(self.mgr.registry["system_version"], "1.0.0")
        self.assertIn("active_versions", self.mgr.registry)
        self.assertIn("feature_schema", self.mgr.registry["active_versions"])
        self.assertIn("change_history", self.mgr.registry)

    def test_register_change_logging(self):
        entry = self.mgr.register_change(
            new_version="1.1.0",
            changes=["Updated scoring weights for compliance adherence"],
            author="Auditing Team Lead",
            approved_by="Chief Quality Officer",
            component_versions={"scoring_config": "1.1.0"},
        )
        self.assertEqual(self.mgr.registry["system_version"], "1.1.0")
        self.assertEqual(self.mgr.registry["active_versions"]["scoring_config"], "1.1.0")
        self.assertEqual(len(self.mgr.registry["change_history"]), 1)
        self.assertEqual(entry["approval_status"], "APPROVED")
        self.assertEqual(entry["approved_by"], "Chief Quality Officer")

    def test_hash_computation(self):
        sample_file = os.path.join(self.temp_dir, "sample.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("Deterministic hash test payload")

        h1 = self.mgr.compute_file_hash(sample_file)
        h2 = self.mgr.compute_file_hash(sample_file)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_controlled_update_workflow(self):
        from deploy.update_process import ControlledUpdateManager
        updater = ControlledUpdateManager(registry_path=self.reg_path)
        pkg_dir = tempfile.mkdtemp()
        cfg_file = os.path.join(pkg_dir, "test_config.json")
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump({"setting": 123}, f)

        # Test rejected without approved_by
        with self.assertRaises(PermissionError):
            updater.stage_and_apply_update(
                update_package_dir=pkg_dir,
                new_version="1.2.0",
                author="QA Team",
                approved_by="",
                change_summary=["Test change"],
            )

        # Test approved update
        entry = updater.stage_and_apply_update(
            update_package_dir=pkg_dir,
            new_version="1.2.0",
            author="QA Team",
            approved_by="Auditing Board",
            change_summary=["Verified approved update"],
        )
        self.assertEqual(updater.registry_mgr.registry["system_version"], "1.2.0")
        self.assertEqual(entry["approved_by"], "Auditing Board")


if __name__ == "__main__":
    unittest.main()
