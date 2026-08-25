#!/usr/bin/env python3
"""
Air-Gapped Offline Package Installer.
Installs vendored Python packages from deploy/offline_package/ without requiring network access.
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OFFLINE_PACKAGE_DIR = os.path.join(PROJECT_ROOT, "deploy", "offline_package")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "deploy", "requirements.txt")


def install_offline(verbose: bool = True) -> bool:
    """Executes pip install targeting local vendored wheel directory only."""
    if not os.path.exists(OFFLINE_PACKAGE_DIR):
        print(f"ERROR: Offline package directory not found: {OFFLINE_PACKAGE_DIR}")
        return False

    wheel_files = [f for f in os.listdir(OFFLINE_PACKAGE_DIR) if f.endswith(".whl")]
    if not wheel_files:
        print(f"ERROR: No .whl packages found in {OFFLINE_PACKAGE_DIR}")
        return False

    print(f"Found {len(wheel_files)} vendored wheels in {OFFLINE_PACKAGE_DIR}:")
    for w in wheel_files:
        print(f" - {w}")

    # Construct pip install command with --no-index and local find-links
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--break-system-packages",
        "--no-index",
        "--find-links",
        OFFLINE_PACKAGE_DIR,
        "scikit-learn",
        "numpy",
        "scipy",
        "joblib",
        "threadpoolctl",
    ]

    print("\nExecuting air-gapped installation command:")
    print(" ".join(cmd))

    res = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(res.stdout)
    if res.returncode != 0:
        print(f"Installation failed with error code {res.returncode}:")
        print(res.stderr)
        return False

    print("Air-gapped offline package installation completed successfully.")
    return True


if __name__ == "__main__":
    success = install_offline()
    sys.exit(0 if success else 1)
