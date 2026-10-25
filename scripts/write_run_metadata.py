from __future__ import annotations

"""Record the software and platform environment for an archived experiment run."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "run_metadata.json"

def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "validation"], default="full")
    args = parser.parse_args()

    config_buffer = io.StringIO()
    with redirect_stdout(config_buffer):
        np.show_config()

    packages = {
        name: package_version(name)
        for name in [
            "numpy",
            "scipy",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "PyYAML",
            "pytest",
            "setuptools",
        ]
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "packages": packages,
        "numpy_configuration": config_buffer.getvalue(),
        "git_commit": git_commit(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote run metadata to {OUTPUT}")


if __name__ == "__main__":
    main()
