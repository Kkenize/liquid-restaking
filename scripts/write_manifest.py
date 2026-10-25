from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "REPRODUCIBILITY_MANIFEST.txt"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist", "results/logs"}


def excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel == OUTPUT.relative_to(ROOT):
        return True
    text = rel.as_posix()
    excluded_parts = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}
    if any(part in rel.parts for part in excluded_parts):
        return True
    if any(part.endswith(".egg-info") for part in rel.parts):
        return True
    return text.startswith("results/logs/")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and not excluded(path))
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {OUTPUT}")


if __name__ == "__main__":
    main()
