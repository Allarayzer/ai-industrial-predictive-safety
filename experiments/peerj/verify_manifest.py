"""Verify imported source/reference hashes without running experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    entries = manifest["scripts"] + manifest["reference_results"]
    failures = []
    for entry in entries:
        path = root / entry["file"]
        if not path.is_file():
            failures.append(f"Missing: {entry['file']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"Hash mismatch: {entry['file']}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {len(entries)} source/reference files. No experiments were run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
