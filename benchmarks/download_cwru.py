"""Instructions and layout check for the CWRU Bearing dataset.
The Case Western Reserve University Bearing Data Center distributes
`.mat` files individually; the user must download them after accepting
the data use terms. This script verifies the expected layout.
"""
from __future__ import annotations
import pathlib
import sys

DATA_DIR = pathlib.Path(__file__).parent / "data" / "cwru"
SUBDIRS = ("normal", "fault")

INSTRUCTIONS = """\
CWRU Bearing Dataset download checklist
=======================================
1. Visit https://engineering.case.edu/bearingdatacenter/download-data-file
   and register for access.
2. Download a subset of .mat files. A reasonable minimum for binary
   classification:
       - Several "Normal Baseline Data" recordings (no fault).
       - Several fault recordings from any class (inner race, outer
         race, or ball faults).
3. Organize them under benchmarks/data/cwru/:
       benchmarks/data/cwru/
           normal/
               *.mat
           fault/
               *.mat
4. Re-run this script to verify.
Notes:
- The channel of interest in each .mat file is typically the drive-end
  vibration signal (variable names ending in '_DE_time').
- Sampling rate is 12 kHz for most recordings, 48 kHz for a subset.
  The benchmark detects this automatically.
"""

def main() -> int:
    print(INSTRUCTIONS)
    if not DATA_DIR.exists():
        print(f"Expected directory does not exist: {DATA_DIR}")
        return 1
    empty = []
    for sub in SUBDIRS:
        d = DATA_DIR / sub
        if not d.exists():
            print(f"Missing subdirectory: {d}")
            return 1
        if not list(d.glob("*.mat")):
            empty.append(d)
    if empty:
        print("The following directories exist but contain no .mat files:")
        for d in empty:
            print(f"  - {d}")
        return 1
    print(f"All expected subdirectories populated under {DATA_DIR}.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
