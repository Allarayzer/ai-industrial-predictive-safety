"""Instructions and layout check for the NASA C-MAPSS dataset.
This script does not automatically download the dataset, because NASA's
data portal requires acceptance of its Dataset Licensing Agreement. Run
this script once to verify the expected layout after manual download.
"""
from __future__ import annotations
import pathlib
import sys

DATA_DIR = pathlib.Path(__file__).parent / "data" / "cmapss"
EXPECTED_FILES = (
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
)

INSTRUCTIONS = """\
NASA C-MAPSS download checklist
===============================
1. Visit https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
   (or the mirror at https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).
2. Accept the data use terms.
3. Download the CMAPSSData archive.
4. Extract and copy (or symlink) the following files into
   benchmarks/data/cmapss/:
       - train_FD001.txt
       - test_FD001.txt
       - RUL_FD001.txt
5. Re-run this script to verify.
Optional: also copy FD002/FD003/FD004 if you want to run the multi-regime
variants. Only FD001 is required by the reference benchmark.
"""

def main() -> int:
    print(INSTRUCTIONS)
    if not DATA_DIR.exists():
        print(f"Expected directory does not exist: {DATA_DIR}")
        return 1
    missing = [f for f in EXPECTED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print("Missing files:")
        for name in missing:
            print(f"  - {DATA_DIR / name}")
        return 1
    print(f"All expected files present in {DATA_DIR}.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
