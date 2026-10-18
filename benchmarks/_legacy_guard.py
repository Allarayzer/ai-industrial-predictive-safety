"""Standard-library-only CLI containment for historical benchmark protocols."""

import argparse
import sys
from pathlib import Path


def parse_legacy_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Require acknowledgement and reserve a new output directory before imports.

    This prevents accidental runs/overwrites; it does not repair or validate the
    historical scientific protocol. Call before importing numerical/API modules.
    """
    parser.epilog = (
        "Historical protocol with known evaluation limitations. See "
        "docs/benchmarks.md. Opting in does not make its results validated evidence."
    )
    parser.add_argument(
        "--allow-legacy", action="store_true",
        help="Acknowledge the known limitations and run this historical protocol.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Required for a legacy run: a new directory that does not yet exist.",
    )
    args = parser.parse_args()
    if not args.allow_legacy:
        parser.error(
            "legacy benchmark disabled by default; review docs/benchmarks.md and "
            "pass --allow-legacy with --output-dir for an intentional historical run"
        )
    if args.output_dir is None:
        parser.error("--output-dir is required with --allow-legacy")

    output_dir = args.output_dir.expanduser().absolute()
    try:
        # exist_ok=False also rejects existing empty directories and symlinks.
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output-dir must be a new directory; refusing an existing path")
    except OSError as exc:
        parser.error(f"cannot create --output-dir: {exc}")
    args.output_dir = output_dir
    print(
        "WARNING: running a legacy benchmark with known evaluation limitations; "
        "outputs are historical demonstrations, not validated performance claims.",
        file=sys.stderr,
    )
    return args
