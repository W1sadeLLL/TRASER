"""Public, portable entry point for the TRASER release."""

from pathlib import Path
import os, sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
if __name__ == "__main__":
    from traser.cli import main

    raise SystemExit(main())
