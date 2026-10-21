# Installation and environment

Python 3.11+ is required. The local release was checked with Python 3.13.9 and
CPU-only PyTorch 2.11.0; exact imported versions are recorded in
`environments/constraints-cpu-verified.txt` and `verification/environment.json`.

```sh
python -m venv .venv
# Activate .venv with the command appropriate for your shell.
python -m pip install -e .
python run.py doctor
python run.py test
```

Editable installation preserves access to the repository's frozen assets. When
importing an installed package from another working directory, the repository
location is resolved from its source by default. `TRASER_ROOT` may explicitly
point to a complete compatible repository/asset tree. No absolute path to the
authors' computer is required.

The default dependency ranges support inspection and rendering. Scientific
replay requires a compatible CUDA PyTorch build and the full assets described
in `REPRODUCIBILITY.md`. Use the official PyTorch installation selector for
the target CUDA runtime. The recorded CPU constraints are not a CUDA environment
lock, and CPU operator tests are not full scientific reproduction.

The renderer uses Matplotlib's bundled DejaVu Sans and exports editable SVG and
PDF at large logical text size. It runs without the authors' private figure
tools. Optional `TRASER_FIGURE_AUDIT_PATH` can enable an external local geometry
auditor during export; the portable renderer does not require it.
