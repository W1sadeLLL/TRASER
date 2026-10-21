# Contributing

Install this repository in editable mode and keep all generated artifacts in
`outputs/`. Run commands through `python run.py`. `python run.py test` checks
deterministic operator and publication contracts; `python run.py audit-data`
checks every published data record; `python run.py verify-release` checks
the publication payload against its SHA256 manifest.

Changes to data normalization, phase boundaries, seeds, selected parameters,
PCA/mapping identities, target mass or numerical definitions require an explicit
new experiment identity and complete evaluation. Do not replace frozen paper
results with test-driven parameter choices or synthetic examples. Keep sparse
density values missing when mathematically undefined.

Do not commit credentials, acquired BLM text/user fields, private model service
settings, or generated checkpoints. Public BLM data remains the complete native
ID set. See `docs/DATA.md` for the available source and rights information.

The authors have not selected a code license. This file does not grant an
open-source license or a license for third-party data or model weights.
