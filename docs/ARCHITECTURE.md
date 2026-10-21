# Project organization

TRASER follows one installable `src/traser` package, one `run.py` entry, explicit
case configurations, immutable scientific inputs, and separate generated output.
This organization was informed by the official AlphaFold 3, Graphormer,
DINOv2 and Latent Diffusion repositories, inspected on 2026-10-21. Their code,
licenses, weights and datasets have not been copied into TRASER.

## Source responsibilities

| Package | Responsibility |
|---|---|
| `traser.data` | Full CSV auditing, native-ID reconstruction and occurrence calendars |
| `traser.representations` | Encoder text preparation and occurrence-weighted PCA64 |
| `traser.semantics` | Eleven masked Ridge axes, clipping and semantic diagnostics |
| `traser.dynamics` | Initialization/seeding and full-mass HK/Gaussian numerical operators |
| `traser.assimilation` | Causal strength, regularized transport, experiment behavior |
| `traser.metrics` | Full-mass marginal density errors and sparse-target handling |
| `traser.baselines` | WAR, WR and LQD fitting/prediction |
| `traser.experiments` | Complete calendar chains, case-specific development search, sensitivity and scenarios |
| `traser.visualization` | Current frozen-array paper panels, layout and export |
| `traser.utils` | Runtime capability inspection and original host/container resource reserves |
| `traser.io` | Project-relative artifact I/O, SHA256 and timestamps |
| `traser.cli` | Argument parsing and workflow dispatch |

`configs/experiments` holds the two current case identities and common scenario
design. `configs/numerics` holds shared operator/resource settings.
`configs/search` holds the declared BLM mechanical grid. All current seeds,
budgets, phase boundaries and choices are explicit; configuration directories
do not contain retired Weibo experiments.

`data/` contains the two published datasets. `assets/` contains the selected
representation/mapping instruments and calendars. `evidence/` contains frozen
reported results and reference displays. These have different roles from
`outputs/`, which is Git-ignored. `local_only/` contains local preparation
backups and the complete BLM master CSV and is also Git-ignored.

`provenance/module_map.json` maps each former implementation module to its new
responsibility. The migration rewrites imports and layout; it does not redefine
the original numerical operators. Exact historical result keys remain in data
artifacts for traceability. Tests and migrated-definition comparison check this
boundary without claiming an empirical scientific re-run.

## Official projects studied

| Project | Publication / role | Pattern adopted for TRASER |
|---|---|---|
| [AlphaFold 3](https://github.com/google-deepmind/alphafold3) | Nature; official inference implementation | `src` package, input/output documentation, explicit external model assets, CPU data preparation vs GPU inference |
| [Graphormer](https://github.com/microsoft/Graphormer) | NeurIPS 2021; official implementation | Task-specific examples and clear data/model/experiment responsibility |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Official representation-learning implementation | Reusable package, configuration separation, independent feature use and training/evaluation requirements |
| [Latent Diffusion](https://github.com/CompVis/latent-diffusion) | CVPR 2022; official implementation | Distinct configuration, data/model asset locations and runnable experiment recipes |

These are structural precedents, not scientific baselines for TRASER and not a
claim that their publication standing certifies this repository's experiments.

Complete scientific commands share an SQLite single-owner transaction in `outputs/runtime`, so a second job is rejected and process exit releases ownership without a stale PID override. The original RAM/swap reserves are applied before transitions.
