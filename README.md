# TRASER

Research implementation for **Tracking Online Public Opinion Distributions through Representation, HK Dynamics, and Assimilation**.

**Chenhe Cao · Xiaofei Xu · Xiaofei Zhao · Mingxing Chen · Zhen Lu**

TRASER follows event-related online expression from normalized text and a frozen latent representation to bounded-confidence distribution dynamics, observation assimilation, one-step forecasts, semantic interpretation and conditional scenarios. This repository contains the current BLM/X and Hong Kong fire Weibo experiments, their published data interfaces, and the numerical evidence used for the paper displays.

[Installation](docs/INSTALLATION.md) · [Reproduction guide](docs/REPRODUCIBILITY.md) · [Data](docs/DATA.md) · [Architecture](docs/ARCHITECTURE.md) · [Command examples](examples/README.md)

## Start here

```sh
git clone https://github.com/W1sadeLLL/TRASER.git
cd TRASER
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -e .
python run.py doctor
python run.py test
python run.py audit-data
python run.py verify-release
python run.py figures
```

Python 3.11+ is required. `run.py` is the single workflow entry. Installation and deterministic unit tests support CPU execution; scientific replay requires compatible CUDA PyTorch. Locally verified versions are recorded in [the CPU constraints](environments/constraints-cpu-verified.txt).

## Choose a reproduction level

| Goal | Entry | Required assets |
|---|---|---|
| Reproduce figures and inspect table cells | `python run.py figures` | Included complete numerical arrays and score tables; CPU |
| Replay frozen representation and choices | `python run.py replay --dataset hk_weibo` | Included Weibo frozen instruments and occurrence indices; CUDA |
| Replay BLM | `python run.py replay --dataset blm_x --asset-root /path/to/assets` | External compatible BLM coordinates/indices; CUDA |
| Rebuild from original text | `prepare-occurrences`, `build-representation`, `fit-mapping`, `select-parameters` | Official encoder implementation/weights and original isolation/label inputs, some externally obtained |

Use `replay --check-only` to inspect asset availability without running a scientific experiment. Full new GPU runs were not executed during repository preparation. Exact original raw-from-scratch reconstruction also requires the encoder revision and isolation inputs described in the reproduction guide. The original archive references 22,269 server state files not present locally; this repository does not claim that entire computational cache is available.

## Project structure

```text
TRASER/
├── run.py                     # single workflow entry
├── pyproject.toml             # installable src package
├── src/traser/
│   ├── data/                  # CSV audit, occurrence preparation, IDs
│   ├── representations/       # text protocol and weighted PCA64
│   ├── semantics/             # eleven-axis Ridge mapping and diagnostics
│   ├── dynamics/              # initialization, HK/Gaussian operators
│   ├── assimilation/          # causal strength and regularized updates
│   ├── metrics/               # marginal density evaluation
│   ├── baselines/             # WAR, WR, LQD
│   ├── experiments/           # full-calendar chains, selection, scenarios
│   ├── visualization/         # current frozen-data paper figures
│   ├── utils/                 # environment and resource guards
│   ├── io.py                  # relative paths, atomic I/O, hashes
│   └── cli.py                 # command parsing and dispatch
├── configs/{experiments,numerics,search}/
├── data/{blm,weibo}/           # complete public dataset payloads
├── assets/                    # frozen PCA/mapping and selected instruments
├── evidence/                  # results, table cells, reference PDFs
├── environments/              # verified local constraints
├── examples/                  # complete command recipes
├── tests/                     # deterministic numerical/release contracts
├── docs/                      # installation, architecture, data, reproduction
├── provenance/                # module lineage and SHA256 manifest
└── verification/              # recorded preparation checks
```

Generated `outputs/` and local preparation/credential material in `local_only/` are Git-ignored. Reusable implementation, experiment configurations, frozen assets and generated outputs have distinct locations. Structural precedents from official research repositories are documented in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data and frozen experiment identities

| Case | Dataset payload | Observation windows | Main particles | Test origins |
|---|---|---|---:|---|
| BLM/X | 5,416,377 native post IDs, six ordered CSV parts | 6/12/24 h | 10,000 | 11/5/2 |
| Weibo | All 62,804 occurrences in one privacy-processed UTF-8 CSV | 3/6/12 h | 1,000 | 235/123/63 |

BLM data contains native post IDs only, retained unchanged. These IDs permit lookup of original posts subject to availability and platform access; the shared data are not described as fully anonymous. The six parts reconstruct the complete master byte-for-byte using `python run.py merge-blm-ids --output local_only/blm_complete.csv`. No BLM text or user/account table is published.

The Weibo CSV is a **privacy-processed public copy**: `username`, `user_id`, `location` and `image_urls` are removed; identifying body text, links, contact details and detected personal addresses are masked. All 62,804 rows, timestamps, engagement counts, occurrence IDs and original window assignments are retained. Original analysis inputs and frozen representations remain separate; embedding the redacted text does not recreate the original embeddings. Public frozen coordinates retain their numerical values, with original text hashes removed. `hk_weibo:<row>` is a local occurrence ID. See the [data provenance and field dictionary](docs/DATA.md) and [privacy verification checklist](docs/RELEASE.md).

Scientific identities are `hssc_adaptive_v3_sparse_v1` for BLM and `hssc_weibo_3d2d_3h6h12h_v1` for Weibo. Seeds are 11, 29 and 47. Sensitivity budgets are 5,000/10,000/20,000 and 500/1,000/2,000 respectively. Each observation series forecasts one window ahead within the same phase and retains the complete calendar and target occurrence mass. Internal artifact keys remain `H0` = Pure BCM, `HF` = Fixed assimilation, `HA` = Selected assimilation / TRASER.

Complete evidence retains negative findings: Weibo's selected assimilation does not improve the mean primary distance at 6 h; it improves at 3 h and 12 h. The 12-hour statistical comparison contains 5 BLM and 63 Weibo origins, 1,020 seed-method rows, 340 origin-method rows, and thirteen error metrics. Undefined singleton/degenerate KDE values remain missing.

## Experiments and displays

```sh
python run.py replay --dataset hk_weibo --check-only
python run.py replay --dataset hk_weibo --references
python run.py replay --dataset hk_weibo --sensitivity
python run.py scenarios --dataset hk_weibo
```

These scientific commands process the full declared design. `--references` fits/selects WAR, WR and LQD on training/development data at 12 h. Conditional scenarios use all declared seeds, five topics, three fractions, matched controls, source multipliers 0.5/0/2 and paired noise. They describe conditional model behavior, not real-world causal effects.

`figures` reads complete frozen arrays and generates 21 quantitative PDF/SVG figures plus the preserved framework PDF. Archived manuscript reference PDFs retain original bytes. The portable renderer uses DejaVu Sans and readable logical 11-point text; it is not claimed to be pixel-identical to the original font exports. Cells for all 3 main and 19 supplementary tables are in `evidence/table_data`.

## Citation and terms

Use [CITATION.cff](CITATION.cff) for the repository software citation. The manuscript title and author order are taken from the current author manuscript. No publication DOI or venue acceptance is asserted.

The authors have not selected an open-source code license. Public availability does not assign a license to code, third-party data or model weights. Source/collection facts not established by local evidence remain explicitly unknown in the data guide; no upstream URL or collection date is guessed.
