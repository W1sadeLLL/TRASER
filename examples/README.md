# Reproduction commands

All commands below are run from the repository root. They use the full declared
event, seeds and budgets. They are command recipes, not precomputed claims that
the corresponding scientific run was executed during packaging.

## Inspect and reproduce paper displays (CPU)

```sh
python -m pip install -e .
python run.py doctor
python run.py test
python run.py audit-data
python run.py verify-release
python run.py figures
```

## Prepare original Weibo occurrences (CPU)

```sh
python run.py prepare-occurrences --dataset hk_weibo --input data/weibo/weibo_posts.csv --output outputs/prepared_weibo
```

## Replay frozen scientific instruments (CUDA)

```sh
python run.py replay --dataset hk_weibo --check-only
python run.py replay --dataset blm_x --check-only
python run.py replay --dataset hk_weibo
python run.py replay --dataset hk_weibo --references
python run.py replay --dataset hk_weibo --sensitivity
python run.py scenarios --dataset hk_weibo
```

BLM requires externally supplied compatible per-occurrence coordinates/indices;
the default check reports the four absent files explicitly. After obtaining
these assets, use `--asset-root /path/to/assets` with BLM commands. Private data
and acquired text must stay outside tracked publication files.

## Rebuild representation/mapping and development searches

```sh
python run.py build-representation --embeddings embeddings.npy --fit-weights fit_weights.npy --output outputs/new_representation
python run.py fit-mapping --input isolated_label_coordinates.npz --output outputs/new_mapping
python run.py select-parameters --dataset hk_weibo
```

These interfaces require the official encoder output and explicit original
isolation inputs. New fits are not automatically the frozen paper instruments.
Read `docs/REPRODUCIBILITY.md` before running them.
