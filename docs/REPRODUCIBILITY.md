# Reproducibility layers and asset boundaries

## Layer 1: start from original data

The public Weibo CSV retains all occurrences but contains privacy-processed text. It preserves the original calendars, not original text/embedding identity. BLM starts from the unchanged native ID parts; author text/timestamps must be acquired separately. `prepare-occurrences` accepts the published Weibo CSV, or a complete externally hydrated BLM CSV with `pid`, `content`, `publish_time_utc`. It verifies the full BLM ID set before proceeding. It does not implement a platform downloader.

```sh
python run.py prepare-occurrences --dataset hk_weibo --input data/weibo/weibo_posts.csv --output outputs/prepared_weibo
```

Derived text normalization follows the original pipeline: Weibo applies Unicode NFKC and whitespace normalization; BLM additionally replaces URLs and user mentions with `[URL]` and `[USER]` and truncates encoder input at 12,000 characters. The public Weibo text has already undergone privacy redaction, so preparing or embedding it creates a new text identity. It must not be matched to original frozen text hashes or described as exact paper reconstruction. The original analysis inputs remain private and unchanged.

Encoding uses **Qwen3-VL-Embedding-8B**, 4,096-dimensional L2-normalized embeddings. Model weights and the original inference service state are not bundled. This release consumes an externally created `embeddings.npy`, ordered exactly like `encoder_texts.parquet`. Model acquisition/inference requires the official model implementation, its dependencies and terms. The release does not silently call a private endpoint or substitute another encoder. No claim of exact encoder reproduction is made without its original model revision, preprocessing/tokenization settings and original inference implementation.

`build-representation` consumes all unique 4,096D embedding rows plus `fit_weights.npy`. Weights are integer eligible training-occurrence multiplicities after the original account and duplicate-component exclusions; all non-fit rows have weight zero. PCA64 uses weighted training moments, descending eigenspectrum, deterministic component sign, and one scalar training RMS. It uses no whitening or post-PCA L2 normalization.

```sh
python run.py build-representation --embeddings embeddings.npy --fit-weights fit_weights.npy --output outputs/new_representation
python run.py fit-mapping --input isolated_label_coordinates.npz --output outputs/new_mapping
```

The mapping input has `coordinates` (N×64), `labels` (N×11, NaN for missing measurements) and string `split` (`fit`, `val`, `prob`, `eval`). Inputs must already enforce pre-test cutoff and the original account/content component isolation, including the exclusions identified during the Weibo diagnostic audit. Eleven independently masked Ridge axes use alpha 0.01/0.1/1/10/100; exact validation ties prefer larger alpha. Mapped values are clipped to [0,1]. Original labels are machine reference measurements, not independently validated human judgments. Original label text is not shipped with the ID-only BLM data.

The raw preparation/PCA/mapping interfaces create a new explicitly labeled build. Recreating the exact paper identity also requires the original isolation/fit-label manifests, original encoder revision and row identity binding. It is not enough to refit PCA or Ridge on a convenient subset. The complete raw-from-scratch pipeline is therefore not certified by the local preparation checks.

## Layer 2: replay frozen representations and selections

The included frozen PCA identities are BLM `06cbcb66a3fb9b329117`, Weibo `60ca7ef7b025335da946`. `assets/hk_weibo` includes all original numerical coordinates and the three current full occurrence/window indices, stored separately from privacy-processed public text. Original per-post text hashes are excluded; numerical coordinates and row/window bindings are unchanged. BLM includes trained coefficients, selections, calendars, features and numerical results; its individual coordinates/indices must be supplied externally.

An external asset tree must mirror `assets/<dataset>/representation`, `mapping`, `w<hours>`. For BLM it needs the frozen 595,024,582-byte `coordinates.parquet` and each full-resolution `observations.parquet` with `coordinate_row` and `window`, preserving every occurrence and the current calendar. Do not use the 328,306,917-byte original occurrence-account table as public data; create a minimal index locally if authorized. `--check-only` reports missing assets without running a substitute computation.

Full replay requires CUDA PyTorch and sufficient GPU/RAM for the complete targets and 5,000–20,000 BLM or 500–2,000 Weibo particles. Import success on CPU is not evidence that the GPU path has run. Original operators use CUDA float64 accumulation for distance integrals and KDE; TF32 is disabled. Runtime/resource differences and nondeterministic GPU details require comparison of newly generated results to the frozen tables.

The constructor in `traser.experiments.replay` supplies portable paths to the original selected numerical class methods. `traser.dynamics.base` retains only the currently used initialization, HK/Gaussian, assimilation, metrics and statistical reference code. The old residual experiment entry is not present. The transition is called with residual disabled. `H0` = Pure BCM, `HF` = Fixed assimilation, `HA` = Selected assimilation / TRASER.

```sh
python run.py replay --dataset hk_weibo
python run.py replay --dataset blm_x --asset-root /path/to/compatible/assets
python run.py replay --dataset hk_weibo --sensitivity
python run.py replay --dataset hk_weibo --references
python run.py scenarios --dataset hk_weibo
```

`--sensitivity` processes all declared budgets with frozen choices. `--references` refits WAR/WR/LQD using training observations, selects on complete development origins, then scores the complete 12-hour test calendar with thirteen metrics. Conditional scenarios require the final selected-assimilation state from replay, frozen pre-test prototypes, five topics, fractions .05/.10/.20, matching controls, source multipliers .5/0/2, self weight one and paired noise. Their interpretation is conditional model behavior, not a real-world causal intervention estimate.

`select-parameters` retains the original development search in new output directories. BLM retains its duration-scaled earlier mechanical anchor; Weibo uses its newly defined coarse grid and two refinements. Neither uses test outcomes to select. Do not confuse this optional new calculation with the frozen choices shipped here.

BLM boundaries UTC: 2020-05-25 / 06-02 / 06-05 / 06-08. Weibo boundaries UTC: 2025-11-25 / 11-28 / 11-30 / 2026-01-01. Every series uses one-step predictions and retains the full calendar, empty windows and partial endpoints. Scored transitions stay in the same phase and require two full windows and a nonempty target. Mathematically undefined sparse/degenerate KDE entries remain missing; other numerical failures must remain visible.

## Layer 3: regenerate paper displays

`python run.py figures` uses archived arrays, complete score tables and current calendars. It executes no encoder, PCA/Ridge fit, mechanical selection or forecasting. Large `exact_distributions.json` is losslessly gzipped, then read on demand. The archived per-cell table CSVs reproduce all 3 main and 19 supplementary tables; they retain full source precision rather than extracting rounded PDF text.

The renderer preserves the current plotting calculations, panel membership and full sample counts. Adaptations are portable path bindings, removal of manuscript writers/old workflows, a bundled Matplotlib font, and public output names. Original figure-source hashes are preserved. The existing framework vector PDF is copied, not redrawn or AI-regenerated. The source framework script/layout is retained locally for review; the published framework asset is the authoritative schematic reference.

## What was verified locally

See `verification/release_checks.json` and related JSON reports. Local checks include all dataset records and IDs, complete calendars, numerical imports, original-definition equivalence, portable command help, asset inventory, all 21 quantitative render paths and the preserved framework. Full new GPU forecasts, full label rebuilding, remote checkpoint replay and empirical validity were not executed or established during this packaging task.

The original result archive references **22,269 server model-state files not available locally**. All current plotting arrays and reported errors are present. This release does not claim the original computational cache is complete, and does not manufacture absent states.
