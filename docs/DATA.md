# Data contents and provenance

## Weibo: complete privacy-processed public copy

`data/weibo/weibo_posts.csv` contains **all 62,804 occurrences**, in their original order. This is a privacy-processed release, not the original analysis text. The private source JSON and original analysis inputs remain separate and unchanged. The source JSON SHA256 is `259504f12a093efa59dcb57f7b1c3d30cf7be0a971b34f8b7a7f0bf39f9fd914`; the current public CSV hash is in `data/weibo/schema.json` and the publication manifest.

| Public column | Meaning / storage |
|---|---|
| `source_row` | One-based position in the original JSON array; unchanged |
| `occurrence_id` | `hk_weibo:<source_row>`, local occurrence ID, not a native Weibo post ID; unchanged |
| `content` | Unicode text with user identifiers, links, contacts and personal addresses redacted |
| `publish_time` | Original UTC+08 timestamp string; unchanged |
| `publish_time_utc` | Derived UTC ISO8601 timestamp; unchanged |
| `repost_count`, `comment_count`, `like_count` | Original integer engagement counts; unchanged |

The entire `username`, `user_id`, `location` and `image_urls` columns have been removed. Redaction replaces account mentions, source author names/nicknames and detected personal names with `[USER]`; web/profile/original-post links with `[LINK]`; phone numbers, email addresses, identity numbers and detected contact handles with `[CONTACT]`; and detected detailed/personal addresses with `[ADDRESS]`. All links are conservatively masked, including links that might not identify a person. A private author lexicon, deterministic patterns and a local Chinese named-entity model are applied to the complete text collection. No post text is sent to an external recognition service. The private lexicon and model weights are not distributed. Detection can also mask public names or place names; redaction is not a guarantee that retained prose cannot be searched or linked.

Read the CSV with a CSV parser: multiline posts mean physical newline counts are not record counts. `schema.json` describes the public fields. `verification/privacy_transformation.json` and `verification/privacy_validation.json` record the full-collection checks without private example text or author dictionaries.

All occurrences retain their times and **original 3/6/12-hour window assignments**: train 24,204, development 17,503, test 21,097. UTC phase boundaries remain 2025-11-25 / 11-28 / 11-30 / 2026-01-01. The 229-record distinction between the training calendar and PCA fitting reflects the original evaluation account/component exclusions (227 + 2), not removal of forecasting targets. Publication timestamps range from 2025-11-25 10:51 to 2025-12-31 22:51 UTC+08; they are not crawler operation dates.

Frozen numerical coordinates and coefficients describe the **original analysis**, and are stored separately under `assets/`. Original text hashes have been removed from the public per-post coordinate/index Parquet files; all numerical coordinates, row bindings, occurrence IDs and window indices are preserved. The complete original frozen files are retained privately. Re-embedding redacted text creates a different representation and must not be described as reconstruction of the original paper embeddings. The public CSV does not recover the private account-isolation inputs.

The authors identify the source as a topic-based Weibo collection. Exact collection/access dates, historical request settings and redistribution conditions require the authors' source/rights statement; post dates, current crawler settings and privacy processing do not independently establish those facts. No third-party open license is asserted.

## BLM/X: native post IDs only, retained unchanged

The dataset contains **5,416,377 unique native post IDs** in one column named `pid`. This is the subset used in the study, not the entire upstream corpus. Phase counts are train 1,938,847, development 1,187,580, test 2,289,950. The complete master is 119,160,300 bytes and is retained privately in Git-ignored `local_only/blm_used_pid.csv`.

The first five ordered CSV parts in `data/blm/ids` contain 1,000,000 IDs each; the sixth contains 416,377. `data/blm/manifest.json` records order, sizes, counts and SHA256. **The privacy revision does not alter any BLM ID or part bytes.** Reconstruct the complete byte-identical CSV with:

```sh
python run.py merge-blm-ids --output local_only/blm_reconstructed.csv
```

The output must not already exist. Preserve `pid` as a string; floating-point conversion can corrupt long IDs:

```python
import pandas as pd
ids = pd.read_csv('data/blm/ids/blm_used_pid.part-001.csv', dtype={'pid': 'string'})
```

BLM `pid` identifies a post, not a user. It can nevertheless be used to retrieve the original post and associated public account information, subject to current availability and platform access conditions. **The shared datasets are not described as fully anonymous.** BLM post text, user/account fields, source JSONL, text-bearing annotations/caches and individual BLM embedding coordinates/occurrence-account tables are excluded. Numerical aggregates and trained coefficients are included separately. The upstream corpus is Giorgi et al. (2022), *Twitter Corpus of the #BlackLivesMatter Movement and Counter Protests: 2013 to 2021*; an unverified upstream download URL is not supplied.

Readers must obtain source access through an appropriate route and follow applicable platform conditions. Six parts avoid GitHub's ordinary single-file limit without sampling or truncation; see [GitHub's large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
