# Release scope and verification

Repository: [W1sadeLLL/TRASER](https://github.com/W1sadeLLL/TRASER).
The author authorized uploading the complete experiment release after organizing
its code structure with reference to official journal/conference projects.
The package includes both full requested datasets: all 62,804 Weibo occurrences
in a privacy-processed CSV and the unchanged complete BLM native post-ID set
in six ordered parts. Post IDs remain linkable; no full-anonymity claim is made.

The local complete BLM master and preparation/authentication files are excluded
by `.gitignore`. Every publication payload file is listed with bytes and SHA256
in `provenance/publication_manifest.csv`. No credential, private server deployer,
BLM text or account table belongs in that manifest.

The source is one editable-installable package with one `run.py` entry. The
organization decision and official repositories studied are documented in
`ARCHITECTURE.md`. Full source-import and deterministic numerical tests, complete
data auditing, source-equivalence checks, and all current figure render paths
were exercised locally. New complete GPU scientific forecasts were not run.
See `verification/reorganized_release.json` for the exact scope and checks.

The authors have not assigned an open-source code license. The data guide
explicitly records source/collection/rights facts not substantiated locally.
The exact encoder revision/inference state, original isolation/label inputs,
external BLM replay coordinates/indices and 22,269 original server states remain
outside this local release where described. Public availability is not a claim
that every original computational cache or raw reconstruction prerequisite is
included, or that scientific validity was independently established.

## Privacy revision checklist (software 0.1.1)

- Remove the four direct-identifier/location/image-link columns from the public Weibo CSV.
- Apply author-name/account-ID matching, mentions/link/contact/address rules and local Chinese name/address recognition to the entire collection; retain no private dictionaries or raw examples in public receipts.
- Verify all 62,804 row IDs, timestamps and engagement counts against the private original; verify every 3/6/12-hour occurrence-to-window binding and phase count.
- Keep original analysis inputs and original frozen files private and unchanged. Remove original text hashes from public Weibo coordinate/index files while preserving all remaining values.
- Inspect every tracked payload, including JSON, gzip contents, Parquet schemas/string fields, examples and figure/screenshot/PDF assets, for identity-bearing data copies.
- Preserve all 5,416,377 native BLM post IDs and all six part hashes exactly.
- Replace the identity-bearing publication commit rather than retaining it as an ancestor; inspect all public branches/tags and the complete reachable history.
- Regenerate schemas, provenance hashes, the complete publication manifest and the downloadable candidate archive; rerun release and data checks.
- State that privacy-processed text does not recreate the original frozen text/embedding identity and that BLM post IDs permit lookup.

Aggregate results are recorded in `verification/privacy_validation.json` and
`verification/privacy_transformation.json`. Historical preparation checks refer
to the private original inputs where explicitly marked; current public-file
integrity is determined by the current publication manifest and data audit.

The local-only redaction implementation can be reviewed in `traser.data.privacy`.
For authorized private inputs, install `python -m pip install -e '.[privacy]'`,
obtain the model weights separately, then use the sole workflow entry:

```sh
python run.py redact-weibo --input local_only/private_analysis/weibo_posts.original.csv --output data/weibo/weibo_posts.csv --ner-model local_only/privacy_models/cluener --private-cache local_only/private_analysis/redacted_cache.jsonl
```

The command requires the complete private CSV and an existing local copy of
`uer/roberta-base-finetuned-cluener2020-chinese`. It never downloads a model or
uploads text. This is a privacy transformation, not a scientific experiment.
