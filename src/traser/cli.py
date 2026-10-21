"""Single command dispatcher for TRASER research workflows."""

from traser.data.audit import audit, merge_ids
from traser.utils.environment import doctor
from pathlib import Path
import argparse, csv, hashlib, importlib, json, sys
from traser.io import ROOT, read, write, sha


def main():
    p = argparse.ArgumentParser(
        description="TRASER: data audit, frozen scientific replay, and paper figures. No network or SSH dependency."
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor", help="Import current numerical modules and inspect dependencies; no experiment"
    )
    sub.add_parser(
        "audit-data", help="Verify ALL published IDs, CSV records, hashes and full calendars"
    )
    sub.add_parser(
        "verify-release", help="Verify every publication payload against its SHA256 manifest"
    )
    sub.add_parser("test", help="Run deterministic engineering/unit tests; no paper experiment")
    privacy = sub.add_parser(
        "redact-weibo",
        help="Redact a complete PRIVATE author CSV using a locally supplied NER model; no network",
    )
    privacy.add_argument("--input", type=Path, required=True)
    privacy.add_argument("--output", type=Path, required=True)
    privacy.add_argument("--ner-model", type=Path, required=True)
    privacy.add_argument("--private-cache", type=Path, required=True)
    m = sub.add_parser("merge-blm-ids", help="Reconstruct complete pid CSV with byte identity")
    m.add_argument("--output", required=True)
    f = sub.add_parser(
        "figures", help="Regenerate quantitative figures from frozen complete arrays"
    )
    r = sub.add_parser(
        "replay", help="Full-calendar GPU replay using frozen selections, never parameter search"
    )
    r.add_argument("--dataset", choices=["blm_x", "hk_weibo"], required=True)
    r.add_argument(
        "--asset-root", type=Path, help="External compatible asset tree; defaults to assets/"
    )
    r.add_argument("--check-only", action="store_true")
    r.add_argument(
        "--references",
        action="store_true",
        help="Also refit/select WAR/WR/LQD on training/development at 12 h only",
    )
    r.add_argument(
        "--sensitivity",
        action="store_true",
        help="Replay ALL three declared particle budgets with frozen choices",
    )
    b = sub.add_parser(
        "build-representation",
        help="Fit training-only weighted PCA to supplied normalized 4096D embeddings and explicit isolation weights",
    )
    b.add_argument("--embeddings", type=Path, required=True)
    b.add_argument("--fit-weights", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    raw = sub.add_parser(
        "prepare-occurrences",
        help="Prepare full event calendars from Weibo CSV or externally hydrated BLM author CSV",
    )
    raw.add_argument("--dataset", choices=["blm_x", "hk_weibo"], required=True)
    raw.add_argument("--input", type=Path, required=True)
    raw.add_argument("--output", type=Path, required=True)
    mapping = sub.add_parser(
        "fit-mapping",
        help="Fit eleven masked Ridge axes to externally supplied isolated fit/val machine labels",
    )
    mapping.add_argument("--input", type=Path, required=True)
    mapping.add_argument("--output", type=Path, required=True)
    scenario = sub.add_parser(
        "scenarios", help="Full conditional design from replay final states and frozen prototypes"
    )
    scenario.add_argument("--dataset", choices=["blm_x", "hk_weibo"], required=True)
    scenario.add_argument("--asset-root", type=Path)
    selection = sub.add_parser(
        "select-parameters",
        help="Rebuild development parameter search on complete frozen representations (new outputs)",
    )
    selection.add_argument("--dataset", choices=["blm_x", "hk_weibo"], required=True)
    selection.add_argument("--asset-root", type=Path)
    args = p.parse_args()
    if args.command == "redact-weibo":
        from traser.data.privacy import redact

        result = redact(args.input, args.output, args.ner_model, args.private_cache)
        write(ROOT / "verification/privacy_transformation.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "test":
        import unittest

        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    elif args.command == "doctor":
        doctor()
    elif args.command == "audit-data":
        audit()
    elif args.command == "verify-release":
        with (ROOT / "provenance/publication_manifest.csv").open(encoding="utf-8", newline="") as f:
            entries = list(csv.DictReader(f))
        for row in entries:
            path = (ROOT / row["path"]).resolve()
            assert path.is_relative_to(ROOT)
            assert path.stat().st_size == int(row["bytes"]) and sha(path) == row["sha256"], row[
                "path"
            ]
        print(
            json.dumps(
                {
                    "publication_payload_files": len(entries),
                    "sha256": "PASS",
                    "scientific_reproduction_executed": False,
                }
            )
        )
    elif args.command == "merge-blm-ids":
        merge_ids(args.output)
    elif args.command == "figures":
        from traser.visualization.render import render

        render()
    elif args.command == "replay":
        from traser.experiments.replay import run

        if args.sensitivity and (not args.check_only):
            cfg = read(
                ROOT / f"configs/experiments/{('blm' if args.dataset == 'blm_x' else 'weibo')}.json"
            )
            for n in cfg["particle_sensitivity"][args.dataset]:
                run(args.dataset, args.asset_root, False, False, n)
        else:
            run(args.dataset, args.asset_root, args.check_only, args.references)
    elif args.command == "build-representation":
        from traser.representations.pca import build

        build(args.embeddings, args.fit_weights, args.output)
    elif args.command == "prepare-occurrences":
        from traser.data.preprocessing import build

        build(args.dataset, args.input, args.output)
    elif args.command == "fit-mapping":
        from traser.semantics.ridge import fit

        fit(args.input, args.output)
    elif args.command == "scenarios":
        from traser.experiments.scenarios import run

        run(args.dataset, args.asset_root)
    elif args.command == "select-parameters":
        from traser.experiments.selection import run

        run(args.dataset, args.asset_root)
    return 0
