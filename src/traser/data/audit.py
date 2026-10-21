import csv, hashlib, json
from pathlib import Path
from traser.io import ROOT, read, write, sha
from traser.data.privacy import PUBLIC_FIELDS, REMOVED_FIELDS, rule_redact


def audit():
    manifest = read(ROOT / "data/blm/manifest.json")
    seen = set()
    digest = hashlib.sha256()
    for i, part in enumerate(manifest["parts"]):
        p = ROOT / part["file"]
        assert sha(p) == part["sha256"]
        with p.open("rb") as f:
            if i:
                f.readline()
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        with p.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["pid"]
            n = 0
            for r in reader:
                pid = r["pid"]
                assert pid.isdigit() and pid not in seen
                seen.add(pid)
                n += 1
            assert n == part["rows"]
    assert (
        len(seen) == manifest["rows"] == 5416377
        and digest.hexdigest() == manifest["complete_csv_sha256"]
    )
    schema = read(ROOT / "data/weibo/schema.json")
    p = ROOT / "data/weibo/weibo_posts.csv"
    assert sha(p) == schema["csv_sha256"]
    phases = {"train": 0, "development": 0, "test": 0}
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == PUBLIC_FIELDS
        assert not set(REMOVED_FIELDS).intersection(reader.fieldnames)
        for n, r in enumerate(reader, 1):
            assert r["source_row"] == str(n) and r["occurrence_id"] == f"hk_weibo:{n}"
            assert rule_redact(r["content"]) == r["content"], ("Unredacted pattern", n)
            t = r["publish_time_utc"]
            phase = "train" if t < "2025-11-28" else "development" if t < "2025-11-30" else "test"
            phases[phase] += 1
    assert n == 62804 and phases == {"train": 24204, "development": 17503, "test": 21097}
    calendars = {}
    for d, hs, counts in [
        ("blm_x", [6, 12, 24], [11, 5, 2]),
        ("hk_weibo", [3, 6, 12], [235, 123, 63]),
    ]:
        for h, expected in zip(hs, counts):
            cal = read(ROOT / f"assets/{d}/w{h}/calendar.json")
            ws = cal["windows"]
            legal = [
                (a["window"], b["window"])
                for a, b in zip(ws, ws[1:])
                if a["phase"] == b["phase"] == "test"
                and a["full_duration"]
                and b["full_duration"]
                and b["eligible_occurrences"]
            ]
            assert len(legal) == expected, (d, h, len(legal), expected)
            calendars[f"{d}/{h}h"] = {
                "test_origins": len(legal),
                "occurrences": sum((w["eligible_occurrences"] for w in ws)),
            }
    result = {
        "data_integrity": "PASS",
        "blm_ids": len(seen),
        "weibo_posts": n,
        "weibo_phase_counts": phases,
        "weibo_public_privacy_schema": "PASS",
        "weibo_residual_rule_scan": "PASS",
        "shared_data_fully_anonymous": False,
        "calendars": calendars,
        "scientific_reproduction_executed": False,
    }
    write(ROOT / "outputs/data_audit.json", result)
    print(json.dumps(result, indent=2))
    return result


def merge_ids(output):
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(target)
    m = read(ROOT / "data/blm/manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        for i, p in enumerate(m["parts"]):
            with (ROOT / p["file"]).open("rb") as f:
                if i:
                    f.readline()
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    out.write(block)
    assert sha(target) == m["complete_csv_sha256"]
    print(json.dumps({"rows": m["rows"], "sha256": sha(target)}))
