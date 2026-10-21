"""Public CSV-to-occurrence interface. No downloaded posts or API credentials."""

import csv, hashlib, json, unicodedata
from datetime import datetime, timedelta, timezone
from traser.io import ROOT, read, write
from traser.representations.text import embedding_text_blm


def build(dataset, input_path, output):
    import pyarrow as pa, pyarrow.parquet as pq

    cfg = read(ROOT / f"configs/experiments/{('blm' if dataset == 'blm_x' else 'weibo')}.json")
    bounds = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in cfg["boundaries"][dataset]]
    calendars = {}
    tables = {h: [] for h in cfg["window_hours"]}
    for h in cfg["window_hours"]:
        rows = []
        index = 0
        for phase, start, end in zip(["train", "development", "test"], bounds, bounds[1:]):
            while start < end:
                last = min(start + timedelta(hours=h), end)
                rows.append(
                    {
                        "window": index,
                        "phase": phase,
                        "start_utc": start.isoformat(),
                        "end_exclusive_utc": last.isoformat(),
                        "duration_hours": (last - start).total_seconds() / 3600,
                        "full_duration": last - start == timedelta(hours=h),
                        "eligible_occurrences": 0,
                    }
                )
                index += 1
                start = last
        calendars[h] = rows
    texts = {}
    occurrences = []
    with input_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if dataset == "hk_weibo":
                pid = row["occurrence_id"]
                text = " ".join(unicodedata.normalize("NFKC", row["content"]).split())
                time = row["publish_time_utc"]
            else:
                pid = row["pid"]
                text = embedding_text_blm(row["content"])
                time = row["publish_time_utc"]
            stamp = datetime.fromisoformat(time.replace("Z", "+00:00"))
            assert stamp.tzinfo is not None and bounds[0] <= stamp < bounds[-1]
            if not text.strip():
                raise ValueError(
                    "Empty author text requires the original reference-only disposition interface; do not silently drop it"
                )
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            coordinate_row = texts.setdefault(key, len(texts))
            occurrences.append(
                {
                    "occurrence_id": pid,
                    "protocol_text_sha256": key,
                    "coordinate_row": coordinate_row,
                    "text": text,
                    "publish_time_utc": stamp.isoformat(),
                }
            )
            for h, ws in calendars.items():
                target = next(
                    (
                        w
                        for w in ws
                        if datetime.fromisoformat(w["start_utc"])
                        <= stamp
                        < datetime.fromisoformat(w["end_exclusive_utc"])
                    )
                )
                target["eligible_occurrences"] += 1
                tables[h].append(
                    {
                        "occurrence_id": pid,
                        "protocol_text_sha256": key,
                        "coordinate_row": coordinate_row,
                        "window": target["window"],
                        "eligible": True,
                    }
                )
    expected = 5416377 if dataset == "blm_x" else 62804
    assert len(occurrences) == expected, "Full target occurrence count required; no subset run"
    if dataset == "blm_x":
        ids = set()
        for part in read(ROOT / "data/blm/manifest.json")["parts"]:
            with (ROOT / part["file"]).open(encoding="utf-8", newline="") as f:
                ids.update((r["pid"] for r in csv.DictReader(f)))
        assert len({r["occurrence_id"] for r in occurrences}) == expected and ids == {
            r["occurrence_id"] for r in occurrences
        }
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(occurrences), output / "occurrences.parquet")
    unique = {
        r["coordinate_row"]: {k: r[k] for k in ["coordinate_row", "protocol_text_sha256", "text"]}
        for r in occurrences
    }
    pq.write_table(
        pa.Table.from_pylist([unique[i] for i in range(len(unique))]),
        output / "encoder_texts.parquet",
    )
    for h, rows in tables.items():
        folder = output / f"w{h}"
        folder.mkdir()
        for r in rows:
            r["mass"] = 1 / calendars[h][r["window"]]["eligible_occurrences"]
        pq.write_table(pa.Table.from_pylist(rows), folder / "observations.parquet")
        write(
            folder / "calendar.json",
            {
                "windows": calendars[h],
                "pca_identity": None,
                "coordinate_path": None,
                "status": "raw preparation; PCA identity must be bound after isolated training fit",
            },
        )
    write(
        output / "preparation.json",
        {
            "dataset": dataset,
            "input_text_status": "privacy-processed public text; does not recreate original text embeddings"
            if dataset == "hk_weibo"
            else "externally hydrated text",
            "occurrences": len(occurrences),
            "unique_encoder_texts": len(texts),
            "frozen_paper_identity_recreated": False,
            "next": "Bind original isolation weights, official encoder settings, PCA and mapping; raw preparation alone is not scientific reproduction",
        },
    )
