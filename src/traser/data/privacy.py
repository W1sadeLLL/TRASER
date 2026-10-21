"""Local-only privacy transformation; never downloads or transmits post text.

The author-name lexicon is constructed in memory from a private input CSV.
Only aggregate receipts are written. This is redaction, not an anonymity claim.
"""

import csv
import hashlib
import json
import re
import os
from collections import Counter
from pathlib import Path

PUBLIC_FIELDS = [
    "source_row",
    "occurrence_id",
    "content",
    "publish_time",
    "publish_time_utc",
    "repost_count",
    "comment_count",
    "like_count",
]
REMOVED_FIELDS = ["username", "user_id", "location", "image_urls"]
RULES = [
    (
        "USER",
        re.compile(
            r"(?:姓名|昵称|暱稱|网名|網名|实名|實名)\s*[:：为為叫]?\s*[^\s，,。；;：:！!？?\[\]]{1,24}"
        ),
    ),
    (
        "USER",
        re.compile(
            r"[\u3400-\u9fff]{1,5}(?:先生|女士|小姐|师傅|師傅|太太|夫妇|夫婦|阿姨|伯伯|叔叔)"
        ),
    ),
    ("CONTACT", re.compile(r"(?<!\w)[A-Za-z]{1,2}\d{6}\(?[0-9Aa]\)?(?!\w)")),
    (
        "CONTACT",
        re.compile(r"(?<!\d)\d{6}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}(?!\d)"),
    ),
    ("CONTACT", re.compile(r"(?<!\d)\+\d(?:[ ()-]?\d){7,14}(?!\d)")),
    ("CONTACT", re.compile(r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}")),
    (
        "CONTACT",
        re.compile(
            r"(?<!\d)\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"
        ),
    ),
    ("CONTACT", re.compile(r"(?<![\dA-Za-z])(?:\+?86[\s-]?)?1[3-9](?:[\s-]?\d){9}(?!\d)")),
    ("CONTACT", re.compile(r"(?<!\d)(?:\+?(?:852|853)[\s-]?)?[2-9]\d{3}[ -]?\d{4}(?!\d)")),
    (
        "CONTACT",
        re.compile(
            r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\(?0\d{2,3}\)?[ -]?)\d{7,8}(?:[ -]\d{1,6})?(?!\d)"
        ),
    ),
    (
        "CONTACT",
        re.compile(
            r"(?:电话|電話|手机|手機|联系|聯繫|微信|WeChat|wechat|QQ|电邮|電郵|邮箱|郵箱)\s*[:：号號]?\s*[A-Za-z0-9_+() -]{5,}"
        ),
    ),
    ("LINK", re.compile(r"(?:https?://|www\.)[^\s<>\[\]{}\u3000]+", re.I)),
    (
        "LINK",
        re.compile(r"(?:(?:m\.|www\.)?weibo\.(?:com|cn)|t\.cn|weibo\.tv)/[^\s<>\[\]{}]+", re.I),
    ),
    ("USER", re.compile(r"[@＠][^\s@＠:：,，。!！?？;；<>\[\]（）()\"\u3000]+")),
    (
        "ADDRESS",
        re.compile(
            r"(?:住址|家庭地址|家庭住址|家住|现住|現住|居住地址|通讯地址|通訊地址|收件地址)\s*[:：为為在]?[^\n。！!？?；;]{2,100}"
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"[\u3400-\u9fffA-Za-z0-9]{1,25}(?:路|街|巷|弄|道)\s*[0-9一二三四五六七八九十百]+\s*(?:号|號)[^\n。！!？?；;，,]{0,35}"
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"[\u3400-\u9fffA-Za-z0-9]{1,20}(?:苑|邨|村|小区|小區|花园|花園|大厦|大廈|屋苑|公寓)[^\n。！!？?；;，,]{0,12}(?:[A-Za-z0-9一二三四五六七八九十百]+(?:栋|棟|座|幢|单元|單元|楼|樓|室|房|层|層|号|號)){1,4}"
        ),
    ),
]


def rule_redact(text, counts=None):
    counts = counts if counts is not None else Counter()
    for label, pattern in RULES:
        text, n = pattern.subn("[" + label + "]", text)
        counts[label] += n
    return text


def mask_spans(text, spans):
    """Merge overlapping detections so one substitution cannot expose another."""
    merged = []
    for start, end, label in sorted(spans):
        if end <= start:
            continue
        if merged and start < merged[-1][1]:
            a, b, previous = merged[-1]
            merged[-1] = (a, max(b, end), previous)
        else:
            merged.append((start, end, label))
    for start, end, label in reversed(merged):
        text = text[:start] + "[" + label + "]" + text[end:]
    return text


def normalize_placeholders(text):
    previous = None
    while previous != text:
        previous = text
        text = re.sub(
            r"\[+(?:[A-Z]{1,10})?\[(USER|LINK|CONTACT|ADDRESS)\](?:[A-Z]{0,10}\])?", r"[\1]", text
        )
        text = re.sub(r"(\[(?:USER|LINK|CONTACT|ADDRESS)\])[A-Z]{0,10}\]+", r"\1", text)
    return text


def redact(input_path, output_path, model_path, private_cache):
    # Optional isolated runtime; does not change the research Python installation.
    if os.environ.get("TRASER_PRIVACY_RUNTIME"):
        import sys

        sys.path.insert(0, os.environ["TRASER_PRIVACY_RUNTIME"])
    import ahocorasick
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.init()
    print(json.dumps({"privacy_torch": torch.__version__, "privacy_device": device}), flush=True)
    if os.environ.get("TRASER_PRIVACY_RUNTIME") and device != "cuda":
        raise RuntimeError("The explicitly selected isolated CUDA runtime is unavailable")
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("Keep the private source separate from the public output")
    rows = list(csv.DictReader(input_path.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 62804 or not all(k in rows[0] for k in REMOVED_FIELDS):
        raise ValueError("Expected the complete private source CSV with author fields")
    names = {r["username"].strip() for r in rows if r["username"].strip()}
    automaton = ahocorasick.Automaton()
    for name in names:
        automaton.add_word(name, len(name))
    # Explicit private account IDs also cannot survive in body text.
    for uid in {r["user_id"].strip() for r in rows if len(r["user_id"].strip()) >= 5}:
        automaton.add_word(uid, len(uid))
    automaton.make_automaton()
    counts = Counter()

    def initial(text):
        text = rule_redact(text, counts)
        spans = [(end - size + 1, end + 1, "USER") for end, size in automaton.iter(text)]
        counts["dictionary_detections"] += len(spans)
        return mask_spans(text, spans)

    prepared = [initial(r["content"]) for r in rows]
    unique = list(dict.fromkeys(prepared))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    torch.set_num_threads(8)
    if device == "cuda":
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    cache_path = Path(private_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            completed[item["key"]] = item["text"]
    labels = model.config.id2label
    model_hashes = {
        p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        for p in Path(model_path).iterdir()
        if p.is_file() and p.suffix in {".json", ".txt", ".bin", ".safetensors"}
    }
    cache_scope = (
        hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest()
        + ":ner0.10:v2:"
    )

    def cache_key(text):
        return hashlib.sha256((cache_scope + text).encode()).hexdigest()

    spans_by_text = {}
    segments = []
    for text in unique:
        key = cache_key(text)
        if key in completed:
            continue
        spans_by_text[text] = []
        for start in range(0, max(1, len(text)), 300):
            segments.append((text, start, text[start : start + 380]))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "unique_prepared_texts": len(unique),
                "ner_segments": len(segments),
                "cached": len(completed),
            }
        ),
        flush=True,
    )
    with torch.inference_mode():
        for i in range(0, len(segments), 32):
            batch = segments[i : i + 32]
            encoded = tokenizer(
                [s[2] for s in batch],
                padding=True,
                truncation=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping").tolist()
            logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
            probabilities = logits.float().softmax(-1)
            predicted = logits.argmax(-1)
            # Conservative token threshold also masks plausible entities below top-1.
            name_tags = [k for k, v in labels.items() if v.endswith("name")]
            address_tags = [k for k, v in labels.items() if v.endswith("address")]
            name_p = probabilities[:, :, name_tags].sum(-1)
            address_p = probabilities[:, :, address_tags].sum(-1)
            predicted = torch.where(name_p >= 0.10, name_tags[0], predicted)
            predicted = torch.where(address_p >= 0.10, address_tags[0], predicted).tolist()
            for (text, base, _), bounds, tags in zip(batch, offsets, predicted):
                group = []
                protected = [
                    m.span() for m in re.finditer(r"\[(?:USER|LINK|CONTACT|ADDRESS)\]", text)
                ]
                for (a, b), tag in zip(bounds, tags):
                    label = labels[tag]
                    target = (
                        "USER"
                        if label.endswith("name")
                        else "ADDRESS"
                        if label.endswith("address")
                        else None
                    )
                    if (
                        target
                        and b > a
                        and not any(a + base < end and b + base > start for start, end in protected)
                    ):
                        if group and group[-1][2] == target and a + base <= group[-1][1]:
                            x, y, z = group[-1]
                            group[-1] = (x, b + base, z)
                        else:
                            group.append((a + base, b + base, target))
                spans_by_text[text].extend(group)
            if i % 1024 == 0:
                print(
                    json.dumps(
                        {
                            "ner_segments_done": min(i + len(batch), len(segments)),
                            "ner_segments_total": len(segments),
                        }
                    ),
                    flush=True,
                )
    with cache_path.open("a", encoding="utf-8") as cache:
        for text, spans in spans_by_text.items():
            counts["ner_name_detections"] += sum(s[2] == "USER" for s in spans)
            counts["ner_address_detections"] += sum(s[2] == "ADDRESS" for s in spans)
            key = cache_key(text)
            completed[key] = normalize_placeholders(rule_redact(mask_spans(text, spans)))
            cache.write(json.dumps({"key": key, "text": completed[key]}, ensure_ascii=False) + "\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=PUBLIC_FIELDS, quoting=csv.QUOTE_ALL, lineterminator="\n"
        )
        writer.writeheader()
        for row, text in zip(rows, prepared):
            result = completed[cache_key(text)]
            # A final dictionary pass catches spans revealed by adjacent substitutions.
            result = initial(result)
            changed += result != row["content"]
            writer.writerow({k: result if k == "content" else row[k] for k in PUBLIC_FIELDS})
    report = {
        "rows": len(rows),
        "content_rows_changed": changed,
        "removed_columns": REMOVED_FIELDS,
        "full_local_ner_pass": True,
        "model": "uer/roberta-base-finetuned-cluener2020-chinese",
        "model_artifact_sha256": model_hashes,
        "ner_entity_probability_threshold": 0.10,
        "inference_device": device,
        "local_inference_only": True,
        "counts": dict(counts),
        "fully_anonymous_claim": False,
        "original_analysis_modified": False,
    }
    return report
