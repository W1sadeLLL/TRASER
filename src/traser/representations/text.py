"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

import re, unicodedata


def embedding_text_blm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = re.sub("https?://\\S+|www\\.\\S+", " [URL] ", value, flags=re.I)
    value = re.sub("@([A-Za-z0-9_]{1,30})", " [USER] ", value)
    return " ".join(value.split())[:12000]
