from pathlib import Path
import hashlib, json, os, time
from datetime import datetime, timezone

"Current paper implementation; provenance is recorded in provenance/module_map.json."
ROOT = Path(os.environ.get("TRASER_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    for attempt in range(100):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.1)
