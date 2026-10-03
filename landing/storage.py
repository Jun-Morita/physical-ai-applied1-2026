"""UTF-8 JSON persistence and bounded-memory file hashing."""

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    temp = path.with_suffix(".tmp")
    try:
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows_at(path):
    with (Path(path) / "samples.jsonl").open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]
