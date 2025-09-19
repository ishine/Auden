import json
from pathlib import Path


def load_id2label(path):
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def save_id2label(mapping, out_dir):
    out = Path(out_dir) / "id2label.json"
    with open(out, "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
