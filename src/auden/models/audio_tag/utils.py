import json
from pathlib import Path


def load_id2label(path):
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def save_id2label(mapping, out_dir, name="id2label.json"):
    out = Path(out_dir) / name
    with open(out, "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
