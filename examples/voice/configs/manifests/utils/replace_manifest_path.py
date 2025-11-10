#!/usr/bin/env python3
import json
import gzip
from pathlib import Path

# ============================================================
# Fill in your absolute dataset paths here:
USER_PATHS = {
    "CREMA": "/home/username/datasets/CREMA-D",
    "RAVDESS": "/home/username/datasets/RAVDESS",
    "TESS": "/home/username/datasets/TESS",
    "IEMOCAP": "/home/username/datasets/IEMOCAP",
    "VoxCeleb2": "/home/username/datasets/vox2",
}
# ============================================================

# Manifest directory (relative to where the script is executed)
MANIFEST_DIR = Path("manifests")

# Output directory for fixed manifests
OUTPUT_DIR = Path("manifest_fixed")

# Default placeholder prefix used in template manifests
OLD_PREFIX = "/path"


def fix_manifest_jsonl(manifest_path: Path, old_prefix: str, new_prefix: str, output_dir: Path):
    """Replace old path prefix with a new dataset path inside a .jsonl.gz manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / manifest_path.name

    with gzip.open(manifest_path, "rt", encoding="utf-8") as fin, \
         gzip.open(output_path, "wt", encoding="utf-8") as fout:
        count = 0
        replaced = 0
        for line in fin:
            item = json.loads(line)

            def replace_prefix(path_str: str):
                nonlocal replaced
                if path_str.startswith(old_prefix):
                    replaced += 1
                    return path_str.replace(old_prefix, new_prefix, 1)
                return path_str

            rec = item.get("recording", {})
            for src in rec.get("sources", []):
                src["source"] = replace_prefix(src["source"])

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    print(f"  ✅ {manifest_path.name}: {count} lines processed, {replaced} paths replaced → {output_path}")


def main():
    print("Starting manifest path replacement...\n")

    for dataset, new_prefix in USER_PATHS.items():
        print(f"Processing dataset: {dataset}")

        manifests = sorted(MANIFEST_DIR.glob(f"{dataset}*.jsonl.gz"))
        if not manifests:
            print(f"  ⚠️  No manifests found for {dataset} in {MANIFEST_DIR}")
            continue

        for manifest_path in manifests:
            fix_manifest_jsonl(manifest_path, OLD_PREFIX, new_prefix, OUTPUT_DIR)

        print(f"✅ Finished dataset: {dataset}\n")

    print("All manifests processed successfully!")
    print(f"Fixed files saved in: {OUTPUT_DIR.resolve()}")
    print(f"You may delete the original manifests after the replacement.")


if __name__ == "__main__":
    main()
