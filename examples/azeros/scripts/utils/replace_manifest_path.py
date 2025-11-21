#!/usr/bin/env python3
import json
import gzip
from pathlib import Path

# ============================================================
# ! Comment out the irrelevant datasets.

# Relative path in the provided manifest
# (adjust here to suit your local path)
OLD_PATHS = {
    "iemocap": 'myfolder/data/iemocap/extracted/train/wavs',
    "crema": 'myfolder/data/crema/AudioWAV',
    "meld": 'myfolder/data/meld/MELD_train_wav',
    "ravdess": 'myfolder/data/ravdess/Audio_speech_Actors',
    "tess": 'myfolder/data/tess',
    "dailytalk": 'myfolder/data/dailytalk/data',
    "aishell1": 'myfolder/data/aishell1/data_aishell/wav/train',
    "emotiontalk": 'myfolder/data/emotiontalk/Audio/wav',
    "cs_dialogue": 'myfolder/data/cs_dialogue/short_wav/WAVE/C0',
    "voxceleb2": 'myfolder/data/voxceleb2/dev/wav',
    "wenetspeech": 'myfolder/data/wenetspeech/wavs/train_l',
    "gigaspeech": 'myfolder/data/gigaspeech/train',
    "commonvoice": 'myfolder/data/commonvoice', # cover 10 langs
}

# Fill in your absolute dataset paths here
# (should be compatible with the provided paths)
USER_PATHS = {
    "iemocap": 'your_path_to_datasets/iemocap/...',
    "crema": 'your_path_to_datasets/crema/...',
    "meld": 'your_path_to_datasets/meld/...',
    "ravdess": 'your_path_to_datasets/ravdess/...',
    "tess": 'your_path_to_datasets/tess/...',
    "dailytalk": 'your_path_to_datasets/dailytalk/...',
    "aishell1": 'your_path_to_datasets/aishell1/...',
    "emotiontalk": 'your_path_to_datasets/emotiontalk/...',
    "cs_dialogue": 'your_path_to_datasets/cs_dialogue/...',
    "voxceleb2": 'your_path_to_datasets/voxceleb2/...',
    "wenetspeech": 'your_path_to_datasets/wenetspeech/...',
    "gigaspeech": 'your_path_to_datasets/gigaspeech/...',
    "commonvoice": 'your_path_to_datasets/commonvoice/...',
}
# ============================================================

# Manifest directory (relative to where the script is executed)
MANIFEST_DIR = Path("myfolder/manifests")

# Output directory for fixed manifests
OUTPUT_DIR = Path("configs/manifests")


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
                assert path_str.startswith(old_prefix), (path_str, old_prefix)
                replaced += 1
                return path_str.replace(old_prefix, new_prefix, 1)
                return path_str

            rec = item.get("recording", {})
            for src in rec.get("sources", []):
                src["source"] = replace_prefix(src["source"])
            if replaced > 1000:
                break

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    print(f" ✅ {manifest_path.name}: {count} lines processed, {replaced} paths replaced → {output_path}")


def main():
    print("Starting manifest path replacement...\n")

    for dataset, new_prefix in USER_PATHS.items():
        old_prefix = OLD_PATHS[dataset]

        manifests = sorted(MANIFEST_DIR.glob(f"{dataset}*.jsonl.gz"))
        if not manifests:
            print(f" ⚠️ No manifests found for {dataset} in {MANIFEST_DIR}")
            continue

        for manifest_path in manifests:
            print(f"Processing dataset: {dataset}")
            fix_manifest_jsonl(manifest_path, old_prefix, new_prefix, OUTPUT_DIR)

        print(f" ✅ Finished dataset: {dataset}\n")

    print("All manifests processed successfully!")
    print(f"Fixed files saved in: {OUTPUT_DIR.resolve()}")
    print(f"You may delete the original manifests after the replacement.")


if __name__ == "__main__":
    main()
