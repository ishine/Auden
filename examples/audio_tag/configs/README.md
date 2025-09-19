# Data & Lhotse Manifests (Audio Tagging)

This guide describes how to prepare data configs and Lhotse CutSet manifests for the audio tagging examples.

## 1) Data config files

Referenced by your experiment configs (e.g., `train.yaml`,`finetune.yaml` or `evaludate.yaml`), provide a list of datasets:

```yaml
# {train,valid}_data_config YAML structure (each entry is a dataset)
- name: "audioset_valid"          # optional for train, used in logs for valid
  manifest: /abs/path/to/cuts.jsonl
  hours: 100                       # optional, for logging/mixing weight
  weights: 1                       # optional, train-side repetition/mix weight
```

Notes:
- Train: multiple datasets will be mixed via `CutSet.mux(...)` with weights = `hours` * `weights`.
- Valid: each dataset is evaluated separately; metrics are logged per `name`.

## 2) Lhotse CutSet manifest requirements

Each cut should contain one supervision used for labeling. If multiple supervisions are present,
the datamodule will keep only the first one.

Tags are read from a configurable supervision field `label_field` (default: `audio_tag`).
You can store it either under `supervision.custom[label_field]` or as `supervision.<label_field>`.

Multi-label is supported via list values, or a semicolon-separated string. The dataset will join
lists using `;` to form a single string per cut. If you already have a string like `"Blues;Music"`,
it will be used as-is.

```json
{
  "id": "cut-0001",
  "start": 0.0,
  "duration": 3.2,
  "recording_id": "rec-0001",
  "recording": {
    "id": "rec-0001",
    "sources": [
      { "type": "file", "channels": [0], "source": "/abs/path/to/audio.wav" }
    ],
    "sampling_rate": 16000,
    "num_samples": 51200,
    "duration": 3.2,
    "channel_ids": [0]
  },
  "supervisions": [
    {
      "id": "utt-0001",
      "recording_id": "rec-0001",
      "start": 0.0,
      "duration": 3.2,
      "channel": 0,
      "custom": {
        "audio_tag": ["Blues", "Music"],
        "audio_event": "251;137"
      }
    }
  ]
}
```

Important:
- The datamodule resamples cuts to the configured sampling rate.
- Tags must match `id2label.json` (e.g., `configs/*/id2label_*.json`), as the model maps
  labels to class indices using this file.
- Additional fields can coexist (e.g., `custom.audio_event`). They are ignored unless
  you set `label_field` in your config to that field name.
 - Recording info can be embedded per cut (as shown for `MonoCut`) or referenced via a
   separate Lhotse RecordingSet; both are supported by Lhotse. Use absolute paths in `sources`.

## 3) id2label.json

The classifier head size and label mapping are derived from `id2label.json`.

- Format: a JSON dict mapping string indices to label strings.

```json
{
  "0": "Music",
  "1": "Speech",
  "2": "Dog"
}
```

Requirements:
- Keys must be consecutive stringified integers starting from `"0"`.
- Values must exactly match the tag strings that appear in your manifests
  (after joining multi-label lists with `;`).
- The number of entries defines `num_classes` for the model classifier.

Usage:
- Set `model.id2label` in your experiment config to point to this file.
- Ensure `data.label_field` is consistent with the tag source used in manifests.

