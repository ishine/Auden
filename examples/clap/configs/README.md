# Data & Lhotse Manifests (CLAP)

This guide describes how to prepare data configs and Lhotse CutSet manifests for the CLAP (audio–text) example.

## 1) Data config files

Referenced by your experiment configs (e.g., `train.yaml` or `evaluate.yaml`), provide a list of datasets:

```yaml
# {train,valid,test}_data_config YAML structure (each entry is a dataset)
- name: "wavcaps_valid"
  manifest: /abs/path/to/cuts.jsonl
  hours: 100          # optional, for logging/mixing weight (train)
  weights: 1          # optional, train-side repetition/mix weight
```

Notes:
- Train: multiple datasets will be mixed via `CutSet.mux(...)` with weights = `hours` * `weights`.
- Valid/Test: each dataset is evaluated separately; metrics are logged per `name`.

## 2) Lhotse CutSet manifest requirements

Each cut should contain at least one supervision used for labeling. If multiple supervisions are present, the datamodule keeps only the first one. Captions are read from a configurable supervision field `label_field` (default: `caption`). You can store it either under `supervision.custom[label_field]` or as `supervision.<label_field>`.

```json
{
  "id": "cut-0001",
  "start": 0.0,
  "duration": 3.2,
  "recording_id": "rec-0001",
  "recording": {
    "id": "rec-0001",
    "sources": [ { "type": "file", "channels": [0], "source": "/abs/path/to/audio.wav" } ],
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
        "caption": ["A dog barking", "A person speaking"]
      }
    }
  ]
}
```

Important:
- The datamodule resamples cuts to the configured sampling rate.
- If a caption field is a list, the dataset flattens all captions per cut; evaluation can repeat audio features accordingly for multi-caption scoring.
- Recording info can be embedded per cut (as shown for `MonoCut`) or referenced via a separate Lhotse RecordingSet; both are supported by Lhotse. Use absolute paths in `sources`.
