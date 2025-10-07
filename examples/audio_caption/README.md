# Audio Caption Example

This example demonstrates training an audio-to-text captioning model using Auden.
It mirrors the CLAP example but focuses on caption generation with a pretrained
audio encoder.

## Quickstart

- Single GPU

```bash
Audit python examples/audio_caption/train.py exp_dir=examples/audio_caption/exp/demo
```

- Multi-GPU (DDP)

```bash
GPUS=8 bash examples/audio_caption/scripts/train.sh examples/audio_caption/exp/demo
```

## Configuration

See `examples/audio_caption/configs/train.yaml`.

Key entries:
- `model.model_type: audio_caption`
- `model.audio_encoder`: `{ model_type: zipformer, pretrained_model?: <path_or_id>, frozen?: bool }`
- `data.label_field`: supervision field where captions are stored (default: `caption`)

## Data

Provide YAML lists in:
- `data.train_data_config`: mux training manifest spec
- `data.valid_data_config`: list of validation manifests

Manifests must include captions under `supervision.custom[label_field]` or
`supervision.<label_field>`.

## Notes

- If `audio_encoder.pretrained_model` is set, weights are loaded (strict=True).
- Set `audio_encoder.frozen: true` for linear probing of the encoder.
- Config is saved to `exp_dir` for reproducibility.

## Inference: Load a pretrained model and generate captions

```python
from auden.auto.auto_model import AutoModel
import torch

# Load from a local directory or a Hugging Face repo id
model = AutoModel.from_pretrained("/pretrained/model/dir_or_hf_repo").eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# 1) Provide WAV paths
captions = model.generate(input=["/abs/a.wav", "/abs/b.wav"], max_length=128)
print(captions)

# 2) Provide precomputed features (x, x_lens)
# Ensure feature extraction matches training
# x, x_lens = feature_extractor(...)
# captions = model.generate(input=(x, x_lens), max_length=128)
```

Inputs supported by `model.generate`:
- Precomputed features `(x, x_lens)` — ensure feature extraction matches training
- WAV paths `List[str]`
- Mono waveforms `List[Tensor/ndarray]`
