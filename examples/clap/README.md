# CLAP (Audio–Text) Example

This example shows how to train, evaluate, and run inference for audio–text contrastive learning (CLAP) using Hydra configs.

## Data configs

See detailed data preparation and manifest format in [configs/README.md](configs/README.md).

Set these keys in your experiment config (e.g., configs/train.yaml):
- data.train_data_config: path to a train data-config YAML
- data.valid_data_config: path to a valid data-config YAML
- data.test_data_config: path to a test data-config YAML (for evaluation)
- data.label_field: caption field name in the manifest (default: caption)
- exp_dir: experiment directory for logs/checkpoints

## Training

Single GPU:
```
python train.py \
  exp_dir=./exp/your_exp_dir
```

Multi-GPU DDP (8 GPUs):
```
torchrun --nproc_per_node=8 train.py \
  exp_dir=./exp/your_exp_dir
```

Notes:
- model.audio_encoder: model_type (e.g., zipformer) or a path/HF repo with config.json.
- model.text_encoder: HF text encoder name or path (e.g., bert-base-uncased).
- You can optionally load/freeze pretrained encoders; see comments in configs/train.yaml.

## Retrieval evaluation on test data

Evaluate a pretrained/exported model checkpoint:
```
python retrieval_eval.py \
  exp_dir=./examples/clap/exp \
  checkpoint.filename=/abs/path/to/model.pt \
  data.test_data_config=examples/clap/configs/wav_ac_cl/test_data_config.yaml
```

Average from trainer checkpoints, save to the same exp_dir, then evaluate:
```
python retrieval_eval.py \
  exp_dir=./examples/clap/exp \
  checkpoint.epoch=10 checkpoint.avg=5 \
  data.test_data_config=examples/clap/configs/wav_ac_cl/test_data_config.yaml
```

Multi-caption evaluation:
- Set multi_caption_eval=true to evaluate with multiple captions per audio.
- The dataloader flattens captions and repeats audio features to align.
- Metrics use global negatives with mAP@k computed as MRR@k.

## Load a pretrained CLAP model and encode

```python
from auden.auto.auto_model import AutoModel
import torch

model = AutoModel.from_pretrained("/pretrained/clap/dir").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Encode audio (features)
audio_embeds = model.encode_audio(x=features, x_lens=lengths)

# Encode text (list of captions)
text_embeds = model.encode_text(text=["a dog barking", "children laughing"]) 
```

## Inference with model.generate

Use the CLAP model’s `generate` API to compute cross-modal similarities.

- Input may be a tuple of precomputed features `(x, x_lens)`, or raw inputs that your audio encoder can consume directly (e.g., WAV paths) if supported by `audio_encoder.extract_feature`.
- Returns two similarity matrices: `sim_a2t` (audio→text) of shape (B, M) and `sim_t2a` (text→audio) of shape (M, B), where B is number of audio inputs and M is number of texts.

```python
# With features
sim_a2t, sim_t2a = model.generate((features, lengths), [
    "a dog barking",
    "children laughing",
])

# Example: top-5 text for each audio by similarity
values, indices = sim_a2t.topk(k=5, dim=1)

# With WAV paths (if your audio encoder supports extract_feature on paths)
wav_paths = ["/abs/a.wav", "/abs/b.wav"]
sim_a2t, sim_t2a = model.generate(wav_paths, [
    "a crowd cheering",
    "rain falling",
])
```
Inputs supported by CLAP:
- Precomputed features (x, x_lens) — ensure feature extraction matches training
- WAV paths List[str] (if your model.generate supports direct WAVs)
- Mono waveforms List[Tensor/ndarray]
