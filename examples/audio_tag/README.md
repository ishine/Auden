# Audio Tagging Example

This example shows how to train, finetune, evaluate, and run inference for audio tagging using Hydra configs.

## Data configs

See detailed data preparation and manifest format in [configs/README.md](configs/README.md).

Set these keys in your experiment config (e.g., `configs/train.yaml` or `configs/finetune.yaml`):
- data.train_data_config: path to a train data-config YAML
- data.valid_data_config: path to a valid data-config YAML
- data.test_data_config: path to a test data-config YAML (for evaluation)
- data.label_field: tag field name in the manifest (default: `audio_tag`; e.g., `audio_event`)
- model.id2label: path to an `id2label.json` matching your labels
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
- `model.encoder`: model_type (e.g., zipformer) or a path/HF repo with config.json.
- `model.id2label`: path to JSON mapping like {"0": "label_0", ...}.

## Finetuning or Linear Probing with a Pretrained Encoder
```
python finetune.py \
  exp_dir=./exp/your_exp_dir
```
In `examples/audio_tag/configs/finetune.yaml`:
```yaml
encoder:
  model_type: zipformer
  pretrained_model: path_or_name_to_pretrained_model  # either a full model with a compatible encoder or a separate encoder checkpoint
  freeze_encoder: False  # False for full finetuning; True for linear probing
```

## Batch evaluation on test data and save results

- With a prepared model .pt (e.g., averaged or exported):
```
python evaluate.py \
  exp_dir=./examples/audio_tag/exp \
  checkpoint.filename=/abs/path/to/model.pt \
  data.test_data_config=examples/audio_tag/configs/*/test_data_config.yaml
```

- Average from trainer checkpoints, save to the same exp_dir, then evaluate:
```
python evaluate.py \
  exp_dir=./examples/audio_tag/exp \
  checkpoint.epoch=10 checkpoint.avg=5 \
  data.test_data_config=examples/audio_tag/configs/*/test_data_config.yaml
```

## Load a pretrained model and run audio tagging

```python
from auden.auto.auto_model import AutoModel
import torch

model = AutoModel.from_pretrained("/pretrained/model/dir").eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# list of WAV paths
labels, topk_logits, topk_probs = model.generate(input=["/abs/a.wav", "/abs/b.wav"], topk=5, threshold=0.0)
```

Inputs supported by `model.generate`:
- Precomputed features `(x, x_lens)` — ensure feature extraction matches training
- WAV paths `List[str]`
- Mono waveforms `List[Tensor/ndarray]`
