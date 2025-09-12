# ASR Example

This example shows how to run training and decoding using Hydra configs (no external scripts required).

## Data configs

Demo data config YAMLs live under `configs/data_configs/` and use absolute placeholder paths:
- `train_data_config.yaml`: list of training manifests with `manifest`, `hours`, optional `weights`, and optional `lang`.
- `valid_data_config.yaml`: list of validation manifests with `name` and `manifest`.
- `test_data_config.yaml`: list of test manifests with `name` and `manifest`.

Edit these files to point `manifest` to your Lhotse CutSet jsonl.gz files.

## Training

Single GPU:
```
python examples/asr/train.py \
  exp_dir=your/exp/dir \
  tokenizer=your/tokenizer/dir \
  model.model_type=asr \
  model.encoder=zipformer \ # either a model name string or a HF repo/local_dir
  data.train_data_config=examples/asr/configs/data_configs/train_data_config.yaml \
  data.valid_data_config=examples/asr/configs/data_configs/valid_data_config.yaml
```

Multi-GPU ddp (8 GPUs):
```
torchrun --nproc_per_node=8 examples/asr/train.py \
  exp_dir=your/exp/dir \
  tokenizer=your/tokenizer/dir \
  model.model_type=asr \
  model.encoder=zipformer \ # either a model name string or a HF repo/local_dir
  data.train_data_config=examples/asr/configs/data_configs/train_data_config.yaml \
  data.valid_data_config=examples/asr/configs/data_configs/valid_data_config.yaml
```

## Batch decoding on test data and save WER&results

- With a prepared model .pt (e.g., averaged or exported):
```
python examples/asr/decode.py \
  exp_dir=./examples/asr/exp \
  checkpoint.filename=/abs/path/to/model.pt \
  data.test_data_config=examples/asr/configs/data_configs/test_data_config.yaml
```

- Average from trainer checkpoints, save to the same exp_dir, then decode:
```
python examples/asr/decode.py \
  exp_dir=./examples/asr/exp \
  checkpoint.epoch=10 checkpoint.avg=5 \
  data.test_data_config=examples/asr/configs/data_configs/test_data_config.yaml
```

## Load a pretrained model and run ASR
```
from auden.auto.auto_model import AutoModel
import torch

model = AutoModel.from_pretrained("/pretrained/model/dir").eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
```

- Precomputed features (x, x_lens) — make sure your precomputed features align with training:
```
hyps = model.generate(input=(x, x_lens), decoding_method="greedy_search")
```

- WAV paths (List[str]):
```
hyps = model.generate(input=["/abs/a.wav", "/abs/b.wav"], decoding_method="greedy_search")
```

- Mono waveforms (List[Tensor/ndarray]):
```
hyps = model.generate(input=[torch.randn(16000*5), torch.randn(16000*3)], decoding_method="greedy_search")
```
