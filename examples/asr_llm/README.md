# ASR‑LLM (Audio→Text) Example

This example shows how to train, evaluate, and run inference for audio‑to‑text generation using Hydra configs. The model couples an audio encoder (Zipformer or Whisper), a projector, and a Hugging Face causal LLM (e.g., Qwen2).

## Data configs

See detailed data preparation and manifest format in [data_module.py](data_module.py).

## Training

Single GPU:
```
python train.py \
  exp_dir=./exp/your_exp_dir
```

Multi‑GPU DDP (8 GPUs):
```
torchrun --nproc_per_node=8 train.py \
  exp_dir=./exp/your_exp_dir
```

Notes:
- model.audio_encoder: model_type (zipformer/whisper) or a path/HF repo with config.json (for Whisper).
- model.llm: HF LLM name or path (e.g., qwen2 or a local Qwen2.5‑7B‑Instruct).
- You can optionally load/freeze pretrained encoders; see comments in configs/train.yaml.
- Special token: The audio token (default `<|AUDIO|>`) is added to the tokenizer; the projector maps encoder frames to LLM hidden size and replaces embeddings at audio token positions.

Examples:
```
# Train Zipformer + Qwen2
torchrun --nproc_per_node=8 train.py \
  exp_dir=./exp/zipformer_qwen2 \
  model.audio_encoder.model_type=zipformer \
  model.audio_encoder.pretrained_model=/abs/path/to/zipformer_based_model \
  model.llm.model_type=qwen2 \
  model.llm.pretrained_model=/abs/path/to/Qwen2.5-7B-Instruct \
  data.train_data_config=configs/aishell2/train_data_config.yaml \
  data.valid_data_config=configs/aishell2/valid_data_config.yaml

# Train Whisper‑encoder + Qwen2 with Whisper FBanks and 30s padding
torchrun --nproc_per_node=8 train.py \
  exp_dir=./exp/whisper_qwen2 \
  model.audio_encoder.model_type=whisper \
  model.audio_encoder.pretrained_model=/abs/path/to/whisper_hf \
  model.llm.model_type=qwen2 \
  model.llm.pretrained_model=/abs/path/to/Qwen2.5-7B-Instruct \
  data.whisper_fbank=true \
  data.pad_to_30s=true
```

## Evaluation on test data

Evaluate a pretrained/exported model checkpoint:
```
python evaluate.py \
  exp_dir=./exp/your_exp_dir \
  checkpoint.model.filename=/abs/path/to/model.pt \
  data.test_data_config=examples/asr_llm/configs/test_data_config.yaml \
  decoding_method=greedy_search
```

Average from trainer checkpoints, save to the same exp_dir, then evaluate:
```
python evaluate.py \
  exp_dir=/exp/your_exp_dir \
  checkpoint.model.iter=100000 checkpoint.model.avg=5 \
  data.test_data_config=./configs/test_data_config.yaml
```

Checkpoint resolution policy (evaluate.py):
- If checkpoint.model.filename is set, use that file (absolute or under exp_dir)
- Else if iter>0 and avg>0, create/use `averaged-iter-{iter}-avg-{avg}.pt`
- Else if epoch>0 and avg>0, create/use `averaged-epoch-{epoch}-avg-{avg}.pt`
- Else fallback to `exp_dir/pretrained.pt`

## Inference with model.generate

Use the Audio‑LLM model’s `generate` API for batched decoding. Input can be precomputed features `(x, x_lens)` or raw inputs that your encoder supports via `audio_encoder.extract_feature`.

```python
from auden.auto.auto_model import AutoModel
import torch

model = AutoModel.from_pretrained("/pretrained/audio_llm/dir").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Given features and lengths
messages = [[{"role": "user", "content": f"{model.config.audio_token} 请重复以上内容"}]] * B
texts = model.generate((features, lengths), messages, max_new_tokens=200)
```

## Tips
- For Whisper, set `data.whisper_fbank=true` and `data.pad_to_30s=true`.
- The model enforces left padding during generation; training can use right padding by default.

## Prompts (prompt.yaml)

During training we use short text prompts to steer the LLM output format. The trainer reads a prompt file and, for each sample, randomly picks one prompt to concatenate after the audio token.

- Configure the prompt file path with `prompt_file` (either in the YAML or via CLI):
```
python train.py \
  exp_dir=./exp/your_exp_dir \
  prompt_file=examples/asr_llm/configs/prompt.yaml \
  ...
```

- File format: plain text, one prompt per line (the extension can be .yaml or .txt). Example `examples/asr_llm/configs/prompt.yaml`:
```
Transcribe the audio in Chinese.
Repeat the audio content verbatim.
```

- How it’s used internally (simplified):
  - The trainer builds messages like:
    - user: `<|AUDIO|> <random_prompt_from_file>`
    - assistant: `<reference_text>`
  - At inference, you can craft your own messages; see the generate() example above.
