# AZeroS Example

This example demonstrates training and evaluating an AZeroS model.

It follows the same layout as other examples and integrates with Auden's Auto* APIs and BaseTrainer.

## Data Configs

Edit YAMLs under `configs/` to point to your Lhotse CutSet `jsonl.gz` manifests.
See `data_module.py` docstring for a minimal supervision schema with optional translation.

## Training

Multi-GPU (4 GPUs):
```bash
torchrun --nproc_per_node=4 train.py \
  exp_dir=your/exp/dir \
  model.model_type=azeros \
  model.encoder.model_type=zipformer \
  data.train_data_config=configs/train_data_config.yaml \
  data.valid_data_config=configs/valid_data_config.yaml
```

## Evaluation (batch decoding)

Demo prompts (greedy):
```bash
python evaluate.py \
  exp_dir=your/exp/dir \
  checkpoint.filename=your_model_file_name.pt \
  data.test_data_config=configs/test_data_config.yaml
```

## 🔥 Release of AZeroS
### Extended from Qwen2.5-7B-Instruct
### Training Data: WenetSpeech + GigaSpeech + Public Age/Gender/Emotion Data
<!-- <p>
  <img src="assets/data.png" width="60%" />
</p> -->

### 📊 Performance: AIRBench

**Highlights:**


### 📊 Performance: VoiceBench

**Highlights:**

---
<!-- 
### 🎧 Cross-Lingual Speech Retrieval

TTA shows significantly higher **retrieval accuracy** than Whisper Large-v2, especially among Indo-European languages — validating its **language-agnostic semantic alignment** capability.

<p>
  <img src="assets/retrieval.png" width="60%" />
  
  
  
</p>

---

### Usage

```python
from auden.auto.auto_model import AutoModel

# 1) Load a model checkpoint directory (contains config.json + weights)
model_dir = "AudenAI/auden-tta-m10"  # or any exported directory / HF repo id
model = AutoModel.from_pretrained(model_dir)
model = model.to("cuda")
model.eval()

# 2) Prepare input features (x, x_lens). If you have raw audio, you can use
#    model.speech_encoder.extract_feature(wav) to get (x, x_lens).
x, x_lens = ...  # Tensor shapes: (B, T, F), (B,)

inputs = (x, x_lens)
# Alternatively, you can pass WAV inputs directly:
# - List of WAV paths (str):
#   inputs = ["/abs/a.wav", "/abs/b.wav"]
# - List of mono waveforms (Tensor/ndarray), 16 kHz:
#   inputs = [torch.randn(16000*5), torch.randn(16000*3)]

# 3a) Transcribe (RNNT greedy)
out = model.generate(inputs, task="transcribe", blank_penalty=0.0, return_timestamps=False)
print(out["hypotheses"])  # list[str]

# 3b) Translate (attention beam search). Language can be a single str or a list[str] per utterance
out = model.generate(
    inputs,
    task="translate",
    beam_size=5,
    source_language=["zh"] * x.size(0),
    target_language=["en"] * x.size(0),
)
print(out["hypotheses"])      # list[str]
print(out["source_language"]) # list[str], model-predicted or provided
print(out["target_language"]) # list[str], model-predicted or provided

# 3c) Align (audio-text similarity)
texts = ["hello world", "good morning"]
out = model.generate(inputs, task="align", texts=texts)
print(out["similarities"])  # (B, len(texts))
print(out["audio_emb"]) # (B, emb_dim)
print(out["text_emb"]) # (B, emb_dim)

```

## 🧩 TTA Encoder 
### ASR-LLM Encoder Evaluation

| Encoder | Aishell CER↓ | LibriSpeech WER↓ |
|----------|---------------|------------------|
| Whisper-Medium | 5.47 | 4.66 |
| Whisper-Large | 4.87 | 3.64 |
| ZT-AED | 2.92 | 2.30 |
| **TTA (Ours)** | **1.92** | **1.95** |

TTA’s encoder achieves **state-of-the-art semantic representation**, enabling seamless LLM integration with high recognition accuracy.

### Usage
```python
from auden.auto.auto_model import AutoModel
encoder = AutoModel.from_pretrained("AudenAI/auden-encoder-tta-m10")
encoder = encoder.to("cuda")

# 2) Prepare input features (x, x_lens). If you have raw audio, you can use
#    encoder.extract_feature(wav) to get (x, x_lens).
x, x_lens = ...  # Tensor shapes: (B, T, F), (B,)

encoder_output = encoder(x, x_lens)
print(encoder_output.encoder_out) # (B, T//4, D)
print(encoder_output.encoder_out_lens) # (B)

```
 -->

