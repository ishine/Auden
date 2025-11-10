# AZeroS Example

This example demonstrates training and evaluating an AZeroS model.

It follows the same layout as other examples and integrates with Auden's Auto* APIs and BaseTrainer.

## Data Configs
Please refer to `configs/README.md` for data preparation and formats.

Edit YAMLs under `configs/` to point to your Lhotse CutSet `jsonl.gz` manifests.
See `data_module.py` docstring for a minimal supervision schema with optional translation.

## Training
We adopt a 2-stage training procedure as follows.

- `scripts/train_stage1.sh`: Training a semantic projector.
- `scripts/train_stage2.sh`: Training a paralinguistic projector with the pretrained semantic branch.

## Evaluation


## Inference Guide
Run `model.generate` for batch decoding.

Demo on ASR (greedy):
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

### 📊 Performance: VoiceBench

**Highlights:**


### 📊 Performance: AIRBench

**Highlights:**

---
