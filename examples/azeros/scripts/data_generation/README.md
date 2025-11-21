# Genration of Self-generated Instruction(-Free) Tuning data

We provide a `run_data_generation.py` script to generate the same data used in AZeroS.

Refer to `run_data_generation.sh` for demo usage.

Arguments:

- `--manifests`: provide a `yaml` file containing `name` (the dataset name), `manifest` (lhotse manifest path), `lang` (language identifier to choose instruction).
- `--model-name`: the LLM model name, default to `Qwen/Qwen2.5-7B-Instruct`
- `mode`: data generation mode, chosen from `sift_s`, `sift_sp`, `sit_sp`, `sift_ssp`, and `sit_ssp`. *Please refer to the [AZeroS paper]() for detailed description.* Recommend to use `sift_s` and `sift_sp` for speech understanding model.
- `--nshards` & `--shard`: split each dataset into N shards and process the i-th shard only, when dealing with large datasets. Use `lhotse combine xx/*.jsonl.gz xx.jsonl.gz` to combine them.
- `--max-new-tokens`: determine the `max_new_tokens` parameter for `model.generate`. Empirically, `256` can be enough for AZeroS traing. **A `<|truncated|>` token will be added when the response is truncated, and it will be cut off in later training to avoid false ending supervision.**
- `--batch-size`: size of one batch. We use simple padded batch here.
- `--output-dir`: directory to save results.
