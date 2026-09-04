# Model Backends

This folder now contains fresh end-to-end backend implementations:

- `dp_switch_transformer.py`: full DP-MoE Switch Transformer training pipeline.
- `dp_olmoe.py`: full DP-MoE OLMoE training pipeline.
- `dp_mistral.py`: fresh Mixtral variant derived from the OLMoE reference pipeline.
- `dp_qwen.py`: fresh Qwen-MoE variant derived from the OLMoE reference pipeline.
- `dp_deepseek.py`: DeepSeek-VL2 VLM backend with inference/eval and full DP-MoE record-routing finetuning methods.

Notes:
- No file in this folder is a scaffold placeholder anymore.
- Mistral/Qwen defaults are pre-configured for their model families.
- DeepSeek backend is VLM-oriented (image+text) and now supports finetuning methods:
  `upper_bound`, `upper_bound_no_lora`, `naive_dp`, `naive_dp_no_lora`, `ours`, `all`.
