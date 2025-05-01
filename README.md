# CS5260_Project

## Global Requirements

## Scripts

### 🧩 Model Merger CLI
---
This script merges a PEFT adapter (e.g., QLoRA) into a base **vision-language model** and saves the result for inference. It uses Hugging Face's `transformers` and `peft` libraries.

Example of model ids: "Xkev/Llama-3.2V-11B-cot", "meta-llama/Llama-3.2-11B-Vision-Instruct"

---
```
cd vlm-helper-scripts
python merge_model_file.py \
  --base_model_id "<MODEL_ID>" \
  --adapter_path "<filepath>" \
  --output_dir "<filepath>" \
  --hf_token "<token>"
```