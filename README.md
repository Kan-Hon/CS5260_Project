# CS5260_Project

## Global Requirements

## Scripts

### 🧩 Model Merger CLI
---
This script merges a PEFT adapter (e.g., QLoRA) into a base **vision-language model** and saves the result for inference. It uses Hugging Face's `transformers` and `peft` libraries.

Example of model ids: "Xkev/Llama-3.2V-11B-cot", "meta-llama/Llama-3.2-11B-Vision-Instruct"

---
```
cd llava-cot-vlm/vlm-helper-scripts
python merge_model_file.py \
  --base_model_id "<MODEL_ID>" \
  --adapter_path "<filepath>" \
  --output_dir "<filepath>" \
  --hf_token "<token>"
```

### 🧩 Upload HF model
---
This script allows you to upload a model folder to the Hugging Face Hub using the Hugging Face API.

---
```
cd llava-cot-vlm/vlm-helper-scripts
python hugging_face_uploader.py \
  --folder_path "/path/to/your/model/folder" \
  --repo_id "your_username/your_repo_name" \
  --hf_token "your_huggingface_token"
```


### 🧩 Inference
---

---


### 🧩 Evaluation CLI
---
This script is designed to **evaluate a pre-merged vision-language model** using the **VLMEvalKit** evaluation framework. It runs evaluation tasks on various datasets and generates detailed reports, storing both **predictions** and **ground truth**. You can easily run the evaluation task directly from the command line interface (CLI).

---

### 🔧 Ensure these are completed:

- **Pre-merged Model**: Ensure you have already merged a model (e.g., using the `merge_model_file.py` script).
- **API Token**: You will need a Hugging Face token for model access and Open AI token for the judge
- **Python Libraries**: The script relies on libraries such as `torch`, `transformers`, and `evalscope`.

### 📥 Example Model IDs

- **Example of model ids**: `"Llama-3.2-11B-Vision-Instruct"`
- These are **vision-language models** that you can evaluate.

---

### ⚙️ Running the Evaluation from CLI

1. **Navigate to the Script's Directory**:
   ```bash
   cd llava-cot-vlm/vlm-helper-scripts
    ```

2. **Running eval file**
```
python eval_model_file.py \
  --task_config_path <config_name>.json \
  --model_path "<model_path>"
```

