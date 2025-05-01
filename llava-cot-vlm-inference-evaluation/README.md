# CS5260_Project

## Global Requirements
---
This guide explains how to set up a dedicated Python environment for this project using the provided `requirements.txt` file. Using a virtual environment is highly recommended to avoid conflicts with other Python projects or your system's Python installation.

---

**Steps:**

1.  **Navigate to Project Directory:**
    Open your terminal or command prompt and navigate to the directory where you have saved the project files (including `requirements.txt`, `inference_metrics.py`, etc.).
    ```bash
    cd /path/to/your/project_directory
    ```

2.  **Create a Virtual Environment:**
    Create a new virtual environment within your project directory. We'll name it `venv` here, but you can choose another name.
    ```bash
    python -m venv venv
    # Or if the above doesn't work, try:
    # python3 -m venv venv
    ```
    This will create a `venv` folder in your project directory containing a copy of the Python interpreter and `pip`.

3.  **Activate the Virtual Environment:**
    Before installing packages, you need to activate the environment. The activation command differs based on your operating system and shell:

    *   **On Linux or macOS (bash/zsh):**
        ```bash
        source venv/bin/activate
        ```

    *   **On Windows (Command Prompt - CMD):**
        ```bash
        .\venv\Scripts\activate.bat
        ```

    *   **On Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
        *(Note: If you get an execution policy error in PowerShell, you might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in that PowerShell window first, then try activating again).*

    Once activated, your terminal prompt should change to indicate the active environment, usually by prefixing it with `(venv)`.

4.  **Install Dependencies:**
    With the virtual environment active, use `pip` to install all the packages listed in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This might take some time depending on the number of packages and your internet speed.



## Trained Model

There are 4 main models uploaded as per experiment conducted:
- mlimb/qlora-model-base01_32
- mlimb/qlora-model-base02_16
- mlimb/qLoRA-LLama-CoT-ft01-32
- mlimb/qlora-model-ft02_16

1. Download from hugging face
```
https://huggingface.co/mlimb/<model_name>/tree/main
https://huggingface.co/mlimb/qLoRA-LLama-CoT-ft01-32/tree/main
```
2. Input this model id in the inference script/ download the model locally
```
mlimb/<model_name>
mlimb/qLoRA-LLama-CoT-ft01-32
```

## Evaluation/Inference Scripts

### 🧩 Model Merger CLI
---
This script merges a PEFT adapter (e.g., QLoRA) into a base **vision-language model** and saves the result for inference. It uses Hugging Face's `transformers` and `peft` libraries.

Example of model ids: "Xkev/Llama-3.2V-11B-cot", "meta-llama/Llama-3.2-11B-Vision-Instruct"

---
```bash
cd llava-cot-vlm/vlm-helper-scripts
python merge_model_file.py \
  --base_model_id "<MODEL_ID>" \
  --adapter_path "<filepath>" \
  --output_dir "<filepath>" \
  --hf_token "<token>"
```

The evaluation results on MMStar for the different models can be accessed at `https://drive.google.com/drive/folders/1J6XPMpyiOEVBpae2cQosJjGatv602xaM?usp=sharing`


### 🧩 Upload HF model
---
This script allows you to upload a model folder to the Hugging Face Hub using the Hugging Face API.

---
```bash
cd llava-cot-vlm/vlm-helper-scripts
python hugging_face_uploader.py \
  --folder_path "/path/to/your/model/folder" \
  --repo_id "your_username/your_repo_name" \
  --hf_token "your_huggingface_token"
```


### 🧩 Inference
---
1.  `inference_metrics.py`: Runs inference for a single image and prompt, measures performance metrics (model size, FLOPs estimation, inference time), and outputs the generated text.
2.  `gradio_app.py`: Launches an interactive web demo using Gradio, allowing users to upload images, enter prompts, and get responses from the LLaVA-CoT model.
---

### 1. Running Inference and Metrics (`inference_metrics.py`)
This script performs a single inference run and gathers performance metrics.

---
**Command:**

```bash
    cd llava-cot-vlm/vlm-helper-scripts
    python inference_metrics.py \
        --model_path <path/to/your/model_directory> \
        --image_path <path/to/your/input_image.jpg> \
        --prompt "<Your text prompt here>" \
```
Optional Arguments:
- cot_type <type>: The CoT strategy to use. Choices: stage, sentence, best_of_N. Default: stage.
- beam_size <int>: Beam size for the selected CoT strategy. Default: 2.
- device <device>: Compute device to use. Choices: cuda, cpu. Default: cuda (falls back to cpu if CUDA is unavailable).
- dtype <dtype>: Data type for model parameters. Choices: bfloat16, float16, float32. Default: bfloat16 (falls back to float32 if not supported).
- num_runs <int>: Number of times to execute the inference for timing measurements. Default: 1.
- warmup_runs <int>: Number of initial inference runs to perform before starting timing measurements (to warm up GPU/cache). Default: 0.
- skip_flops: If present, disables the estimation of FLOPs (useful if it causes errors or is slow). No value needed after the flag.

### 2. Gradio Application (`gradio_app.py`)
This script runs the gradio interface to use the model

---
```bash
cd llava-cot-vlm/vlm-helper-scripts
python gradio_app.py \
    --model_path "/<path/to/your/model_directory>" \
    --device cuda \
    --dtype bfloat16 \
    --server_port 7861 \
    --share
```

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
```bash
python eval_model_file.py \
  --task_config_path <config_name>.json \
  --model_path "<model_path>"
```

