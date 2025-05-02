"# CS5260_Project" 



## PART II Effective Fine-Tuning for VQA

Training Llama-CoT: For qLoRA fine-tuning with Llama-3.2-11B-Vision-Instruct as the base model, please follow the following steps:
- Make a request to access the gated GitHub repo on Hugging Face (https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- Once your request is approved, `huggingface-cli login` to login with your access tokens
- Move to `./LLaVA-CoT` and pip install 'requirements.txt' in your environment
- Follow instructions in `https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k` to download the train datasets
- Modify the paths on lines 56-57 in `.\LLaVA-CoT\train\datasets\cot_dataset.py` to where your train datasets are stored
- Run the following in `./LLaVA-CoT/train` to replicate qLoRA r=32 Llama-CoT (we used a single node of two A40 with 48 GB VRAM)
```
torchrun --nnodes 1 --nproc_per_node 2 --master_port 29500 finetuning.py \
  --enable_fsdp True \
  --use_peft True \
  --peft_method "lora" \
  --lora_config.r 32 \
  --lora_config.lora_alpha 64 \
  --lora_config.lora_dropout 0.05 \
  --lora_config.target_modules ["q_proj","k_proj","v_proj","o_proj"] \
  --lora_config.inference_mode False \
  --freeze_layers False \
  --fsdp_config.fsdp_cpu_offload False \
  --use_wandb True \
  --lr 1e-5  \
  --num_epochs 2 \
  --batch_size_training 2 \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --output_dir ./base02_3e \
  --use_fast_kernels True \
  --dataset "custom_dataset" \
  --custom_dataset.test_split "test" \
  --custom_dataset.file "datasets/cot_dataset.py"  \
  --run_validation False \
  --batching_strategy padding
  ```
