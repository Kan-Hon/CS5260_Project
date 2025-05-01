# CS5260 Final Project

This repository contains the code, experiments, and models developed for our CS5260 final project, which investigates multimodal Chain-of-Thought (CoT) reasoning in Vision–Language Models (VLMs). The project is split into two parts:

## Part I: Multimodal CoT using Small Vision Language Models

### Link to repository
[MM-COT](https://github.com/Kan-Hon/CS5260_Project/tree/main/mm-cot)

- Model architecture (encoder–decoder vs. decoder-only)
- Two-stage (rationale → answer) vs. direct answer generation
- Parameter-efficient fine-tuning (LoRA) vs. full fine-tuning
- Expt 7 and 8 for MMStar Evaluation

[MM-STAR](https://github.com/Kan-Hon/CS5260_Project/tree/main/mmstar)
- Expt 1 and 2 for MMStar Evaluation

## Part II: Llama-CoT – Lean CoT Fine-Tuning of LLaMA-3.2-11B-Vision

### Link to repository
[Llama-CoT-Train](https://github.com/Kan-Hon/CS5260_Project/tree/main/Llama-CoT)
- Training 2-stage Llama CoT


[Llama-CoT-Evaluation](https://github.com/Kan-Hon/CS5260_Project/tree/main/llava-cot-vlm-inference-evaluation)
- Evaluating Llama-CoT on MMStar
- Merging LoRA adapters to model
- Gradio app for inference