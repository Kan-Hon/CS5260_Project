import os
import torch
import argparse
from peft import PeftModel
from huggingface_hub import login
from transformers import AutoModelForVision2Seq, AutoProcessor


class ModelMerger:
    def __init__(self,
                 base_model_id,
                 adapter_path,
                 output_dir,
                 hf_token=None,
                 dtype=torch.float16,
                 device_map="auto"):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.output_dir = output_dir
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.dtype = dtype
        self.device_map = device_map

        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
            login(self.hf_token)

    def load_base_model(self):
        print(f"Loading base model: {self.base_model_id}")
        self.base_model = AutoModelForVision2Seq.from_pretrained(
            self.base_model_id,
            torch_dtype=self.dtype,
            device_map=self.device_map
        )

    def load_adapter(self):
        print(f"Loading adapter from: {self.adapter_path}")
        self.peft_model = PeftModel.from_pretrained(self.base_model, self.adapter_path)

    def merge_and_save(self):
        print("Merging adapter weights...")
        merged_model = self.peft_model.merge_and_unload()
        print("Merge complete.")

        print(f"Saving merged model to: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        merged_model.save_pretrained(self.output_dir)

        print(f"Saving processor/tokenizer to: {self.output_dir}")
        try:
            processor = AutoProcessor.from_pretrained(self.base_model_id)
            processor.save_pretrained(self.output_dir)
        except Exception as e:
            print(f"Warning: Processor not saved automatically - {e}")

        print("All components saved successfully.")

    def run(self):
        self.load_base_model()
        self.load_adapter()
        self.merge_and_save()


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model and save to disk.")
    parser.add_argument("--base_model_id", type=str, required=True, help="Hugging Face ID or path to the base model.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the adapter directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN env variable).")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"],
                        help="Torch dtype for loading the model.")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for model loading (e.g. 'auto').")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    merger = ModelMerger(
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        dtype=dtype_map[args.dtype],
        device_map=args.device_map
    )
    merger.run()
