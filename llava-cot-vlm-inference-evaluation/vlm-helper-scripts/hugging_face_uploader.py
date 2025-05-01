import os
import argparse
from huggingface_hub import login, HfApi

class HuggingFaceUploader:
    def __init__(self, hf_token, folder_path, repo_id):
        self.hf_token = hf_token
        self.folder_path = folder_path
        self.repo_id = repo_id

        # Set the Hugging Face API token and login
        os.environ["HF_TOKEN"] = self.hf_token
        login(self.hf_token)

        # Initialize the Hugging Face API
        self.api = HfApi(token=self.hf_token)

    def upload_model(self):
        # Upload the model folder to the specified repository
        print(f"Uploading model from {self.folder_path} to {self.repo_id}...")
        try:
            self.api.upload_folder(
                folder_path=self.folder_path,
                repo_id=self.repo_id,
                repo_type="model"
            )
            print(f"Model uploaded successfully to {self.repo_id}")
        except Exception as e:
            print(f"An error occurred during upload: {e}")

    @classmethod
    def from_args(cls, args):
        return cls(args.hf_token, args.folder_path, args.repo_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a model folder to the Hugging Face Hub")
    parser.add_argument(
        "--folder_path", 
        type=str, 
        required=True, 
        help="Path to the model folder that you want to upload"
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True, 
        help="Repository ID where the model will be uploaded (e.g., <username>/<repo_name>)"
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        required=True, 
        help="Your Hugging Face API token"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    uploader = HuggingFaceUploader.from_args(args)
    uploader.upload_model()


if __name__ == "__main__":
    main()
