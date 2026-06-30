import json
from datasets import Dataset
from huggingface_hub import HfApi

def build_and_push(files: list[str], repo_id: str):
    """Concatenate JSONL files and push to HF Hub."""
    all_rows = []
    
    # Simple merge of all entries
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                all_rows.append(json.loads(line))
    
    # Create dataset
    dataset = Dataset.from_list(all_rows)
    
    # 1. Push dataset
    dataset.push_to_hub(repo_id)
    
    # 2. Upload README without local modification
    HfApi().upload_file(
        path_or_fileobj="scripts/pretrain/readme.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"Dataset and README pushed to {repo_id}")

if __name__ == "__main__":
    files_to_merge = ["simple-wiki.jsonl", "combined_dataset.jsonl"]
    build_and_push(files_to_merge, "ffurfaro/keep-it-simple")