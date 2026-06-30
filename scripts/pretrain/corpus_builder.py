import json
from tqdm import tqdm
from datasets import load_dataset

def process_datasets(output_file: str):
    """Process OPUS and Cosmopedia datasets and save to a single JSONL file."""
    
    # Load both datasets
    opus = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")
    cosmo = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")

    with open(output_file, "w", encoding="utf-8") as f:
        # Process OPUS Books
        for row in tqdm(opus, desc="Processing OPUS", unit="row"):
            entry = {
                "prompt": row["translation"]["en"],
                "text": row["translation"]["fr"],
                "seed_data": "opus_books_en-fr"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Process Cosmopedia
        for row in tqdm(cosmo, desc="Processing Cosmopedia", unit="row"):
            entry = {
                "prompt": row["prompt"],
                "text": row["text"],
                "seed_data": "cosmopedia"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process_datasets("combined_dataset.jsonl")