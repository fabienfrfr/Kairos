from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import HfApi
import json


def extract_function_block(system: str):
    if not system:
        return None

    start = system.find("[")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(system)):
        c = system[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                block = system[start:i + 1]
                try:
                    json.loads(block)
                    return block
                except Exception:
                    return None
    return None


def extract_all_user_and_toolcalls(conversations):
    inputs, outputs = [], []
    last_user = None

    for msg in conversations:
        sender = msg.get("from")
        value = msg.get("value", "").strip()

        if sender == "user":
            last_user = msg

        elif sender == "assistant" and value.startswith("["):
            if last_user:
                inputs.append(last_user)
                outputs.append(msg)

    return inputs, outputs


def build_toolace_view():
    raw = load_dataset("Team-ACE/ToolACE", split="train")
    records = []

    for row in raw:
        instruction = extract_function_block(row.get("system", ""))
        inputs, outputs = extract_all_user_and_toolcalls(row.get("conversations", []))

        if not instruction or not inputs or not outputs:
            continue

        records.append({
            "instruction": instruction,
            "input": json.dumps(inputs, ensure_ascii=False),   # JSON STRING
            "output": json.dumps(outputs, ensure_ascii=False), # JSON STRING
            "seed_data": "ToolACE",
        })

    return Dataset.from_list(records)


def build_alpaca_view():
    raw = load_dataset("yahma/alpaca-cleaned", split="train")
    records = []

    for row in raw:
        records.append({
            "instruction": row.get("instruction", ""),
            "input": row.get("input", "") or "",
            "output": row.get("output", "") or "",
            "seed_data": "alpaca-cleaned",
        })

    return Dataset.from_list(records)


def main():
    repo_id = "ffurfaro/simple-tool-instruct"

    toolace_ds = build_toolace_view()
    alpaca_ds = build_alpaca_view()

    combined = concatenate_datasets([toolace_ds, alpaca_ds])

    combined.push_to_hub(repo_id)

    HfApi().upload_file(
        path_or_fileobj="scripts/instruct-tool/readme.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Dataset and README pushed to {repo_id}")


if __name__ == "__main__":
    main()
