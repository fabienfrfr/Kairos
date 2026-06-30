import random
import torch
from torch.utils.data import Dataset
from datasets import get_dataset_config_names, load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset

# ByteToken with stride=3
MAX_LEN = 3*2048

# =========================
# Dataset Full Diffusion
# =========================
class KairosPretrainingDataset(Dataset):
    """
    Full diffusion pretraining dataset.

    Text is tokenized and split into fixed-length chunks.
    All tokens are part of the diffusion process (no prompt/generation split).
    Mask only distinguishes real tokens (1) vs padding (0).

    Key idea:
        Model learns to denoise entire sequences uniformly.

    Example:
        "Hello world" → tokens → padded → mask=[1,1,0,...]
        (all non-pad tokens are diffused)
    """

    def __init__(self, texts, tokenizer, max_len=MAX_LEN, stride=3):
        self.tokenizer = tokenizer
        self.stride = stride
        # self.max_len % stride == 0
        self.target_len = max_len
        self.max_len = (max_len // stride) * stride

        if texts is None:
            configs = get_dataset_config_names("HuggingFaceTB/cosmopedia")
            parts = [
                load_dataset("HuggingFaceTB/cosmopedia", c, split="train[:98.00%]")
                for c in configs
            ]
            self.ds = concatenate_datasets(parts)
        else:
            self.ds = HFDataset.from_dict({"text": texts})

        # batched + remove old columns
        self.ds = self.ds.map(
            self.preprocess,
            batched=True,
            remove_columns=self.ds.column_names,
        )

        self.ds.set_format("torch")

    def preprocess(self, examples):

        all_input_ids = []
        all_masks = []

        # safe fallback if no prompt field
        prompts = examples.get("prompt", [""] * len(examples["text"]))
        texts   = examples.get("text", [""] * len(examples["text"]))

        for prompt, text in zip(prompts, texts):

            # basic anti‑Reversal Curse
            merged = " ".join(
                [prompt, text]
                if random.random() < 0.5
                else [text, prompt]
            ).strip()

            tokens = self.tokenizer.encode(merged, add_special_tokens=False)

            # chunking
            for i in range(0, len(tokens), self.max_len):
                chunk = tokens[i:i + self.max_len]

                pad_len = self.target_len - len(chunk)
                chunk = chunk + [self.tokenizer.pad_token_id] * pad_len

                mask = [1]*(len(chunk) - pad_len) + [0]*pad_len

                all_input_ids.append(chunk)
                all_masks.append(mask)

        return {
            "input_ids": all_input_ids,
            "mask": all_masks,
            "prompt_len": [0] * len(all_input_ids)  # pretraining only
        }

    def __getitem__(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)


# ==============================
# Diffusion Mask instruct & tool
# ==============================
class KairosSFTDataset(Dataset):
    """
    SFT dataset for instruction following and tool calling.
 
    Supports Team-ACE/ToolACE and yahma/alpaca-cleaned formats,
    or custom examples passed directly.
 
    Each sample is a conversation flattened into a single sequence:
        <system>...</system>
        <user>...</user>
        <assistant>...</assistant>
        ...
 
    Prompt tokens (everything except the last assistant turn) are never
    noised (gen_mask=0). The last assistant turn is the diffusion region
    (gen_mask=1).
    """
 
    def __init__(self, tokenizer, max_len=512, examples=None, source="toolace"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._build(examples, source)
 
    def _build(self, examples, source):
        if examples is not None:
            return [self._process(ex) for ex in examples]
 
        if source == "toolace":
            ds = load_dataset("Team-ACE/ToolACE", split="train")
            return [self._process(ex) for ex in ds]
 
        if source == "alpaca":
            ds = load_dataset("yahma/alpaca-cleaned", split="train")
            return [self._process_alpaca(ex) for ex in ds]
 
        raise ValueError(f"Unknown source: {source}")
 
    def _process(self, ex):
        """
        Handle ToolACE format: {system: str, conversations: [{from, value}]}
        or custom format: {system, conversations} with same structure.
        """
        system = ex.get("system", "")
        turns = ex.get("conversations", [])
 
        # Build the flat text, tracking where the last assistant turn starts
        parts = [f"<system>\n{system}\n</system>\n"] if system else []
 
        last_assistant_start = None
 
        for turn in turns:
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", ""))
 
            if role in ("user", "human"):
                parts.append(f"<user>\n{value}\n</user>\n")
            elif role in ("assistant", "gpt"):
                #last_assistant_start = sum(len(p) for p in parts)
                prefix_text = "".join(parts)
                prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                last_assistant_start = len(prefix_ids)
                parts.append(f"<assistant>\n{value}\n</assistant>\n")
            elif role == "tool":
                parts.append(f"<tool_result>\n{value}\n</tool_result>\n")
 
        full_text = "".join(parts)
 
        # Tokenize full text
        all_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
 
        # Find the last assistant turn boundary in token space
        if last_assistant_start is not None:
            prompt_len = last_assistant_start
        else:
            prompt_len = len(all_ids)

 
        # Truncate
        all_ids = all_ids[:self.max_len]
        prompt_len = min(prompt_len, len(all_ids))
 
        # Pad
        pad_len = self.max_len - len(all_ids)
        gen_len = len(all_ids) - prompt_len
        all_ids += [self.tokenizer.pad_token_id] * pad_len
 
        gen_mask = [0] * prompt_len + [1] * gen_len + [0] * pad_len
 
        return {
            "input_ids":  all_ids,
            "gen_mask":   gen_mask,
            "prompt_len": prompt_len,
        }
 
    def _process_alpaca(self, ex):
        """
        Handle alpaca-cleaned format: {instruction, input, output}
        """
        user = ex["instruction"]
        if ex.get("input", "").strip():
            user += f"\n\n{ex['input']}"
 
        conversations = [
            {"from": "user",      "value": user},
            {"from": "assistant", "value": ex["output"]},
        ]
        return self._process({"system": "", "conversations": conversations})
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "input_ids":  torch.tensor(s["input_ids"],  dtype=torch.long),
            "gen_mask":   torch.tensor(s["gen_mask"],   dtype=torch.long),
            "prompt_len": torch.tensor(s["prompt_len"], dtype=torch.long),
        }


# ==============================
# Diffusion Dataset Human Pref
# ==============================
class KairosDPODataset(Dataset):
    """
    DPO dataset for preference alignment.

    Supports argilla/ultrafeedback-binarized-preferences-cleaned format,
    or custom examples passed directly.

    Each sample contains a prompt tokenized as the fixed context (gen_mask=0),
    plus a chosen and a rejected response (gen_mask=1) tokenized separately.
    The trainer computes the DPO loss from (prompt, chosen, rejected).
    """

    def __init__(self, tokenizer, max_len=512, examples=None):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.data = self._build(examples)

    def _build(self, examples):
        if examples is not None:
            return [self._process(ex) for ex in examples]
        ds = load_dataset(
            "argilla/ultrafeedback-binarized-preferences-cleaned",
            split="train"
        )
        return [self._process(ex) for ex in ds]

    def _render_messages(self, messages):
        """Convert [{role, content}] to flat string."""
        return "".join(
            f"<{m['role']}>\n{m['content']}\n</{m['role']}>\n"
            for m in messages
        )

    def _encode_pair(self, prompt_text, response_text):
        """
        Tokenize prompt + response as one sequence.
        prompt_len is derived by encoding prompt alone (safe for ByT5:
        byte-level tokenizer, encode(A)+encode(B) == encode(A+B)).
        """
        prompt_ids   = self.tokenizer.encode(prompt_text,   add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        # Truncate response if needed, never truncate the prompt
        response_ids = response_ids[:self.max_len - len(prompt_ids)]
        prompt_ids   = prompt_ids[:self.max_len]

        all_ids  = prompt_ids + response_ids
        pad_len  = self.max_len - len(all_ids)
        all_ids += [self.tokenizer.pad_token_id] * pad_len

        gen_mask = [0] * len(prompt_ids) + [1] * len(response_ids) + [0] * pad_len

        return all_ids, gen_mask, len(prompt_ids)

    def _process(self, ex):
        prompt   = ex["prompt"]
        chosen   = ex.get("chosen",   [])
        rejected = ex.get("rejected", [])

        # prompt is just the user turn; chosen/rejected are full message lists
        # drop the last assistant turn from chosen/rejected to get the shared prefix
        prompt_text    = f"<user>\n{prompt}\n</user>\n<assistant>\n"
        chosen_text    = self._render_messages(chosen[-1:])   # last assistant turn
        rejected_text  = self._render_messages(rejected[-1:])

        chosen_ids,   chosen_mask,   plen = self._encode_pair(prompt_text, chosen_text)
        rejected_ids, rejected_mask, _    = self._encode_pair(prompt_text, rejected_text)

        return {
            "chosen_ids":    chosen_ids,
            "chosen_mask":   chosen_mask,
            "rejected_ids":  rejected_ids,
            "rejected_mask": rejected_mask,
            "prompt_len":    plen,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "chosen_ids":    torch.tensor(s["chosen_ids"],    dtype=torch.long),
            "chosen_mask":   torch.tensor(s["chosen_mask"],   dtype=torch.long),
            "rejected_ids":  torch.tensor(s["rejected_ids"],  dtype=torch.long),
            "rejected_mask": torch.tensor(s["rejected_mask"], dtype=torch.long),
            "prompt_len":    torch.tensor(s["prompt_len"],    dtype=torch.long),
        }


# =========================
# Diffusion Mask Reasonning
# =========================
class KairosRLDataset(Dataset):
    """
    RL dataset for reasoning fine-tuning via masked diffusion.
    Prompt tokens are never noised (gen_mask=0).
    Generation tokens are noised by the trainer (gen_mask=1).
    """
 
    def __init__(self, tokenizer, max_len=2048, split="train", max_samples=None, examples=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._build(examples, split, max_samples)
 
    def _build(self, examples, split, max_samples):
        if examples is not None:
            return [self._process(ex) for ex in examples]
 
        configs = get_dataset_config_names("ffurfaro/bigbench")
        raw, count = [], 0
        for name in configs:
            for ex in load_dataset("ffurfaro/bigbench", name, split=split, streaming=True):
                if "multiple_choice_targets" not in ex:
                    continue
                raw.append(self._process(ex))
                count += 1
                if max_samples and count >= max_samples:
                    return raw
        return raw
 
    def _process(self, ex):
        question = ex["inputs"]
        choices  = list(ex["multiple_choice_targets"])
        scores   = list(ex.get("multiple_choice_scores", [0] * len(choices)))
        reasoning = ex.get("reasoning", "")
 
        # add uncertainty option
        choices.append("not sure / I don't know")
        scores.append(0.1)
 
        # shuffle choices (anti-position bias)
        paired = list(zip(choices, scores))
        random.shuffle(paired)
        choices, scores = zip(*paired)
 
        best = choices[int(torch.tensor(list(scores)[:-1]).argmax())]
 
        level = random.choice(["low", "medium", "flex"])
        mask_ratio = {"low": 0.25, "medium": 0.5, "flex": random.uniform(0.1, 0.9)}[level]
 
        choice_lines = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(choices))
        prompt = f"<inputs>\n{question}\n<choices>\n{choice_lines}\n"
 
        # anti-Reversal Curse: randomize order of reasoning and answer blocks
        gen_blocks = [f"<reasoning={level}>\n{reasoning}", f"<answer>\n{best}"]
        random.shuffle(gen_blocks)
        generation = "\n".join(gen_blocks) + "\n"
 
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        gen_ids    = self.tokenizer.encode(generation, add_special_tokens=False)
        gen_ids    = gen_ids[:self.max_len - len(prompt_ids)]
 
        ids     = prompt_ids + gen_ids
        pad_len = self.max_len - len(ids)
        ids    += [self.tokenizer.pad_token_id] * pad_len
 
        gen_mask = [0] * len(prompt_ids) + [1] * len(gen_ids) + [0] * pad_len
 
        return {
            "input_ids":  ids,
            "gen_mask":   gen_mask,
            "prompt_len": len(prompt_ids),
            "mask_ratio": mask_ratio,
            "choices":    list(choices),
            "scores":     list(scores),
            "level":      level,
        }
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "input_ids":  torch.tensor(s["input_ids"],  dtype=torch.long),
            "gen_mask":   torch.tensor(s["gen_mask"],   dtype=torch.long),
            "prompt_len": torch.tensor(s["prompt_len"], dtype=torch.long),
            "mask_ratio": torch.tensor(s["mask_ratio"], dtype=torch.float),
            "choices":    s["choices"],
            "scores":     s["scores"],
            "level":      s["level"],
        }
 