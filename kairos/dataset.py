import array
import io
import json
import random
import warnings

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from torch.utils.data import Dataset

from .tokenizer import KairosTokenizer, Modality, MultimodalSegment

MAX_LEN = 3 * 2048


class NonFiniteDataError(ValueError):
    """Raised when a multimodal example contains NaN/Inf — a data-quality issue to."""


def pack_multimodal_data(arrays: dict) -> bytes:
    """Serialize named numpy arrays into one self-describing blob — shape/dtype travel with."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def unpack_multimodal_data(data: bytes) -> dict:
    """Inverse of pack_multimodal_data."""
    with np.load(io.BytesIO(data)) as npz:
        return {k: npz[k] for k in npz.files}


def _pad_and_gen_mask(ids, prompt_len, max_len, pad_token_id):
    """Pad `ids` to max_len and build a gen_mask that's 0 on the."""
    pad_len = max_len - len(ids)
    gen_len = len(ids) - prompt_len
    ids = ids + [pad_token_id] * pad_len
    gen_mask = [0] * prompt_len + [1] * gen_len + [0] * pad_len
    return ids, gen_mask


class KairosPretrainingDataset(Dataset):
    """Full diffusion pretraining dataset: text or multimodal, chunked to token ids."""

    def __init__(
        self,
        texts=None,
        tokenizer=None,
        max_len=MAX_LEN,
        stride=3,
        multimodal_examples=None,
        multimodal_path=None,
        pack=False,
    ):
        self.tokenizer = tokenizer
        self.stride = stride
        self.target_len = max_len
        self.max_len = (max_len // stride) * stride
        self.pack = pack

        if multimodal_examples is not None or multimodal_path is not None:
            if multimodal_examples is None:
                multimodal_examples = torch.load(multimodal_path, weights_only=False)
            self._build_multimodal(multimodal_examples)
            return

        if texts is None:
            configs = get_dataset_config_names("HuggingFaceTB/cosmopedia")
            parts = [load_dataset("HuggingFaceTB/cosmopedia", c, split="train[:98.00%]") for c in configs]
            self.ds = concatenate_datasets(parts)
        else:
            self.ds = HFDataset.from_dict({"text": texts})

        self.ds = self.ds.map(self.preprocess, batched=True, remove_columns=self.ds.column_names)
        self.ds.set_format("torch")

    def _chunk(self, token_ids, modality_ids, family_ids):
        """Fixed-length windowing shared by text and multimodal, padded to self.target_len."""
        for i in range(0, len(token_ids), self.max_len):
            ids_chunk = token_ids[i : i + self.max_len]
            mod_chunk = modality_ids[i : i + self.max_len]
            fam_chunk = list(family_ids[i : i + self.max_len])
            pad_len = self.target_len - len(ids_chunk)
            ids_chunk = ids_chunk + [self.tokenizer.pad_token_id] * pad_len
            mod_chunk = mod_chunk + [int(Modality.TEXT)] * pad_len
            fam_chunk = fam_chunk + [0] * pad_len
            mask = [1] * (len(ids_chunk) - pad_len) + [0] * pad_len
            yield ids_chunk, mod_chunk, fam_chunk, mask

    def _collect_chunks(self, chunk_sources):
        """Run each (ids, modality_ids, family_ids) triple through self._chunk and flatten."""
        all_input_ids, all_modality_ids, all_family_ids, all_masks = [], [], [], []
        if self.pack:
            packed_ids, packed_mods = [], []
            packed_fams = array.array("B")  # uint8 — family ids are 0..255, saves ~27x vs Python ints
            for ids, mods, fams in chunk_sources:
                packed_ids += ids
                packed_mods += mods
                packed_fams.extend(fams)
            chunk_sources = [(packed_ids, packed_mods, packed_fams)]
        for ids, mods, fams in chunk_sources:
            for ids_chunk, mod_chunk, fam_chunk, mask in self._chunk(ids, mods, fams):
                all_input_ids.append(ids_chunk)
                all_modality_ids.append(mod_chunk)
                all_family_ids.append(fam_chunk)
                all_masks.append(mask)
        return all_input_ids, all_modality_ids, all_family_ids, all_masks

    def preprocess(self, examples):
        prompts = examples.get("prompt", [""] * len(examples["text"]))
        texts = examples.get("text", [""] * len(examples["text"]))

        def sources():
            for prompt, text in zip(prompts, texts):
                # anti-Reversal Curse: randomize prompt/text order
                merged = " ".join([prompt, text] if random.random() < 0.5 else [text, prompt]).strip()
                if not merged:
                    continue  # empty example: nothing to learn from
                tokens = self.tokenizer.encode(merged, add_special_tokens=False)
                if not tokens:
                    continue
                yield tokens, [int(Modality.TEXT)] * len(tokens), [0] * len(tokens)

        all_input_ids, all_modality_ids, all_family_ids, all_masks = self._collect_chunks(sources())
        return {
            "input_ids": all_input_ids,
            "modality_ids": all_modality_ids,
            "octet_family_ids": all_family_ids,
            "mask": all_masks,
            "prompt_len": [0] * len(all_input_ids),
        }

    _KNOWN_MULTIMODAL_MODALITIES = frozenset(
        {"image_caption", "audio_caption", "video_caption", "lidar", "imu", "control"}
    )

    def _segments_for(self, ex):
        """Dispatch by `modality` (see build_keep_it_simple_multimodal.py)."""
        modality = ex["modality"]

        if modality == "text":
            return [MultimodalSegment(Modality.TEXT, ex["text"].encode("utf-8"))]

        if modality not in self._KNOWN_MULTIMODAL_MODALITIES:
            raise ValueError(f"unknown example modality: {modality!r}")

        arrays = unpack_multimodal_data(ex["data"])
        for name, arr in arrays.items():
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                # NaN/Inf in a raw modality array
                raise NonFiniteDataError(f"non-finite values in {modality!r} field {name!r}")
        meta = json.loads(ex["meta"]) if ex.get("meta") else {}
        caption = ex.get("caption") or ""

        if modality == "image_caption":
            img_markers = KairosTokenizer.encode_image(arrays["image"])
            return [
                MultimodalSegment(Modality.TEXT, caption.encode("utf-8")),
                MultimodalSegment(Modality.IMAGE, img_markers),
            ]

        if modality == "audio_caption":
            sr = meta.get("sample_rate", KairosTokenizer.AUDIO_SAMPLE_RATE)
            audio_markers = KairosTokenizer.encode_audio(arrays["audio"], tick_samples=sr)
            return [
                MultimodalSegment(Modality.TEXT, caption.encode("utf-8")),
                MultimodalSegment(Modality.AUDIO, audio_markers),
            ]

        if modality == "video_caption":
            video_markers = KairosTokenizer.encode_video(arrays["video"])
            return [
                MultimodalSegment(Modality.TEXT, caption.encode("utf-8")),
                MultimodalSegment(Modality.VIDEO, video_markers),
            ]

        if modality == "lidar":
            return [MultimodalSegment(Modality.LIDAR, KairosTokenizer.encode_lidar(arrays["points"]))]

        if modality == "imu":
            flat = np.clip(arrays["signal"].flatten(), -1.0, 1.0).astype(np.float32)
            markers = KairosTokenizer.encode_signal(flat, family="STA")
            return [MultimodalSegment(Modality.STATE, markers)]

        if modality == "control":
            sample_rate = meta.get("sample_rate", KairosTokenizer.AUDIO_SAMPLE_RATE)
            action = arrays["action"]
            state = arrays["state"]
            action_markers = KairosTokenizer.encode_signal(action, family="ACT", tick_samples=sample_rate)
            state_markers = KairosTokenizer.encode_signal(state, family="STA", tick_samples=sample_rate)
            segments = []
            if caption:
                segments.append(MultimodalSegment(Modality.TEXT, caption.encode("utf-8")))
            segments.append(MultimodalSegment(Modality.ACTION, action_markers))
            segments.append(MultimodalSegment(Modality.STATE, state_markers))
            return segments

    def _build_multimodal(self, examples):
        if self.tokenizer is None:
            self.tokenizer = KairosTokenizer()

        def sources():
            skipped = 0
            for ex in examples:
                try:
                    segments = self._segments_for(ex)
                except NonFiniteDataError as e:
                    skipped += 1
                    warnings.warn(f"skipping corrupt example ({e}); {skipped} skipped so far", stacklevel=2)
                    continue
                encoded = self.tokenizer.encode_multimodal(segments)
                yield encoded["input_ids"].tolist(), encoded["modality_ids"].tolist(), encoded[
                    "octet_family_ids"
                ].tolist()

        all_input_ids, all_modality_ids, all_family_ids, all_masks = self._collect_chunks(sources())
        self.ds = HFDataset.from_dict(
            {
                "input_ids": all_input_ids,
                "modality_ids": all_modality_ids,
                "octet_family_ids": all_family_ids,
                "mask": all_masks,
                "prompt_len": [0] * len(all_input_ids),
            }
        )
        self.ds.set_format("torch")

    def __getitem__(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)


class KairosSFTDataset(Dataset):
    """SFT dataset: flattens a conversation to tags and diffuses only the last."""

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
        system = ex.get("system", "")
        turns = ex.get("conversations", [])
        parts = [f"<system>\n{system}\n</system>\n"] if system else []
        last_assistant_start = None

        for turn in turns:
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", ""))
            if role in ("user", "human"):
                parts.append(f"<user>\n{value}\n</user>\n")
            elif role in ("assistant", "gpt"):
                prefix_ids = self.tokenizer.encode("".join(parts), add_special_tokens=False)
                last_assistant_start = len(prefix_ids)
                parts.append(f"<assistant>\n{value}\n</assistant>\n")
            elif role == "tool":
                parts.append(f"<tool_result>\n{value}\n</tool_result>\n")

        all_ids = self.tokenizer.encode("".join(parts), add_special_tokens=False)
        prompt_len = last_assistant_start if last_assistant_start is not None else len(all_ids)

        all_ids = all_ids[: self.max_len]
        prompt_len = min(prompt_len, len(all_ids))
        all_ids, gen_mask = _pad_and_gen_mask(all_ids, prompt_len, self.max_len, self.tokenizer.pad_token_id)

        return {"input_ids": all_ids, "gen_mask": gen_mask, "prompt_len": prompt_len}

    def _process_alpaca(self, ex):
        user = ex["instruction"]
        if ex.get("input", "").strip():
            user += f"\n\n{ex['input']}"
        conversations = [
            {"from": "user", "value": user},
            {"from": "assistant", "value": ex["output"]},
        ]
        return self._process({"system": "", "conversations": conversations})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "gen_mask": torch.tensor(s["gen_mask"], dtype=torch.long),
            "prompt_len": torch.tensor(s["prompt_len"], dtype=torch.long),
        }


class KairosDPODataset(Dataset):
    """DPO dataset: fixed prompt plus separately-tokenized chosen/rejected responses."""

    def __init__(self, tokenizer, max_len=512, examples=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._build(examples)

    def _build(self, examples):
        if examples is not None:
            return [self._process(ex) for ex in examples]
        ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
        return [self._process(ex) for ex in ds]

    def _render_messages(self, messages):
        return "".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>\n" for m in messages)

    def _encode_pair(self, prompt_text, response_text):
        # byte-level tokenizer: encode(A) + encode(B) == encode(A + B)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        response_ids = response_ids[: self.max_len - len(prompt_ids)]
        prompt_ids = prompt_ids[: self.max_len]
        all_ids = prompt_ids + response_ids
        all_ids, gen_mask = _pad_and_gen_mask(all_ids, len(prompt_ids), self.max_len, self.tokenizer.pad_token_id)
        return all_ids, gen_mask, len(prompt_ids)

    def _process(self, ex):
        prompt_text = f"<user>\n{ex['prompt']}\n</user>\n<assistant>\n"
        chosen_text = self._render_messages(ex.get("chosen", [])[-1:])
        rejected_text = self._render_messages(ex.get("rejected", [])[-1:])
        chosen_ids, chosen_mask, plen = self._encode_pair(prompt_text, chosen_text)
        rejected_ids, rejected_mask, _ = self._encode_pair(prompt_text, rejected_text)
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
            "prompt_len": plen,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "chosen_ids": torch.tensor(s["chosen_ids"], dtype=torch.long),
            "chosen_mask": torch.tensor(s["chosen_mask"], dtype=torch.long),
            "rejected_ids": torch.tensor(s["rejected_ids"], dtype=torch.long),
            "rejected_mask": torch.tensor(s["rejected_mask"], dtype=torch.long),
            "prompt_len": torch.tensor(s["prompt_len"], dtype=torch.long),
        }


class KairosRLDataset(Dataset):
    """RL dataset for reasoning via masked diffusion: prompt tokens are never noised."""

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
        choices = list(ex["multiple_choice_targets"])
        scores = list(ex.get("multiple_choice_scores", [0] * len(choices)))
        reasoning = ex.get("reasoning", "")

        choices.append("not sure / I don't know")
        scores.append(0.1)

        paired = list(zip(choices, scores))
        random.shuffle(paired)  # anti-position bias
        choices, scores = zip(*paired)
        best = choices[int(torch.tensor(list(scores)[:-1]).argmax())]

        level = random.choice(["low", "medium", "flex"])
        mask_ratio = {"low": 0.25, "medium": 0.5, "flex": random.uniform(0.1, 0.9)}[level]

        choice_lines = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))
        prompt = f"<inputs>\n{question}\n<choices>\n{choice_lines}\n"

        gen_blocks = [f"<reasoning={level}>\n{reasoning}", f"<answer>\n{best}"]
        random.shuffle(gen_blocks)  # anti-Reversal Curse
        generation = "\n".join(gen_blocks) + "\n"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        gen_ids = self.tokenizer.encode(generation, add_special_tokens=False)
        gen_ids = gen_ids[: self.max_len - len(prompt_ids)]

        ids = prompt_ids + gen_ids
        ids, gen_mask = _pad_and_gen_mask(ids, len(prompt_ids), self.max_len, self.tokenizer.pad_token_id)

        return {
            "input_ids": ids,
            "gen_mask": gen_mask,
            "prompt_len": len(prompt_ids),
            "mask_ratio": mask_ratio,
            "choices": list(choices),
            "scores": list(scores),
            "level": level,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "gen_mask": torch.tensor(s["gen_mask"], dtype=torch.long),
            "prompt_len": torch.tensor(s["prompt_len"], dtype=torch.long),
            "mask_ratio": torch.tensor(s["mask_ratio"], dtype=torch.float),
            "choices": s["choices"],
            "scores": s["scores"],
            "level": s["level"],
        }