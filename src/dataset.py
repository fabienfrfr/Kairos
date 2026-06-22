import random
import concurrent.futures

from datasets import get_dataset_config_names, load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset

# =========================
# Dataset Full Diffusion
# =========================
class KairosPretrainingDataset:
    def __init__(self, texts, tokenizer, max_len=2048, stride=3):
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


# =========================
# Diffusion Mask Reasonning
# =========================
class KairosRLDataset:
    def __init__(
        self,
        max_len=2048,
        split="train",
        max_samples=None,
        timeout=30,
    ):
        self.max_len = max_len
        self.max_samples = max_samples
        self.split = split
        self.timeout = timeout
        self._data: list | None = None

    def _build(self) -> list:
        def _fetch():
            configs = get_dataset_config_names("ffurfaro/bigbench")
            data = []
            count = 0
            for name in configs:
                stream = load_dataset(
                    "ffurfaro/bigbench", name,
                    split=self.split,
                    streaming=True,
                )
                for example in stream:
                    if "multiple_choice_targets" not in example:
                        continue
                    data.append(self.preprocess(example))
                    count += 1
                    if self.max_samples is not None and count >= self.max_samples:
                        return data
            return data

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = concurrent.futures.Future()
            future = executor.submit(_fetch)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"KairosRLDataset: no data fetched after {self.timeout}s — "
                    f"check network access to ffurfaro/bigbench"
                )

    @property
    def data(self):
        if self._data is None:
            self._data = self._build()
        return self._data

    def preprocess(self, example):
        question = example["inputs"]
        choices = list(example["multiple_choice_targets"])
        scores = list(example.get("multiple_choice_scores", [0]*len(choices)))

        # add fallback
        choices.append("not sure / I don't know")
        scores.append(0.1)

        # shuffle choices + scores together
        paired = list(zip(choices, scores))
        random.shuffle(paired)
        choices, scores = zip(*paired)
        choices, scores = list(choices), list(scores)

        # reasoning level
        level = random.choice(["low", "medium", "flex"])

        if level == "low":
            mask_ratio = 0.25
        elif level == "medium":
            mask_ratio = 0.5
        else:
            mask_ratio = random.uniform(0.1, 0.9)

        # build sections
        sections = [
            ("inputs", question),
            ("reasoning", level),
            ("choices", choices),
        ]

        random.shuffle(sections)

        text = ""

        for name, content in sections:
            if name == "inputs":
                text += f"<inputs>\n{content}\n"
            elif name == "reasoning":
                text += f"<reasoning={content}>\n"
            elif name == "choices":
                text += "<choices>\n"
                for i, c in enumerate(content):
                    text += f"{chr(65+i)}) {c}\n"

        return {
            "text": text,
            "choices": choices,
            "scores": scores,
            "level": level,
            "mask_ratio": mask_ratio,
        }

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)