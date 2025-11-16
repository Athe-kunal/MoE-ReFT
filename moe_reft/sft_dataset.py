import re
from typing import Any, Callable, Mapping, Optional, Protocol, Mapping, cast, Literal
from loguru import logger
from datasets import load_dataset
import torch
import dataclasses
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

CROSS_ENTROPY_IGNORE_INDEX = -100
PROMPT_TEMPLATE = [
    {"role": "system", "content": "<SYSTEMT>"},
    {"role": "user", "content": "<USER>"},
    {"role": "assistant", "content": "<ASSISTANT>"},
]


def extract_response_template(tokenizer: PreTrainedTokenizerBase) -> str | None:
    text = cast(str, tokenizer.apply_chat_template(PROMPT_TEMPLATE, tokenize=False))
    match = re.search(r"<USER>(.*?)<ASSISTANT>", text, flags=re.DOTALL)
    if not match:
        return None
    segment = match.group(1)
    return segment


class Transform(Protocol):
    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]: ...


# Whatever you already use in SFTTransform
CROSS_ENTROPY_IGNORE_INDEX = -100  # or import from your constants


@dataclasses.dataclass
class SFTDataCollator:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = CROSS_ENTROPY_IGNORE_INDEX
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # 1. Separate labels so tokenizer.pad only sees model inputs
        labels_list: list[Any] = [f["labels"] for f in features]
        features_for_pad: list[dict[str, Any]] = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        # 2. Let tokenizer.pad handle input_ids / attention_mask
        batch = self.tokenizer.pad(
            features_for_pad,
            padding=True,  # pad to max length in this batch
            max_length=None,  # or a fixed max_length if you want
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 3. Manually pad labels to match seq_len of input_ids
        seq_len: int = batch["input_ids"].size(1)
        padded_labels: list[list[int]] = []

        for lbl in labels_list:
            # convert to python list of ints
            if isinstance(lbl, torch.Tensor):
                lbl_list = lbl.tolist()
            else:
                lbl_list = list(lbl)

            # truncate if somehow longer than seq_len
            if len(lbl_list) > seq_len:
                lbl_list = lbl_list[:seq_len]

            pad_len = seq_len - len(lbl_list)
            if pad_len > 0:
                lbl_list = lbl_list + [self.label_pad_token_id] * pad_len

            padded_labels.append(lbl_list)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


class SFTDataset(Dataset):
    def __init__(
        self,
        *,
        source: str,
        tokenizer_model_name: str,
        system_key: str | None,
        user_key: str,
        assistant_key: str,
        system_message: str | None,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.response_template = extract_response_template(self.tokenizer)
        logger.info(
            f"For the {tokenizer_model_name=} automatically assigned the response template to {self.response_template}"
        )
        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            if filter_kwargs is None:
                filter_kwargs = {}
            self._data = self._data.filter(filter_fn, **filter_kwargs)
        self._prepare_sample = SFTTransform(
            tokenizer=self.tokenizer,
            response_template=self.response_template,
            system_message=system_message,
            system_key=system_key,
            user_key=user_key,
            assistant_key=assistant_key,
        )
        validate_data_kwargs = load_dataset_kwargs.copy()
        validate_data_kwargs.pop("split", None)
        self._validate_data = load_dataset(source, split="train[:1]", **validate_data_kwargs)
        self.validate_one_sample()

    def validate_one_sample(self) -> None:
        sample = self._prepare_sample(self._validate_data[0])
        decoded_input = self.tokenizer.decode(sample["input_ids"].tolist(), skip_special_tokens=False)
        logger.info(f"Decoded Input (Full Prompt + Response)\n{decoded_input}")

        label_ids = [l for l in sample["labels"].tolist() if l != CROSS_ENTROPY_IGNORE_INDEX]
        decoded_labels = self.tokenizer.decode(label_ids, skip_special_tokens=False)
        logger.info(f"Decoded Labels (Assistant Response Only)\n{decoded_labels}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)


def find_after_subseq_batched(
    input_ids: torch.Tensor,  # [B, T]
    response_ids: torch.Tensor,  # [R]
) -> torch.Tensor:  # [B], indices after match, -1 if not found
    B, T = input_ids.shape
    R = response_ids.shape[0]

    if R > T:
        return input_ids.new_full((B,), -1, dtype=torch.long)

    # [B, T-R+1, R]
    windows = input_ids.unfold(dimension=1, size=R, step=1)

    # [B, T-R+1, R] == [1,1,R] -> [B, T-R+1, R]
    eq = windows == response_ids.view(1, 1, -1)
    matches = eq.all(dim=-1)  # [B, T-R+1]

    Lw = matches.shape[1]
    positions = torch.arange(Lw, device=input_ids.device)  # [T-R+1]
    pos = positions.unsqueeze(0).expand(B, -1)  # [B, T-R+1]

    # set non-matches to big sentinel
    sentinel = Lw + 1
    pos = pos.masked_fill(~matches, sentinel)

    first_pos, _ = pos.min(dim=1)  # [B]
    idx_after = torch.where(
        first_pos <= Lw,
        first_pos + R,
        input_ids.new_full((B,), -1, dtype=torch.long),
    )
    return idx_after


class SFTTransform(Transform):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        response_template: str,
        system_key: str | None,
        user_key: str,
        assistant_key: str,
        system_message: str | None,
        max_seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        if system_key and system_message:
            raise ValueError(
                f"Can't set both `system_message` and `system_key`, but they are set {system_key=} and {system_message=}"
            )
        self.system_key = system_key
        self.system_message = system_message
        self.user_key = user_key
        self.assistant_key = assistant_key
        self.response_template = response_template
        self.response_template_ids = self.tokenizer.encode(self.response_template, return_tensors="pt").squeeze(0)
        self.max_seq_len = max_seq_len

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        user_message = sample[self.user_key]
        if self.system_message:
            system_message = self.system_message
        else:
            system_message = sample[self.system_key]
        assistant_message = sample[self.assistant_key]

        assert system_message
        assert user_message
        assert assistant_message

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        tokenized_dict = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            truncation=True,
            max_seq_len=self.max_seq_len,
            return_tensors="pt",
        )

        input_ids: torch.Tensor = tokenized_dict["input_ids"]
        attention_mask: torch.Tensor = tokenized_dict["attention_mask"]

        idx_after = find_after_subseq_batched(input_ids, self.response_template_ids)
        label_ids = torch.arange(0, input_ids.shape[1])
        labels = torch.where(label_ids[None, :] > idx_after[:, None], input_ids, CROSS_ENTROPY_IGNORE_INDEX)

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "labels": labels[0],
        }


if __name__ == "__main__":
    tokenizer_model = "allenai/OLMoE-1B-7B-0125-Instruct"
    ds = SFTDataset(
        source="openai/gsm8k",
        tokenizer_model_name=tokenizer_model,
        system_key=None,
        system_message="You are a helpful math tutor. Solve step by step.",
        user_key="question",
        assistant_key="answer",
        split="train",
        name="main",
    )
