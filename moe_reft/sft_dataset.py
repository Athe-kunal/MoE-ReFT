import re
from typing import Any, Callable, Mapping, Optional, Protocol, Mapping, cast, Literal
from loguru import logger
from datasets import load_dataset
import torch
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


class SFTDataset(Dataset):
    def __init__(
        self,
        *,
        source: str,
        tokenizer_model_name: str,
        train_on_input: bool = False,
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
            train_on_input=train_on_input,
        )
        validate_data_kwargs = load_dataset_kwargs.copy()
        validate_data_kwargs.pop("split", None)
        self._validate_data = load_dataset(source, split="train[:1]", **validate_data_kwargs)
        self.validate_one_sample()

    def validate_one_sample(self) -> None:
        sample = self._prepare_sample(self._validate_data)
        decoded_input = self.tokenizer.decode(sample["input_ids"].tolist()[0], skip_special_tokens=False)
        logger.info(f"Decoded Input (Full Prompt + Response)\n{decoded_input}")

        label_ids = [l for l in sample["labels"].tolist()[0] if l != CROSS_ENTROPY_IGNORE_INDEX]
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
        train_on_input: bool,
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
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        user_messages = sample[self.user_key]
        if self.system_message:
            system_messages = [self.system_message] * len(user_messages)
        else:
            system_messages = sample[self.system_key]
        assistant_messages = sample[self.assistant_key]

        assert system_messages
        assert user_messages
        assert assistant_messages

        messages = [
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_mesage},
            ]
            for system_message, user_message, assistant_mesage in zip(
                system_messages, user_messages, assistant_messages, strict=True
            )
        ]
        tokenized_dict = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            padding=True,
            truncation=True,
            max_seq_len=self.max_seq_len,
            return_tensors="pt",
        )

        input_ids: torch.Tensor = tokenized_dict["input_ids"]
        attention_mask: torch.Tensor = tokenized_dict["attention_mask"]

        idx_after = find_after_subseq_batched(input_ids, self.response_template_ids)
        label_ids = torch.arange(0, input_ids.shape[1])
        labels = torch.where(label_ids[None, :] >= idx_after[:, None], input_ids, CROSS_ENTROPY_IGNORE_INDEX)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
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
        train_on_input=False,
        split="train",
        name="main",
    )
