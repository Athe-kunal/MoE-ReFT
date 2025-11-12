import re
from typing import Any, Callable, Mapping, Optional, Protocol, Mapping, cast
from loguru import logger
from datasets import load_dataset
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
    """
    Loose interface for all data and model transforms. Transforms operate at the
    sample level and perform operations on a sample dict, returning the updated dict.
    For an example implementation of this protocol, see
    :class:`~torchtune.modules.transforms.VisionCrossAttentionMask`.
    """

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        pass


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

        decoded_input = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        logger.info(f"[Decoded Input (Full Prompt + Response)]\n{decoded_input}")

        label_ids = [tid for tid, l in zip(sample["input_ids"], sample["labels"]) if l != -100]
        decoded_labels = self.tokenizer.decode(label_ids, skip_special_tokens=False)
        logger.info(f"[Decoded Labels (Assistant Response Only)]\n{decoded_labels}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)


def _find_subseq(h: list[int], n: list[int]) -> int:
    for i in range(len(h) - len(n) + 1):
        if h[i : i + len(n)] == n:
            return i
    return -1


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
        self.response_template_ids = self.tokenizer.encode(self.response_template)
        self.train_on_input = train_on_input

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        system_message = self.system_message if self.system_message else sample[self.system_key]
        user_message = sample[self.user_key]
        assistant_message = sample[self.assistant_key]
        assert system_message
        assert user_message
        assert assistant_message

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        print(messages)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        enc = self.tokenizer(
            text,
            add_special_tokens=False,
        )
        input_ids: list[int] = enc["input_ids"]
        attention_mask: list[int] = enc["attention_mask"]

        labels = [-100] * len(input_ids)

        start = _find_subseq(input_ids, self.response_template_ids)
        if start != -1:
            start += len(self.response_template_ids)
            labels[start:] = input_ids[start:]

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
