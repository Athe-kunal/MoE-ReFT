from typing import Tuple

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from trl.data_utils import DataCollatorForCompletionOnlyLM


def build_gsm8k_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 4,
    split: str = "train",
    pad_to_multiple_of: int = 8,
) -> Tuple[Dataset, DataLoader]:
    """
    Build a GSM8K dataset + DataLoader with completion-only loss masking.

    - Uses openai/gsm8k, config 'main'
    - Format: "Q: {question}\nA: {answer}"
    - DataCollatorForCompletionOnlyLM will mask everything before 'A:'.
    """

    raw_dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=split,
    )

    original_cols = raw_dataset.column_names

    def preprocess(example: dict) -> dict:
        question: str = example["question"].strip()
        answer: str = example["answer"].strip()
        # Standard SFT-style prompt/completion
        prompt = f"Q: {question}\nA:"
        # Make sure there's a space after A: so response_template matches exactly "A:"
        completion = " " + answer
        text = prompt + completion
        return {
            "text": text,
            "prompt": prompt,
            "completion": completion,
        }

    dataset = raw_dataset.map(
        preprocess,
        remove_columns=original_cols,
    )

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="A:",  # everything before "A:" is masked out in labels
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    return dataset, dataloader


# Example usage:
if __name__ == "__main__":
    model_name = "allenai/OLMoE-1B-7B-0125-Instruct"  # or whatever you're using
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gsm8k_dataset, gsm8k_loader = build_gsm8k_dataloader(
        tokenizer=tokenizer,
        batch_size=8,
        split="train",
    )

    batch = next(iter(gsm8k_loader))
    print(batch.keys())  # input_ids, attention_mask, labels
    print(batch["labels"].shape)
