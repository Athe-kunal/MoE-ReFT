from __future__ import annotations

from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TinySFTDataset(Dataset):
    """
    Very small SFT-style dataset for testing.
    Each item returns:
      - input_ids: [max_length]
      - attention_mask: [max_length]
      - labels: [max_length] (padding positions set to -100)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        examples: List[str],
        max_length: int = 64,
        ignore_index: int = -100,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.examples: List[str] = examples
        self.max_length: int = max_length
        self.ignore_index: int = ignore_index
        if self.tokenizer.pad_token_id is None:
            # For GPT2-like tokenizers, pad with eos
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text: str = self.examples[idx]

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids: torch.Tensor = enc["input_ids"][0]  # [T]
        attention_mask: torch.Tensor = enc["attention_mask"][0]

        # Standard causal-LM SFT: labels = input_ids with padding masked as ignore_index
        labels: torch.Tensor = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_tiny_sft_dataloader(
    *,
    model_name: str = "gpt2",
    batch_size: int = 2,
    max_length: int = 64,
    ignore_index: int = -100,
) -> Tuple[DataLoader, PreTrainedTokenizerBase]:
    """
    Build a tiny DataLoader you can pass into train_sft_interventions_only.
    """

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tiny set of toy SFT examples
    examples: List[str] = [
        "Instruction: Rewrite the sentence more politely.\n"
        "Input: Shut up.\n"
        "Output: Could you please be quiet?\n",
        "Instruction: Answer the question concisely.\n" "Input: What is 2 + 2?\n" "Output: 4.\n",
        "Instruction: Summarize the sentence.\n"
        "Input: The weather is nice and I am going for a walk.\n"
        "Output: Nice weather, going for a walk.\n",
        "Instruction: Translate to French.\n"
        "Input: I love machine learning.\n"
        "Output: J'aime l'apprentissage automatique.\n",
    ]

    dataset = TinySFTDataset(
        tokenizer=tokenizer,
        examples=examples,
        max_length=max_length,
        ignore_index=ignore_index,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader, tokenizer
