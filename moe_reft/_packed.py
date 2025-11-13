from typing import Optional

import torch
import torch.distributed
from torch.nn import functional as F

from torch.utils.data import Dataset
from tqdm import tqdm

PackType = dict[str, torch.Tensor | list[int]]
CROSS_ENTROPY_IGNORE_IDX = -100


class PackedDataset(Dataset):
    def __init__(
        self,
        ds: Dataset,
        *,
        max_seq_len: int,
        padding_idx: int = 0,
        max_packs: int | None = None,
        split_across_pack: bool = False,
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        # Where final samples will be held
        self.packs: list[PackType] = []
        self.previous_sample_boundary: int = 0
        self._pack()

    def _pack(self) -> None:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset."""
        # Buffer to hold samples until they are long enough to be added to self.packs
        current_pack = {
            "input_ids": [],
            "labels": [],
            "input_pos": [],
            "seq_lens": [],
        }

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )

        if rank == 0:
            pbar = tqdm(total=len(self.ds), desc="Packing Dataset", dynamic_ncols=True)

        for sample in self.ds:
            input_ids, labels = sample["input_ids"], sample["labels"]

            seq_len = len(input_ids)
            if seq_len > self.max_seq_len and not self.split_across_pack:
                raise ValueError(
                    f"Dataset sample is too long ({seq_len} > {self.max_seq_len}). "
                    "Please set `split_across_pack=True` or increase `max_seq_len`."
                )

            current_pack["input_ids"] += input_ids
            current_pack["labels"] += labels
            current_pack["input_pos"] += [x % self.max_seq_len for x in range(seq_len)]
            current_pack["seq_lens"] += [seq_len]

            while len(current_pack["input_ids"]) > self.max_seq_len and not self._should_stop_packing():
                current_pack = self._split_and_add_pack(current_pack)

            if rank == 0:
                pbar.update()

            self.previous_sample_boundary = len(current_pack["input_ids"])

            if self._should_stop_packing():
                break

            if len(current_pack["input_ids"]) > 0 and (self.max_packs is None or len(self.packs) < self.max_packs):
                # No need to handle splitting at this point so we can just add the current pack
                self._add_pack(current_pack)

    def _should_stop_packing(self) -> bool:
        """If max packs is set, stop packing when we reach that number."""

        if self.max_packs is not None and len(self.packs) == self.max_packs:
            return True
        return False

    def _split_and_add_pack(self, current_pack: PackType) -> PackType:
        if self.split_across_pack:
            boundary = self.max_seq_len

            leftover_seq_len = self.max_seq_len - sum(current_pack["seq_lens"][:-1])
            seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []

        else:
            boundary = self.previous_sample_boundary
            # If we aren't splitting across packs, we leave out the last sample b/c
            # it will go into the next pack
            seq_len_padding = []

        pack = {
            "input_ids": current_pack["input_ids"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "input_pos": current_pack["input_pos"][:boundary],
            "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
        }

        # Process and add the pack
        self._add_pack(pack)

        # Return the length of the first sample in next pack if we are splitting across packs,
        # otherwise return the length of the last sample in the current pack
        next_seq_len = (
            len(current_pack["input_ids"][boundary:]) if self.split_across_pack else current_pack["seq_lens"][-1]
        )

        return {
            "input_ids": current_pack["input_ids"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "input_pos": current_pack["input_pos"][boundary:],
            "seq_lens": [next_seq_len],
        }

    def _add_pack(self, pack: PackType) -> None:
        """Processes, pads and adds a pack to ``self.packs``."""
        pack = self._convert_to_tensors(pack)
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _convert_to_tensors(self, pack: PackType) -> PackType:
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors."""
        return {
            "input_ids": torch.tensor(pack["input_ids"], dtype=torch.long),
            "labels": torch.tensor(pack["labels"], dtype=torch.long),
            "input_pos": torch.tensor(pack["input_pos"], dtype=torch.long),
            "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
        }

    def _pad_pack(self, pack: PackType, padding_idx: int) -> PackType:
        """Pads a pack to ``self.max_seq_len``."""
        # Pad tokens
        num_padding_tokens = self.max_seq_len - len(pack["input_ids"])
        padded_tokens = F.pad(
            pack["input_ids"],
            (0, num_padding_tokens),
            value=padding_idx,
        )

        # Pad labels
        padded_labels = F.pad(
            pack["labels"],
            (0, self.max_seq_len - len(pack["labels"])),
            value=CROSS_ENTROPY_IGNORE_IDX,
        )

        # Add padding tokens as a last seq len to ensure sum is max_seq_len
        padded_seq_lens = (
            torch.cat([pack["seq_lens"], torch.tensor([num_padding_tokens])])
            if num_padding_tokens > 0
            else pack["seq_lens"]
        )

        # Pad input_pos continuing the sequence from last value
        # in input_pos
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
        num_range = torch.arange(
            pack["input_pos"][-1] + 1,
            pack["input_pos"][-1] + self.max_seq_len - len(pack["input_pos"]) + 1,
        )
        # Clamp to max_seq_len - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.max_seq_len - 1)
        padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

        return {
            "input_ids": padded_tokens,
            "labels": padded_labels,
            "input_pos": padded_input_pos,
            "seq_lens": padded_seq_lens,
        }

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.packs[idx]


if __name__ == "__main__":
    from moe_reft import sft_dataset

    tokenizer_model = "allenai/OLMoE-1B-7B-0125-Instruct"

    ds = sft_dataset.SFTDataset(
        source="openai/gsm8k",
        tokenizer_model_name=tokenizer_model,
        system_key=None,
        system_message="You are a helpful math tutor. Solve step by step.",
        user_key="question",
        assistant_key="answer",
        split="test",
        name="main",
    )

    packed_ds = PackedDataset(ds=ds, max_seq_len=8192)
    for i in packed_ds:
        print(i)
        break
