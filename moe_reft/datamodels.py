import dataclasses
import torch


@dataclasses.dataclass
class TrainConfig:
    save_dir: str
    epochs: int
    learning_rate: float
    grad_accum_steps: int
    max_grad_norm: float | None
    amp: bool
    amp_dtype: torch.dtype
    log_every: int
    ignore_index: int  # standard for HF causal LMs
    num_warmup_steps: int
    device: torch.device | None  # auto-detected if None
    batch_size: int
    test_batch_size: int
