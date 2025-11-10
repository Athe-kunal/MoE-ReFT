import dataclasses
import torch


@dataclasses.dataclass
class TrainConfig:
    save_dir: str
    epochs: int = 1
    learning_rate: float = 5e-5
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    log_every: int = 50
    ignore_index: int = -100  # standard for HF causal LMs
    num_warmup_steps: int = 0
    device: torch.device | None = None  # auto-detected if None
    batch_size: int = 32
    test_batch_size: int = 64
