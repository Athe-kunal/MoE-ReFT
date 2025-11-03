from __future__ import annotations

from typing import Iterable, Generator, Any
import os
import dataclasses
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from loguru import logger
from transformers import get_scheduler

from moe_reft.interventions import LoreftIntervention, DireftIntervention  # adjust path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


_INTERVENTION_TYPES = (LoreftIntervention, DireftIntervention)


@dataclasses.dataclass
class TrainConfig:
    epochs: int = 1
    learning_rate: float = 5e-5
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16  # use torch.float16 on older GPUs if needed
    log_every: int = 50
    ignore_index: int = -100  # standard for HF causal LMs
    num_warmup_steps: int = 0
    device: torch.device | None = None  # auto-detected if None


def is_non_identity_intervention(m: nn.Module) -> bool:
    """True if module is one of the known intervention types and has parameters."""
    if isinstance(m, nn.Identity):
        return False
    if _INTERVENTION_TYPES and isinstance(m, _INTERVENTION_TYPES):
        # double-check it actually has parameters
        return any(p.requires_grad or p.is_leaf for p in m.parameters(recurse=True))
    # Fallback when classes aren't importable: match by attribute name (robust enough).
    return False  # prefer class-based detection when available


def iter_target_interventions(model: nn.Module) -> Generator[nn.Module, None, None]:
    """
    Yield all submodules that are named 'pre_moe_intervention' or 'after_moe_intervention'
    and are not nn.Identity.
    """
    for name, module in model.named_modules():
        if name.endswith(".pre_moe_intervention") or name.endswith(".after_moe_intervention"):
            if not isinstance(module, nn.Identity):
                yield module


def collect_intervention_params(model: nn.Module) -> Iterable[nn.Parameter]:
    """Collect parameters from the two intervention slots only (skips Identity)."""
    for m in iter_target_interventions(model):
        for p in m.parameters():
            yield p


def freeze_all_unfreeze_interventions(model: nn.Module) -> None:
    """Freeze everything, then unfreeze just the intervention params."""
    for p in model.parameters():
        p.requires_grad = False
    for p in collect_intervention_params(model):
        p.requires_grad = True


def _unpack_batch(
    batch: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if isinstance(batch, dict):
        labels = batch["labels"].to(device)
        model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        return model_inputs, labels

    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
            return {"input_ids": x.to(device)}, y.to(device)
        if len(batch) == 3:
            x, attn, y = batch
            return {"input_ids": x.to(device), "attention_mask": attn.to(device)}, y.to(device)

    # Fallback: assume (X, Y)
    x, y = batch
    return {"input_ids": x.to(device)}, y.to(device)


def _manual_shifted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def build_intervention_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    params = list(collect_intervention_params(model))
    if not params:
        raise RuntimeError(
            "No intervention parameters found. Ensure pre_moe_intervention/after_moe_intervention "
            "are not nn.Identity and are instantiated in __init__."
        )
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def train_sft_interventions_only(
    model: nn.Module,
    dataloader: DataLoader,
    train_config: TrainConfig,
) -> None:

    device = train_config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = build_intervention_optimizer(model=model, lr=train_config.learning_rate)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=train_config.num_warmup_steps)

    global_step: int = 0
    running_loss: float = 0.0
    total_tokens: int = 0

    for epoch in range(train_config.epochs):
        for step, batch in enumerate(dataloader):
            model_inputs, labels = _unpack_batch(batch, device)
            if "labels" not in model_inputs:
                model_inputs["labels"] = labels
            total_tokens += model_inputs["input_ids"].size(0)
            outputs = model(**model_inputs)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = _manual_shifted_ce_loss(outputs[0], labels)

            loss.backward()

            running_loss += loss.item()

            optimizer.step()
            scheduler.step()

            global_step += 1

            if step % train_config.log_every == 0:
                avg_loss = running_loss / train_config.log_every
                logger.info(f"Epoch {epoch} | Step {global_step} | Loss: {avg_loss:.4f}")
                running_loss = 0.0
        logger.info(f"Completed for epoch {epoch}")
