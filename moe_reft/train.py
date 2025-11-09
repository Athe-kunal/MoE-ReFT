from __future__ import annotations

from typing import Any
import os
import dataclasses
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from loguru import logger
from transformers import get_scheduler
from moe_reft import read_config

from moe_reft import interventions_config, tiny_sft
from moe_reft.olmoe import modeling_olmoe, configuration_olmoe, load_weights

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclasses.dataclass
class TrainConfig:
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


def build_optimizer_from_requires_grad(
    model: nn.Module,
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    """Build an optimizer using only parameters with requires_grad=True."""
    params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]

    if not params:
        raise RuntimeError(
            "No trainable parameters found (requires_grad=True). "
            "Did you forget to unfreeze the intervention layers?"
        )

    logger.info(f"Building optimizer over {sum(p.numel() for p in params):,} trainable params")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def train_sft(
    model: nn.Module,
    dataloader: DataLoader,
    train_config: TrainConfig,
) -> None:
    device = train_config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = build_optimizer_from_requires_grad(model=model, lr=train_config.learning_rate)
    num_training_steps = (len(dataloader) // train_config.grad_accum_steps) * train_config.epochs
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=train_config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    global_step: int = 0
    running_loss: float = 0.0
    total_tokens: int = 0

    for epoch in range(train_config.epochs):
        for step, batch in enumerate(dataloader):
            model_inputs, labels = _unpack_batch(batch, device)
            if "labels" not in model_inputs:
                model_inputs["labels"] = labels
            total_tokens += model_inputs["input_ids"].size(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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


def run_main(config_path: str) -> None:
    train_config, interventions_config_, olmoe_config = read_config.load_all_configs(config_path)

    custom_model = modeling_olmoe.OlmoeForCausalLM(
        configuration_olmoe.OlmoeInterventionsConfig(interventios_config=interventions_config_)
    )

    # 2) Load HF weights into the overlapping parts, skipping interventions
    report = load_weights.load_hf_into_custom_model(
        hf_model_name_or_path="allenai/OLMoE-1B-7B-0125-Instruct",
        custom_model=custom_model,
        intervention_patterns=["*.pre_moe_intervention.*", "*.after_moe_intervention.*"],
        map_dtype=torch.bfloat16,  # optional casting
        map_device=torch.device("cuda"),  # optional device move
        trust_remote_code=False,
    )
    logger.info(f"{report.summary()}")

    for name, param in custom_model.named_parameters():
        if load_weights._matches_any(name, interventions_config.INTERVENTION_PATTERNS):
            param.requires_grad = True

    # 5) Print parameter stats
    total_params, trainable_params = load_weights._count_parameters(custom_model)
    print(f"Total parameters:     {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    logger.info(f"Parameter stats — total: {total_params}, trainable: {trainable_params}")
    dataloader, _ = tiny_sft.build_tiny_sft_dataloader(model_name="allenai/OLMoE-1B-7B-0125-Instruct")
    train_sft(model=custom_model, dataloader=dataloader, train_config=train_config)


if __name__ == "__main__":
    run_main("moe_reft/configs/olmoe.yaml")
