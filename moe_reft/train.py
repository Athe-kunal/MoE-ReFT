from __future__ import annotations

from typing import Any
import dataclasses
import os
import time
import torch
from torch import nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from moe_reft import interventions_config, tiny_sft, read_config, datamodels
from moe_reft.olmoe import modeling_olmoe, configuration_olmoe, load_weights

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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


def train_sft(model: nn.Module, dataloader: DataLoader, train_config: datamodels.TrainConfig) -> None:
    device = train_config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = build_optimizer_from_requires_grad(model=model, lr=train_config.learning_rate)
    num_training_steps = (len(dataloader) // train_config.grad_accum_steps) * train_config.epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_config.num_warmup_steps)

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
            optimizer.zero_grad()
        logger.info(f"Completed for epoch {epoch}")
    torch.save(model, train_config.save_dir + "model.pt")


def setup():

    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def train_sft_ddp(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_config: datamodels.TrainConfig,
) -> None:
    # Assume setup() does dist.init_process_group(...)
    setup()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        os.makedirs(train_config.save_dir, exist_ok=True)

    dist.barrier()

    # Set current device and move model
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model.to(device)

    # IMPORTANT: use local_rank for device_ids/output_device
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # or True if needed
    )

    optimizer = optim.AdamW(
        [p for p in ddp_model.parameters() if p.requires_grad],
        lr=train_config.learning_rate,
    )

    # Cosine scheduler over steps; T_0 here is warmup / first cycle length
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_config.num_warmup_steps,
    )

    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    common_loader_kwargs = {
        "num_workers": 2,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.test_batch_size,
        sampler=val_sampler,
        shuffle=False,
        **common_loader_kwargs,
    )

    total_params = sum(p.numel() for p in ddp_model.module.parameters())
    trainable_params = sum(p.numel() for p in ddp_model.module.parameters() if p.requires_grad)

    writer: SummaryWriter | None = None
    wandb_run: Any = None

    if rank == 0:
        tensorboard_dir = os.path.join(train_config.save_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tensorboard_dir)

        wandb_project = os.getenv("WANDB_PROJECT", "moe-reft")
        wandb_run_name = os.getenv("WANDB_RUN_NAME")
        wandb_config = dataclasses.asdict(train_config)
        if wandb_config.get("device") is not None:
            wandb_config["device"] = str(wandb_config["device"])
        wandb_config["amp_dtype"] = str(wandb_config.get("amp_dtype"))
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            dir=train_config.save_dir,
            config=wandb_config,
            reinit=True,
        )

        if wandb_run is not None:
            wandb_run.summary["model/total_parameters"] = total_params
            wandb_run.summary["model/trainable_parameters"] = trainable_params

        logger.info(f"Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")

        if writer is not None:
            writer.add_text("model/parameter_stats", f"total={total_params}, trainable={trainable_params}")

    global_step: int = 0
    running_loss: float = 0.0
    total_tokens: int = 0
    start_time = time.time()

    for epoch in range(train_config.epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # important for proper shuffling each epoch

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            model_inputs, labels = _unpack_batch(batch, device)
            if "labels" not in model_inputs:
                model_inputs["labels"] = labels

            # this is actually counting batches, not tokens — up to you
            total_tokens += model_inputs["input_ids"].size(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # IMPORTANT: use ddp_model, not bare model
                outputs = ddp_model(**model_inputs)

            router_logits = getattr(outputs, "router_logits", None)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = _manual_shifted_ce_loss(outputs[0], labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            # Log only on rank 0 to avoid spam
            if (step + 1) % train_config.log_every == 0 and rank == 0:
                avg_loss = running_loss / train_config.log_every
                current_time = time.time() - start_time
                last_lr = scheduler.get_last_lr()
                lr = last_lr[0] if last_lr else None

                log_dict: dict[str, Any] = {
                    "train/loss": avg_loss,
                    "train/time": current_time,
                    "train/step": global_step,
                }
                if lr is not None:
                    log_dict["train/learning_rate"] = lr

                if wandb_run is not None:
                    wandb_run.log(log_dict, step=global_step)

                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/time", current_time, global_step)
                    if lr is not None:
                        writer.add_scalar("train/learning_rate", lr, global_step)

                if router_logits is not None and writer is not None:
                    router_log_items: dict[str, Any] = {}
                    for layer_idx, layer_logits in enumerate(router_logits):
                        if layer_logits is None:
                            continue
                        logits_cpu = layer_logits.detach().float().cpu()
                        writer.add_histogram(f"router_logits/layer_{layer_idx}", logits_cpu, global_step)
                        mean_value = logits_cpu.mean().item()
                        writer.add_scalar(f"router_logits/layer_{layer_idx}_mean", mean_value, global_step)
                        log_key = f"router_logits/layer_{layer_idx}_mean"
                        router_log_items[log_key] = mean_value
                        if wandb_run is not None:
                            router_log_items[f"router_logits/layer_{layer_idx}_hist"] = wandb.Histogram(
                                logits_cpu.view(-1).numpy()
                            )
                    if wandb_run is not None and router_log_items:
                        wandb_run.log(router_log_items, step=global_step)

                logger.info(f"[Train] Epoch {epoch} | Step {global_step} | Loss: {avg_loss:.4f}")

                running_loss = 0.0

        # ===== Validation after each epoch =====
        ddp_model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                model_inputs, labels = _unpack_batch(batch, device)
                if "labels" not in model_inputs:
                    model_inputs["labels"] = labels

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = ddp_model(**model_inputs)

                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    loss = _manual_shifted_ce_loss(outputs[0], labels)

                val_loss_sum += loss.item()
                val_steps += 1

        # Reduce val loss across all ranks so rank0 can report global mean
        loss_tensor = torch.tensor(
            [val_loss_sum, val_steps],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_val_loss = (loss_tensor[0] / loss_tensor[1]).item()

        if rank == 0:
            current_time = time.time() - start_time
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/loss": global_val_loss,
                        "val/time": current_time,
                        "val/epoch": epoch,
                        "train/step": global_step,
                    },
                    step=global_step,
                )

            if writer is not None:
                writer.add_scalar("val/loss", global_val_loss, global_step)
                writer.add_scalar("val/time", current_time, global_step)
                writer.add_scalar("val/epoch", epoch, global_step)

            logger.info(f"[Val] Epoch {epoch} | Val Loss: {global_val_loss:.4f}")
            logger.info(f"Completed epoch {epoch}")

            last_ckpt_path = os.path.join(train_config.save_dir, f"checkpoint-epoch{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": global_val_loss,
                    "model_state_dict": ddp_model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_config": getattr(train_config, "__dict__", None),
                },
                last_ckpt_path,
            )

    # Optional: barrier and cleanup()
    dist.barrier()

    if rank == 0:
        if writer is not None:
            writer.flush()
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()


def run_main_olmoe(
    config_path: str,
    model_name: str = "allenai/OLMoE-1B-7B-0125",
    tokenizer_model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
) -> None:
    train_config, interventions_config_, _ = read_config.load_all_configs(config_path)

    custom_model = modeling_olmoe.OlmoeForCausalLM(
        configuration_olmoe.OlmoeInterventionsConfig(interventios_config=interventions_config_)
    )

    report = load_weights.load_hf_into_custom_model(
        hf_model_name_or_path=model_name,
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
    dataloader, _, dataset = tiny_sft.build_tiny_sft_dataloader(model_name=tokenizer_model_name)
    # train_sft(model=custom_model, dataloader=dataloader, train_config=train_config)
    # train_sft_fsdp(model=custom_model, train_dataset=dataset, val_dataset=dataset, train_config=train_config)
    custom_model.config.output_router_logits = True

    train_sft_ddp(model=custom_model, train_dataset=dataset, val_dataset=dataset, train_config=train_config)


if __name__ == "__main__":
    run_main_olmoe("moe_reft/configs/olmoe.yaml")
