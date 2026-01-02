from __future__ import annotations

import os

# Disable tokenizers parallelism to avoid deadlock warnings with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any
import dataclasses
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
from torch.profiler import profile, ProfilerActivity, record_function

import wandb
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from moe_reft import interventions_config, tiny_sft, read_config, datamodels, sft_dataset
from moe_reft.olmoe import modeling_olmoe, configuration_olmoe, load_weights

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _unpack_batch(
    batch: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    labels = batch["labels"].to(device)
    model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    return model_inputs, labels


def _manual_shifted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logger.info("Going into manual loss calculation")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def build_optimizer_from_requires_grad(
    model: nn.Module,
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
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
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_config.num_warmup_steps)

    global_step: int = 0
    running_loss: float = 0.0
    total_tokens: int = 0

    # Remove the profiler for lighter memory and compute use
    for epoch in range(train_config.epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            model_inputs, labels = _unpack_batch(batch, device)
            if "labels" not in model_inputs:
                model_inputs["labels"] = labels
            total_tokens += model_inputs["input_ids"].size(0)
            with record_function("forward_pass"):
                # with torch.autocast(device_type="cuda", dtype=torch.float32):
                outputs = model(**model_inputs)
            with record_function("backward_pass"):

                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                    loss.backward()
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
    # prof.export_chrome_trace("trace.json")
    torch.save(model, train_config.save_dir + "model.pt")


def setup():

    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def train_sft_ddp(
    model: nn.Module,
    train_dataset: sft_dataset.SFTDataset,
    val_dataset: sft_dataset.SFTDataset,
    train_config: datamodels.TrainConfig,
    config_path: str,
) -> None:
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

    optimizer = build_optimizer_from_requires_grad(ddp_model, lr=train_config.learning_rate)
    # Cosine scheduler over steps; T_0 here is warmup / first cycle length
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_config.num_warmup_steps,
    )

    # Distributed samplers
    train_sampler: DistributedSampler[Any] = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    val_sampler: DistributedSampler[Any] = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    common_loader_kwargs = {
        "num_workers": 16,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    collate_fn = sft_dataset.SFTDataCollator(train_dataset.tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.test_batch_size,
        collate_fn=collate_fn,
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
            artifact = wandb.Artifact(
                name="training-config",
                type="config",
            )
            artifact.add_file(config_path)
            wandb_run.log_artifact(artifact)

        logger.info(f"Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")

        if writer is not None:
            writer.add_text("model/parameter_stats", f"total={total_params}, trainable={trainable_params}")

    global_step: int = 0
    running_loss: float = 0.0
    total_tokens: int = 0
    start_time = time.time()

    # with profile(activities=[ProfilerActivity.CUDA], profile_memory=False, record_shapes=False) as prof:
    for epoch in range(train_config.epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # important for proper shuffling each epoch

        for step, batch in enumerate(train_loader):
            if step % train_config.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            model_inputs, labels = _unpack_batch(batch, device=device)
            if "labels" not in model_inputs:
                model_inputs["labels"] = labels

            total_tokens += model_inputs["input_ids"].size(0)
            # num_supervised = int((model_inputs["labels"] != -100).sum().item())
            # logger.info(f"{rank=} {step=} {num_supervised=}")
            # with record_function("forward_pass"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = ddp_model(**model_inputs)

            router_logits = getattr(outputs, "router_logits", None)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
                loss = loss / train_config.grad_accum_steps
                # logger.info(f"{loss=}")
            else:
                loss = _manual_shifted_ce_loss(outputs[0], labels)
                loss = loss / train_config.grad_accum_steps

            # with record_function("backward_pass"):
            loss.backward()
            running_loss += float(loss.item()) * train_config.grad_accum_steps

            if (step + 1) % train_config.grad_accum_steps == 0 or step + 1 == len(
                train_loader
            ):  # last batch needs to be updated
                # Clip gradients to prevent explosion
                if train_config.max_grad_norm is not None and train_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in ddp_model.parameters() if p.requires_grad],
                        max_norm=train_config.max_grad_norm,
                    )
                optimizer.step()
                total_norm_sq = 0.0
                for p in ddp_model.parameters():
                    if p.requires_grad and p.grad is not None:
                        total_norm_sq += p.grad.norm(2).item() ** 2

                grad_norm = total_norm_sq**0.5

                # logger.info(f"Calling optimizer step with {grad_norm=}")
                # wandb.log({"train/grad_norm": grad_norm})
                scheduler.step()
                # for n, p in model.named_parameters():
                #     if p.requires_grad and "pre_moe_intervention" in n:
                #         print(n, p.grad.abs().mean().item())

                global_step += 1

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

                        if router_logits is not None:
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

        ddp_model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                model_inputs, labels = _unpack_batch(batch, device)
                if "labels" not in model_inputs:
                    model_inputs["labels"] = labels

                # with torch.autocast(device_type="cuda", dtype=torch.float32):
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

        # Sync all ranks before starting next epoch
        dist.barrier()

    if rank == 0:
        if writer is not None:
            writer.flush()
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()


# if rank == 0:
# prof.export_chrome_trace("trace.json")


def run_main_olmoe(
    config_path: str,
    model_name: str = "allenai/OLMoE-1B-7B-0125",
    tokenizer_model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
) -> None:
    train_config, interventions_config_, olmoe_config = read_config.load_all_configs(config_path)

    custom_model = modeling_olmoe.OlmoeForCausalLM(olmoe_config)

    intervention_patterns: list[str] = []
    if not olmoe_config.full_parameter_finetuning:
        intervention_patterns = interventions_config.INTERVENTION_PATTERNS

    report = load_weights.load_hf_into_custom_model(
        hf_model_name_or_path=model_name,
        custom_model=custom_model,
        intervention_patterns=intervention_patterns,
        full_parameter_finetuning=olmoe_config.full_parameter_finetuning,
        map_dtype=torch.float32,  # optional casting
        map_device=torch.device("cuda"),  # optional device move
        trust_remote_code=False,
    )
    logger.info(f"{report.summary()}")

    # 5) Print parameter stats
    total_params, trainable_params = load_weights._count_parameters(custom_model)
    print(f"Total parameters:     {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    percent_trainable = (trainable_params / total_params) * 100 if total_params > 0 else 0
    logger.info(
        f"Parameter stats — total: {total_params}, trainable: {trainable_params} "
        f"({percent_trainable:.2f}% trainable)"
    )
    # dataloader, _, dataset = tiny_sft.build_tiny_sft_dataloader(model_name=tokenizer_model_name)
    # train_sft(model=custom_model, dataloader=dataloader, train_config=train_config)
    # train_sft_ddp(model=custom_model, train_dataset=dataset, val_dataset=dataset, train_config=train_config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    response_template = sft_dataset.extract_response_template(tokenizer)
    response_template_ids = tokenizer(response_template)["input_ids"]
    train_dataset = sft_dataset.SFTDataset(
        # source="meta-math/MetaMathQA",
        source="openai/gsm8k",
        tokenizer=tokenizer,
        response_template_ids=response_template_ids,
        system_key=None,
        system_message="You are a helpful math tutor. Solve step by step.",
        # user_key="query",
        user_key="question",
        assistant_key="answer",
        # assistant_key="response",
        split="train",
        name="main",
    )
    val_dataset = sft_dataset.SFTDataset(
        source="openai/gsm8k",
        tokenizer=tokenizer,
        response_template_ids=response_template_ids,
        system_key=None,
        system_message="You are a helpful math tutor. Solve step by step.",
        user_key="question",
        assistant_key="answer",
        split="test",
        name="main",
    )
    train_sft_ddp(
        model=custom_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_config=train_config,
        config_path=config_path,
    )


if __name__ == "__main__":
    run_main_olmoe("moe_reft/configs/olmoe.yaml")
