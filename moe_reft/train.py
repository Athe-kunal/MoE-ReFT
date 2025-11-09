from __future__ import annotations

from typing import Any
import os
import dataclasses
import torch
from torch import nn
import tqdm
import datetime
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from loguru import logger
from transformers import get_scheduler
import torch.distributed as dist
import torch.optim as optim
import time

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from moe_reft import read_config

from moe_reft import interventions_config, tiny_sft
from moe_reft.olmoe import modeling_olmoe, configuration_olmoe, load_weights

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclasses.dataclass
class TrainConfig:
    save_path: str
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


def train_sft(model: nn.Module, dataloader: DataLoader, train_config: TrainConfig, save_path: str) -> None:
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
    torch.save(model, save_path)


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / 10e9
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def train(
    model: nn.Module,
    rank: int,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    sampler=None,
):
    model.train()
    local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank == 0:
        inner_pbar = tqdm.tqdm(range(len(train_loader)), colour="blue", desc="r0 Training Epoch")
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(
            input_ids=batch["source_ids"], attention_mask=batch["source_mask"], labels=batch["target_ids"]
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank == 0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]

    if rank == 0:
        inner_pbar.close()
        print(f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}")
    return train_accuracy


def validation(model: nn.Module, rank: int, val_loader: DataLoader):
    model.eval()
    correct = 0
    local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(range(len(val_loader)), colour="green", desc="Validation Epoch")
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(
                input_ids=batch["source_ids"], attention_mask=batch["source_mask"], labels=batch["target_ids"]
            )
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank == 0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def train_sft_fsdp(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_config: TrainConfig,
) -> None:
    model.train()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD  # for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)
    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    # init_start_event.record()

    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        mixed_precision=bfSixteen,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    setup()

    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_config.num_warmup_steps)

    fsdp_loss = torch.zeros(2).to(local_rank)
    train_kwargs = {"batch_size": train_config.batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": train_config.test_batch_size, "sampler": sampler2}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **test_kwargs)
    if rank == 0:
        inner_pbar = tqdm.tqdm(range(len(train_loader)), colour="blue", desc="r0 Training Epoch")
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    if rank == 0:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    for epoch in range(1, train_config.epochs + 1):
        t0 = time.time()
        train_accuracy = train(model, rank, train_loader, optimizer, epoch, sampler=sampler1)
        curr_val_loss = validation(model, rank, val_loader)
        scheduler.step()

        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            val_acc_tracking.append(curr_val_loss.item())

            mem_alloc_tracker.append(format_metrics_to_gb(torch.cuda.memory_allocated()))
            mem_reserved_tracker.append(format_metrics_to_gb(torch.cuda.memory_reserved()))
            print(f"completed save and stats zone...")

        if curr_val_loss < best_val_loss:

            # save
            if rank == 0:
                print(f"--> entering save model state")

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state = model.state_dict()
            # print(f"saving process: rank {rank}  done w state_dict")

            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                print(f"--> attempting to save model prefix {currEpoch}")
                save_name = f"{train_config.save_path}" + "-" + time_of_run + "-" + currEpoch
                print(f"--> saving as model name {save_name}")

                torch.save(cpu_state, save_name)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank == 0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()


def run_main_olmoe(config_path: str, save_path: str) -> None:
    train_config, interventions_config_, _ = read_config.load_all_configs(config_path)

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
    train_sft(model=custom_model, dataloader=dataloader, train_config=train_config, save_path=save_path)


if __name__ == "__main__":
    run_main_olmoe("moe_reft/configs/olmoe.yaml")
