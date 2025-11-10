import tqdm
import datetime
import time
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from moe_reft import datamodels


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


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


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
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
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
    train_config: datamodels.TrainConfig,
) -> None:
    model.train()
    setup()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD  # for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)
    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    # init_start_event.record()

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
    )

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=train_config.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_config.num_warmup_steps)

    train_kwargs = {"batch_size": train_config.batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": train_config.test_batch_size, "sampler": sampler2}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **test_kwargs)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    if rank == 0:
        time_of_run = get_date_of_run()
        dur: list[float] = []
        train_acc_tracking: list[float] = []
        val_acc_tracking: list[float] = []
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
            print(f"saving process: rank {rank}  done w state_dict")

            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                print(f"--> attempting to save model prefix {currEpoch}")
                save_name = f"{train_config.save_dir}" + "-" + time_of_run + "-" + currEpoch
                print(f"--> saving as model name {save_name}")

                torch.save(cpu_state, save_name)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank == 0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()
