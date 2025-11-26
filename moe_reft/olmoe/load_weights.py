from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence
import tqdm
import fnmatch
import torch
from loguru import logger
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel


@dataclass
class TransferReport:
    copied: list[str]
    skipped_shape: list[tuple[str, torch.Size, torch.Size]]
    skipped_missing: list[str]
    skipped_intervention: list[str]

    def summary(self) -> str:
        return (
            f"Copied: {len(self.copied)} | "
            f"Skipped (shape): {len(self.skipped_shape)} | "
            f"Skipped (missing): {len(self.skipped_missing)} | "
            f"Skipped (intervention): {len(self.skipped_intervention)}"
        )

    @classmethod
    def from_empty(cls) -> TransferReport:
        return TransferReport(copied=[], skipped_shape=[], skipped_missing=[], skipped_intervention=[])


def matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def build_partial_state_dict(
    src_sd: Mapping[str, torch.Tensor],
    dst_module: nn.Module,
    *,
    # kept in signature for compatibility, but ignored
    intervention_patterns: Sequence[str],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], TransferReport]:
    dst_sd: MutableMapping[str, torch.Tensor] = dst_module.state_dict()
    out: dict[str, torch.Tensor] = {}

    report = TransferReport.from_empty()

    # tqdm over keys so we can get len(src_sd)
    for src_name in tqdm.tqdm(
        src_sd.keys(),
        total=len(src_sd),
        desc="Building partial state dict",
    ):
        src_tensor: torch.Tensor = src_sd[src_name]

        # Skip intervention weights from HF
        if matches_any(src_name, intervention_patterns):
            report.skipped_intervention.append(src_name)
            continue

        # We now ignore rename_rules entirely; direct name match only
        matched_dst: str | None = src_name if src_name in dst_sd else None

        if matched_dst is None:
            report.skipped_missing.append(src_name)
            logger.info(f"Missing for {src_name=}")
            continue

        if matches_any(matched_dst, intervention_patterns):
            report.skipped_intervention.append(src_name)
            continue

        dst_t: torch.Tensor = dst_sd[matched_dst]

        if src_tensor.shape != dst_t.shape:
            report.skipped_shape.append((matched_dst, src_tensor.shape, dst_t.shape))
            logger.error(
                f"The shape for {src_name=} with {src_tensor.shape=} "
                f"didn't match {matched_dst=} with {dst_t.shape=}"
            )
            continue

        if dtype is not None or device is not None:
            src_tensor = src_tensor.to(
                device=device if device is not None else dst_t.device,
                dtype=dtype if dtype is not None else dst_t.dtype,
            )

        out[matched_dst] = src_tensor
        report.copied.append(matched_dst)

    return out, report


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params: int = 0
    trainable_params: int = 0
    for p in model.parameters():
        numel: int = p.numel()
        total_params += numel
        if p.requires_grad:
            trainable_params += numel
    return total_params, trainable_params


def load_hf_into_custom_model(
    *,
    hf_model_name_or_path: str,
    custom_model: nn.Module,
    intervention_patterns: Sequence[str] | None = None,
    map_dtype: torch.dtype | None = None,
    map_device: torch.device | None = None,
    trust_remote_code: bool = False,
) -> TransferReport:
    """
    Load a HF Causal LM and copy overlapping weights into `custom_model`,
    skipping intervention modules, then freeze all parameters except those
    matching `intervention_patterns`.

    `intervention_patterns`: fnmatch patterns to mark modules as *trainable*
                             (e.g., "*.pre_moe_intervention.*").
    """

    if intervention_patterns is None:
        intervention_patterns = [
            "*.pre_moe_intervention.*",
            "*.after_moe_intervention.*",
            "*.pre_moe_intervenetion.*",  # typo fallback
        ]
    # 1) Load HF model & grab its state dict
    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        hf_model_name_or_path,
        dtype=map_dtype if map_dtype is not None else None,
        trust_remote_code=trust_remote_code,
    )
    src_sd = hf_model.state_dict()

    # 2) Build filtered state dict compatible with your custom model
    filtered_sd, report = build_partial_state_dict(
        src_sd=src_sd,
        dst_module=custom_model,
        intervention_patterns=intervention_patterns,
        device=map_device,
        dtype=map_dtype,
    )

    # 3) Load with strict=False (so missing keys — e.g., interventions — are fine)
    missing, unexpected = custom_model.load_state_dict(filtered_sd, strict=False)

    # Merge loader feedback into the report
    report.skipped_missing.extend(missing)
    if unexpected:
        report.skipped_missing.extend(unexpected)

    # 4) Freeze everything, then unfreeze only intervention layers
    for param in custom_model.parameters():
        param.requires_grad = False

    for name, param in custom_model.named_parameters():
        if matches_any(name, intervention_patterns):
            param.requires_grad = True

    # 5) Print parameter stats
    total_params, trainable_params = _count_parameters(custom_model)
    print(f"Total parameters:     {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    logger.info(f"Parameter stats — total: {total_params}, trainable: {trainable_params}")

    return report


if __name__ == "__main__":
    from moe_reft.olmoe import configuration_olmoe, modeling_olmoe

    custom_model = modeling_olmoe.OlmoeForCausalLM(configuration_olmoe.OlmoeInterventionsConfig())

    # 2) Load HF weights into the overlapping parts, skipping interventions
    report = load_hf_into_custom_model(
        hf_model_name_or_path="allenai/OLMoE-1B-7B-0125-Instruct",
        custom_model=custom_model,
        intervention_patterns=["*.pre_moe_intervention.*", "*.after_moe_intervention.*"],
        map_dtype=torch.bfloat16,  # optional casting
        map_device=torch.device("cuda"),  # optional device move
        trust_remote_code=False,
    )

    print(report.summary())
