from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence
import tqdm
import fnmatch
import torch
from loguru import logger
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from moe_reft.olmoe import configuration_olmoe, modeling_olmoe
from moe_reft import interventions_config as ic


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


def count_parameters(model: nn.Module) -> tuple[int, int]:
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
    hf_model_name_or_path: str | None = None,
    pt_file: str | None = None,
    custom_model: nn.Module,
    intervention_patterns: Sequence[str] | None = None,
    full_parameter_finetuning: bool = False,
    map_dtype: torch.dtype | None = None,
    map_device: torch.device | None = None,
    trust_remote_code: bool = False,
) -> tuple[TransferReport, nn.Module]:
    """
    Load either a HF Causal LM (by name/path) or a torch.pt file, copy overlapping weights into `custom_model`,
    skipping intervention modules, then freeze all parameters except those
    matching `intervention_patterns`.

    You must specify either `hf_model_name_or_path` or `pt_file` (but not both).

    `intervention_patterns`: fnmatch patterns to mark modules as *trainable*
                             (e.g., "*.pre_moe_intervention.*").
    """

    if (hf_model_name_or_path is None) == (pt_file is None):
        raise ValueError("You must specify exactly one of hf_model_name_or_path or pt_file.")
    if intervention_patterns is None:
        intervention_patterns = [
            "*.pre_moe_intervention.*",
            "*.after_moe_intervention.*",
            "*.pre_moe_intervenetion.*",  # typo fallback
        ]

    # 1) Load state dict from HF or torch file
    if hf_model_name_or_path is not None:
        hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            hf_model_name_or_path,
            dtype=map_dtype if map_dtype is not None else None,
            trust_remote_code=trust_remote_code,
        )
        src_sd = hf_model.state_dict()
    else:
        assert pt_file
        src_sd = torch.load(pt_file, map_location="cpu")
        if isinstance(src_sd, dict) and "model_state_dict" in src_sd:
            src_sd = src_sd["model_state_dict"]

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

    # 4) Freeze everything, then unfreeze only intervention layers unless doing full-parameter finetuning
    if full_parameter_finetuning:
        for param in custom_model.parameters():
            param.requires_grad = True
    else:
        for param in custom_model.parameters():
            param.requires_grad = False

        for name, param in custom_model.named_parameters():
            if matches_any(name, intervention_patterns):
                param.requires_grad = True

    # 5) Print parameter stats
    total_params, trainable_params = count_parameters(custom_model)
    print(f"Total parameters:     {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    logger.info(f"Parameter stats — total: {total_params}, trainable: {trainable_params}")

    return report, custom_model


def load_pretrained_with_interventions_from_checkpoint(
    *,
    pt_file: str,
    hf_model_name_or_path: str,
    intervention_patterns: Sequence[str] | None = None,
    map_dtype: torch.dtype | None = None,
    map_device: torch.device | None = None,
    trust_remote_code: bool = False,
    full_parameter_finetuning: bool = False,
) -> tuple[nn.Module, TransferReport]:
    """
    Create an OlmoeForCausalLM model with interventions and load weights from two sources:
    - Base model weights from a HuggingFace pretrained model
    - Intervention weights from a .pt checkpoint file

    This function:
    1. Loads the .pt file to extract model_state_dict and config
    2. Initializes OlmoeForCausalLM with the intervention config
    3. Loads pretrained HF weights into non-intervention parameters
    4. Loads intervention weights from the .pt checkpoint

    Args:
        pt_file: Path to the .pt checkpoint containing intervention weights and config.
                 Expected to have 'model_state_dict' and optionally 'config' keys.
        hf_model_name_or_path: HuggingFace model name or path for base pretrained weights.
        intervention_patterns: fnmatch patterns identifying intervention modules
                              (e.g., "*.pre_moe_intervention.*").
        map_dtype: Optional dtype to cast tensors to.
        map_device: Optional device to move tensors to.
        trust_remote_code: Whether to trust remote code when loading HF model.
        full_parameter_finetuning: If True, all parameters are trainable.
                                   If False, only intervention params are trainable.

    Returns:
        A tuple of (model, TransferReport) where model is the initialized OlmoeForCausalLM
        with loaded weights, and TransferReport contains details about the weight transfer.
    """

    if intervention_patterns is None:
        intervention_patterns = ic.INTERVENTION_PATTERNS

    # 1) Load checkpoint from .pt file
    checkpoint = torch.load(pt_file, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        pt_state_dict = checkpoint["model_state_dict"]
    else:
        pt_state_dict = checkpoint

    # 2) Extract or create intervention config from checkpoint
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
        if isinstance(config, configuration_olmoe.OlmoeInterventionsConfig):
            model_config = config
        elif isinstance(config, dict):
            # Reconstruct config from dict
            interventions_cfg = ic.InterventionsConfig(**config.get("intervention_config", {}))
            model_config = configuration_olmoe.OlmoeInterventionsConfig(
                interventions_cfg, **{k: v for k, v in config.items() if k != "intervention_config"}
            )
        else:
            raise ValueError(f"Unexpected config type in checkpoint: {type(config)}")
    else:
        # Try to infer intervention config from state dict keys
        # Default to a reasonable config if not found
        logger.warning("No config found in checkpoint, using default intervention config")
        interventions_cfg = ic.InterventionsConfig(
            intervention_type="LoreftIntervention",
            intervention_layers="all",
            intervention_places="pre_moe",
            low_rank_dimension=64,
            dropout=0.0,
            act_fn=None,
            init_orth=True,
        )
        model_config = configuration_olmoe.OlmoeInterventionsConfig(interventions_cfg)

    # 3) Initialize custom model with interventions
    custom_model = modeling_olmoe.OlmoeForCausalLM(model_config)

    # 4) Load HF pretrained weights (excluding interventions)
    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        hf_model_name_or_path,
        dtype=map_dtype if map_dtype is not None else None,
        trust_remote_code=trust_remote_code,
    )
    hf_state_dict = hf_model.state_dict()

    # Build filtered state dict for base model weights (skip intervention weights)
    base_filtered_sd, base_report = build_partial_state_dict(
        src_sd=hf_state_dict,
        dst_module=custom_model,
        intervention_patterns=intervention_patterns,
        device=map_device,
        dtype=map_dtype,
    )

    # 5) Extract intervention weights from .pt checkpoint
    intervention_report = TransferReport.from_empty()
    intervention_sd: dict[str, torch.Tensor] = {}
    dst_sd = custom_model.state_dict()

    for name, tensor in tqdm.tqdm(
        pt_state_dict.items(),
        total=len(pt_state_dict),
        desc="Extracting intervention weights from checkpoint",
    ):
        if matches_any(name, intervention_patterns):
            if name in dst_sd:
                dst_tensor = dst_sd[name]
                if tensor.shape == dst_tensor.shape:
                    if map_dtype is not None or map_device is not None:
                        tensor = tensor.to(
                            device=map_device if map_device is not None else dst_tensor.device,
                            dtype=map_dtype if map_dtype is not None else dst_tensor.dtype,
                        )
                    intervention_sd[name] = tensor
                    intervention_report.copied.append(name)
                else:
                    intervention_report.skipped_shape.append((name, tensor.shape, dst_tensor.shape))
                    logger.error(
                        f"Shape mismatch for intervention {name}: "
                        f"checkpoint {tensor.shape} vs model {dst_tensor.shape}"
                    )
            else:
                intervention_report.skipped_missing.append(name)
                logger.warning(f"Intervention weight {name} not found in model")

    # 6) Merge both state dicts (base + interventions)
    merged_sd = {**base_filtered_sd, **intervention_sd}

    # 7) Load merged weights into model
    missing, unexpected = custom_model.load_state_dict(merged_sd, strict=False)

    # Create combined report
    combined_report = TransferReport(
        copied=base_report.copied + intervention_report.copied,
        skipped_shape=base_report.skipped_shape + intervention_report.skipped_shape,
        skipped_missing=base_report.skipped_missing + intervention_report.skipped_missing + list(missing),
        skipped_intervention=base_report.skipped_intervention,  # These were intentionally skipped from HF
    )

    if unexpected:
        combined_report.skipped_missing.extend(unexpected)

    # 8) Set parameter trainability
    if full_parameter_finetuning:
        for param in custom_model.parameters():
            param.requires_grad = True
    else:
        for param in custom_model.parameters():
            param.requires_grad = False

        for name, param in custom_model.named_parameters():
            if matches_any(name, intervention_patterns):
                param.requires_grad = True

    # 9) Print stats
    total_params, trainable_params = count_parameters(custom_model)
    print(f"Total parameters:     {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Base model weights loaded:  {len(base_report.copied)}")
    print(f"Intervention weights loaded: {len(intervention_report.copied)}")

    logger.info(f"Parameter stats — total: {total_params}, trainable: {trainable_params}")
    logger.info(
        f"Loaded {len(base_report.copied)} base weights from HF, "
        f"{len(intervention_report.copied)} intervention weights from checkpoint"
    )

    return custom_model, combined_report


if __name__ == "__main__":
    # from moe_reft.olmoe import configuration_olmoe, modeling_olmoe

    # custom_model = modeling_olmoe.OlmoeForCausalLM(configuration_olmoe.OlmoeInterventionsConfig())

    # # 2) Load HF weights into the overlapping parts, skipping interventions
    # report = load_hf_into_custom_model(
    #     hf_model_name_or_path="allenai/OLMoE-1B-7B-0125-Instruct",
    #     custom_model=custom_model,
    #     intervention_patterns=["*.pre_moe_intervention.*", "*.after_moe_intervention.*"],
    #     map_dtype=torch.bfloat16,  # optional casting
    #     map_device=torch.device("cuda"),  # optional device move
    #     trust_remote_code=False,
    # )

    # print(report.summary())
    # Test the function load_pretrained_with_interventions_from_checkpoint with the given path
    model, report = load_pretrained_with_interventions_from_checkpoint(
        pt_file="math_sft/checkpoint-epoch0.pt",
        hf_model_name_or_path="allenai/OLMoE-1B-7B-0125",
        intervention_patterns=["*.pre_moe_intervention.*", "*.after_moe_intervention.*"],
        map_dtype=torch.bfloat16,  # or choose appropriate dtype
        map_device=torch.device("cuda"),  # or choose appropriate device
        trust_remote_code=False,
        full_parameter_finetuning=False,
    )
    print(report.summary())
