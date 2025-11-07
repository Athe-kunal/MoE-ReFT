from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

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


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _apply_rename_rules(name: str, rename_rules: Sequence[tuple[str, str]]) -> str:
    new_name = name
    for old, new in rename_rules:
        if old and old in new_name:
            new_name = new_name.replace(old, new)
    return new_name


def build_partial_state_dict(
    src_sd: Mapping[str, torch.Tensor],
    dst_module: nn.Module,
    *,
    rename_rules: Sequence[tuple[str, str]],
    intervention_patterns: Sequence[str],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], TransferReport]:
    dst_sd: MutableMapping[str, torch.Tensor] = dst_module.state_dict()
    out: dict[str, torch.Tensor] = {}

    report = TransferReport.from_empty()

    for src_name, src_tensor in src_sd.items():
        if _matches_any(src_name, intervention_patterns):
            report.skipped_intervention.append(src_name)
            continue

        cand_names: list[str] = []

        cand_names.append(src_name)

        renamed = _apply_rename_rules(src_name, rename_rules)

        if renamed != src_name:
            cand_names.append(renamed)

        matched_dst: str | None = next((n for n in cand_names if n in dst_sd), None)

        if matched_dst is None:
            report.skipped_missing.append(src_name)
            logger.info(f"Missing for {matched_dst=}")
            continue

        if _matches_any(matched_dst, intervention_patterns):
            report.skipped_intervention.append(src_name)
            continue

        dst_t: torch.Tensor = dst_sd[matched_dst]
        if src_tensor.shape != dst_t.shape:
            report.skipped_shape.append((matched_dst, src_tensor.shape, dst_t.shape))
            logger.error(
                f"The shape for {src_name=} with {src_tensor.shape=} didn't match with for {matched_dst=} and shape {dst_t.shape=}"
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


def load_hf_into_custom_model(
    *,
    hf_model_name_or_path: str,
    custom_model: nn.Module,
    rename_rules: Sequence[tuple[str, str]] | None = None,
    intervention_patterns: Sequence[str] | None = None,
    map_dtype: torch.dtype | None = None,
    map_device: torch.device | None = None,
    trust_remote_code: bool = False,
) -> TransferReport:
    """
    Load a HF Causal LM and copy overlapping weights into `custom_model`,
    skipping intervention modules.

    `rename_rules`: list of (old_substring, new_substring) replacements applied to HF names
                    to match custom model names (e.g., ("model.", ""), ("transformer.", "")).
    `intervention_patterns`: fnmatch patterns to exclude (e.g., "*.pre_moe_intervention.*").
    """
    if rename_rules is None:
        # Common cases: HF uses "model." or "transformer." prefixes that your model may not.
        rename_rules = [
            ("model.", ""),
            ("transformer.", ""),
        ]

    if intervention_patterns is None:
        # Exclude your intervention slots wherever they live in the tree
        intervention_patterns = [
            "*.pre_moe_intervention.*",
            "*.after_moe_intervention.*",
            "*.pre_moe_intervenetion.*",  # in case of typos in existing checkpoints
        ]

    # 1) Load HF model & grab its state dict
    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        hf_model_name_or_path,
        dtype=map_dtype if map_dtype is not None else None,
        trust_remote_code=trust_remote_code,
    )
    src_sd = hf_model.state_dict()

    # 2) Build filtered, renamed state dict compatible with your custom model
    filtered_sd, report = build_partial_state_dict(
        src_sd=src_sd,
        dst_module=custom_model,
        rename_rules=rename_rules,
        intervention_patterns=intervention_patterns,
        device=map_device,
        dtype=map_dtype,
    )

    # 3) Load with strict=False (so missing keys — e.g., interventions — are fine)
    missing, unexpected = custom_model.load_state_dict(filtered_sd, strict=False)

    # Merge loader feedback into the report
    # Anything still "missing" after our pass are genuine unmatched dst params (e.g., interventions)
    report.skipped_missing.extend(missing)
    # "unexpected" should be empty because we feed an exact subset; still include for visibility
    if unexpected:
        # Not typical here, but surface them as "missing" from dst naming perspective
        report.skipped_missing.extend(unexpected)

    return report


if __name__ == "__main__":
    from moe_reft import configuration_olmoe, modeling_olmoe

    custom_model = modeling_olmoe.OlmoeForCausalLM(
        configuration_olmoe.OlmoeInterventionsConfig()
    )  # must define the full module tree

    # 2) Load HF weights into the overlapping parts, skipping interventions
    report = load_hf_into_custom_model(
        hf_model_name_or_path="allenai/OLMoE-1B-7B-0125-Instruct",
        custom_model=custom_model,
        rename_rules=[("model.", ""), ("transformer.", ""), ("layers.", "layers.")],  # tweak as needed
        intervention_patterns=["*.pre_moe_intervention.*", "*.after_moe_intervention.*"],
        map_dtype=torch.bfloat16,  # optional casting
        map_device=torch.device("cuda"),  # optional device move
        trust_remote_code=False,
    )

    print(report.summary())
