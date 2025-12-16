from __future__ import annotations

from typing import Optional, Tuple
import torch
from omegaconf import OmegaConf

from moe_reft.datamodels import TrainConfig
from moe_reft import interventions_config
from moe_reft.olmoe import configuration_olmoe


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    """
    Map a string from YAML (e.g., "bfloat16") to a torch.dtype.
    Extend this if you want to support more names.
    """
    mapping: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported amp_dtype string: {dtype_str!r}")
    return mapping[dtype_str]


def load_all_configs(
    yaml_path: str,
) -> Tuple[TrainConfig, interventions_config.InterventionsConfig, configuration_olmoe.OlmoeInterventionsConfig]:
    """
    Load TrainConfig, InterventionsConfig, and OlmoeInterventionsConfig
    from a single nested YAML file using OmegaConf.
    """
    cfg = OmegaConf.load(yaml_path)
    # ------------------------
    # Train config
    # ------------------------
    train_section = cfg.train

    amp_dtype: torch.dtype = _dtype_from_str(train_section.amp_dtype)

    device_obj: Optional[torch.device]
    if train_section.device is None:
        device_obj = None
    else:
        device_obj = torch.device(str(train_section.device))

    train_config = TrainConfig(
        save_dir=str(train_section.save_dir),
        epochs=int(train_section.epochs),
        learning_rate=float(train_section.learning_rate),
        grad_accum_steps=int(train_section.grad_accum_steps),
        max_grad_norm=train_section.max_grad_norm,
        amp=bool(train_section.amp),
        amp_dtype=amp_dtype,
        log_every=int(train_section.log_every),
        ignore_index=int(train_section.ignore_index),
        num_warmup_steps=int(train_section.num_warmup_steps),
        device=device_obj,
        batch_size=int(train_section.batch_size),
        test_batch_size=int(train_section.test_batch_size),
    )

    # ------------------------
    # Interventions config
    # ------------------------
    inter_section = cfg.interventions

    interventions_config_ = interventions_config.InterventionsConfig(
        intervention_type=inter_section.intervention_type,
        intervention_layers=inter_section.intervention_layers,
        intervention_places=inter_section.intervention_places,
        low_rank_dimension=int(inter_section.low_rank_dimension),
        dropout=float(inter_section.dropout),
        act_fn=(None if inter_section.act_fn is None else str(inter_section.act_fn)),
        init_orth=bool(inter_section.init_orth),
    )
    # ------------------------
    # Rope parameters (optional)
    # ------------------------
    rope_params_cfg = cfg.model.get("rope_parameters", None)
    rope_parameters: Optional[configuration_olmoe.RopeParameters] = None

    if rope_params_cfg is not None:
        rope_parameters = configuration_olmoe.RopeParameters(
            rope_theta=float(rope_params_cfg.rope_theta),
            rope_type=(None if rope_params_cfg.rope_type is None else str(rope_params_cfg.rope_type)),
            factor=(None if rope_params_cfg.factor is None else float(rope_params_cfg.factor)),
            original_max_position_embeddings=(
                None
                if rope_params_cfg.original_max_position_embeddings is None
                else int(rope_params_cfg.original_max_position_embeddings)
            ),
            attention_factor=(
                None if rope_params_cfg.attention_factor is None else float(rope_params_cfg.attention_factor)
            ),
            beta_fast=(None if rope_params_cfg.beta_fast is None else float(rope_params_cfg.beta_fast)),
            beta_slow=(None if rope_params_cfg.beta_slow is None else float(rope_params_cfg.beta_slow)),
            short_factor=(None if rope_params_cfg.short_factor is None else list(rope_params_cfg.short_factor)),
            long_factor=(None if rope_params_cfg.long_factor is None else list(rope_params_cfg.long_factor)),
            low_freq_factor=(
                None if rope_params_cfg.low_freq_factor is None else float(rope_params_cfg.low_freq_factor)
            ),
            high_freq_factor=(
                None if rope_params_cfg.high_freq_factor is None else float(rope_params_cfg.high_freq_factor)
            ),
        )

    # ------------------------
    # Model / OlmoeInterventionsConfig
    # ------------------------
    model_section = cfg.model

    olmoe_config = configuration_olmoe.OlmoeInterventionsConfig(
        interventions_config=interventions_config_,
        vocab_size=int(model_section.vocab_size),
        hidden_size=int(model_section.hidden_size),
        intermediate_size=int(model_section.intermediate_size),
        num_hidden_layers=int(model_section.num_hidden_layers),
        num_attention_heads=int(model_section.num_attention_heads),
        num_key_value_heads=(
            None if model_section.num_key_value_heads is None else int(model_section.num_key_value_heads)
        ),
        hidden_act=str(model_section.hidden_act),
        max_position_embeddings=int(model_section.max_position_embeddings),
        initializer_range=float(model_section.initializer_range),
        rms_norm_eps=float(model_section.rms_norm_eps),
        use_cache=bool(model_section.use_cache),
        pad_token_id=int(model_section.pad_token_id),
        bos_token_id=(None if model_section.bos_token_id is None else int(model_section.bos_token_id)),
        eos_token_id=int(model_section.eos_token_id),
        tie_word_embeddings=bool(model_section.tie_word_embeddings),
        rope_theta=float(model_section.rope_theta),
        rope_scaling=model_section.get("rope_scaling", None),
        attention_bias=bool(model_section.attention_bias),
        rope_parameters=rope_parameters,
        attention_dropout=float(model_section.attention_dropout),
        clip_qkv=(None if model_section.clip_qkv is None else float(model_section.clip_qkv)),
        num_experts_per_tok=int(model_section.num_experts_per_tok),
        num_experts=int(model_section.num_experts),
        output_router_logits=bool(model_section.output_router_logits),
        router_aux_loss_coef=float(model_section.router_aux_loss_coef),
        norm_topk_prob=bool(model_section.norm_topk_prob),
    )

    return train_config, interventions_config_, olmoe_config


if __name__ == "__main__":
    print(load_all_configs("moe_reft/configs/olmoe.yaml"))
