import torch
import pydantic
from typing import Literal, Self

# Types for interventions
InterventionLayers = Literal["all", "odd_only", "even_only", "alternate"]

# Intervention place
InterventionPlace = Literal["pre_moe", "after_moe"]

# Intervention Type
InterventionType = Literal["LoreftIntervention", "DireftIntervention"]

INTERVENTION_PATTERNS = [
    "*.pre_moe_intervention.*",
    "*.after_moe_intervention.*",
    "*.pre_moe_intervenetion.*",  # typo fallback
]


class InterventionsConfig(pydantic.BaseModel):
    intervention_type: InterventionType = "LoreftIntervention"
    intervention_layers: InterventionLayers = "all"
    intervention_places: InterventionPlace = "pre_moe"
    low_rank_dimension: int = 8
    dropout: float = 0.0
    act_fn: str | None = None  # e.g. "gelu", "relu", or None/"linear"
    init_orth: bool = True  # if your rotate layer supports it

    @pydantic.model_validator(mode="after")
    def validate_interventions_config(self) -> Self:
        assert self.low_rank_dimension >= 1, f"{self.low_rank_dimension=} cannot be less than 1"
        return self

    def to_json(self, **kwargs) -> str:
        """Convenient shortcut for JSON serialization."""
        return self.model_dump_json(indent=2, **kwargs)
