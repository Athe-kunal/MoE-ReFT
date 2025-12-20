import pydantic
from typing import Annotated, Literal, Self

# Types for interventions (keep for type hints elsewhere)
InterventionLayers = Literal["all", "odd_only", "even_only", "alternate"]
InterventionPlace = Literal["pre_moe", "after_moe"]
InterventionType = Literal["LoreftIntervention", "DireftIntervention"]

INTERVENTION_PATTERNS = [
    "*.pre_moe_intervention.*",
    "*.after_moe_intervention.*",
]


class InterventionsConfig(pydantic.BaseModel):
    """Configuration settings controlling REFT / DIREFT interventions applied to MoE layers."""

    intervention_type: Annotated[
        Literal["LoreftIntervention", "DireftIntervention"],
        pydantic.Field(
        description=(
            "Specifies which intervention module to inject. "
            "'LoreftIntervention' applies low-rank feature transformations "
            "(LoReFT-style REFT), while 'DireftIntervention' applies "
            "directional interventions (DiReFT) using directional vectors."
        ),
        ),
    ]

    intervention_layers: Annotated[
        Literal["all", "odd_only", "even_only", "alternate"],
        pydantic.Field(
        description=(
            "Controls which transformer layers receive interventions:\n"
            "- 'all': apply to every layer\n"
            "- 'odd_only': apply only to odd-numbered layers\n"
            "- 'even_only': apply only to even-numbered layers\n"
            "- 'alternate': apply in alternating pattern depending on index\n"
            "Useful for ablations and reducing compute overhead."
        ),
        ),
    ]

    intervention_places: Annotated[
        Literal["pre_moe", "after_moe"],
        pydantic.Field(
        description=(
            "Where to inject the intervention relative to the MoE block:\n"
            "- 'pre_moe': before the MoE router + expert blocks\n"
            "- 'after_moe': after the expert outputs are combined\n"
            "Choosing pre/post allows control over how interventions influence routing or expert mixing."
        ),
        ),
    ]

    low_rank_dimension: Annotated[
        int,
        pydantic.Field(description="Rank of the low-rank projection used for intervention layers."),
    ]

    dropout: Annotated[
        float,
        pydantic.Field(description="Dropout probability applied inside the intervention layer. Useful for regularization"),
    ]

    act_fn: Annotated[
        str | None,
        pydantic.Field(
        description=(
            "Optional activation function used inside the intervention module. "
                "Examples: 'gelu', 'relu', 'silu'. If None, the intervention is linear."
            ),
        ),
    ]

    init_orth: Annotated[
        bool,
        pydantic.Field(
        default=True,
            description="Whether to orthogonally initialize the low-rank projection matrices.",
        ),
    ]

    @pydantic.model_validator(mode="after")
    def validate_interventions_config(self) -> Self:
        assert self.low_rank_dimension >= 1, f"{self.low_rank_dimension=} cannot be less than 1"
        return self

    def to_json(self, **kwargs) -> str:
        """Convenient shortcut for JSON serialization."""
        return self.model_dump_json(indent=2, **kwargs)
