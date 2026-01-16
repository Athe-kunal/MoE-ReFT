import torch
from collections import OrderedDict

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from transformers.activations import ACT2FN


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m, dtype=torch.float32), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x, self.weight)


class LoreftIntervention(SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float32
        )
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = (
            ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Overwrite for data-efficiency.
        Properly integrates with PyTorch's hierarchical state_dict mechanism.
        """
        if destination is None:
            destination = OrderedDict()

        # Add learned_source parameters with proper prefix
        for k, v in self.learned_source.state_dict(keep_vars=keep_vars).items():
            destination[prefix + "learned_source." + k] = v

        # Add rotate_layer weight
        if keep_vars:
            destination[prefix + "rotate_layer"] = self.rotate_layer.weight
        else:
            destination[prefix + "rotate_layer"] = self.rotate_layer.weight.data

        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Overwrite for data-efficiency.
        Handles both new format (learned_source.weight) and old format (weight) for backward compatibility.
        """
        # Build learned_source state dict - handle both old and new key formats
        learned_source_sd = {}
        for k, v in state_dict.items():
            if k.startswith("learned_source."):
                # New format: learned_source.weight -> weight
                learned_source_sd[k[len("learned_source.") :]] = v
            elif k in ("weight", "bias"):
                # Old format: direct keys
                learned_source_sd[k] = v

        if learned_source_sd:
            self.learned_source.load_state_dict(learned_source_sd, strict=False)

        # Load rotate_layer weight
        rotate_layer_key = "rotate_layer"
        if rotate_layer_key not in state_dict:
            if strict:
                raise KeyError(f"Missing key: {rotate_layer_key}")
            return

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict[rotate_layer_key].to(self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:, :overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True  # we must match!


class NoreftIntervention(SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.float32)
        self.learned_source = torch.nn.Linear(self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float32
        )
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = (
            ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base, source=None, subspaces=None):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul((self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight)
        return self.dropout(output)


class ConsreftIntervention(SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)

    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul((self.learned_source - rotated_base), self.rotate_layer.weight.T)
        return output


class LobireftIntervention(SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention):
    """
    LobiReFT(h) = h + R^T(b)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)

    def forward(self, base, source=None, subspaces=None):
        output = base + torch.matmul(self.learned_source, self.rotate_layer.weight.T)
        return self.dropout(output)


class DireftIntervention(SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention):
    """
    DiReFT(h) = h + R^T(Wh + b)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float32
        )
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = (
            ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base, source=None, subspaces=None):
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)),
            self.rotate_layer.weight.T,
        )
        return self.dropout(output)


class NodireftIntervention(SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.float32)
        self.learned_source = torch.nn.Linear(self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float32
        )
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = (
            ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base, source=None, subspaces=None):
        output = base + torch.matmul(self.act_fn(self.learned_source(base)), self.proj_layer.weight)
        return self.dropout(output)
