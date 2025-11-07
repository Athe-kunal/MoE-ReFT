from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

import fnmatch
import torch
from loguru import logger
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel

map_dtype = torch.bfloat16  # optional casting
map_device = torch.device("cuda")  # optional device move

# hf_model_name_or_path="allenai/OLMoE-1B-7B-0125-Instruct"
hf_model_name_or_path = "allenai/OLMoE-1B-7B-0924-Instruct"

hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    hf_model_name_or_path,
    dtype=map_dtype if map_dtype is not None else None,
    trust_remote_code=True,
)

src_sd = hf_model.state_dict()
