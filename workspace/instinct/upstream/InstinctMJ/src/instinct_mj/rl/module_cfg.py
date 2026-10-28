"""Instinct-RL module configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class InstinctRlParallelBlockCfg:
    """Configuration for an encoder parallel block."""

    class_name: str | None = None
    component_names: list[str] = field(default_factory=list)
    output_size: int | None = None
    takeout_input_components: bool = True


@dataclass
class InstinctRlMlpCfg(InstinctRlParallelBlockCfg):
    """Configuration for an MLP encoder block."""

    class_name: str = "MlpModel"
    hidden_sizes: list[int] = field(default_factory=list)
    nonlinearity: str | None = None


@dataclass
class InstinctRlConv2dHeadCfg(InstinctRlParallelBlockCfg):
    """Configuration for a Conv2d encoder block."""

    class_name: str = "Conv2dHeadModel"
    channels: list[int] = field(default_factory=list)
    kernel_sizes: list[int] = field(default_factory=list)
    strides: list[int] = field(default_factory=list)
    hidden_sizes: list[int] = field(default_factory=list)
    paddings: list[int] = field(default_factory=list)
    nonlinearity: str | None = None
    use_maxpool: bool = False


@dataclass
class InstinctRlTransformerHeadCfg(InstinctRlParallelBlockCfg):
    """Configuration for a Transformer encoder block."""

    class_name: str = "TransformerHeadModel"
    num_heads: int = 4
    num_layers: int = 1
    d_model: int = 256
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "relu"
    nonlinearity: str = "ReLU"
    layer_norm_eps: float = 1.0e-5
    batch_first: bool = True
    norm_first: bool = False
    mask_from_input_dim: int = -1
    output_selection: Literal["maxpool", "smallest_positive", "smallest_nonnegative"] = "maxpool"
    input_hidden_sizes: list[int] = field(default_factory=list)
    output_hidden_sizes: list[int] = field(default_factory=list)


__all__ = [
    "InstinctRlParallelBlockCfg",
    "InstinctRlMlpCfg",
    "InstinctRlConv2dHeadCfg",
    "InstinctRlTransformerHeadCfg",
]
