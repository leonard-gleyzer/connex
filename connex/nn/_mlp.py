from collections.abc import Callable, Sequence

import jax.nn as jnn
import jax.random as jr
import numpy as np
from jax import Array

from .. import ops as cnx_ops
from .._model import NeuralDAG
from .._spec import DropoutLike, GraphSpec
from ._utils import _identity


class MLP(NeuralDAG):
    """A standard multi-layer perceptron represented as a Connex DAG."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        depth: int,
        *,
        activation: Callable = jnn.gelu,
        output_transform: Callable = _identity,
        dropout: DropoutLike = 0.0,
        ops: Sequence[cnx_ops.Op] | None = None,
        key: Array | None = None,
    ):
        spec = GraphSpec(
            _mlp_adjacency(input_size, output_size, width, depth),
            inputs=np.arange(input_size, dtype=int),
            outputs=np.arange(output_size, dtype=int)
            + (width * depth + input_size),
            topo_sort=range(width * depth + input_size + output_size),
            dropout=dropout,
        )
        ops = (
            cnx_ops.default_ops(
                activation=activation,
                output_transform=output_transform,
            )
            if ops is None
            else tuple(ops)
        )
        super().__init__(spec, ops=ops, key=jr.key(0) if key is None else key)


def _mlp_adjacency(
    input_size: int, output_size: int, width: int, depth: int
) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {}
    layer_sizes = [input_size] + ([width] * depth) + [output_size]
    neuron = 0
    for layer_index in range(len(layer_sizes) - 1):
        in_size = layer_sizes[layer_index]
        out_size = layer_sizes[layer_index + 1]
        sources = range(neuron, neuron + in_size)
        targets = range(neuron + in_size, neuron + in_size + out_size)
        for source in sources:
            adjacency[source] = list(targets)
        neuron += in_size
    return adjacency
