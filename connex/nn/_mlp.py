from typing import Callable, Optional

import jax.nn as jnn
import jax.random as jr
import numpy as np

from .._network import NeuralNetwork
from ._utils import _identity


class MLP(NeuralNetwork):
    """
    A standard Multi-Layer Perceptron with constant layer width.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        depth: int,
        hidden_activation: Callable = jnn.silu,
        output_transformation: Callable = _identity,
        *,
        key: Optional[jr.PRNGKey] = None,
        **kwargs,
    ):
        """**Arguments**:

         - `input_size`: The number of neurons in the input layer.
         - `output_size`: The number of neurons in the output layer.
         - `width`: The number of neurons in each hidden layer.
         - `depth`: The number of hidden layers.
         - `hidden_activation`: The activation function applied element-wise to
            the hidden (i.e. non-input, non-output) neurons. It can itself be a
            trainable `equinox Module`.
        - `output_transformation`: The transformation applied group-wise to the
            output neurons, e.g. `jax.nn.softmax`. It can itself be a trainable
            `equinox.Module`.
         - `key`: The `PRNGKey` used to initialize parameters. Optional, keyword-only
             argument. Defaults to `jax.random.PRNGKey(0)`.
        """
        key = key if key is not None else jr.PRNGKey(0)
        num_neurons = width * depth + input_size + output_size
        input_neurons = np.arange(input_size, dtype=int)
        output_neurons_start = num_neurons - output_size
        output_neurons = np.arange(output_size, dtype=int) + output_neurons_start
        adjacency_dict = {}
        layer_sizes = [input_size] + ([width] * depth) + [output_size]
        neuron = 0
        for layer_size in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[layer_size], layer_sizes[layer_size + 1]
            row_idx = range(neuron, neuron + in_size)
            col_idx = range(neuron + in_size, neuron + in_size + out_size)
            for r in row_idx:
                adjacency_dict[r] = list(col_idx)
            neuron += in_size
        topo_sort = list(range(num_neurons))

        super().__init__(
            adjacency_dict,
            input_neurons,
            output_neurons,
            hidden_activation,
            output_transformation,
            topo_sort=topo_sort,
            key=key,
            **kwargs,
        )
