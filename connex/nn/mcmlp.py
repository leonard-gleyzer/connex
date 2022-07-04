from typing import Callable, Optional

import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr

from .. import NeuralNetwork
from ..utils import _identity


class MCMLP(NeuralNetwork):
    """
    A "Maximally-Connected Multi-Layer Perceptron". Like a standard MLP, but
    every neuron is connected to every other neuron in all later layers, rather
    than only the next layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        depth: int,
        hidden_activation: Callable = jnn.silu,
        output_activation_elem: Callable = _identity,
        output_activation_group: Callable = _identity,
        *,
        key: Optional[jr.PRNGKey] = None,
        **kwargs,
    ):
        """**Arguments**:

        - `input_size`: The number of neurons in the input layer.
        - `output_size`: The number of neurons in the output layer.
        - `width`: The number of neurons in each hidden layer.
        - `depth`: The number of hidden layers.
        - `hidden_activation`: The activation function applied element-wise to the 
            hidden (i.e. non-input, non-output) neurons. It can itself be a 
            trainable equinox Module.
        - `output_activation_elem`: The activation function applied element-wise to 
            the  output neurons. It can itself be a trainable `equinox.Module`.
        - `output_activation_group`: The activation function applied to the output 
            neurons as a whole after applying `output_activation_elem` element-wise, 
            e.g. `jax.nn.softmax`. It can itself be a trainable `equinox.Module`.
            the  output neurons. It can itself be a trainable equinox Module.
        - `key`: The `PRNGKey` used to initialize parameters. Optional, keyword-only 
            argument. Defaults to `jax.random.PRNGKey(0)`.
        """
        key = key if key is not None else jr.PRNGKey(0)
        num_neurons = width * depth + input_size + output_size
        input_neurons = jnp.arange(input_size)
        output_neurons_start = num_neurons - output_size
        output_neurons = jnp.arange(output_size) + output_neurons_start
        adjacency_dict = {}
        layer_sizes = [input_size] + ([width] * depth) + [output_size]
        neuron = 0
        for layer_size in layer_sizes[:-1]:
            row_idx = range(neuron, neuron + layer_size)
            col_idx = range(neuron + layer_size, num_neurons)
            for r in row_idx:
                adjacency_dict[r] = list(col_idx)
            neuron += layer_size

        super().__init__(
            num_neurons,
            adjacency_dict,
            input_neurons,
            output_neurons,
            hidden_activation,
            output_activation_elem,
            output_activation_group,
            key=key,
            **kwargs
        )