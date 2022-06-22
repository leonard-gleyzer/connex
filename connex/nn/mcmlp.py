from typing import Callable

import jax.numpy as jnp
import jax.nn as jnn

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
        activation: Callable=jnn.silu,
        output_activation: Callable=_identity,
        seed: int=0,
        **kwargs,
    ):
        """**Arguments**:

        - `input_size`: The number of neurons in the input layer.
        - `output_size`: The number of neurons in the output layer.
        - `width`: The number of neurons in each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function applied element-wise to the 
            hidden (i.e. non-input, non-output) neurons. It can itself be a 
            trainable equinox Module.
        - `output_activation`: The activation function applied element-wise to 
            the  output neurons. It can itself be a trainable equinox Module.
        - `seed`: The random seed used to initialize parameters.
        """
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
            activation,
            output_activation,
            seed=seed,
            **kwargs
        )