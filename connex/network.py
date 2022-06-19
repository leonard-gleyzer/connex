from typing import Callable, Mapping, Optional, Sequence, Union

import equinox.experimental as eqxe
from equinox import Module, filter_jit, static_field

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax

import numpy as np

from .custom_types import Array
from .utils import keygen, _adjacency_dict_to_matrix, _identity


class NeuralNetwork(Module):
    """
    A neural network whose structure is primarily specified by an adjecency matrix
    representing a directed acyclic graph (DAG) and sequences of ints specifying
    which neurons are input and output neurons. Create your model by inheriting from
    this.
    """
    parameter_matrix: jnp.array
    activation: Callable
    output_activation: Callable
    topo_batches: Sequence[jnp.array] = static_field()
    adjacency_dict: Mapping[int, Sequence[int]] = static_field()
    adjacency_matrix: jnp.array = static_field()
    input_neurons: jnp.array = static_field()
    output_neurons: jnp.array = static_field()
    num_neurons: int = static_field()
    dropout_p: eqxe.StateIndex = static_field()
    seed: int = static_field()

    def __init__(
        self,
        num_neurons: int,
        adjacency_dict: Mapping[int, Sequence[int]],
        input_neurons: Sequence[int],
        output_neurons: Sequence[int],
        activation: Callable = jnn.silu,
        output_activation: Callable = _identity,
        dropout_p: Union[float, Sequence[float]] = 0.,
        seed: int = 0,
        parameter_matrix: Optional[Array] = None,
        **kwargs,
    ):
        """**Arguments:**

        - `num_neurons`: The number of neurons in the network.
        - `adjacency_dict`: A dictionary that maps a neuron id to the ids of its
            outputs. Neurons must be ordered from `0` to `num_neurons - 1`. Neurons
            with no outgoing connections do not need to be included.
        - `input_neurons`: A sequence of `int` indicating the ids of the 
            input neurons. The order here matters, as the input data will be
            passed into the input neurons in the order passed in here.
        - `output_neurons`: A sequence of `int` indicating the ids of the 
            output neurons. The order here matters, as the output data will be
            read from the output neurons in the order passed in here.
        - `activation`: The activation function applied element-wise to the 
            hidden (i.e. non-input, non-output) neurons. It can itself be a 
            trainable `equinox.Module`.
        - `output_activation`: The activation function applied element-wise to 
            the  output neurons. It can itself be a trainable `equinox.Module`.
        - `dropout_p`: Dropout probability. If a single `float`, the same dropout
            probability will be applied to all hidden neurons. If a `Sequence[float]`,
            the sequence must have length `num_neurons`, where `dropout_p[i]` is the
            dropout probability for neuron `i`. Note that this allows dropout to be 
            applied to input and output neurons as well.
        - `seed`: The random seed used to initialize parameters.
        - `parameter_matrix`: A `jnp.array` of shape `(N, N + 1)` where entry `[i, j]` is 
            neuron `i`'s weight for neuron `j`, and entry `[i, N]` is neuron `i`'s bias. 
            Optional argument -- used primarily for plasticity functionality.
        """
        super().__init__(**kwargs)
        input_neurons = jnp.array(input_neurons, dtype=int)
        output_neurons = jnp.array(output_neurons, dtype=int)
        self._check_input(num_neurons, adjacency_dict, input_neurons, output_neurons)
        self.adjacency_dict = adjacency_dict
        adjacency_matrix = _adjacency_dict_to_matrix(num_neurons, adjacency_dict)
        self.topo_batches = self._topological_batching(adjacency_matrix)
        self.adjacency_matrix = adjacency_matrix
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.num_neurons = num_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.seed = seed

        # Set dropout probabilities. We use eqxe.StateIndex here so that
        # dropout probabilities can later be modified, if desired.
        self.dropout_p = eqxe.StateIndex()
        self.set_dropout_p(dropout_p)

        # Set parameters
        if parameter_matrix is None:
            # Initialize parameters as being drawn iid ~ N(0, 0.01)
            parameter_matrix = jr.normal(
                jr.PRNGKey(seed),
                (self.num_neurons, self.num_neurons + 1)
            ) * 0.1
        shape = parameter_matrix.shape
        assert len(shape) == 2
        assert shape[0] == self.num_neurons
        assert shape[1] == self.num_neurons + 1
        self.parameter_matrix = parameter_matrix


    @filter_jit
    def __call__(self, x: Array) -> Array:
        """The forward pass of the network. Neurons are "fired" in topological batch
        order (see https://arxiv.org/pdf/1911.06904.pdf), with `jax.vmap` 
        vectorization used within each topological batch.
        
        **Arguments**:
        
        - `x`: The input array to the network for the forward pass. The individual
            values will be written to the input neurons in the order passed in during
            initialization.

        **Returns**:

        The result array from the forward pass. The order of the array elements will be
        the order of the output neurons passed in during initialization.
        """
        # Set input values
        values = jnp.zeros((self.num_neurons,)).at[self.input_neurons].set(x)

        # Function to parallelize neurons within a topological batch
        fire_neurons = vmap(self._fire_neuron, in_axes=[0, None, 0])

        # Forward pass in topological batch order
        for tb in self.topo_batches:
            keys = keygen(n_keys=jnp.size(tb))
            output_values = fire_neurons(tb, values, keys)
            values = values.at[tb].set(output_values)

        # Return values pertaining to output neurons
        return values[self.output_neurons]


    def _fire_neuron(
        self, 
        id: int, 
        neuron_values: jnp.array, 
        key: jr.PRNGKey,
    ) -> float:
        """Compute the output of neuron `id`."""
        def _affine_transformation() -> jnp.array:
            # Get parameters
            parameters = self.parameter_matrix[id]

            # Use the respective column of the adjacency matrix as a binary mask
            # so vmap can work with neurons with different numbers of inputs
            inputs = neuron_values * self.adjacency_matrix[:, id]

            # Apply weights and bias
            affine_transformation = jnp.dot(parameters, jnp.append(inputs, 1))

            # Add a dimension in case the activation functions are themselves
            # neural networks (i.e. equinox Modules)
            return jnp.expand_dims(affine_transformation, 0)

        rand = jr.uniform(key, minval=0, maxval=1)
        dropout_p = self.get_dropout_p()
        use_dropout = jnp.less_equal(rand, dropout_p[id])

        def exec_if_dropout() -> float:
            return 0.

        def exec_if_not_dropout() -> float:

            def exec_if_input_neuron() -> float:
                return neuron_values[id]

            def exec_if_not_input_neuron() -> float:
                affine_transformation = _affine_transformation()
                output = lax.cond(
                    jnp.isin(id, self.output_neurons),
                    lambda: self.output_activation(affine_transformation),
                    lambda: self.activation(affine_transformation)
                )
                return jnp.squeeze(output)

            return lax.cond(
                jnp.isin(id, self.input_neurons),
                exec_if_input_neuron,
                exec_if_not_input_neuron
            )
        
        return lax.cond(
            use_dropout,
            exec_if_dropout,
            exec_if_not_dropout
        )


    def _check_input(
        self,
        num_neurons: int, 
        adjacency_dict: Mapping[int, Sequence[int]],
        input_neurons: jnp.array,
        output_neurons: jnp.array,
    ) -> None:
        """
        Check to make sure the input neurons, output neurons,
        and adjacency dict are valid.
        """
        # Check that the input and output neurons are 1D and nonempty
        assert len(input_neurons.shape) == 1 and len(input_neurons) > 0
        assert len(output_neurons.shape) == 1 and len(output_neurons) > 0

        # Check that the input and output neurons are disjoint
        assert len(jnp.intersect1d(input_neurons, output_neurons)) == 0

        # Check that neuron ids are in the range [0, num_neurons)
        # and that ids are not repeated
        for input, outputs in adjacency_dict.items():
            assert 0 <= input < num_neurons
            if outputs:
                assert len(set(outputs)) == len(outputs)
                assert min(outputs) >= 0 and max(outputs) < num_neurons


    def _topological_batching(self, adjacency_matrix: jnp.array,
    ) -> Sequence[jnp.array]:
        """
        Topologically sort/batch neurons and also check that 
        the network is acyclic.
        """
        inputs_per_neuron = np.sum(adjacency_matrix, axis=0)
        neurons_with_no_input = np.argwhere(inputs_per_neuron == 0).flatten()
        queue = list(neurons_with_no_input)
        adjacency_matrix = np.copy(adjacency_matrix)
        topo_batches = [list(neurons_with_no_input)]
        while queue:
            neuron = queue.pop(0)
            outputs = np.argwhere(adjacency_matrix[neuron]).flatten()
            adjacency_matrix[neuron, outputs] = 0
            topo_batch = []
            for output in outputs:
                if np.all(adjacency_matrix[:, output] == 0):
                    queue.append(output)
                    topo_batch.append(output)
            topo_batches.append(topo_batch)
        topo_batches = [jnp.array(b) for b in topo_batches if b]

        # Check that the graph is acyclic
        row_sums = np.sum(adjacency_matrix, axis=1)
        col_sums = np.sum(adjacency_matrix, axis=0)
        bad_in_neurons = set(np.argwhere(col_sums).flatten())
        bad_out_neurons = set(np.argwhere(row_sums).flatten())
        union = bad_in_neurons | bad_out_neurons
        assert len(union) == 0, f'Cycle(s) found involving neurons {union}'

        return topo_batches


    def get_dropout_p(self) -> Array:
        """Get the per-neuron dropout probabilities.
        
        **Returns**:

        A 1D `jnp.array` with shape `((num_neurons,))` where element `i` 
        is the dropout probability of neuron `i`.
        """
        dropout_p = eqxe.get_state(
            self.dropout_p, 
            jnp.arange(self.num_neurons, dtype=float)
        )
        return dropout_p


    def set_dropout_p(self, dropout_p: Union[float, Sequence[float]]) -> None:
        """Set the per-neuron dropout probabilities.
        
        **Arguments**:

        - `dropout_p`: Dropout probability. If a single `float`, the same dropout
            probability will be applied to all hidden neurons. If a `Sequence[float]`,
            the sequence must have length `num_neurons`, where `dropout_p[i]` is the
            dropout probability for neuron `i`. Note that this allows dropout to be 
            applied to input and output neurons as well.
        """
        if isinstance(dropout_p, int):
            # if user enters 0 or 1
            dropout_p = float(dropout_p)
        if isinstance(dropout_p, float):
            dropout_p = jnp.ones((self.num_neurons,)) * dropout_p
            dropout_p = dropout_p.at[self.input_neurons].set(0.)
            dropout_p = dropout_p.at[self.output_neurons].set(0.)
        else:
            assert len(dropout_p) == self.num_neurons
            dropout_p = jnp.array(dropout_p, dtype=float)
        assert jnp.all(jnp.greater_equal(dropout_p, 0))
        assert jnp.all(jnp.less_equal(dropout_p, 1))
        eqxe.set_state(self.dropout_p, dropout_p)