from typing import Callable, Mapping, Optional, Sequence, Union

import equinox.experimental as eqxe
from equinox import Module, filter_jit, static_field

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

import networkx as nx
import numpy as np

from custom_types import Array
from utils import _identity, _invert_dict


class NeuralNetwork(Module):
    """
    A neural network whose structure is primarily specified by an adjecency dict
    representing a directed acyclic graph (DAG) and sequences of ints specifying
    which neurons are input and output neurons. Create your model by inheriting from
    this.
    """
    weights: Sequence[jnp.array]
    bias: jnp.array
    hidden_activation: Callable
    output_activation_elem: Callable
    output_activation_group: Callable
    hidden_activation_: Callable
    output_activation_elem_: Callable
    adjacency_dict: Mapping[int, Sequence[int]] = static_field()
    adjacency_dict_inv: Mapping[int, Sequence[int]] = static_field()
    topo_batches: Sequence[jnp.array] = static_field()
    topo_sort_inv: jnp.array = static_field()
    masks: Sequence[jnp.array] = static_field()
    idxs: Sequence[jnp.array] = static_field()
    input_neurons: jnp.array = static_field()
    output_neurons: jnp.array = static_field()
    num_neurons: int = static_field()
    dropout_p: eqxe.StateIndex = static_field()
    key: eqxe.StateIndex = static_field()

    def __init__(
        self,
        num_neurons: int,
        adjacency_dict: Mapping[int, Sequence[int]],
        input_neurons: Sequence[int],
        output_neurons: Sequence[int],
        hidden_activation: Callable = jnn.silu,
        output_activation_elem: Callable = _identity,
        output_activation_group: Callable = _identity,
        dropout_p: Union[float, Sequence[float]] = 0.,
        *,
        key: Optional[jr.PRNGKey] = None,
        **kwargs,
    ):
        """**Arguments**:

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
        - `hidden_activation`: The activation function applied element-wise to the 
            hidden (i.e. non-input, non-output) neurons. It can itself be a 
            trainable `equinox.Module`.
        - `output_activation_elem`: The activation function applied element-wise to 
            the  output neurons. It can itself be a trainable `equinox.Module`.
        - `output_activation_group`: The activation function applied to the output 
            neurons as a whole after applying `output_activation_elem` element-wise, 
            e.g. `jax.nn.softmax`. It can itself be a trainable `equinox.Module`.
        - `dropout_p`: Dropout probability. If a single `float`, the same dropout
            probability will be applied to all hidden neurons. If a `Sequence[float]`,
            the sequence must have length `num_neurons`, where `dropout_p[i]` is the
            dropout probability for neuron `i`. Note that this allows dropout to be 
            applied to input and output neurons as well.
        - `key`: The `jax.random.PRNGKey` used for parameter initialization and dropout. 
            Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.
        """
        super().__init__(**kwargs)
        for n in range(num_neurons):
            if n not in adjacency_dict:
                adjacency_dict[n] = []
        adjacency_dict_inv = _invert_dict(adjacency_dict)
        self._check_input(
            num_neurons, adjacency_dict, adjacency_dict_inv, input_neurons, output_neurons
        )
        self.adjacency_dict = adjacency_dict
        self.adjacency_dict_inv = adjacency_dict_inv
        self.input_neurons = jnp.array(input_neurons, dtype=int)
        self.output_neurons = jnp.array(output_neurons, dtype=int)
        self.num_neurons = num_neurons

        # Activations may themselves be `eqx.Module`s, so we do this to ensure
        # that both `Module` and non-`Module` activations work with the same
        # input shape. The copies (hidden_activation_ and output_activation_elem_) 
        # are for plasticity functionality.
        self.hidden_activation = hidden_activation \
            if isinstance(hidden_activation, Module) else vmap(hidden_activation)
        self.output_activation_elem = output_activation_elem \
            if isinstance(output_activation_elem, Module) else vmap(output_activation_elem)
        self.hidden_activation_ = hidden_activation
        self.output_activation_elem_ = output_activation_elem
        self.output_activation_group = output_activation_group

        # Set dropout probabilities. We use `eqxe.StateIndex`` here so that
        # dropout probabilities can later be modified, if desired.
        self.dropout_p = eqxe.StateIndex()
        self.set_dropout_p(dropout_p)

        # Get topological information.
        topo_batches = self._topological_batching()
        topo_sort = topo_batches[0]
        for tb in topo_batches[1:]:
            topo_sort = np.append(topo_sort, tb)
        # Maps a neuron id to its position in the topological sort.
        self.topo_sort_inv = jnp.array(np.argsort(topo_sort), dtype=int)
        self.topo_batches = [jnp.array(tb, dtype=int) for tb in topo_batches[1:]]
        num_topo_batches = len(self.topo_batches)

        # Here, `mins[i]` (`maxs[i]`) is the index representing the minimum
        # (maximum) position -- with respect to the topological order -- of
        # those neurons strictly necessary to process `self.topo_batches[i]`
        # from the previous topological batch. If `i == 0`, the previous
        # topological batch is the input neurons.
        mins = np.array([])
        maxs = np.array([])
        for tb in self.topo_batches:
            input_locs = [jnp.array(adjacency_dict_inv[int(i)], dtype=int) for i in tb]
            input_locs_topo = [self.topo_sort_inv[loc] for loc in input_locs]
            mins_ = np.array([np.amin(locs) for locs in input_locs_topo])
            maxs_ = np.array([np.amax(locs) for locs in input_locs_topo])
            mins = np.append(mins, np.amin(mins_))
            maxs = np.append(maxs, np.amax(maxs_))

        # Set the random key. We use eqxe.StateIndex here so that the key 
        # can be updated after every forward pass. This ensures that the
        # random values generated for determining dropout application
        # are different for each forward pass.
        self.key = eqxe.StateIndex()
        key = key if key is not None else jr.PRNGKey(0)
        *wkeys, bkey, dkey = jr.split(key, num_topo_batches + 2)
        eqxe.set_state(self.key, dkey)

        # Set the network parameters. Here, `self.bias` is a `jnp.array` of shape `(num_neurons,)`,
        # where `self.bias[i]` is the bias of neuron `i + num_input_neurons`. `self.weights`,
        # however, is a list of 2D `jnp.array`s, where `self.weights[i]` includes the weights used 
        # by the neurons in `self.topo_batches[i]`. More specifically, `self.weights[i][j, k]` is 
        # the weight of the connection from the neuron with topological index `k + mins[i]`, i.e. 
        # neuron `topo_sort[k + mins[i]]`, to neuron `self.topo_batches[i][j]`. The weights are stored 
        # this way in order to use minimal memory while allowing for maximal `vmap` parallelism during 
        # the forward pass, since the minimum and maximum neurons needed to process a topological batch 
        # in parallel will be closest together when in topological order.
        weight_lengths = np.array(maxs - mins, dtype=int) + 1
        self.weights = [
            jr.normal(
                wkeys[i], (jnp.size(self.topo_batches[i]), weight_lengths[i])
            ) * 0.1
            for i in range(num_topo_batches)
        ]
        self.bias = jr.normal(bkey, (num_neurons,)) * 0.1

        # Here, `self.masks` is a list of 2D binary `jnp.array` with identical structure
        # to `self.weights`. These are multiplied by the weights during the forward pass
        # to mask out weights for connections that are not present in the actual network.
        masks = []
        for (tb, weights, min_) in zip(self.topo_batches, self.weights, mins):
            mask = np.zeros_like(weights)
            for (i, neuron) in enumerate(tb):
                inputs = jnp.array(adjacency_dict_inv[int(neuron)], dtype=int)
                inputs_topo = self.topo_sort_inv[inputs] - int(min_)
                mask[i, inputs_topo] = 1
            masks.append(jnp.array(mask, dtype=int))
        self.masks = masks

        # Here, `self.idxs[i]` includes the indices -- in topological order -- of the 
        # neurons needed to process `self.topo_batches[i]`. This is done for the same
        # memory/parallelism reason as the structure of `self.weights`.
        self.idxs = [
            jnp.arange(mins[i], maxs[i] + 1, dtype=int) for i in range(num_topo_batches)
        ]


    @filter_jit
    def __call__(
        self, x: Array, *, key: Optional[jr.PRNGKey] = None,
    ) -> Array:
        """The forward pass of the network. Neurons are "fired" in topological batch
        order (see Section 2.2 of https://arxiv.org/pdf/2101.07965.pdf), with `jax.vmap` 
        vectorization used within each topological batch.
        
        **Arguments**:
        
        - `x`: The input array to the network for the forward pass. The individual
            values will be written to the input neurons in the order passed in during
            initialization.
        - `key`: A `jax.random.PRNGKey` used for dropout. Optional, keyword-only argument.
            If `None`, a key will be generated by getting the key in `self.key` (which is
            an `eqxe.StateIndex`) and then splitting and updating the key for the next
            forward pass.

        **Returns**:

        The result array from the forward pass. The order of the array elements will be
        the order of the output neurons passed in during initialization.
        """
        # Neuron value array, updated as neurons are "fired".
        values = jnp.zeros((self.num_neurons,))

        # Dropout.
        if key is None:
            key = eqxe.get_state(self.key, jr.PRNGKey(0))
            _, new_key = jr.split(key)
            eqxe.set_state(self.key, new_key)
        dropout_p = self.get_dropout_p()
        rand = jr.uniform(key, dropout_p.shape, minval=0, maxval=1)
        dropout_keep = jnp.greater(rand, dropout_p)
        
        # Set input values.
        input_neurons_topo = self.topo_sort_inv[self.input_neurons]
        values = values.at[input_neurons_topo].set(
            x * dropout_keep[self.input_neurons]
        )

        # Function to apply the activation for a single neuron.
        def _apply_activation(id: int, affine: jnp.array) -> float:
            affine = jnp.expand_dims(affine, 0)
            return jnp.squeeze(lax.cond(
                jnp.isin(id, self.output_neurons),
                lambda: self.output_activation_elem(affine),
                lambda: self.hidden_activation(affine)
            ))

        # Forward pass in topological batch order.
        for (tb, weights, mask, idx) in zip(self.topo_batches, self.weights, self.masks, self.idxs):
            # Affine transformation, wx + b.
            affine = vmap(jnp.dot, in_axes=[0, None])(weights * mask, values[idx]) + self.bias[tb]
            # Apply activations/dropout.
            output_values = vmap(_apply_activation)(tb, affine) * dropout_keep[tb]
            # Set new values.
            values = values.at[self.topo_sort_inv[tb]].set(output_values)

        # Return values pertaining to output neurons, with the group-wise 
        # output activation applied.
        output_neurons_topo = self.topo_sort_inv[self.output_neurons]
        return self.output_activation_group(values[output_neurons_topo])


    def _check_input(
        self,
        num_neurons: int, 
        adjacency_dict: Mapping[int, Sequence[int]],
        adjacency_dict_inv: Mapping[int, Sequence[int]],
        input_neurons: Sequence[int],
        output_neurons: Sequence[int],
    ) -> None:
        """
        Check to make sure the input neurons, output neurons,
        and adjacency dict are valid.
        """
        input_neurons = np.array(input_neurons, dtype=int)
        output_neurons = np.array(output_neurons, dtype=int)

        # Check that the input and output neurons are 1D and nonempty.
        assert np.size(input_neurons.shape) == 1 and np.size(input_neurons) > 0
        assert np.size(output_neurons.shape) == 1 and np.size(output_neurons) > 0

        # Check that the input and output neurons are disjoint.
        assert np.size(np.intersect1d(input_neurons, output_neurons)) == 0

        # Check that input neurons and neurons with no input are equivalent.
        neurons_with_no_input = np.array([i for i in adjacency_dict_inv if not adjacency_dict_inv[i]])
        assert np.size(np.setdiff1d(neurons_with_no_input, input_neurons)) == 0

        # Check that output neurons and neurons with no output are equivalent.
        neurons_with_no_output = np.array([i for i in adjacency_dict if not adjacency_dict[i]])
        assert np.size(np.setdiff1d(neurons_with_no_output, output_neurons)) == 0

        # Check that neuron ids are in the range [0, num_neurons)
        # and that ids are not repeated.
        for input, outputs in adjacency_dict.items():
            assert 0 <= input < num_neurons
            if outputs:
                assert len(set(outputs)) == len(outputs)
                assert min(outputs) >= 0 and max(outputs) < num_neurons


    def _topological_batching(self) -> Sequence[np.array]:
        """
        Topologically sort/batch neurons;
        see Section 2.2 of https://arxiv.org/pdf/2101.07965.pdf.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(self.adjacency_dict)))
        for (neuron, outputs) in self.adjacency_dict.items():
            for output in outputs:
                graph.add_edge(neuron, output)
        topo_sort = nx.topological_sort(graph)
        topo_batches = []
        topo_batch = []
        nodes_to_remove = []
        for node in topo_sort:
            if graph.in_degree(node) == 0:
                topo_batch.append(node)
                nodes_to_remove.append(node)
            else:
                topo_batches.append(np.array(topo_batch, dtype=int))
                graph.remove_nodes_from(nodes_to_remove)
                topo_batch = [node]
                nodes_to_remove = [node]
        topo_batches.append(np.array(topo_batch, dtype=int))

        return topo_batches


    def get_dropout_p(self) -> Array:
        """Get the per-neuron dropout probabilities.
        
        **Returns**:

        A `jnp.array` with shape `(num_neurons,)` where element `i` 
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


    def to_networkx_graph(self) -> nx.DiGraph:
        """NetworkX (https://networkx.org) is a popular Python library for network 
        analysis. This function returns an instance of NetworkX's directed graph object 
        `networkx.DiGraph` that represents the structure of the neural network. This may be 
        useful for analyzing and/or debugging the connectivity structure of the network.

        **Returns**:

        A `networkx.DiGraph` object that represents the structure of the network, where
        the neurons are nodes with the same numbering. 
        
        The nodes have the following field(s):

        - `group`: One of {`'input'`, `'hidden'`, `'output'`} (a string).
        - `bias`: The corresponding neuron's bias (a float).

        The edges have the following field(s):

        - `weight`: The corresponding network weight (a float).
        """            
        graph = nx.DiGraph()

        for id in range(self.num_neurons):
            if jnp.isin(id, self.input_neurons):
                group = 'input'
                neuron_bias = None
            elif jnp.isin(id, self.output_neurons):
                group = 'output'
                neuron_bias = self.parameter_matrix[id, -1] 
            else:
                group = 'hidden'
                neuron_bias = self.parameter_matrix[id, -1] 
            graph.add_node(id, group=group, bias=neuron_bias)
            
        for (neuron, outputs) in self.adjacency_dict.items():
            for output in outputs:
                weight = self.parameter_matrix[output, neuron]
                graph.add_edge(neuron, output, weight=weight)

        return graph