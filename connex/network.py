from typing import Callable, Hashable, Mapping, Optional, Sequence, Tuple, Union

import equinox.experimental as eqxe
from equinox import Module, filter_jit, static_field

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

import networkx as nx
import numpy as np

from custom_types import Array
from utils import _identity, _invert_dict, _nx_digraph_to_adjacency_dict


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
    graph: nx.DiGraph = static_field()
    adjacency_dict: Mapping[int, Sequence[int]] = static_field()
    adjacency_dict_inv: Mapping[int, Sequence[int]] = static_field()
    neuron_to_id: Mapping[Hashable, int] = static_field()
    topo_batches: Sequence[jnp.array] = static_field()
    neuron_to_topo_batch_idx: Sequence[Tuple[int, int]] = static_field()
    topo_sort: np.array = static_field()
    mins: np.array = static_field()
    maxs: np.array = static_field()
    masks: Sequence[jnp.array] = static_field()
    idxs: Sequence[jnp.array] = static_field()
    input_neurons: jnp.array = static_field()
    output_neurons: jnp.array = static_field()
    num_neurons: int = static_field()
    num_input_neurons: int = static_field()
    dropout_p: eqxe.StateIndex = static_field()
    _hidden_activation: Callable = static_field()
    _output_activation_elem: Callable = static_field()
    key: eqxe.StateIndex = static_field()

    def __init__(
        self,
        graph: nx.DiGraph,
        input_neurons: Sequence[Hashable],
        output_neurons: Sequence[Hashable],
        hidden_activation: Callable = jnn.silu,
        output_activation_elem: Callable = _identity,
        output_activation_group: Callable = _identity,
        dropout_p: Union[float, Mapping[Hashable, float]] = 0.,
        *,
        key: Optional[jr.PRNGKey] = None,
    ):
        """**Arguments**:

        - `graph`: A `networkx.DiGraph` object that represents the DAG structure of the 
            neural network.
        - `input_neurons`: A sequence of nodes from `graph` indicating the input neurons. 
            The order here matters, as the input data will be passed into the input neurons 
            in the order specified here.
        - `output_neurons`: A sequence of nodes from `graph` indicating the output neurons. 
            The order here matters, as the output data will be read from the output neurons 
            in the order specified here.
        - `hidden_activation`: The activation function applied element-wise to the hidden 
            (i.e. non-input, non-output) neurons. It can itself be a trainable `equinox.Module`.
        - `output_activation_elem`: The activation function applied element-wise to the output 
            neurons. It can itself be a trainable `equinox.Module`.
        - `output_activation_group`: The activation function applied to the output neurons as 
            a whole after applying `output_activation_elem` element-wise, e.g. `jax.nn.softmax`. 
            It can itself be a trainable `equinox.Module`.
        - `dropout_p`: Dropout probability. If a single `float`, the same dropout
            probability will be applied to all hidden neurons. If a `Mapping[Hashable, float]`,
            `dropout_p[i]` refers to the dropout probability of neuron `i`. All neurons default 
            to zero unless otherwise specified. Note that this allows dropout to be applied to 
            input and output neurons as well.
        - `key`: The `jax.random.PRNGKey` used for parameter initialization and dropout. 
            Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.
        """
        super().__init__()
        self._set_topo_info(graph)
        self.adjacency_dict = _nx_digraph_to_adjacency_dict(graph)
        self.adjacency_dict_inv = _invert_dict(self.adjacency_dict)
        self._set_input_output_neurons(input_neurons, output_neurons)
        self._set_activations(hidden_activation, output_activation_elem, output_activation_group)
        self._set_dropout_p(dropout_p)
        self._set_parameters(key)


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
        key = self._keygen() if key is None else key
        rand = jr.uniform(key, self.dropout_p.shape, minval=0, maxval=1)
        dropout_keep = jnp.greater(rand, self.dropout_p)
        
        # Set input values.
        values = values.at[self.input_neurons].set(x * dropout_keep[self.input_neurons])

        # Function to apply the activation for a single neuron.
        def _apply_activation(id: int, affine: jnp.array) -> jnp.array:
            affine = jnp.expand_dims(affine, 0)
            output = lax.cond(
                jnp.isin(id, self.output_neurons),
                lambda: self.output_activation_elem(affine),
                lambda: self.hidden_activation(affine)
            )
            return jnp.squeeze(output)

        # Forward pass in topological batch order.
        for (tb, weights, mask, idx) in zip(self.topo_batches, self.weights, self.masks, self.idxs):
            # Affine transformation, wx + b.
            affine = (weights * mask) @ values[idx] + self.bias[tb - self.num_input_neurons]
            # Apply activations/dropout.
            output_values = vmap(_apply_activation)(tb, affine) * dropout_keep[tb]
            # Set new values.
            values = values.at[tb].set(output_values)

        # Return values pertaining to output neurons, with the group-wise 
        # output activation applied.
        return self.output_activation_group(values[self.output_neurons])


    def to_networkx_graph(self) -> nx.DiGraph:
        """Returns a `networkx.DiGraph` represention of the network with parameters
        and other relevant information included as node/edge attributes.

        **Returns**:

        A `networkx.DiGraph` object that represents the network.
        The nodes have the following `str` field(s):

        - `'id'`: The neuron's position in the topological sort (an `int`)
        - `'group'`: One of {`'input'`, `'hidden'`, `'output'`} (a `str`).
        - `'bias'`: The corresponding neuron's bias (a `float`).

        The edges have the following field(s):

        - `weight`: The corresponding network weight (a `float`).
        """            
        graph = self.graph.copy()

        node_attrs = {}
        for (neuron, id) in self.neuron_to_id.items():
            if jnp.isin(id, self.input_neurons):
                node_attrs[neuron] = {'id': id, 'group': 'input', 'bias': None}
            elif jnp.isin(id, self.output_neurons):
                bias = self.bias[id - self.num_input_neurons]
                node_attrs[neuron] = {'id': id, 'group': 'output', 'bias': bias}
            else:
                bias = self.bias[id - self.num_input_neurons]
                node_attrs[neuron] = {'id': id, 'group': 'hidden', 'bias': bias}
        nx.set_node_attributes(graph, node_attrs)
            
        edge_attrs = {}
        for (neuron, outputs) in self.adjacency_dict.items():
            topo_batch_idx, pos_idx = self.neuron_to_topo_batch_idx[neuron]
            for output in outputs:
                col_idx = output - self.mins[topo_batch_idx]
                assert self.masks[topo_batch_idx][pos_idx, col_idx]
                weight = self.weights[topo_batch_idx][pos_idx, col_idx]
                edge_attrs[(neuron, output)] = {'weight': weight}
        nx.set_edge_attributes(graph, edge_attrs)

        return graph


    def _set_topo_info(self, graph: nx.DiGraph) -> None:
        """Assign a unique `int` id to each neuron such that the neuron ids are 
        contiguous, and return the adjacency dict reflecting the internal ids along
        with a dict mapping each neuron to its id.
        """
        assert isinstance(graph, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(graph)
        self.graph = graph
        topo_sort = nx.lexicographical_topological_sort(graph)
        # Map a neuron to its `int` id, which is its position in the topological sort.
        neuron_to_id = {neuron: id for (id, neuron) in enumerate(topo_sort)}
        topo_sort_ids = [neuron_to_id[neuron] for neuron in topo_sort]
        self.topo_sort = np.array(topo_sort_ids, dtype=int)
        self.num_neurons = len(topo_sort)

        # Topological batching.
        # See Section 2.2 of https://arxiv.org/pdf/2101.07965.pdf.
        topo_batches = []
        topo_batch = []
        neurons_to_remove = []
        for neuron in topo_sort:
            if graph.in_degree(neuron) == 0:
                topo_batch.append(neuron_to_id[neuron])
                neurons_to_remove.append(neuron)
            else:
                topo_batches.append(np.array(topo_batch, dtype=int))
                graph.remove_nodes_from(neurons_to_remove)
                topo_batch = [neuron_to_id[neuron]]
                neurons_to_remove = [neuron]
        topo_batches.append(np.array(topo_batch, dtype=int))
        # The first topo batch is technically the input neurons, but we don't include
        # those here, since they are handled separately in the forward pass.
        self.topo_batches = [jnp.array(tb, dtype=int) for tb in topo_batches[1:]]

        self.num_topo_batches = len(self.topo_batches)
        self.neuron_to_id = neuron_to_id

        # Maps a neuron id to its topological batch and position within that batch.
        neuron_to_topo_batch_idx = [None] * self.num_neurons
        for i in range(self.num_topo_batches):
            for (j, n) in enumerate(self.topo_batches[i]):
                neuron_to_topo_batch_idx[int(n)] = (i, j)
        assert all(neuron_to_topo_batch_idx)
        self.neuron_to_topo_batch_idx = neuron_to_topo_batch_idx   


    def _set_input_output_neurons(
        self,
        input_neurons: Sequence[int],
        output_neurons: Sequence[int],
    ) -> None:
        """Check to make sure the input neurons and output neurons are valid.
        """
        input_neurons = [self.neuron_to_id[n] for n in input_neurons]
        output_neurons = [self.neuron_to_id[n] for n in output_neurons]

        # Check that the input and output neurons are both non-empty.
        assert input_neurons and output_neurons

        # Check that the input and output neurons are disjoint.
        assert not (set(input_neurons) & set(output_neurons))

        # Check that input neurons themselves have no inputs.
        for neuron in input_neurons:
            assert not self.adjacency_dict_inv[neuron]

        # Check that output neurons themselves have no outputs.
        for neuron in output_neurons:
            assert not self.adjacency_dict[neuron]

        self.num_input_neurons = len(input_neurons)
        self.input_neurons = jnp.array(input_neurons, dtype=int)
        self.output_neurons = jnp.array(output_neurons, dtype=int)

    
    def _set_activations(
        self,
        hidden_activation: Callable, 
        output_activation_elem: Callable,
        output_activation_group: Callable,
    ) -> None:
        """Check that all activations produce correctly-shaped output and
        do not raise exceptions on correctly-shaped, zero-valued input, and
        set the activations of `self`.
        """
        # Activations may themselves be `eqx.Module`s, so we do this to ensure
        # that both `Module` and non-`Module` activations work with the same
        # input shape.
        hidden_activation_ = hidden_activation \
            if isinstance(hidden_activation, Module) else vmap(hidden_activation)
        output_activation_elem_ = output_activation_elem \
            if isinstance(output_activation_elem, Module) else vmap(output_activation_elem)

        x = jnp.zeros((1,))
        try:
            y = hidden_activation_(x)
        except Exception as e:
            raise e
        assert jnp.array_equal(x.shape, y.shape)

        try:
            y = output_activation_elem_(x)
        except Exception as e:
            raise e
        assert jnp.array_equal(x.shape, y.shape)

        x = jnp.zeros_like(self.output_neurons)
        try:
            y = output_activation_group(x)
        except Exception as e:
            raise e
        assert jnp.array_equal(x.shape, y.shape)

        self.hidden_activation = hidden_activation_
        self.output_activation_elem = output_activation_elem_
        self.output_activation_group = output_activation_group
        # Done for plasticity functionality.
        self._hidden_activation = hidden_activation
        self._output_activation_elem = output_activation_elem


    def _set_dropout_p(self, dropout_p: Union[float, Mapping[Hashable, float]]) -> None:
        """Set the per-neuron dropout probabilities.
        """
        if isinstance(dropout_p, float):
            dropout_p = jnp.ones((self.num_neurons,)) * dropout_p
            dropout_p = dropout_p.at[self.input_neurons].set(0.)
            dropout_p = dropout_p.at[self.output_neurons].set(0.)
        else:
            assert isinstance(dropout_p, Mapping)
            dropout_p_ = np.zeros((self.num_neurons,))
            for (n, d) in dropout_p.items():
                dropout_p_[n] = d
            dropout_p = jnp.array(dropout_p_, dtype=float)
        assert jnp.all(jnp.greater_equal(dropout_p, 0))
        assert jnp.all(jnp.less_equal(dropout_p, 1))
        self.dropout_p = dropout_p


    def _set_parameters(self, key: Optional[jr.PRNGKey]) -> None:
        """Set the network parameters and relevent topological/indexing information.
        """
        # Here, `mins[i]` (`maxs[i]`) is the index representing the minimum
        # (maximum) topological index of those neurons strictly necessary to 
        # process `self.topo_batches[i]` from the previous topological batch. 
        # If `i == 0`, the previous topological batch is the input neurons.
        mins = np.array([])
        maxs = np.array([])
        for tb in self.topo_batches:
            input_locs = [self.adjacency_dict_inv[int(i)] for i in tb]
            mins_ = np.array([np.amin(locs) for locs in input_locs])
            maxs_ = np.array([np.amax(locs) for locs in input_locs])
            mins = np.append(mins, np.amin(mins_))
            maxs = np.append(maxs, np.amax(maxs_))
        self.mins = mins.astype(int)
        self.maxs = maxs.astype(int)

        # Set the random key. We use eqxe.StateIndex here so that the key 
        # can be automatically updated after every forward pass. This ensures 
        # that the random values generated for determining dropout application
        # are different for each forward pass if the user does not provide
        # an explicit key.
        key = jr.PRNGKey(0) if key is None else key
        assert isinstance(key, jr.PRNGKey)
        self.key = eqxe.StateIndex()
        *wkeys, bkey, dkey = jr.split(key, self.num_topo_batches + 2)
        eqxe.set_state(self.key, dkey)

        # Set the network parameters. Here, `self.bias` is a `jnp.array` of shape 
        # `(num_neurons - num_input_neurons,)`, where `self.bias[i]` is the bias of the neuron with
        # topological index `i + self.num_input_neurons`. `self.weights`, on the other hand, is a 
        # list of 2D `jnp.array`s, where `self.weights[i]` includes the weights used by the neurons 
        # in `self.topo_batches[i]`. More specifically, `self.weights[i][j, k]` is the weight of the 
        # connection from the neuron with topological index `k + mins[i]` to neuron `self.topo_batches[i][j]`. 
        # The weights are stored this way in order to use minimal memory while allowing for maximal `vmap` 
        # parallelism during the forward pass, since the minimum and maximum neurons needed to process a 
        # topological batch in parallel will be closest together when in topological order. 
        # All parameters are drawn iid ~ N(0, 0.01).
        weight_lengths = np.array(maxs - mins, dtype=int) + 1
        self.weights = [
            jr.normal(
                wkeys[i], (jnp.size(self.topo_batches[i]), weight_lengths[i])
            ) * 0.1
            for i in range(self.num_topo_batches)
        ]
        self.bias = jr.normal(
            bkey, (self.num_neurons - self.num_input_neurons,)
        ) * 0.1

        # Here, `self.masks` is a list of 2D binary `jnp.array` with identical structure
        # to `self.weights`. These are multiplied by the weights during the forward pass
        # to mask out weights for connections that are not present in the actual network.
        masks = []
        for (tb, weights, min_) in zip(self.topo_batches, self.weights, mins):
            mask = np.zeros_like(weights)
            for (i, neuron) in enumerate(tb):
                inputs = jnp.array(self.adjacency_dict_inv[int(neuron)], dtype=int)
                mask[i, inputs - int(min_)] = 1
            masks.append(jnp.array(mask, dtype=int))
        self.masks = masks

        # Here, `self.idxs[i]` includes the indices -- in topological order -- of the 
        # neurons needed to process `self.topo_batches[i]`. This is done for the same
        # memory/parallelism reason as the structure of `self.weights`.
        self.idxs = [
            jnp.arange(mins[i], maxs[i] + 1, dtype=int) for i in range(self.num_topo_batches)
        ]


    def _keygen(self) -> jr.PRNGKey:
        """Get the random key contained in `self.key` (an `eqxe.StateIndex`), 
        split the key, set `self.key` to contain the new key, and return the 
        original key.
        """
        key = eqxe.get_state(self.key, jr.PRNGKey(0))
        _, new_key = jr.split(key)
        eqxe.set_state(self.key, new_key)
        return key