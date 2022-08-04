import sys
from copy import deepcopy
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
from utils import _identity, _invert_dict


class NeuralNetwork(Module):
    """
    A neural network whose structure is specified by a DAG.
    Create your model by inheriting from this.
    """
    weights: Sequence[jnp.array]
    biases: jnp.array
    hidden_activation: Callable
    output_activation: Callable
    output_transform: Callable
    attention_params: Sequence[jnp.array]
    gamma: Optional[jnp.array]
    beta: Optional[jnp.array]
    gain: Optional[jnp.array]
    amplification: Optional[jnp.array]
    graph: nx.DiGraph = static_field()
    adjacency_dict: Mapping[int, Sequence[int]] = static_field()
    adjacency_dict_inv: Mapping[int, Sequence[int]] = static_field()
    neuron_to_id: Mapping[Hashable, int] = static_field()
    topo_batches: Sequence[jnp.array] = static_field()
    neuron_to_topo_batch_idx: Sequence[Tuple[int, int]] = static_field()
    topo_sort: np.array = static_field()
    min_index: np.array = static_field()
    max_index: np.array = static_field()
    masks: Sequence[jnp.array] = static_field()
    attention_masks: Sequence[jnp.array] = static_field()
    indices: Sequence[jnp.array] = static_field()
    input_neurons: jnp.array = static_field()
    output_neurons: jnp.array = static_field()
    num_neurons: int = static_field()
    num_input_neurons: int = static_field()
    num_inputs_per_neuron: jnp.array = static_field()
    dropout_p: jnp.array = static_field()
    use_self_attention: bool = static_field()
    use_neuron_norm: bool = static_field()
    use_adaptive_activations: bool = static_field()
    _hidden_activation: Callable = static_field()
    _output_activation: Callable = static_field()
    key: eqxe.StateIndex = static_field()

    def __init__(
        self,
        graph: nx.DiGraph,
        input_neurons: Sequence[Hashable],
        output_neurons: Sequence[Hashable],
        hidden_activation: Callable = jnn.silu,
        output_activation: Callable = _identity,
        output_transform: Callable = _identity,
        dropout_p: Union[float, Mapping[Hashable, float]] = 0.,
        use_self_attention: bool = False,
        use_neuron_norm: bool = False,
        use_adaptive_activations: bool = False,
        *,
        key: Optional[jr.PRNGKey] = None,
        **kwargs
    ):
        """**Arguments**:

        - `graph`: A `networkx.DiGraph` representing the DAG structure of the neural network.
        - `input_neurons`: A sequence of nodes from `graph` indicating the input neurons. 
            The order here matters, as the input data will be passed into the input neurons 
            in the order specified here.
        - `output_neurons`: A sequence of nodes from `graph` indicating the output neurons. 
            The order here matters, as the output data will be read from the output neurons 
            in the order specified here.
        - `hidden_activation`: The activation function applied element-wise to the hidden 
            (i.e. non-input, non-output) neurons. It can itself be a trainable `equinox.Module`.
        - `output_activation`: The activation function applied element-wise to the output 
            neurons. It can itself be a trainable `equinox.Module`.
        - `output_transform`: The transformation applied to the output neurons as a whole after 
            applying `output_activation` element-wise, e.g. `jax.nn.softmax`. It can itself be a 
            trainable `equinox.Module`.
        - `dropout_p`: Dropout probability. If a single `float`, the same dropout
            probability will be applied to all hidden neurons. If a `Mapping[Hashable, float]`,
            `dropout_p[i]` refers to the dropout probability of neuron `i`. All neurons default 
            to zero unless otherwise specified. Note that this allows dropout to be applied to 
            input and output neurons as well.
        - `use_self_attention`: A `bool` indicating whether to apply neuron-wise self-attention, 
            where each neuron applies self-attention (CITE) to its inputs.
        - `use_neuron_norm`: A `bool` indicating whether neurons should normalize their respective 
            inputs, i.e. scale by mean and variance. If both `use_self_attention` and `use_neuron_norm` 
            are `True`, normalization is applied after self-attention. (CITE)
        - `use_adaptive_activations`: Inspired by (CITE). If `True`, activations undergo
            `σ(x) -> a * σ(b * x) + c`, where `a`, `b`, `c` are trainable scalar parameters unique
            to each neuron.
        - `key`: The `jax.random.PRNGKey` used for parameter initialization and dropout. 
            Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.
        """
        super().__init__(**kwargs)
        print("Compiling network...")
        self._set_topological_info(graph)
        self._set_input_output_neurons(input_neurons, output_neurons)
        self._set_activations(hidden_activation, output_activation, output_transform)
        self._set_parameters(key, use_self_attention, use_adaptive_activations, use_neuron_norm)
        self._set_dropout_p(dropout_p)
        print("Done!\n")


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

        # Forward pass in topological batch order.
        for (tb, weights, mask, indices, attn_params, attn_mask) in zip(
            self.topo_batches, 
            self.weights, 
            self.masks,  
            self.indices,
            self.attention_params,
            self.attention_masks 
        ):
            # Previous neuron values strictly necessary to process the current topological batch.
            vals = values[indices]
            # Neuron-level self-attention.
            if self.use_self_attention:
                apply_self_attention = vmap(self._apply_self_attention, in_axes=[0, 0, 0, None])
                vals = apply_self_attention(tb, attn_params, attn_mask, vals)
            # "Neuron Norm": basically like Layer Norm for each neuron individually.
            if self.use_neuron_norm:
                apply_neuron_norm = vmap(self._apply_neuron_norm, in_axes=[0, 0, None])
                vals = apply_neuron_norm(tb, mask, vals)
            # Affine transformation, wx + b.
            affine = (weights * mask) @ vals + self.biases[tb - self.num_input_neurons]
            # Apply activations/dropout.
            output_values = vmap(self._apply_activation)(tb, affine) * dropout_keep[tb]
            # Set new values.
            values = values.at[tb].set(output_values)

        # Return values pertaining to output neurons, with the group-wise 
        # output activation applied.
        return self.output_transform(values[self.output_neurons])


    ##############################################################################
    ################ Methods used inside forward pass (__call__). ################
    ##############################################################################

    def _apply_self_attention(
        self,
        id: jnp.array, 
        attn_params: jnp.array,
        attn_mask: jnp.array,
        vals: jnp.array
    ) -> jnp.array:
        """Function for a single neuron to apply self-attention to its inputs,
        followed by a skip connection.
        """
        query_params, key_params, value_params = attn_params
        query_weight, query_bias = query_params[:, :-1], query_params[:, -1]
        key_weight, key_bias = key_params[:, :-1], key_params[:, -1]
        value_weight, value_bias = value_params[:, :-1], value_params[:, -1]
        query = query_weight @ vals + query_bias
        key = key_weight @ vals + key_bias
        value = value_weight @ vals + value_bias
        rsqrt = lax.rsqrt(self.num_inputs_per_neuron[id])
        scaled_outer_product = jnp.outer(query, key) * rsqrt
        attention_weights = jnn.softmax(scaled_outer_product - attn_mask)
        return attention_weights @ value + vals


    def _apply_neuron_norm(
        self, id: jnp.array, mask: jnp.array, vals: jnp.array, eps: float = 1e-5
    ) -> jnp.array:
        """Neuron Norm -- like Layer Norm but for each neuron individually.
        """
        num_inputs = self.num_inputs_per_neuron[id]
        masked_vals = vals * mask
        mean = jnp.sum(masked_vals) / num_inputs
        # Var[X] = E[X^2] - E[X]^2
        var = jnp.sum(masked_vals * vals) / num_inputs - jnp.square(mean)
        normalized = (vals - mean) * lax.rsqrt(var + eps)
        idx = id - self.num_input_neurons
        gamma, beta = self.gamma[idx], self.beta[idx]
        return gamma * normalized + beta


    def _apply_activation(self, id: jnp.array, affine: jnp.array) -> jnp.array:
        """Function for a single neuron to apply its activation.
        """
        idx = id - self.num_input_neurons
        gain, amplification = lax.cond(
            self.use_adaptive_activations,
            lambda: (self.gain[idx], self.amplification[idx]),
            lambda: (1., 1.)
        )
        affine_with_gain = jnp.expand_dims(affine * gain, 0)
        output = lax.cond(
            jnp.isin(id, self.output_neurons),
            lambda: self.output_activation(affine_with_gain),
            lambda: self.hidden_activation(affine_with_gain)
        )
        return jnp.squeeze(output) * amplification


    def _keygen(self) -> jr.PRNGKey:
        """Get the random key contained in `self.key` (an `eqxe.StateIndex`), 
        split the key, set `self.key` to contain the new key, and return the 
        original key.
        """
        key = eqxe.get_state(self.key, jr.PRNGKey(0))
        _, new_key = jr.split(key)
        eqxe.set_state(self.key, new_key)
        return key


    ###########################################################################
    ################ Methods used to set network attributes. ##################
    ###########################################################################
        
    def _set_topological_info(self, graph: nx.DiGraph) -> None:
        """Set the topological information and relevant attributes.
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

        # Create an adjacency dict that maps a neuron id to its output ids and  
        # an inverse adjacency dict that maps a neuron id to its input ids.
        adjacency_dict = {}
        adjacency_dict_ = nx.to_dict_of_lists(graph)
        for (input, outputs) in adjacency_dict_.items():
            adjacency_dict[neuron_to_id[input]] = [neuron_to_id[o] for o in outputs]
        self.adjacency_dict = adjacency_dict
        self.adjacency_dict_inv = _invert_dict(adjacency_dict)
        self.num_inputs_per_neuron = jnp.array(
            [len(self.adjacency_dict_inv[i]) for i in range(self.num_neurons)]
        )

        # Topological batching.
        # See Section 2.2 of https://arxiv.org/pdf/2101.07965.pdf.
        topo_batches, topo_batch, neurons_to_remove = [], [], []
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
        input_neurons: Sequence[Hashable],
        output_neurons: Sequence[Hashable],
    ) -> None:
        """Set the input and output neurons.
        """
        assert isinstance(input_neurons, Sequence)
        assert isinstance(output_neurons, Sequence)
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
        output_activation: Callable,
        output_transform: Callable,
    ) -> None:
        """Set the activation functions.
        """
        # Activations may themselves be `eqx.Module`s, so we do this to ensure
        # that both `Module` and non-`Module` activations work with the same
        # input shape.
        hidden_activation_ = hidden_activation \
            if isinstance(hidden_activation, Module) else vmap(hidden_activation)
        output_activation_ = output_activation \
            if isinstance(output_activation, Module) else vmap(output_activation)

        x = jnp.zeros((1,))
        try:
            y = hidden_activation_(x)
        except Exception as e:
            raise e
        assert jnp.array_equal(x.shape, y.shape)

        try:
            y = output_activation_(x)
        except Exception as e:
            raise e
        assert jnp.array_equal(x.shape, y.shape)

        x = jnp.zeros_like(self.output_neurons)
        try:
            y = output_transform(x)
        except Exception as e:
            raise e
        assert jnp.array_equal(x.shape, y.shape)

        self.hidden_activation = hidden_activation_
        self.output_activation = output_activation_
        self.output_transform = output_transform
        
        # Done for plasticity functionality.
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation


    def _set_parameters(
        self, 
        key: Optional[jr.PRNGKey], 
        use_self_attention: bool,
        use_adaptive_activations: bool,
        use_neuron_norm: bool
    ) -> None:
        """Set the network parameters and relevent topological/indexing information.
        """
        # Here, `min_index[i]` (`max_index[i]`) is the index representing the minimum
        # (maximum) topological index of those neurons strictly necessary to 
        # process `self.topo_batches[i]` from the previous topological batch. 
        # If `i == 0`, the previous topological batch is the input neurons.
        min_index = np.array([])
        max_index = np.array([])
        for tb in self.topo_batches:
            input_locs = [self.adjacency_dict_inv[int(i)] for i in tb]
            mins = np.array([np.amin(locs) for locs in input_locs])
            maxs = np.array([np.amax(locs) for locs in input_locs])
            min_index = np.append(min_index, np.amin(mins))
            max_index = np.append(max_index, np.amax(maxs))
        self.min_index = min_index.astype(int)
        self.max_index = max_index.astype(int)

        # Set the random key. We use eqxe.StateIndex here so that the key 
        # can be automatically updated after every forward pass. This ensures 
        # that the random values generated for determining dropout application
        # are different for each forward pass if the user does not provide
        # an explicit key.
        key = jr.PRNGKey(0) if key is None else key
        assert isinstance(key, jr.PRNGKey)
        self.key = eqxe.StateIndex()
        dkey, key = jr.split(key, 2)
        eqxe.set_state(self.key, dkey)

        # Set the network parameters. Here, `self.biases` is a `jnp.array` of shape 
        # `(num_neurons - num_input_neurons,)`, where `self.biases[i]` is the bias of 
        # the neuron with topological index `i + self.num_input_neurons`, and `self.weights` 
        # is a list of 2D `jnp.array`s, where `self.weights[i]` are the weights used by the 
        # neurons in `self.topo_batches[i]`. More specifically, `self.weights[i][j, k]` is 
        # the weight of the connection from the neuron with topological index `k + mins[i]` 
        # to the neuron with index `self.topo_batches[i][j]`. The weights are stored this way 
        # in order to use minimal memory while allowing for maximal `vmap` parallelism during 
        # the forward pass, since the minimum and maximum neurons needed to process a topological 
        # batch in parallel will be closest together when in topological order. 
        # All parameters are drawn iid ~ N(0, 0.01).
        *wkeys, bkey, key = jr.split(key, self.num_topo_batches + 2)
        topo_lengths = self.max_index - self.min_index + 1
        num_non_input_neurons = self.num_neurons - self.num_input_neurons
        self.weights = [
            jr.normal(wkeys[i], (jnp.size(self.topo_batches[i]), topo_lengths[i])) * 0.1
            for i in range(self.num_topo_batches)
        ]
        self.biases = jr.normal(bkey, (num_non_input_neurons,)) * 0.1

        # Here, `self.masks` is a list of 2D binary `jnp.array` with identical structure
        # to `self.weights`. These are multiplied by the weights during the forward pass
        # to mask out weights for connections that are not present in the actual network.
        masks = []
        for (tb, weights, min_idx) in zip(self.topo_batches, self.weights, self.min_index):
            mask = np.zeros_like(weights)
            for (i, neuron) in enumerate(tb):
                inputs = jnp.array(self.adjacency_dict_inv[int(neuron)], dtype=int)
                mask[i, inputs - min_idx] = 1
            masks.append(jnp.array(mask, dtype=int))
        self.masks = masks

        # Self-attention. TODO
        self.use_self_attention = bool(use_self_attention)
        if use_self_attention:
            skey, key = jr.split(key, 2)
            # Set attention parameters.
            self.attention_params = [
                jr.normal(
                    skey, (jnp.size(self.topo_batches[i]), 3, topo_lengths[i], topo_lengths[i] + 1)
                ) * 0.1 for i in range(self.num_topo_batches)
            ]
            outer_product = vmap(lambda x: jnp.outer(x, x))
            self.attention_masks = [
                jnp.where(outer_product(1 - mask), jnp.inf, 0) for mask in self.masks
            ]
        else:
            self.attention_params = [jnp.nan] * self.num_topo_batches
            self.attention_masks = [jnp.nan] * self.num_topo_batches

        # Here, `self.indices[i]` includes the indices of the neurons needed to process 
        # `self.topo_batches[i]`. This is done for the same memory/parallelism reason 
        # as the structure of `self.weights`.
        self.indices = [
            jnp.arange(min_index[i], max_index[i] + 1, dtype=int) 
            for i in range(self.num_topo_batches)
        ]

        # Gain, amplification iid ~ N(1, 0.01), N(1, 0.01)). TODO
        self.use_adaptive_activations = bool(use_adaptive_activations)
        if use_adaptive_activations:
            gkey, akey, key = jr.split(key, 3)
            self.gain = jr.normal(gkey, (num_non_input_neurons,)) * 0.1 + 1
            self.amplification = jr.normal(akey, (num_non_input_neurons,)) * 0.1 + 1
        else:
            self.gain = None
            self.amplification = None

        # Neuron norm.
        self.use_neuron_norm = bool(use_neuron_norm)
        if use_neuron_norm:
            gkey, bkey = jr.split(key, 2)
            self.gamma = jr.normal(gkey, (num_non_input_neurons,)) * 0.1 + 1
            self.beta = jr.normal(bkey, (num_non_input_neurons,)) * 0.1
        else:
            self.gamma = None
            self.beta = None


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
                dropout_p_[self.neuron_to_id[n]] = d
            dropout_p = jnp.array(dropout_p_, dtype=float)
        assert jnp.all(jnp.greater_equal(dropout_p, 0))
        assert jnp.all(jnp.less_equal(dropout_p, 1))
        self.dropout_p = dropout_p


    ###################################################
    ################ Public methods. ##################
    ###################################################

    def to_networkx_graph(self) -> nx.DiGraph:
        """Returns a `networkx.DiGraph` represention of the network with parameters
        and other relevant information included as node/edge attributes.

        **Returns**:

        A `networkx.DiGraph` object that represents the network. The original graph
        used to initialize the network is left unchanged.

        In addition to any field(s) the nodes may already have, the nodes 
        now also have the following additional `str` field(s):

        - `'id'`: The neuron's position in the topological sort (an `int`)
        - `'group'`: One of {`'input'`, `'hidden'`, `'output'`} (a `str`).
        - `'bias'`: The corresponding neuron's bias (a `float`).

        In addition to any field(s) the edges may already have, the edges 
        now also have the following additional `str` field(s):

        - `weight`: The corresponding network weight (a `float`).
        """            
        graph = deepcopy(self.graph)

        node_attrs = {}
        node_attrs_ = dict(graph.nodes(data=True))
        for (neuron, id) in self.neuron_to_id.items():
            if jnp.isin(id, self.input_neurons):
                node_attrs[neuron] = {
                    'id': id, 'group': 'input', 'bias': None, **node_attrs_[neuron]
                }
            elif jnp.isin(id, self.output_neurons):
                bias = self.biases[id - self.num_input_neurons]
                node_attrs[neuron] = {
                    'id': id, 'group': 'output', 'bias': bias, **node_attrs_[neuron]
                }
            else:
                bias = self.biases[id - self.num_input_neurons]
                node_attrs[neuron] = {
                    'id': id, 'group': 'hidden', 'bias': bias, **node_attrs_[neuron]
                }
        nx.set_node_attributes(graph, node_attrs)
            
        edge_attrs = {}
        edge_attrs_ = {(i, o): data for (i, o, data) in graph.edges(data=True)}
        for (neuron, outputs) in self.adjacency_dict.items():
            topo_batch_idx, pos_idx = self.neuron_to_topo_batch_idx[neuron]
            for output in outputs:
                col_idx = output - self.min_index[topo_batch_idx]
                assert self.masks[topo_batch_idx][pos_idx, col_idx]
                weight = self.weights[topo_batch_idx][pos_idx, col_idx]
                edge_attrs[(neuron, output)] = {
                    'weight': weight, **edge_attrs_[(neuron, output)]
                }
        nx.set_edge_attributes(graph, edge_attrs)

        return graph