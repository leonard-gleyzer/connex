from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import equinox.experimental as eqxe
from equinox import Module, filter_jit, static_field

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

from jaxtyping import Array

import networkx as nx
import numpy as np

from .utils import _identity, _invert_dict


class NeuralNetwork(Module):
    """
    A neural network whose structure is specified by a DAG.
    Create your model by inheriting from this.
    """
    weights: List[Array]
    biases: Array
    hidden_activation: Callable
    output_activation: Callable
    gammas: Optional[Array]
    betas: Optional[Array]
    attention_params_topo: List[Array]
    attention_params_neuron: List[Array]
    gain: Optional[Array]
    amplification: Optional[Array]
    graph: nx.DiGraph = static_field()
    adjacency_dict: Dict[int, List[int]] = static_field()
    adjacency_dict_inv: Dict[int, List[int]] = static_field()
    neuron_to_id: Dict[Any, int] = static_field()
    topo_batches: List[Array] = static_field()
    num_topo_batches: int = static_field()
    neuron_to_topo_batch_idx: Dict[int, Tuple[int, int]] = static_field()
    topo_sort: np.ndarray = static_field()
    min_index: np.ndarray = static_field()
    max_index: np.ndarray = static_field()
    masks: List[Array] = static_field()
    attention_masks_neuron: List[Array] = static_field()
    indices: List[Array] = static_field()
    input_neurons: Array = static_field()
    output_neurons: Array = static_field()
    num_neurons: int = static_field()
    num_input_neurons: int = static_field()
    num_inputs_per_neuron: Array = static_field()
    dropout_p: Array = static_field()
    use_topo_norm: bool = static_field()
    use_topo_self_attention: bool = static_field()
    use_neuron_self_attention: bool = static_field()
    use_adaptive_activations: bool = static_field()
    _hidden_activation: Callable = static_field()
    key: eqxe.StateIndex = static_field()

    def __init__(
        self,
        graph: nx.DiGraph,
        input_neurons: Sequence[Any],
        output_neurons: Sequence[Any],
        hidden_activation: Callable = jnn.silu,
        output_activation: Callable = _identity,
        dropout_p: Union[float, Mapping[Any, float]] = 0.,
        use_topo_norm: bool = False,
        use_topo_self_attention: bool = False,
        use_neuron_self_attention: bool = False,
        use_adaptive_activations: bool = False,
        topo_sort: Optional[Sequence[Any]] = None,
        *,
        key: Optional[jr.PRNGKey] = None,
        **kwargs
    ):
        """**Arguments**:

        - `graph`: A `networkx.DiGraph` representing the DAG structure of the neural network.
        - `input_neurons`: An `Sequence` of nodes from `graph` indicating the input neurons. 
            The order here matters, as the input data will be passed into the input neurons 
            in the order specified here.
        - `output_neurons`: An `Sequence` of nodes from `graph` indicating the output neurons. 
            The order here matters, as the output data will be read from the output neurons 
            in the order specified here.
        - `hidden_activation`: The activation function applied element-wise to the hidden 
            (i.e. non-input, non-output) neurons. It can itself be a trainable `equinox.Module`.
        - `output_activation`: The activation function applied group-wise to the output 
            neurons, e.g. `jax.nn.softmax`. It can itself be a trainable `equinox.Module`.
        - `dropout_p`: Dropout probability. If a single `float`, the same dropout
            probability will be applied to all hidden neurons. If a `Mapping[Any, float]`,
            `dropout_p[i]` refers to the dropout probability of neuron `i`. All neurons default 
            to zero unless otherwise specified. Note that this allows dropout to be applied to 
            input and output neurons as well.
        - `use_topo_norm`: A `bool` indicating whether to apply a topo batch version of [Layer Norm](),
            where the collective inputs of each topological batch are normalized, with learnable 
            elementwise-affine parameters `gamma` and `beta`.
        - `use_topo_self_attention`: A `bool` indicating whether to apply self-attention to each topological 
            batch's collective inputs.
        - `use_neuron_self_attention`: A `bool` indicating whether to apply neuron-wise self-attention, 
            where each neuron applies [self-attention](https://arxiv.org/abs/1706.03762) to its inputs. 
            If both `use_neuron_self_attention` and `use_neuron_norm` are `True`, normalization is applied 
            [before self-attention](https://arxiv.org/abs/2002.04745). Warning: this may cause significantly 
            greater memory use.
        - `use_adaptive_activations`: A bool indicating whether to use neuron-wise 
            [adaptive activations](https://arxiv.org/abs/1909.12228). If `True`, activations 
            undergo `σ(x) -> a * σ(b * x)`, where `a`, `b` are trainable scalar parameters 
            unique to each neuron.
        - `topo_sort`: An optional sequence of neurons indicating a topological sort of the graph. If `None`,
            the topological sort will be done via NetworkX, which may be time-consuming for large networks.
        - `key`: The `jax.random.PRNGKey` used for parameter initialization and dropout. 
            Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.
        """
        super().__init__(**kwargs)
        print("Compiling network...")
        self._set_topological_info(graph, topo_sort)
        self._set_input_output_neurons(input_neurons, output_neurons)
        self._set_activations(hidden_activation, output_activation)
        self._set_parameters(
            key, 
            use_topo_norm, 
            use_topo_self_attention, 
            use_neuron_self_attention, 
            use_adaptive_activations
        )
        self._set_dropout_p(dropout_p)
        print("Done!")


    @filter_jit
    def __call__(
        self, x: Array, *, key: Optional[jr.PRNGKey] = None,
    ) -> Array:
        """The forward pass of the network. Neurons are "fired" in topological batch
        order (see Section 2.2 of [this paper](https://arxiv.org/abs/2101.07965)), 
        with `jax.vmap` vectorization used within each topological batch.
        
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
        for (tb, weights, mask, indices, gamma, beta, attn_params_t, attn_params_n, attn_mask_n) in zip(
            self.topo_batches, 
            self.weights, 
            self.masks,  
            self.indices,
            self.gammas,
            self.betas,
            self.attention_params_topo, 
            self.attention_params_neuron,
            self.attention_masks_neuron,
        ):
            # Previous neuron values strictly necessary to process the current topological batch.
            vals = values[indices]
            # Topo Norm.
            if self.use_topo_norm:
                vals = self._apply_topo_norm(gamma, beta, vals)
            # Topo-level self-attention.
            if self.use_topo_self_attention:
                vals = self._apply_topo_self_attention(attn_params_t, vals)
            # Neuron-level self-attention.
            if self.use_neuron_self_attention:
                # vals = jnp.tile(vals, (jnp.size(tb), 1))
                _apply_neuron_self_attention = vmap(self._apply_neuron_self_attention, in_axes=[0, 0, 0, None])
                vals = _apply_neuron_self_attention(tb, attn_params_n, attn_mask_n, vals)
            # Affine transformation, wx + b.
            affine = vmap(jnp.dot)(weights * mask, vals) + self.biases[tb - self.num_input_neurons]
            # Apply activations/dropout.
            output_values = vmap(self._apply_activation)(tb, affine) * dropout_keep[tb]
            # Set new values.
            values = values.at[tb].set(output_values)

        # Return values pertaining to output neurons, with the group-wise 
        # output activation applied.
        return self.output_activation(values[self.output_neurons])


    ###############################################################################
    ################ Methods used during forward pass in __call__. ################
    ###############################################################################

    def _apply_topo_norm(self, gamma: Array, beta: Array, vals: Array) -> Array:
        """Function for a topo batch to standardize its inputs (unless the input array
        has shape (1,), in which case it is left alone), followed by a learnable 
        elementwise-affine transformation.
        """
        return lax.cond(
            jnp.greater(jnp.size(vals), 1),
            lambda: jnn.standardize(vals) * gamma + beta,
            lambda: vals
        )


    def _apply_topo_self_attention(
        self, attn_params: Array, vals: Array
    ) -> Array:
        """Function for a topo batch to apply self-attention to its collective inputs,
        followed by a skip connection.
        """
        query_params, key_params, value_params = attn_params
        query_weight, query_bias = query_params[:, :-1], query_params[:, -1]
        key_weight, key_bias = key_params[:, :-1], key_params[:, -1]
        value_weight, value_bias = value_params[:, :-1], value_params[:, -1]
        query = query_weight @ vals + query_bias
        key = key_weight @ vals + key_bias
        value = value_weight @ vals + value_bias
        rsqrt = lax.rsqrt(jnp.size(vals))
        scaled_outer_product = jnp.outer(query, key) * rsqrt
        attention_weights = jnn.softmax(scaled_outer_product)
        return attention_weights @ value + vals


    def _apply_neuron_self_attention(
        self, id: int, attn_params: Array, attn_mask: Array, vals: Array
    ) -> Array:
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


    def _apply_activation(self, id: int, affine: float) -> Array:
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
            lambda: affine_with_gain,
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


    #############################################################################
    ############ Methods used to set network attributes in __init__. ############
    #############################################################################
        
    def _set_topological_info(
        self, graph: nx.DiGraph, topo_sort: Optional[Sequence[Any]]
    ) -> None:
        """Set the topological information and relevant attributes.
        """
        if topo_sort is None:
            topo_sort = list(nx.lexicographical_topological_sort(graph))
        else:
            graph_copy = nx.DiGraph(graph)
            assert nx.is_directed_acyclic_graph(graph_copy)
            # Check that the provided topological sort is valid.
            for neuron in topo_sort:
                assert graph_copy.in_degree(neuron) == 0
                graph_copy.remove_node(neuron)
        self.graph = graph
        # Map a neuron to its `int` id, which is its position in the topological sort.
        neuron_to_id = {neuron: id for (id, neuron) in enumerate(topo_sort)}
        topo_sort_ids = [neuron_to_id[neuron] for neuron in topo_sort]
        self.topo_sort = np.array(topo_sort_ids, dtype=int)
        self.num_neurons = np.size(self.topo_sort)

        # Create an adjacency dict that maps a neuron id to its output ids and  
        # an inverse adjacency dict that maps a neuron id to its input ids.
        adjacency_dict = {}
        adjacency_dict_ = nx.to_dict_of_lists(graph)
        for (input, outputs) in adjacency_dict_.items():
            adjacency_dict[neuron_to_id[input]] = [neuron_to_id[o] for o in outputs]
        self.adjacency_dict = adjacency_dict
        self.adjacency_dict_inv = _invert_dict(adjacency_dict)
        num_inputs_per_neuron = [len(self.adjacency_dict_inv[i]) for i in range(self.num_neurons)]
        self.num_inputs_per_neuron = jnp.array(num_inputs_per_neuron, dtype=float)

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
        neuron_to_topo_batch_idx = {}
        for i in range(self.num_topo_batches):
            for (j, n) in enumerate(self.topo_batches[i]):
                neuron_to_topo_batch_idx[int(n)] = (i, j)
        self.neuron_to_topo_batch_idx = neuron_to_topo_batch_idx   


    def _set_input_output_neurons(
        self,
        input_neurons: Sequence[Any],
        output_neurons: Sequence[Any],
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
    ) -> None:
        """Set the activation functions.
        """
        # Activations may themselves be `eqx.Module`s, so we do this to ensure
        # that both `Module` and non-`Module` activations work with the same
        # input shape.
        hidden_activation_ = hidden_activation \
            if isinstance(hidden_activation, Module) else vmap(hidden_activation)

        x = jnp.zeros((1,))
        try:
            y = hidden_activation_(x)
        except Exception as e:
            raise e
        assert jnp.array_equal(x.shape, y.shape)

        x = jnp.zeros_like(self.output_neurons)
        try:
            y = output_activation(x)
        except Exception as e:
            raise e
        assert jnp.array_equal(x.shape, y.shape)

        self.hidden_activation = hidden_activation_
        self.output_activation = output_activation
        
        # Done for plasticity functionality.
        self._hidden_activation = hidden_activation


    def _set_parameters(
        self, 
        key: Optional[jr.PRNGKey], 
        use_topo_norm: bool,
        use_topo_self_attention: bool,
        use_neuron_self_attention: bool,
        use_adaptive_activations: bool
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
        self.key = eqxe.StateIndex()
        dkey, key = jr.split(key, 2)
        eqxe.set_state(self.key, dkey)

        # Set the network weights and biases. Here, `self.biases` is a `jnp.ndarray` of 
        # shape `(num_neurons - num_input_neurons,)`, where `self.biases[i]` is the bias of 
        # the neuron with topological index `i + self.num_input_neurons`, and `self.weights` 
        # is a list of 2D `jnp.ndarray`s, where `self.weights[i]` are the weights used by the 
        # neurons in `self.topo_batches[i]`. More specifically, `self.weights[i][j, k]` is 
        # the weight of the connection from the neuron with topological index `k + mins[i]` 
        # to the neuron with index `self.topo_batches[i][j]`. The weights are stored this way 
        # in order to use minimal memory while allowing for maximal `vmap` parallelism during 
        # the forward pass, since the minimum and maximum neurons needed to process a topological 
        # batch in parallel will be closest together when in topological order. 
        # All weights and biases are drawn iid ~ N(0, 0.01).
        *wkeys, bkey, key = jr.split(key, self.num_topo_batches + 2)
        topo_lengths = self.max_index - self.min_index + 1
        num_non_input_neurons = self.num_neurons - self.num_input_neurons
        self.weights = [
            jr.normal(wkeys[i], (jnp.size(self.topo_batches[i]), topo_lengths[i])) * 0.1
            for i in range(self.num_topo_batches)
        ]
        self.biases = jr.normal(bkey, (num_non_input_neurons,)) * 0.1

        # Here, `self.masks` is a list of 2D binary `jnp.ndarray` with identical structure
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

        # Set parameters for TopoNorm
        self.use_topo_norm = bool(use_topo_norm)
        *gkeys, key = jr.split(key, self.num_topo_batches + 1)
        *bkeys, key = jr.split(key, self.num_topo_batches + 1)
        if use_topo_norm:
            self.gammas = [
                jr.normal(gkeys[i], (topo_lengths[i],)) * 0.1 + 1 for i in range(self.num_topo_batches)
            ]
            self.betas = [
                jr.normal(bkeys[i], (topo_lengths[i],)) * 0.1 for i in range(self.num_topo_batches)
            ]
        else:
            self.gammas = [jnp.nan] * self.num_topo_batches
            self.betas = [jnp.nan] * self.num_topo_batches

        # Set parameters and masks for topo-wise self-attention.
        self.use_topo_self_attention = bool(use_topo_self_attention)
        if use_topo_self_attention:
            *akeys, key = jr.split(key, self.num_topo_batches + 1)
            self.attention_params_topo = [
                jr.normal(
                    akeys[i], (3, topo_lengths[i], topo_lengths[i] + 1)
                ) * 0.1 for i in range(self.num_topo_batches)
            ]
        else:
            self.attention_params_topo = [jnp.nan] * self.num_topo_batches

        # Set parameters and masks for neuron-wise self-attention.
        self.use_neuron_self_attention = bool(use_neuron_self_attention)
        if use_neuron_self_attention:
            *akeys, key = jr.split(key, self.num_topo_batches + 1)
            self.attention_params_neuron = [
                jr.normal(
                    akeys[i], (jnp.size(self.topo_batches[i]), 3, topo_lengths[i], topo_lengths[i] + 1)
                ) * 0.1 for i in range(self.num_topo_batches)
            ]
            outer_product = vmap(lambda x: jnp.outer(x, x))
            mask_fn = filter_jit(lambda m: jnp.where(outer_product(m), 0, jnp.inf))
            self.attention_masks_neuron = [mask_fn(mask) for mask in self.masks]
        else:
            self.attention_params_neuron = [jnp.nan] * self.num_topo_batches
            self.attention_masks_neuron = [jnp.nan] * self.num_topo_batches

        # Here, `self.indices[i]` includes the indices of the neurons needed to process 
        # `self.topo_batches[i]`. This is done for the same memory/parallelism reason 
        # as the structure of `self.weights`.
        self.indices = [
            jnp.arange(min_index[i], max_index[i] + 1, dtype=int) 
            for i in range(self.num_topo_batches)
        ]

        # Set trainable parameters for neuron-wise adaptive activations (if applicable), 
        # drawn iid ~ N(1, 0.01).
        self.use_adaptive_activations = bool(use_adaptive_activations)
        gkey, akey, key = jr.split(key, 3)
        self.gain = jr.normal(gkey, (num_non_input_neurons,)) * 0.1 + 1
        self.amplification = jr.normal(akey, (num_non_input_neurons,)) * 0.1 + 1


    def _set_dropout_p(self, dropout_p: Union[float, Mapping[Any, float]]) -> None:
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

        - `'weight'`: The corresponding network weight (a `float`).
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