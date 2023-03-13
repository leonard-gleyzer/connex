import functools as ft
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import equinox.experimental as eqxe
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np
from equinox import filter_jit, Module, static_field, tree_at
from jax import Array, lax, vmap

from .utils import _identity, _invert_dict


class NeuralNetwork(Module):
    """
    A neural network whose structure is specified by a DAG.
    Create your model by inheriting from this.
    """

    weights_and_biases: List[Array]
    hidden_activation: Callable
    output_transformation: Callable
    topo_norm_params: List[Array]
    attention_params_topo: List[Array]
    attention_params_neuron: List[Array]
    adaptive_activation_params: List[Array]
    adjacency_dict: Dict[int, List[int]] = static_field()
    adjacency_dict_inv: Dict[int, List[int]] = static_field()
    neuron_to_id: Dict[Any, int] = static_field()
    topo_batches: List[Array] = static_field()
    topo_lengths: np.ndarray = static_field()
    topo_sizes: List[int] = static_field()
    num_topo_batches: int = static_field()
    neuron_to_topo_batch_idx: Dict[int, Tuple[int, int]] = static_field()
    topo_sort: List[Any] = static_field()
    min_index: np.ndarray = static_field()
    max_index: np.ndarray = static_field()
    masks: List[Array] = static_field()
    attention_masks_neuron: List[Array] = static_field()
    indices: List[Array] = static_field()
    input_neurons: List[Any] = static_field()
    output_neurons: List[Any] = static_field()
    hidden_neurons: List[Any] = static_field()
    input_neurons_id: Array = static_field()
    output_neurons_id: Array = static_field()
    num_neurons: int = static_field()
    num_inputs_per_neuron: Array = static_field()
    dropout_p: Dict[Any, float] = static_field()
    _dropout_p: Array = static_field()
    use_topo_norm: bool = static_field()
    use_topo_self_attention: bool = static_field()
    use_neuron_self_attention: bool = static_field()
    use_adaptive_activations: bool = static_field()
    key_state: eqxe.StateIndex = static_field()

    def __init__(
        self,
        graph: nx.DiGraph,
        input_neurons: Sequence[Any],
        output_neurons: Sequence[Any],
        hidden_activation: Callable = jnn.silu,
        output_transformation: Callable = _identity,
        dropout_p: Union[float, Mapping[Any, float]] = 0.0,
        use_topo_norm: bool = False,
        use_topo_self_attention: bool = False,
        use_neuron_self_attention: bool = False,
        use_adaptive_activations: bool = False,
        topo_sort: Optional[Sequence[Any]] = None,
        *,
        key: Optional[jr.PRNGKey] = None,
        **kwargs,
    ):
        """**Arguments**:

        - `graph`: A `networkx.DiGraph` representing the DAG structure of the neural
            network.
        - `input_neurons`: An `Sequence` of nodes from `graph` indicating the input
            neurons. The order here matters, as the input data will be passed into
            the input neurons in the order specified here.
        - `output_neurons`: An `Sequence` of nodes from `graph` indicating the output
            neurons. The order here matters, as the output data will be read from the
            output neurons in the order specified here.
        - `hidden_activation`: The activation function applied element-wise to the
            hidden (i.e. non-input, non-output) neurons. It can itself be a trainable
            `equinox.Module`.
        - `output_transformation`: The transformation applied group-wise to the output
            neurons, e.g. `jax.nn.softmax`. It can itself be a trainable
            `equinox.Module`.
        - `dropout_p`: Dropout probability. If a single `float`, the same dropout
            probability will be applied to all hidden neurons.
            If a `Mapping[Any, float]`, `dropout_p[i]` refers to the dropout
            probability of neuron `i`. All neurons default to zero unless otherwise
            specified. Note that this allows dropout to be applied to input and output
            neurons as well.
        - `use_topo_norm`: A `bool` indicating whether to apply a topological batch-
            version of Layer Norm,

            ??? cite
                [Layer Normalization](https://arxiv.org/abs/1607.06450)
                ```bibtex
                @article{ba2016layer,
                    author={Jimmy Lei Ba, Jamie Ryan Kriso, Geoffrey E. Hinton},
                    title={Layer Normalization},
                    year={2016},
                    journal={arXiv:1607.06450},
                }
                ```

            where the collective inputs of each topological batch are standardized
            (made to have mean 0 and variance 1), with learnable elementwise-affine
            parameters `gamma` and `beta`.
        - `use_topo_self_attention`: A `bool` indicating whether to apply
            (single-headed) self-attention to each topological batch's collective inputs.

            ??? cite
                [Attention is All You Need](https://arxiv.org/abs/1706.03762)
                ```bibtex
                @inproceedings{vaswani2017attention,
                    author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
                            Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
                            Kaiser, {\L}ukasz and Polosukhin, Illia},
                    booktitle={Advances in Neural Information Processing Systems},
                    publisher={Curran Associates, Inc.},
                    title={Attention is All You Need},
                    volume={30},
                    year={2017}
                }
                ```

        - `use_neuron_self_attention`: A `bool` indicating whether to apply neuron-wise
            self-attention, where each neuron applies (single-headed) self-attention to
            its inputs. If both `use_neuron_self_attention` and `use_neuron_norm` are
            `True`, normalization is applied before self-attention.

            !!! warning

            Neuron-level self-attention will use significantly more memory than
            than topo-level self-attention. You may notice training is slower with
            this enabled.

        - `use_adaptive_activations`: A bool indicating whether to use neuron-wise
            adaptive activations, where all hidden activations transform as
            `σ(x) -> a * σ(b * x)`, where `a`, `b` are trainable scalar parameters
            unique to each neuron.

            ??? cite
                [Locally adaptive activation functions with slope recovery term for
                 deep and physics-informed neural networks](https://arxiv.org/abs/1909.12228)  # noqa: E501
                ```bibtex
                @article{Jagtap_2020,
                    author={Ameya D. Jagtap, Kenji Kawaguchi, George Em Karniadakis},
                    title={Locally adaptive activation functions with slope recovery
                           term for deep and physics-informed neural networks},
                    year={2020},
                    publisher={The Royal Society},
                    journal={Proceedings of the Royal Society A: Mathematical, Physical
                    and Engineering Sciences},
                }
                ```

        - `topo_sort`: An optional sequence of neurons indicating a topological sort of
            the graph. If `None`, a topological sort will be performed on the graph.
            This may be time-consuming for large networks, which is why it is provided
            as an optional argument. The provided topological sort will still be checked
            that it is indeed a valid topological sort (which is far less time-consuming).
        - `key`: The `jax.random.PRNGKey` used for parameter initialization and dropout.
            Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.
        """
        super().__init__(**kwargs)
        self._set_topological_info(graph, input_neurons, output_neurons, topo_sort)
        self._set_activations(hidden_activation, output_transformation)
        self._set_parameters(
            key,
            use_topo_norm,
            use_topo_self_attention,
            use_neuron_self_attention,
            use_adaptive_activations,
        )
        self._set_dropout_p(dropout_p)

    @filter_jit
    def __call__(
        self,
        x: Array,
        *,
        key: Optional[jr.PRNGKey] = None,
    ) -> Array:
        """The forward pass of the network.
        Neurons are "fired" in topological batch order -- see Section 2.2 of

        ??? cite

            [Directed Acyclic Graph Neural Networks](https://arxiv.org/abs/2101.07965)
            ```bibtex
            @inproceedings{thost2021directed,
                author={Veronika Thost and Jie Chen},
                booktitle={International Conference on Learning Representations},
                publisher={Curran Associates, Inc.},
                title={Directed Acyclic Graph Neural Networks},
                year={2021}
            }
            ```

        **Arguments**:

        - `x`: The input array to the network for the forward pass. The individual
            values will be written to the input neurons in the order passed in during
            initialization.
        - `key`: A `jax.random.PRNGKey` used for dropout. Optional, keyword-only
            argument. If `None`, a key will be generated by getting the key in
            `self.key_state` (which is an `eqxe.StateIndex`) and then splitting
            and updating the key for the next forward pass.

        **Returns**:

        The result array from the forward pass. The order of the array elements will be
        the order of the output neurons passed in during initialization.
        """
        # Neuron value array, updated as neurons are "fired"
        values = jnp.zeros((self.num_neurons,))

        # Dropout
        key = self._keygen() if key is None else key
        rand = jr.uniform(key, self._dropout_p.shape, minval=0, maxval=1)
        dropout_keep = jnp.greater(rand, self._dropout_p)

        # Set input values
        values = values.at[self.input_neurons_id].set(
            x * dropout_keep[self.input_neurons_id]
        )

        # Forward pass in topological batch order
        for (
            tb,
            w_and_b,
            mask,
            indices,
            attn_params_t,
            attn_params_n,
            attn_mask_n,
            norm_params,
            ada_params,
        ) in zip(
            self.topo_batches,
            self.weights_and_biases,
            self.masks,
            self.indices,
            self.attention_params_topo,
            self.attention_params_neuron,
            self.attention_masks_neuron,
            self.topo_norm_params,
            self.adaptive_activation_params,
        ):
            # Previous neuron values strictly necessary to process the current
            # topological batch
            vals = values[indices]
            # Topo Norm
            if self.use_topo_norm and jnp.size(vals) > 1:
                vals = self._apply_topo_norm(norm_params, vals)
            # Topo-level self-attention
            if self.use_topo_self_attention:
                vals = self._apply_topo_self_attention(attn_params_t, vals)
            # Neuron-level self-attention
            if self.use_neuron_self_attention:
                _apply_neuron_self_attention = vmap(
                    self._apply_neuron_self_attention, in_axes=[0, 0, 0, None]
                )
                vals = _apply_neuron_self_attention(
                    tb, attn_params_n, attn_mask_n, vals
                )
            # Affine transformation, wx + b
            weights, biases = w_and_b[:, :-1], w_and_b[:, -1]
            affine = (weights * mask) @ vals + biases
            # Apply activations/dropout
            output_values = (
                vmap(self._apply_activation)(tb, affine, ada_params) * dropout_keep[tb]
            )
            # Set new values
            values = values.at[tb].set(output_values)

        # Return values pertaining to output neurons, with the group-wise
        # output activation applied
        return self.output_transformation(values[self.output_neurons_id])

    ##############################################################################
    ################ Methods used during forward pass in __call__ ################  # noqa: E266 E501
    ##############################################################################

    def _apply_topo_norm(self, norm_params: Array, vals: Array) -> Array:
        """Function for a topo batch to standardize its inputs, followed by a
        learnable elementwise-affine transformation.
        """
        gamma, beta = norm_params
        return jnn.standardize(vals) * gamma + beta

    def _apply_topo_self_attention(self, attn_params: Array, vals: Array) -> Array:
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

    def _apply_activation(self, id: int, affine: float, ada_params: Array) -> Array:
        """Function for a single neuron to apply its activation."""
        gain, amplification = lax.cond(
            self.use_adaptive_activations, lambda: ada_params, lambda: jnp.ones((2,))
        )
        _expand = ft.partial(jnp.expand_dims, axis=0)
        output = lax.cond(
            jnp.isin(id, self.output_neurons_id),
            lambda: _expand(affine),
            lambda: self.hidden_activation(_expand(affine * gain)) * amplification,
        )
        return jnp.squeeze(output)

    def _get_key(self) -> jr.PRNGKey:
        """Get the random key stored in `self.key_state` (an `eqxe.StateIndex`)."""
        return eqxe.get_state(self.key_state, jr.PRNGKey(0))

    def _keygen(self) -> jr.PRNGKey:
        """Get the random key contained in `self.key_state` (an `eqxe.StateIndex`),
        split the key, set `self.key_state` to contain the new key, and return the
        original key.
        """
        key = self._get_key()
        _, new_key = jr.split(key)
        eqxe.set_state(self.key_state, new_key)
        return key

    ############################################################################
    ############ Methods used to set network attributes in __init__ ############  # noqa: E266 E501
    ############################################################################

    def _set_topological_info(
        self,
        graph: nx.DiGraph,
        input_neurons: Sequence[Any],
        output_neurons: Sequence[Any],
        topo_sort: Optional[Sequence[Any]],
    ) -> None:
        """Set the topological information and relevant attributes."""
        # Check that the graph is acyclic
        cycles = nx.simple_cycles(graph)
        if cycles:
            raise ValueError(f"`graph` contains cycles: {cycles}.")

        input_neurons = list(input_neurons)
        output_neurons = list(output_neurons)

        # Check that the input and output neurons are both non-empty
        if not input_neurons:
            raise ValueError("`input_neurons` must be a nonempty sequence.")
        if not output_neurons:
            raise ValueError("`output_neurons` must be a nonempty sequence.")

        # Check that the input and output neurons are disjoint
        input_output_intersection = set(input_neurons) & set(output_neurons)
        if input_output_intersection:
            raise ValueError(
                f"""
                `input_neurons` and `output_neurons` must be disjoint, but
                neurons {list(input_output_intersection)} appear in both."""
            )

        # Topological sort
        if topo_sort is None:
            topo_sort = nx.lexicographical_topological_sort(graph)
        else:
            graph_copy = nx.DiGraph(graph)
            # Check that the provided topological sort is valid
            for neuron in topo_sort:
                if graph_copy.in_degree(neuron):
                    raise ValueError(f"Invalid `topo_sort` at neuron {neuron}.")
                graph_copy.remove_node(neuron)
        topo_sort = list(topo_sort)
        # Make sure input neurons appear first in the topo sort
        first_topo_neurons = topo_sort[: len(input_neurons)]
        if set(input_neurons) != set(first_topo_neurons):
            for neuron in input_neurons:
                topo_sort.remove(neuron)
            topo_sort = input_neurons + topo_sort
        # Make sure output neurons appear last in the topo sort
        last_topo_neurons = topo_sort[-len(output_neurons) :]
        if set(output_neurons) != set(last_topo_neurons):
            for neuron in output_neurons:
                topo_sort.remove(neuron)
            topo_sort = topo_sort + output_neurons

        # Create an adjacency dict that maps a neuron id to its output ids and
        # an inverse adjacency dict that maps a neuron id to its input ids
        adjacency_dict = {}
        adjacency_dict_ = nx.to_dict_of_lists(graph)
        for (input, outputs) in adjacency_dict_.items():
            adjacency_dict[self.neuron_to_id[input]] = [
                self.neuron_to_id[o] for o in outputs
            ]
        adjacency_dict_inv = _invert_dict(adjacency_dict)

        # Store the number of inputs per neuron
        num_inputs_per_neuron = [
            len(adjacency_dict_inv[i]) for i in range(graph.number_of_nodes)
        ]
        self.num_inputs_per_neuron = jnp.array(num_inputs_per_neuron, dtype=float)

        # Check that input neurons themselves have no inputs
        for neuron in input_neurons:
            neurons_in = adjacency_dict_inv[neuron]
            if neurons_in:
                raise ValueError(
                    f"""
                    Input neuron {neuron} has input(s) from neuron(s) {neurons_in}.
                    Input neurons cannot themselves have inputs from other neurons.
                    """
                )

        # Check that output neurons themselves have no outputs
        for neuron in output_neurons:
            neurons_out = adjacency_dict[neuron]
            if neurons_out:
                raise ValueError(
                    f"""
                    Output neuron {neuron} has output(s) to neuron(s) {neurons_out}.
                    Output neurons cannot themselves have outputs to other neurons.
                    """
                )

        # Set the network's underlying graph
        self.graph = graph

        # Map a neuron to its `int` id, which is its position in the topo sort
        neuron_to_id = {neuron: id for (id, neuron) in enumerate(topo_sort)}

        self.topo_sort = topo_sort
        self.num_neurons = len(topo_sort)

        # Set the neuron lists of the adjacency dicts to be in topological order
        self.adjacency_dict = {
            neuron: sorted(outputs, key=lambda n: self.neuron_to_id[n])
            for (neuron, outputs) in adjacency_dict.items()
        }
        self.adjacency_dict_inv = {
            neuron: sorted(inputs, key=lambda n: self.neuron_to_id[n])
            for (neuron, inputs) in adjacency_dict_inv.items()
        }

        self.input_neurons = list(input_neurons)
        self.output_neurons = list(output_neurons)
        self.hidden_neurons = self.topo_sort[len(input_neurons) : -len(output_neurons)]
        input_neurons = [self.neuron_to_id[n] for n in input_neurons]
        output_neurons = [self.neuron_to_id[n] for n in output_neurons]
        self.input_neurons_id = jnp.array(input_neurons, dtype=int)
        self.output_neurons_id = jnp.array(output_neurons, dtype=int)

        # Topological batching
        # See Section 2.2 of https://arxiv.org/abs/2101.07965
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

        # The first topo batch technically has no input, so we don't include
        # it here, since it is handled separately in the forward pass
        self.topo_batches = [jnp.array(tb, dtype=int) for tb in topo_batches[1:]]
        self.num_topo_batches = len(self.topo_batches)
        self.neuron_to_id = neuron_to_id

        # Maps a neuron id to its topological batch and position within that batch
        neuron_to_topo_batch_idx = {}
        for i in range(self.num_topo_batches):
            for (j, n) in enumerate(self.topo_batches[i]):
                neuron_to_topo_batch_idx[int(n)] = (i, j)
        self.neuron_to_topo_batch_idx = neuron_to_topo_batch_idx

    def _set_activations(
        self,
        hidden_activation: Callable,
        output_transformation: Callable,
    ) -> None:
        """Set the activation functions."""
        x = jnp.zeros((1,))
        try:
            y = hidden_activation(x)
        except Exception as e:
            raise Exception(f"Exception caught when checking hidden activation:\n\n{e}")
        if not jnp.array_equal(x.shape, y.shape):
            raise ValueError(
                f"""
                Hidden activation output must have shape (1,) for input of shape (1,).
                Output had shape {y.shape}."""
            )

        num_output_neurons = len(self.output_neurons)
        x = jnp.zeros((num_output_neurons,))
        try:
            y = output_transformation(x)
        except Exception as e:
            raise Exception(
                f"Exception caught when checking output transformation:\n\n{e}"
            )
        if not jnp.array_equal(x.shape, y.shape):
            raise ValueError(
                f"""
                Hidden activation output must have shape ({num_output_neurons},)
                for input of shape ({num_output_neurons},).
                Output had shape {y.shape}."""
            )

        self.hidden_activation = hidden_activation
        self.output_transformation = output_transformation

    def _set_parameters(
        self,
        key: Optional[jr.PRNGKey],
        use_topo_norm: bool,
        use_topo_self_attention: bool,
        use_neuron_self_attention: bool,
        use_adaptive_activations: bool,
    ) -> None:
        """Set the network parameters and relevent topological/indexing information."""
        # Here, `min_index[i]` (`max_index[i]`) is the index representing the minimum
        # (maximum) topological index of those neurons strictly necessary to process
        # `self.topo_batches[i]`. If `i == 0`, the previous topological batch is the
        # input neurons.
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

        # Set the random key. We use eqxe.StateIndex here so that the key can be
        # automatically updated after every forward pass. This ensures that the
        # random values generated for determining dropout application are different
        # for each forward pass if the user does not provide an explicit key.
        key = jr.PRNGKey(0) if key is None else key
        self.key_state = eqxe.StateIndex()
        dkey, key = jr.split(key, 2)
        eqxe.set_state(self.key_state, dkey)

        # Set the network weights and biases. Here, `self.weights_and_biases` is a list
        # of 2D `jnp.ndarray`s, where `self.weights_and_biases[i]` are the weights and
        # biases used by the neurons in `self.topo_batches[i]`.
        #
        # More specifically, `self.weights[i][j, k]` is the weight of the connection
        # from the neuron with topological index `k + mins[i]` to the neuron with index
        # `self.topo_batches[i][j]`, and self.weights[i][j, -1] is the bias of the
        # neuron with index `self.topo_batches[i][j]`. The parameters are stored this
        # way in order to use minimal memory while allowing for maximal `vmap`
        # vectorization during the forward pass, since the minimum and maximum
        # neurons needed to process a topological batch in parallel will usually
        # be closest together when in topological order.
        #
        # All weights and biases are drawn iid ~ N(0, 0.01).
        *wbkeys, key = jr.split(key, self.num_topo_batches + 1)
        topo_lengths = self.max_index - self.min_index + 1
        topo_sizes = [jnp.size(tb) for tb in self.topo_batches]
        self.topo_lengths = topo_lengths
        self.topo_sizes = topo_sizes
        self.weights_and_biases = [
            jr.normal(wbkeys[i], (topo_sizes[i], topo_lengths[i] + 1)) * 0.1
            for i in range(self.num_topo_batches)
        ]

        # Here, `self.masks` is a list of 2D binary `jnp.ndarray` with identical
        # structure to `self.weights`. These are multiplied by the weights during
        # the forward pass to mask out weights for connections that are not present
        # in the network.
        masks = []
        for (tb, w_and_b, min_idx) in zip(
            self.topo_batches, self.weights_and_biases, self.min_index
        ):
            weights = w_and_b[:, :-1]
            mask = np.zeros_like(weights)
            for (i, neuron) in enumerate(tb):
                inputs = jnp.array(self.adjacency_dict_inv[int(neuron)], dtype=int)
                mask[i, inputs - min_idx] = 1
            masks.append(jnp.array(mask, dtype=int))
        self.masks = masks

        # Set parameters for Topo Norm
        self.use_topo_norm = bool(use_topo_norm)
        if self.use_topo_norm:
            *nkeys, key = jr.split(key, self.num_topo_batches + 1)
            self.topo_norm_params = [
                jr.normal(nkeys[i], (2, topo_lengths[i])) * 0.1 + 1
                for i in range(self.num_topo_batches)
            ]
        else:
            self.topo_norm_params = [jnp.nan] * self.num_topo_batches

        # Set parameters and masks for topo-wise self-attention
        self.use_topo_self_attention = bool(use_topo_self_attention)
        if self.use_topo_self_attention:
            *akeys, key = jr.split(key, self.num_topo_batches + 1)
            self.attention_params_topo = [
                jr.normal(akeys[i], (3, topo_lengths[i], topo_lengths[i] + 1)) * 0.1
                for i in range(self.num_topo_batches)
            ]
        else:
            self.attention_params_topo = [jnp.nan] * self.num_topo_batches

        # Set parameters and masks for neuron-wise self-attention
        self.use_neuron_self_attention = bool(use_neuron_self_attention)
        if self.use_neuron_self_attention:
            *akeys, key = jr.split(key, self.num_topo_batches + 1)
            self.attention_params_neuron = [
                jr.normal(
                    akeys[i], (topo_sizes[i], 3, topo_lengths[i], topo_lengths[i] + 1)
                )
                * 0.1
                for i in range(self.num_topo_batches)
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

        # Set trainable parameters for neuron-wise adaptive activations
        # (if applicable), drawn iid ~ N(1, 0.01).
        self.use_adaptive_activations = bool(use_adaptive_activations)
        if self.use_adaptive_activations:
            akeys = jr.split(key, self.num_topo_batches)
            self.adaptive_activation_params = [
                jr.normal(akeys[i], (topo_sizes[i], 2)) * 0.1 + 1
                for i in range(self.num_topo_batches)
            ]
        else:
            self.adaptive_activation_params = [jnp.nan] * self.num_topo_batches

    def _set_dropout_p(self, dropout_p: Union[float, Mapping[Any, float]]) -> None:
        """Set the initial per-neuron dropout probabilities."""
        dropout_dict = {neuron: 0.0 for neuron in self.topo_sort}
        if isinstance(dropout_p, float):
            for neuron in self.hidden_neurons:
                dropout_dict[neuron] = dropout_p
            _dropout_p = jnp.ones((self.num_neurons,)) * dropout_p
            _dropout_p = _dropout_p.at[self.input_neurons_id].set(0.0)
            _dropout_p = _dropout_p.at[self.output_neurons_id].set(0.0)
        else:
            assert isinstance(dropout_p, Mapping)
            _dropout_p = np.zeros((self.num_neurons,))
            for (n, d) in dropout_p.items():
                assert n in self.neuron_to_id
                assert isinstance(d, float)
                _dropout_p[self.neuron_to_id[n]] = d
                dropout_dict[neuron] = d
            _dropout_p = jnp.array(_dropout_p, dtype=float)
        assert jnp.all(jnp.greater_equal(_dropout_p, 0))
        assert jnp.all(jnp.less_equal(_dropout_p, 1))
        self._dropout_p = _dropout_p
        self.dropout_p = dropout_dict

    #####################################################
    ################## Public methods ###################  # noqa: E266
    #####################################################

    def copy(self) -> "NeuralNetwork":
        """**Returns:**

        A copy of the network with no modification.
        """
        return tree_at(lambda net: net.num_neurons, self, self.num_neurons)

    def enable_neuron_self_attention(
        self, *, key: Optional[jr.PRNGKey] = None
    ) -> "NeuralNetwork":
        """Enable the network to use neuron-level self-attention, where each neuron
        has its own unique (single-headed) self-attention module for its inputs.

        **Arguments:**

        - `key`: A `jax.random.PRNGKey` for the initialization of the attention
            parameters. Optional, keyword-only argument.
            Defaults to `jax.random.PRNGKey(0)`.

        **Returns:**

        A copy of the current network with new trainable neuron-level self-attention
        parameters. If the network already has neuron-level self-attention enabled,
        a copy of the network is returned without modification.

        Note that if `disable_neuron_self_attention` had previously been called, those
        parameters were wiped out and so the new self-attention parameters will _not_
        revert to the parameters prior to the call to `disable_neuron_self_attention`.
        """
        # If already enabled, return a copy of the network
        if self.use_neuron_self_attention:
            return self.copy()

        # Random key
        key = key if key is not None else jr.PRNGKey(0)

        # Parameters and masks for neuron-wise self-attention
        *akeys, key = jr.split(key, self.num_topo_batches + 1)
        attention_params_neuron = [
            jr.normal(
                akeys[i],
                (
                    self.topo_sizes[i],
                    3,  # query, key, value
                    self.topo_lengths[i],
                    self.topo_lengths[i] + 1,
                ),
            )
            * 0.1
            for i in range(self.num_topo_batches)
        ]
        outer_product = vmap(lambda x: jnp.outer(x, x))
        mask_fn = filter_jit(lambda m: jnp.where(outer_product(m), 0, jnp.inf))
        attention_masks_neuron = [mask_fn(mask) for mask in self.masks]

        # Set parameters and return
        return tree_at(
            lambda network: (
                network.use_neuron_self_attention,
                network.attention_params_neuron,
                network.attention_masks_neuron,
            ),
            self,
            (True, attention_params_neuron, attention_masks_neuron),
        )

    def disable_neuron_self_attention(self) -> "NeuralNetwork":
        """Disable the network from using neuron-level self-attention, where each
        neuron has its own unique (single-headed) self-attention module for its inputs.

        **Returns:**

        A copy of the current network with neuron-level self-attention parameters
        removed. If the network already has neuron-level self-attention disabled,
        a copy of the network is returned without modification.

        Note that this method wipes out the neuron-level self-attention parameters
        (on the returned copy) to minimize unnecessary memory use. If this function
        is called, followed by `enable_neuron_level_self_attention` on the returned
        network, those parameters will not revert to what they previously were in the
        original network.
        """
        # If already disabled, return a copy of the network
        if not self.use_neuron_self_attention:
            return self.copy()

        # Return copy of network with neuron-level self-attention parameters wiped out
        nan_list = [jnp.nan] * self.num_topo_batches
        return tree_at(
            lambda network: (
                network.use_neuron_self_attention,
                network.attention_params_neuron,
                network.attention_masks_neuron,
            ),
            self,
            (False, nan_list, nan_list),
        )

    def enable_topo_self_attention(
        self, *, key: Optional[jr.PRNGKey] = None
    ) -> "NeuralNetwork":
        """Enable the network to use topological-level self-attention, where
        self-attention is applied to each topological batch's collective inputs
        prior to undergoing an affine (linear) transformation.

        **Arguments:**

        - `key`: A `jax.random.PRNGKey` for the initialization of the attention
            parameters. Optional, keyword-only argument.
            Defaults to `jax.random.PRNGKey(0)`.

        **Returns:**

        A copy of the current network with new trainable topological-level
        self-attention parameters. If the network already has topological-level
        self-attention enabled, a copy of the network is returned without
        modification.

        Note that if `disable_topo_self_attention` had previously been called, those
        parameters were wiped out in the returned network, so the new self-attention
        parameters will _not_ revert to the original parameters.
        """
        # If already enabled, return a copy of the network
        if self.use_topo_self_attention:
            return self.copy()

        # Random key
        key = key if key is not None else jr.PRNGKey(0)

        # Parameters for topological-level self-attention
        *akeys, key = jr.split(key, self.num_topo_batches + 1)
        attention_params_topo = [
            jr.normal(akeys[i], (3, self.topo_lengths[i], self.topo_lengths[i] + 1))
            * 0.1
            for i in range(self.num_topo_batches)
        ]

        # Set parameters and return
        return tree_at(
            lambda network: (
                network.use_topo_self_attention,
                network.attention_params_topo,
            ),
            self,
            (True, attention_params_topo),
        )

    def disable_topo_self_attention(self) -> "NeuralNetwork":
        """Disable the network from using topological-level self-attention, where
        self-attention is applied to each topological batch's collective inputs.

        **Returns:**

        A copy of the current network with topological-level self-attention parameters
        removed. If the network already has topological-level self-attention disabled,
        a copy of the network is returned without modification.

        Note that this method wipes out the topological-level self-attention
        parameters on the returned copy to minimize unnecessary memory use. If this
        function is called, followed by `enable_topological_level_self_attention`,
        those parameters will not revert to their original values prior to calling
        this method.
        """
        # If already disabled, return a copy of the network
        if not self.use_topo_self_attention:
            return self.copy()

        # Return copy of network with topological-level self-attention
        # parameters wiped out
        return tree_at(
            lambda network: (
                network.use_topo_self_attention,
                network.attention_params_topo,
            ),
            self,
            (False, [jnp.nan] * self.num_topo_batches),
        )

    def enable_topo_norm(self, *, key: Optional[jr.PRNGKey] = None) -> "NeuralNetwork":
        """Enable the network to use "Topo Norm", the equivalent of Layer Norm for
        topological batches.

        More specifically, each topological batch's collective inputs are standardized
        (centered and rescaled), followed by a learnable element-wise affine
        transformation. This occurs before any further processing of the
        topological batch (e.g. attention, affine transformation).

        **Arguments:**

        - `key`: A `jax.random.PRNGKey` for the initialization of the Topo Norm
            parameters. Optional, keyword-only argument.
            Defaults to `jax.random.PRNGKey(0)`.

        **Returns:**

        A copy of the current network with new trainable Topo Norm parameters.
        If the network already has Topo Norm, a copy of the network is
        returned without modification.

        Note that if `disable_topo_norm` had previously been called, those parameters
        were wiped out in the returned network and so the new Topo Norm parameters will
        _not_ revert to the original parameters prior to the call to
        `disable_topo_norm`.
        """
        # If already enabled, return a copy of the network
        if self.use_topo_norm:
            return self.copy()

        # Random key
        key = key if key is not None else jr.PRNGKey(0)

        # Parameters for Topo Norm
        *nkeys, key = jr.split(key, self.num_topo_batches + 1)
        topo_norm_params = [
            jr.normal(nkeys[i], (2, self.topo_lengths[i])) * 0.1 + 1
            for i in range(self.num_topo_batches)
        ]

        # Set parameters and return
        return tree_at(
            lambda network: (network.use_topo_norm, network.topo_norm_params),
            self,
            (True, topo_norm_params),
        )

    def disable_topo_norm(self) -> "NeuralNetwork":
        """Disable the network from using "Topo Norm", the equivalent of Layer Norm
        for topological batches.

        More specifically, with Topo Norm enabled, each topological batch's collective
        inputs are standardized (centered and rescaled), followed by a learnable
        element-wise affine transformation. This occurs before any further processing
        of the topolical batch (e.g. attention, affine transformation).

        **Returns:**

        A copy of the current network with Topo Norm parameters removed.
        If the network already has Topo Norm disabled, a copy of the network is
        returned without modification.

        Note that this method wipes out the Topo Norm parameters (on the returned
        copy) to minimize unnecessary memory use. If this function is called, followed
        by `enable_topo_norm`, those parameters will not revert to what they previously
        were.
        """
        # If already disabled, return a copy of the network
        if not self.use_topo_norm:
            return self.copy()

        # Return copy of network with Topo Norm parameters wiped out
        return tree_at(
            lambda network: (network.use_topo_norm, network.topo_norm_params),
            self,
            (False, [jnp.nan] * self.num_topo_batches),
        )

    def enable_adaptive_activations(
        self, *, key: Optional[jr.PRNGKey] = None
    ) -> "NeuralNetwork":
        """Enable the network to use adaptive activations, where all hidden activations
        transform as `σ(x) -> a * σ(b * x)`, where `a`, `b` are trainable scalar
        parameters unique to each hidden neuron.

        **Arguments:**

        - `key`: A `jax.random.PRNGKey` for the initialization of the adaptive
            activation parameters. Optional, keyword-only argument.
            Defaults to `jax.random.PRNGKey(0)`.

        **Returns:**

        A copy of the current network with new trainable adaptive activation
        parameters. If the network already has adaptive activations, a copy
        of the network is returned without modification.

        Note that if `disable_adaptive activations` had previously been called, those
        parameters were wiped out in the returned network and so the new adaptive
        activation parameters will _not_ revert to the parameters prior to the call to
        `disable_adaptive_activations`.
        """
        # If already enabled, return a copy of the network
        if self.use_adaptive_activations:
            return self.copy()

        # Random key
        key = key if key is not None else jr.PRNGKey(0)

        # Parameters for adaptive activations
        akeys = jr.split(key, self.num_topo_batches)
        adaptive_activation_params = [
            jr.normal(akeys[i], (self.topo_sizes[i], 2)) * 0.1 + 1
            for i in range(self.num_topo_batches)
        ]

        # Set parameters and return
        return tree_at(
            lambda network: (
                network.use_adaptive_activations,
                network.adaptive_activation_params,
            ),
            self,
            (True, adaptive_activation_params),
        )

    def disable_adaptive_activations(self) -> "NeuralNetwork":
        """Disable the network from using adaptive activations, where all hidden
        activations transform as `σ(x) -> a * σ(b * x)`, where `a`, `b` are
        trainable scalar parameters unique to each hidden neuron.

        **Returns:**

        A copy of the current network with adaptive activation parameters removed.
        If the network already has adaptive activation disabled, a copy of the network
        is returned without modification.

        Note that this method wipes out the adaptive activation parameters (on the
        returned copy) to minimize unnecessary memory use. If this function is called,
        followed by `enable_adaptive_activations`, those parameters will not revert to
        what they previously were prior to calling this method.
        """
        # If already disabled, return a copy of the network
        if not self.use_adaptive_activations:
            return self.copy()

        # Return copy of network with adaptive activation parameters wiped out
        return tree_at(
            lambda network: (
                network.use_adaptive_activations,
                network.adaptive_activation_params,
            ),
            self,
            (False, [jnp.nan] * self.num_topo_batches),
        )

    def set_dropout_p(self, dropout_p: Union[float, Mapping[Any, float]]) -> None:
        """Set the per-neuron dropout probabilities.

        **Arguments:**

        - `dropout_p`: Either a float or mapping from neuron (`Any`) to float. If a
            single float, all hidden neurons will have that dropout probability, and
            all input and output neurons will have dropout probability 0 by default.
            If a `Mapping`, it is assumed that `dropout_p` maps a neuron to its dropout
            probability, and all unspecified neurons will retain their current dropout
            probability.

        **Returns:**

        A copy of the current network with dropout probabilities as specified.
        The original network (including unspecified dropout probabilities) is left
        unchanged.
        """
        dropout_dict = self.dropout_p
        if isinstance(dropout_p, float):
            # Set all hidden neurons to have dropout probability `dropout_p`, and
            # all input/output neurons to have dropout probability 0.
            for neuron in self.hidden_neurons:
                dropout_dict[neuron] = dropout_p
            _dropout_p = jnp.ones((self.num_neurons,)) * dropout_p
            _dropout_p = _dropout_p.at[self.input_neurons_id].set(0.0)
            _dropout_p = _dropout_p.at[self.output_neurons_id].set(0.0)
        else:
            assert isinstance(dropout_p, Mapping)
            _dropout_p = np.array(self._dropout_p)
            for (n, d) in dropout_p.items():
                assert n in self.neuron_to_id
                assert isinstance(d, float)
                _dropout_p[self.neuron_to_id[n]] = d
                dropout_dict[n] = d
            _dropout_p = jnp.array(_dropout_p, dtype=float)
        assert jnp.all(jnp.greater_equal(_dropout_p, 0))
        assert jnp.all(jnp.less_equal(_dropout_p, 1))
        return tree_at(
            lambda network: (network.dropout_p, network._dropout_p),
            self,
            (dropout_dict, _dropout_p),
        )

    def to_networkx_weighted_digraph(self) -> nx.DiGraph:
        """Returns a `networkx.DiGraph` represention of the network with neuron weights
        saved as edge attributes. This may be useful for applying network analysis
        techniques to the neural network.

        **Returns**:

        A `networkx.DiGraph` object that represents the network, with neuron weights
        saved as edge attributes. The original graph used to initialize the network is
        left unchanged.
        """
        graph = deepcopy(self.graph)

        edge_attrs = {}
        edge_attrs_ = {(i, o): data for (i, o, data) in graph.edges(data=True)}
        for (neuron, outputs) in self.adjacency_dict.items():
            topo_batch_idx, pos_idx = self.neuron_to_topo_batch_idx[neuron]
            for output in outputs:
                col_idx = output - self.min_index[topo_batch_idx]
                assert self.masks[topo_batch_idx][pos_idx, col_idx]
                weight = self.weights_and_biases[topo_batch_idx][pos_idx, col_idx]
                edge_attrs[(neuron, output)] = {
                    "weight": weight,
                    **edge_attrs_[(neuron, output)],
                }
        nx.set_edge_attributes(graph, edge_attrs)

        return graph
