import functools as ft
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Optional, Union

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np
from equinox import filter_jit, Module, static_field
from jax import Array, jit, lax, vmap

from ._utils import _identity, _invert_dict, _keygen, DiGraphLike


class NeuralNetwork(Module):
    """A neural network whose structure is specified by a DAG."""

    _weights_and_biases: list[Array]
    _hidden_activation: Callable
    _output_transformation: Callable
    _topo_norm_params: list[Array]
    _attention_params_topo: list[Array]
    _attention_params_neuron: list[Array]
    _adaptive_activation_params: list[Array]
    _dropout_dict: dict[Any, float]
    _dropout_array: Array
    _graph: nx.DiGraph = static_field()
    _adjacency_dict: dict[Any, list[Any]] = static_field()
    _adjacency_dict_inv: dict[Any, list[Any]] = static_field()
    _neuron_to_id: dict[Any, int] = static_field()
    _topo_batches: list[Array] = static_field()
    _topo_lengths: np.ndarray = static_field()
    _topo_sizes: list[int] = static_field()
    _num_topo_batches: int = static_field()
    _neuron_to_topo_batch_idx: dict[int, tuple[int, int]] = static_field()
    _topo_sort: list[Any] = static_field()
    _min_index: np.ndarray = static_field()
    _max_index: np.ndarray = static_field()
    _masks: list[Array] = static_field()
    _attention_masks_neuron: list[Array] = static_field()
    _indices: list[Array] = static_field()
    _input_neurons: list[Any] = static_field()
    _output_neurons: list[Any] = static_field()
    _hidden_neurons: list[Any] = static_field()
    _input_neurons_id: Array = static_field()
    _output_neurons_id: Array = static_field()
    _num_neurons: int = static_field()
    _num_inputs_per_neuron: Array = static_field()
    _use_topo_norm: bool = static_field()
    _use_topo_self_attention: bool = static_field()
    _use_neuron_self_attention: bool = static_field()
    _use_adaptive_activations: bool = static_field()

    def __init__(
        self,
        graph_data: DiGraphLike,
        input_neurons: Sequence[Any],
        output_neurons: Sequence[Any],
        hidden_activation: Callable = jnn.gelu,
        output_transformation: Callable = _identity,
        dropout_p: Union[float, Mapping[Any, float]] = 0.0,
        use_topo_norm: bool = False,
        use_topo_self_attention: bool = False,
        use_neuron_self_attention: bool = False,
        use_adaptive_activations: bool = False,
        topo_sort: Optional[Sequence[Any]] = None,
        *,
        key: Optional[jr.PRNGKey] = None,
    ):
        r"""**Arguments**:

        - `graph_data`: A `networkx.DiGraph`, or data that can be turned into a
            `networkx.DiGraph`  by calling `networkx.DiGraph(graph_data)`
            (such as an adjacency dict) representing the DAG structure of the neural
            network. All nodes of the graph must have the same type.
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
            (single-headed) self-attention to each topological batch's collective
            inputs.

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
            its inputs. If both `_use_neuron_self_attention` and `use_neuron_norm` are
            `True`, normalization is applied before self-attention.

            !!! warning

                Neuron-level self-attention will use significantly more memory than
                than topo-level self-attention.

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
            the graph. If `None`, a topological sort will be performed on the graph, which
            may be time-consuming for some networks.
        - `key`: The `jax.random.PRNGKey` used for parameter initialization and dropout.
            Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.
        """
        if not isinstance(graph_data, nx.DiGraph):
            graph_data = nx.DiGraph(graph_data)
        self._set_topological_info(graph_data, input_neurons, output_neurons, topo_sort)
        self._set_activations(hidden_activation, output_transformation)
        self._set_parameters(
            use_topo_norm,
            use_topo_self_attention,
            use_neuron_self_attention,
            use_adaptive_activations,
            key,
        )
        self._set_dropout_p_initial(dropout_p)

    @filter_jit
    def __call__(self, x: Array, *, key: Optional[jr.PRNGKey] = None) -> Array:
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
            argument. If `None`, a key will be generated using the current time.

        **Returns**:

        The result array from the forward pass. The order of the array elements will be
        the order of the output neurons passed in during initialization.
        """
        # Neuron value array, updated as neurons are "fired"
        values = jnp.zeros((self._num_neurons,))

        # Dropout
        key = _keygen() if key is None else key
        rand = jr.uniform(key, self._dropout_array.shape, minval=0, maxval=1)
        dropout_mask = jnp.greater(rand, self._dropout_array)

        # Set input values
        values = values.at[self._input_neurons_id].set(
            x * dropout_mask[self._input_neurons_id]
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
            self._topo_batches,
            self._weights_and_biases,
            self._masks,
            self._indices,
            self._attention_params_topo,
            self._attention_params_neuron,
            self._attention_masks_neuron,
            self._topo_norm_params,
            self._adaptive_activation_params,
        ):
            # Previous neuron values strictly necessary to process the current
            # topological batch
            vals = values[indices]

            # Topo Norm
            if self._use_topo_norm:
                vals = self._apply_topo_norm(norm_params, vals)

            # Topo-level self-attention
            if self._use_topo_self_attention:
                vals = self._apply_topo_self_attention(attn_params_t, vals)

            # Neuron-level self-attention
            if self._use_neuron_self_attention:
                _apply_neuron_self_attention = vmap(
                    self._apply_neuron_self_attention, in_axes=(0, 0, 0, None)
                )
                vals = _apply_neuron_self_attention(
                    tb, attn_params_n, attn_mask_n, vals
                )

            # Affine transformation, wx + b
            weights, biases = w_and_b[:, :-1], w_and_b[:, -1]
            if self._use_neuron_self_attention:
                affine = vmap(jnp.dot)(weights * mask, vals) + biases
            else:
                affine = (weights * mask) @ vals + biases

            # Apply activations/dropout
            if not self._use_adaptive_activations:
                ada_params = jnp.ones((tb.shape[0], 2))
            output_values = (
                vmap(self._apply_activation)(tb, affine, ada_params) * dropout_mask[tb]
            )

            # Set new values
            values = values.at[tb].set(output_values)

        # Return values pertaining to output neurons, with the group-wise
        # output activation applied
        return self._output_transformation(values[self._output_neurons_id])

    ##############################################################################
    ################ Methods used during forward pass in __call__ ################  # noqa: E266 E501
    ##############################################################################

    def _apply_topo_norm(self, norm_params: Array, vals: Array) -> Array:
        """Function for a topo batch to standardize its inputs, followed by a
        learnable elementwise-affine transformation.
        """
        gamma, beta = norm_params[:, 0], norm_params[:, 1]
        return lax.cond(
            jnp.size(vals) > 1,  # Don't standardize if there is only one neuron
            lambda: jnn.standardize(vals) * gamma + beta,
            lambda: vals,
        )

    def _apply_topo_self_attention(self, attn_params: Array, vals: Array) -> Array:
        """Function for a topo batch to apply self-attention to its collective inputs,
        followed by a skip connection.
        """
        query_params, key_params, value_params = attn_params[:]
        query_weight, query_bias = query_params[:, :-1], query_params[:, -1]
        key_weight, key_bias = key_params[:, :-1], key_params[:, -1]
        value_weight, value_bias = value_params[:, :-1], value_params[:, -1]
        query = query_weight @ vals + query_bias
        key = key_weight @ vals + key_bias
        value = value_weight @ vals + value_bias
        rsqrt = lax.rsqrt(float(jnp.size(vals)))
        scaled_outer_product = jnp.outer(query, key) * rsqrt
        attention_weights = jnn.softmax(scaled_outer_product)
        return attention_weights @ value + vals

    def _apply_neuron_self_attention(
        self, id: Array, attn_params: Array, attn_mask: Array, vals: Array
    ) -> Array:
        """Function for a single neuron to apply self-attention to its inputs,
        followed by a skip connection.
        """
        query_params, key_params, value_params = attn_params[:]
        query_weight, query_bias = query_params[:, :-1], query_params[:, -1]
        key_weight, key_bias = key_params[:, :-1], key_params[:, -1]
        value_weight, value_bias = value_params[:, :-1], value_params[:, -1]
        query = query_weight @ vals + query_bias
        key = key_weight @ vals + key_bias
        value = value_weight @ vals + value_bias
        rsqrt = lax.rsqrt(self._num_inputs_per_neuron[id])
        scaled_outer_product = jnp.outer(query, key) * rsqrt
        attention_weights = jnn.softmax(scaled_outer_product - attn_mask)
        return attention_weights @ value + vals

    def _apply_activation(self, id: Array, affine: Array, ada_params: Array) -> Array:
        """Function for a single neuron to apply its activation."""
        a, b = ada_params
        _expand = ft.partial(jnp.expand_dims, axis=0)
        output = lax.cond(
            jnp.isin(id, self._output_neurons_id),
            lambda: _expand(affine),
            lambda: self._hidden_activation(_expand(affine * b)) * a,
        )
        return jnp.squeeze(output)

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

        def _validate_neurons(graph, input_neurons, output_neurons):
            for neuron in input_neurons:
                if neuron not in graph.nodes():
                    raise ValueError(
                        f"`input_neurons` contains a neuron not in the network: {neuron}"  # noqa: E501
                    )
            for neuron in output_neurons:
                if neuron not in graph.nodes():
                    raise ValueError(
                        f"`output_neurons` contains a neuron not in the network: {neuron}"  # noqa: E501
                    )

            if not (len(input_neurons) and len(output_neurons)):
                raise ValueError(
                    "`input_neurons` and `output_neurons` must be nonempty sequences."
                )

            input_output_intersection = set(input_neurons) & set(output_neurons)
            if input_output_intersection:
                raise ValueError(
                    f"Neurons {list(input_output_intersection)} appear in both input and output neurons."  # noqa: E501
                )

        def _validate_topological_sort(graph, topo_sort, input_neurons, output_neurons):
            if topo_sort is None:
                topo_sort = list(nx.topological_sort(graph))
            else:
                # Check if provided topo_sort is valid
                graph = nx.DiGraph(graph)
                for neuron in topo_sort:
                    if graph.in_degree(neuron):
                        raise ValueError(f"Invalid `topo_sort` at neuron {neuron}.")
                    graph.remove_node(neuron)
                # Check that the graph is acyclic
                cycles = list(nx.simple_cycles(graph))
                if cycles:
                    raise ValueError(f"`graph` contains cycles: {cycles}.")

            # Make sure input and output neurons are at the beginning and end of
            # topo_sort
            topo_sort_ = list(deepcopy(topo_sort))
            input_neurons, output_neurons = list(input_neurons), list(output_neurons)
            for n in input_neurons + output_neurons:
                topo_sort_.remove(n)
            topo_sort = input_neurons + topo_sort_ + output_neurons
            self._neuron_to_id = {neuron: id for (id, neuron) in enumerate(topo_sort)}

            return topo_sort

        def _compute_topological_batches(topo_sort):
            # See Section 2.2 of https://arxiv.org/abs/2101.07965
            graph_copy = nx.DiGraph(self._graph)
            topo_batches, topo_batch, neurons_to_remove = [], [], []
            for neuron in topo_sort:
                if graph_copy.in_degree(neuron) == 0:
                    topo_batch.append(self._neuron_to_id[neuron])
                    neurons_to_remove.append(neuron)
                else:
                    topo_batches.append(np.array(topo_batch, dtype=int))
                    graph_copy.remove_nodes_from(neurons_to_remove)
                    topo_batch = [self._neuron_to_id[neuron]]
                    neurons_to_remove = [neuron]
            topo_batches.append(np.array(topo_batch, dtype=int))

            self._topo_batches = [jnp.array(tb, dtype=int) for tb in topo_batches[1:]]
            self._num_topo_batches = len(self._topo_batches)

            # Map each neuron id to that neuron's topological batch
            # and index within that batch
            self._neuron_to_topo_batch_idx = {
                int(n): (i, j)
                for i, batch in enumerate(self._topo_batches)
                for j, n in enumerate(batch)
            }

        def _process_topological_information(
            graph, topo_sort, input_neurons, output_neurons
        ):
            adjacency_dict = nx.to_dict_of_lists(graph)
            adjacency_dict_inv = _invert_dict(adjacency_dict)

            self._num_inputs_per_neuron = jnp.array(
                [len(adjacency_dict_inv[neuron]) for neuron in topo_sort], dtype=float
            )

            for neuron in input_neurons:
                if adjacency_dict_inv[neuron]:
                    raise ValueError(
                        f"Input neuron {neuron} has input(s) from neuron(s) \
                            {adjacency_dict_inv[neuron]}."
                    )

            for neuron in output_neurons:
                if adjacency_dict[neuron]:
                    raise ValueError(
                        f"Output neuron {neuron} has output(s) to neuron(s) \
                            {adjacency_dict[neuron]}."
                    )

            self._graph = graph
            self._topo_sort = topo_sort
            self._num_neurons = len(topo_sort)

            # Reorder the adjacency dicts so the neuron lists are in topological order
            self._adjacency_dict = {
                neuron: sorted(outputs, key=lambda n: self._neuron_to_id[n])
                for (neuron, outputs) in adjacency_dict.items()
            }
            self._adjacency_dict_inv = {
                neuron: sorted(inputs, key=lambda n: self._neuron_to_id[n])
                for (neuron, inputs) in adjacency_dict_inv.items()
            }

            self._input_neurons = list(input_neurons)
            self._output_neurons = list(output_neurons)
            self._hidden_neurons = topo_sort[len(input_neurons) : -len(output_neurons)]

            self._input_neurons_id = jnp.array(
                [self._neuron_to_id[n] for n in input_neurons], dtype=int
            )
            self._output_neurons_id = jnp.array(
                [self._neuron_to_id[n] for n in output_neurons], dtype=int
            )

            _compute_topological_batches(topo_sort)

        # Check that the graph is acyclic
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            raise ValueError(f"`graph` contains cycles: {cycles}.")

        _validate_neurons(graph, input_neurons, output_neurons)

        topo_sort = _validate_topological_sort(
            graph, topo_sort, input_neurons, output_neurons
        )

        # Process and store topological information
        _process_topological_information(
            graph, topo_sort, input_neurons, output_neurons
        )

    def _set_activations(
        self, hidden_activation: Callable, output_transformation: Callable
    ) -> None:
        """Set the activation functions."""

        def _validate_activation_function(
            activation_func: Callable, input_shape: tuple[int]
        ) -> None:
            x = jnp.zeros(input_shape)

            try:
                y = activation_func(x)
            except Exception as e:
                raise Exception(
                    f"Exception caught when checking activation function:\n\n{e}"
                )

            if not jnp.array_equal(x.shape, y.shape):
                raise ValueError(
                    f"Activation function output must have shape {input_shape} \
                        for input of shape {input_shape}. Output had shape {y.shape}."
                )

        _validate_activation_function(hidden_activation, (1,))
        _validate_activation_function(
            output_transformation, (len(self._output_neurons),)
        )

        self._hidden_activation = hidden_activation
        self._output_transformation = output_transformation

    def _set_parameters(
        self,
        use_topo_norm: bool,
        use_topo_self_attention: bool,
        use_neuron_self_attention: bool,
        use_adaptive_activations: bool,
        key: Optional[jr.PRNGKey],
    ) -> None:
        """Set the network parameters and relevant topological/indexing information."""

        # Set the random key
        key = jr.PRNGKey(0) if key is None else key

        def _compute_min_max_indices():
            """
            Compute the minimum and maximum topological index for neurons strictly
            needed to process each topological batch.
            """
            min_indices, max_indices = [], []
            for tb in self._topo_batches:
                input_locs = [
                    [
                        self._neuron_to_id[n]
                        for n in self._adjacency_dict_inv[self._topo_sort[int(i)]]
                    ]
                    for i in tb
                ]
                min_indices.append(
                    np.min([np.min(locs) for locs in input_locs if locs])
                )
                max_indices.append(
                    np.max([np.max(locs) for locs in input_locs if locs])
                )
            return np.array(min_indices, dtype=int), np.array(max_indices, dtype=int)

        self._min_index, self._max_index = _compute_min_max_indices()

        self._topo_lengths = self._max_index - self._min_index + 1
        self._topo_sizes = [jnp.size(tb) for tb in self._topo_batches]

        def _initialize_weights_and_biases(key):
            """
            Set the network weights and biases. Here, `self._weights_and_biases` is a
            list of 2D `jnp.ndarray`s, where `self._weights_and_biases[i]` are the
            weights and biases used by the neurons in `self._topo_batches[i]`.

            More specifically, `self.weights[i][j, k]` is the weight of the connection
            from the neuron with topological index `k + mins[i]` to the neuron with
            index `self._topo_batches[i][j]`, and self.weights[i][j, -1] is the bias of
            the neuron with index `self._topo_batches[i][j]`. The parameters are stored
            this way in order to use minimal memory while allowing for maximal `vmap`
            vectorization during the forward pass, since the minimum and maximum
            neurons needed to process a topological batch in parallel will usually
            be closest together when in topological order.

            All weights and biases are drawn iid ~ N(0, 0.01).
            """
            *wbkeys, key = jr.split(key, self._num_topo_batches + 1)
            return [
                jr.normal(wbkeys[i], (self._topo_sizes[i], self._topo_lengths[i] + 1))
                * 0.1
                for i in range(self._num_topo_batches)
            ]

        def _initialize_masks():
            """Initial the masks for the weight matrices."""
            masks = []
            for tb, w_and_b, min_idx in zip(
                self._topo_batches, self._weights_and_biases, self._min_index
            ):
                weights = w_and_b[:, :-1]
                mask = np.zeros_like(weights)
                for i, id in enumerate(tb):
                    neuron = self._topo_sort[int(id)]
                    inputs = jnp.array(
                        list(
                            map(
                                self._neuron_to_id.get, self._adjacency_dict_inv[neuron]
                            )
                        ),
                        dtype=int,
                    )
                    mask[i, inputs - min_idx] = 1
                masks.append(jnp.array(mask, dtype=int))
            return masks

        self._weights_and_biases = _initialize_weights_and_biases(key)
        self._masks = _initialize_masks()

        def _initialize_topo_norm_params(key):
            if not self._use_topo_norm:
                return [jnp.nan] * self._num_topo_batches

            *nkeys, key = jr.split(key, self._num_topo_batches + 1)
            return [
                jr.normal(nkeys[i], (self._topo_lengths[i], 2)) * 0.1 + 1
                for i in range(self._num_topo_batches)
            ]

        def _initialize_attention_params_topo(key):
            if not self._use_topo_self_attention:
                return [jnp.nan] * self._num_topo_batches

            *akeys, key = jr.split(key, self._num_topo_batches + 1)
            return [
                jr.normal(  # query, key, value
                    akeys[i], (3, self._topo_lengths[i], self._topo_lengths[i] + 1)
                )
                * 0.1
                for i in range(self._num_topo_batches)
            ]

        def _initialize_attention_params_neuron(key):
            if not self._use_neuron_self_attention:
                return [jnp.nan] * self._num_topo_batches, [
                    jnp.nan
                ] * self._num_topo_batches

            *akeys, key = jr.split(key, self._num_topo_batches + 1)
            attention_params_neuron = [
                jr.normal(
                    akeys[i],
                    (
                        self._topo_sizes[i],
                        3,  # query, key, value
                        self._topo_lengths[i],
                        self._topo_lengths[i] + 1,
                    ),
                )
                * 0.1
                for i in range(self._num_topo_batches)
            ]
            outer_product = vmap(lambda x: jnp.outer(x, x))
            mask_fn = jit(lambda m: jnp.where(outer_product(m), 0, jnp.inf))
            attention_masks_neuron = [mask_fn(mask) for mask in self._masks]

            return attention_params_neuron, attention_masks_neuron

        def _initialize_adaptive_activation_params(key):
            if not self._use_adaptive_activations:
                return [jnp.nan] * self._num_topo_batches

            akeys = jr.split(key, self._num_topo_batches)
            return [
                jr.normal(akeys[i], (self._topo_sizes[i], 2)) * 0.1 + 1
                for i in range(self._num_topo_batches)
            ]

        self._use_topo_norm = bool(use_topo_norm)
        self._use_topo_self_attention = bool(use_topo_self_attention)
        self._use_neuron_self_attention = bool(use_neuron_self_attention)
        self._use_adaptive_activations = bool(use_adaptive_activations)

        key1, key2, key3, key4 = jr.split(key, 4)
        self._topo_norm_params = _initialize_topo_norm_params(key1)
        self._attention_params_topo = _initialize_attention_params_topo(key2)
        (
            self._attention_params_neuron,
            self._attention_masks_neuron,
        ) = _initialize_attention_params_neuron(key3)

        self._indices = [
            jnp.arange(self._min_index[i], self._max_index[i] + 1, dtype=int)
            for i in range(self._num_topo_batches)
        ]

        self._adaptive_activation_params = _initialize_adaptive_activation_params(key4)

    def _set_dropout_p_initial(
        self, dropout_p: Union[float, Mapping[Any, float]]
    ) -> None:
        """Set the initial per-neuron dropout probabilities."""

        def initialize_dropout_probabilities():
            if isinstance(dropout_p, float):
                hidden_dropout = {neuron: dropout_p for neuron in self._hidden_neurons}
                input_output_dropout = {
                    neuron: 0.0 for neuron in self._input_neurons + self._output_neurons
                }
                return {**hidden_dropout, **input_output_dropout}
            else:
                assert isinstance(dropout_p, Mapping)
                for n, d in dropout_p.items():
                    if n not in self._graph.nodes:
                        raise ValueError(f"'{n}' is not present in the network.")
                    if not isinstance(d, float):
                        raise TypeError(f"Invalid dropout value of {d} for neuron {n}.")
                for neuron in set(self._topo_sort) - set(dropout_p.keys()):
                    dropout_p[neuron] = 0.0
                return dropout_p

        dropout_dict = initialize_dropout_probabilities()
        dropout_array = jnp.array(
            [dropout_dict[neuron] for neuron in self._topo_sort], dtype=float
        )

        assert jnp.all(jnp.greater_equal(dropout_array, 0))
        assert jnp.all(jnp.less_equal(dropout_array, 1))

        self._dropout_array = dropout_array
        self._dropout_dict = dropout_dict

    #####################################################
    ################## Public methods ###################  # noqa: E266
    #####################################################

    def to_networkx_weighted_digraph(self) -> nx.DiGraph:
        """Returns a `networkx.DiGraph` represention of the network with neuron weights
        saved as edge attributes."""

        def _get_edge_attributes():
            edge_attrs = {}
            for neuron, inputs in self._adjacency_dict_inv.items():
                if neuron in self._input_neurons:
                    continue
                topo_batch_idx, pos_idx = self._neuron_to_topo_batch_idx[neuron]
                for input_neuron in inputs:
                    col_idx = (
                        self._neuron_to_id[input_neuron]
                        - self._min_index[topo_batch_idx]
                    )
                    assert self._masks[topo_batch_idx][pos_idx, col_idx]
                    weight = self._weights_and_biases[topo_batch_idx][pos_idx, col_idx]
                    edge_attrs[(input_neuron, neuron)] = {"weight": weight}
            return edge_attrs

        graph = deepcopy(self._graph)
        edge_attrs = _get_edge_attributes()
        nx.set_edge_attributes(graph, edge_attrs)

        return graph
