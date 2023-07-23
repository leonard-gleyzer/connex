from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np

from ._network import NeuralNetwork
from ._utils import _edges_to_adjacency_dict


def _get_id_mappings_old_new(
    old_network: NeuralNetwork, new_network: NeuralNetwork
) -> tuple[np.ndarray, np.ndarray]:
    old_neurons, new_neurons = old_network._graph.nodes(), new_network._graph.nodes()
    ids_old_to_new = -np.ones((old_network._num_neurons,), dtype=int)
    for neuron in old_neurons:
        if neuron in new_neurons:
            old_id = old_network._neuron_to_id[neuron]
            new_id = new_network._neuron_to_id[neuron]
            ids_old_to_new[old_id] = new_id
    ids_new_to_old = -np.ones((new_network._num_neurons,), dtype=int)
    for neuron in new_neurons:
        if neuron in old_neurons:
            new_id = new_network._neuron_to_id[neuron]
            old_id = old_network._neuron_to_id[neuron]
            ids_new_to_old[new_id] = old_id
    return ids_old_to_new, ids_new_to_old


def add_connections(
    network: NeuralNetwork,
    connections: Union[Sequence[tuple[Any, Any]], Mapping[Any, Sequence[Any]]],
    *,
    key: Optional[jr.PRNGKey] = None,
) -> NeuralNetwork:
    """Add connections to the network.

    **Arguments:**

    - `network`: A `NeuralNetwork` object.
    - `connections`: The directed edges to add. Must be a sequence of 2-tuples, or
        an adjacency dict mapping an existing neuron (by its NetworkX id) to its
        new outgoing connections. Connections that already exist are ignored.
    - `key`: The `jax.random.PRNGKey` used for new weight initialization.
        Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.

    **Returns:**

    A `NeuralNetwork` object with the specified connections added and original
    parameters retained.
    """
    # If there is nothing to change, return the network as given
    if len(connections) == 0:
        return deepcopy(network)

    # Make sure connections are in adjacency dict format
    if isinstance(connections, Sequence):
        connections = _edges_to_adjacency_dict(connections)

    # Check that all new connections are between neurons actually in the network
    # and that no neuron outputs to an input neuron
    existing_neurons = network._graph.nodes
    for input, outputs in connections.items():
        if input not in existing_neurons:
            raise ValueError(f"Neuron {input} does not exist in the network.")
        for output in outputs:
            if output not in existing_neurons:
                raise ValueError(f"Neuron {output} does not exist in the network.")
            if output in network._input_neurons:
                raise ValueError(
                    f"""
                    Input neurons cannot receive output from other neurons.
                    The neuron in this case was neuron {input} attempting to
                    add a connection to input neuron {output}.
                    """
                )

    # Update connectivity information
    new_graph = nx.DiGraph(network._graph)
    new_edges = []
    for input, outputs in connections.items():
        new_edges += [(input, output) for output in outputs]
    new_graph.add_edges_from(new_edges)

    def _update_topo_sort(topo_sort: list, new_edges: list) -> list:
        # Find affected nodes
        affected_nodes = set()
        for u, v in new_edges:
            affected_nodes.add(u)
            affected_nodes.add(v)

        # Identify the positions of the affected nodes in the topo_sort list
        node_positions = deepcopy(network._neuron_to_id)

        # For each affected node, update its position in topo_sort
        for node in affected_nodes:
            pos = node_positions[node]
            while pos > 0 and new_graph.in_degree(
                topo_sort[pos - 1]
            ) > new_graph.in_degree(node):
                # Swap the positions of the current node and the previous node
                topo_sort[pos], topo_sort[pos - 1] = topo_sort[pos - 1], topo_sort[pos]
                # Update the node_positions dictionary
                node_positions[topo_sort[pos]] = pos
                node_positions[topo_sort[pos - 1]] = pos - 1
                pos -= 1

        return topo_sort

    topo_sort = deepcopy(network._topo_sort)
    topo_sort = _update_topo_sort(topo_sort, new_edges)

    # Random key
    key = key if key is not None else jr.PRNGKey(0)

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        network._input_neurons,
        network._output_neurons,
        network._hidden_activation,
        network._output_transformation,
        network._dropout_dict,
        network._use_topo_norm,
        network._use_topo_self_attention,
        network._use_neuron_self_attention,
        network._use_adaptive_activations,
        topo_sort=topo_sort,
        key=key,
    )

    old_network = network
    ids_old_to_new, ids_new_to_old = _get_id_mappings_old_new(old_network, new_network)
    assert np.all(ids_old_to_new >= 0)
    assert np.all(ids_new_to_old >= 0)

    # Copy parameters
    new_weights_and_biases = [np.array(w) for w in new_network._weights_and_biases]
    new_attention_params_neuron = [
        np.array(w) for w in new_network._attention_params_neuron
    ]
    new_attention_params_topo = [
        np.array(w) for w in new_network._attention_params_topo
    ]
    new_topo_norm_params = [np.array(w) for w in new_network._topo_norm_params]
    new_adaptive_activation_params = [
        np.array(w) for w in new_network._adaptive_activation_params
    ]

    # Loop through each topo batch in the new network and copy the corresponding
    # parameters present in the old network to the new network
    for i, tb_new in enumerate(new_network._topo_batches):
        # Get the indices of the topological batch in the old network
        tb_old = ids_new_to_old[tb_new]
        # Get the index of the old network topo batch `tb_old` is a subset of.
        # Note that each topo batch in the new network is a subset of a topo batch
        # in the old network.
        tb_index, _ = old_network._neuron_to_topo_batch_idx[tb_old[0]]

        # Make sure it is actually a subset
        intersection = np.intersect1d(
            tb_old, np.array(old_network._topo_batches[tb_index])
        )
        assert intersection.size == tb_old.size

        # For the current topological batch in the new network, get the minimum and
        # maximum index of the topo batch's collective inputs, and get the indices
        # of the corresponding contiguous range in terms of old network ids.
        min_new, max_new = new_network._min_index[i], new_network._max_index[i]
        range_new = np.arange(min_new, max_new + 1)
        min_old, max_old = (
            old_network._min_index[tb_index],
            old_network._max_index[tb_index],
        )
        range_old = np.arange(min_old, max_old + 1)
        input_indices_old = ids_new_to_old[range_new]
        intersection_old = np.intersect1d(range_old, input_indices_old)
        assert intersection_old.size > 0

        # Re-order the values of `intersection_old` to reflect the order in
        # `input_indices_old`
        intersection_old = input_indices_old[
            np.in1d(input_indices_old, intersection_old)
        ]
        intersection_new = ids_old_to_new[intersection_old]
        intersection_old_ = intersection_old - min_old
        intersection_new_ = intersection_new - min_new

        # Get the respective neuron positions within the old topological batches
        pos_old = [old_network._neuron_to_topo_batch_idx[id][1] for id in tb_old]
        pos_old = np.array(pos_old, dtype=int)

        # Copy parameters
        old_weights_and_biases = old_network._weights_and_biases[tb_index][pos_old][
            :, np.append(intersection_old_, -1)
        ]
        new_weights_and_biases[i][
            :, np.append(intersection_new_, -1)
        ] = old_weights_and_biases

        if new_network._use_neuron_self_attention:
            old_attention_params_neuron = old_network._attention_params_neuron[
                tb_index
            ][pos_old][
                :, :, intersection_old_.reshape(-1, 1), np.append(intersection_old_, -1)
            ]

            new_attention_params_neuron[i][
                :, :, intersection_new_.reshape(-1, 1), np.append(intersection_new_, -1)
            ] = old_attention_params_neuron

        if new_network._use_topo_self_attention:
            old_attention_params_topo = old_network._attention_params_topo[tb_index][
                :, intersection_old_.reshape(-1, 1), np.append(intersection_old_, -1)
            ]
            new_attention_params_topo[i][
                :, intersection_new_.reshape(-1, 1), np.append(intersection_new_, -1)
            ] = old_attention_params_topo

        if new_network._use_topo_norm:
            old_topo_norm_params = old_network._topo_norm_params[tb_index][
                intersection_old_
            ]
            new_topo_norm_params[i][intersection_new_] = old_topo_norm_params

        if new_network._use_adaptive_activations:
            old_adaptive_activation_params = old_network._adaptive_activation_params[
                tb_index
            ][pos_old]
            new_adaptive_activation_params[i] = old_adaptive_activation_params

    new_weights_and_biases = [jnp.array(w) for w in new_weights_and_biases]
    new_attention_params_neuron = (
        [jnp.array(w) for w in new_attention_params_neuron]
        if new_network._use_neuron_self_attention
        else [jnp.nan] * new_network._num_topo_batches
    )
    new_attention_params_topo = (
        [jnp.array(w) for w in new_attention_params_topo]
        if new_network._use_topo_self_attention
        else [jnp.nan] * new_network._num_topo_batches
    )
    new_topo_norm_params = (
        [jnp.array(w) for w in new_topo_norm_params]
        if new_network._use_topo_norm
        else [jnp.nan] * new_network._num_topo_batches
    )
    new_adaptive_activation_params = (
        [jnp.array(w) for w in new_adaptive_activation_params]
        if new_network._use_adaptive_activations
        else [jnp.nan] * new_network._num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network._weights_and_biases,
            network._attention_params_neuron,
            network._attention_params_topo,
            network._topo_norm_params,
            network._adaptive_activation_params,
        ),
        new_network,
        (
            new_weights_and_biases,
            new_attention_params_neuron,
            new_attention_params_topo,
            new_topo_norm_params,
            new_adaptive_activation_params,
        ),
    )


def remove_connections(
    network: NeuralNetwork,
    connections: Union[Sequence[tuple[Any, Any]], Mapping[Any, Sequence[Any]]],
) -> NeuralNetwork:
    """Remove connections from the network.

    **Arguments:**

    - `network`: A `NeuralNetwork` object.
    - `connections`: The directed edges to remove. Must be a sequence of 2-tuples, or
        an adjacency dict mapping an existing neuron (by its NetworkX id) to its
        new outgoing connections. Connections that already exist are ignored.

    **Returns:**

    A `NeuralNetwork` object with the specified connections removed and original
    parameters retained.
    """
    # If there is nothing to change, return a copy of the network
    if len(connections) == 0:
        return deepcopy(network)

    # Make sure connections are in adjacency dict format
    if isinstance(connections, Sequence):
        connections = _edges_to_adjacency_dict(connections)

    # Check that all new connections are between neurons actually in the network
    existing_neurons = network._graph.nodes
    for input, outputs in connections.items():
        if input not in existing_neurons:
            raise ValueError(f"Neuron {input} does not exist in the network.")
        for output in outputs:
            if output not in existing_neurons:
                raise ValueError(f"Neuron {output} does not exist in the network.")

    # Update connectivity information
    new_graph = nx.DiGraph(network._graph)
    for input, outputs in connections.items():
        edges_to_remove = [(input, output) for output in outputs]
        new_graph.remove_edges_from(edges_to_remove)

    # Topological sort remains unchanged by removing connections
    topo_sort = deepcopy(network._topo_sort)

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        network._input_neurons,
        network._output_neurons,
        network._hidden_activation,
        network._output_transformation,
        network._dropout_dict,
        network._use_topo_norm,
        network._use_topo_self_attention,
        network._use_neuron_self_attention,
        network._use_adaptive_activations,
        topo_sort=topo_sort,
    )

    old_network = network
    ids_old_to_new, ids_new_to_old = _get_id_mappings_old_new(old_network, new_network)
    assert np.all(ids_old_to_new) >= 0
    assert np.all(ids_new_to_old) >= 0

    # Copy parameters
    new_weights_and_biases = [np.array(w) for w in new_network._weights_and_biases]
    new_attention_params_neuron = [
        np.array(w) for w in new_network._attention_params_neuron
    ]
    new_attention_params_topo = [
        np.array(w) for w in new_network._attention_params_topo
    ]
    new_topo_norm_params = [np.array(w) for w in new_network._topo_norm_params]
    new_adaptive_activation_params = [
        np.array(w) for w in new_network._adaptive_activation_params
    ]

    # Loop through each topo batch in the old network and copy the corresponding
    # parameters present in the new network from the old network
    for i, tb_old in enumerate(old_network._topo_batches):
        # Get the indices of the old topological batch in the new network
        tb_new = ids_old_to_new[tb_old]

        # Get the index of the new network topo batch `tb_new` is a subset of.
        # Note that each topo batch in the old network is a subset of a topo batch
        # in the new network.
        if tb_new[0] not in new_network._neuron_to_topo_batch_idx:
            continue
        tb_index, _ = new_network._neuron_to_topo_batch_idx[tb_new[0]]

        # Make sure it is actually a subset
        intersection = np.intersect1d(
            tb_new, np.array(new_network._topo_batches[tb_index])
        )
        assert intersection.size == tb_new.size

        # For the current topological batch in the old network, get the minimum and
        # maximum index of the topo batch's collective inputs, and get the indices
        # of the corresponding contiguous range in terms of new network ids.
        min_old, max_old = old_network._min_index[i], old_network._max_index[i]
        range_old = np.arange(min_old, max_old + 1)
        min_new, max_new = (
            new_network._min_index[tb_index],
            new_network._max_index[tb_index],
        )
        range_new = np.arange(min_new, max_new + 1)
        input_indices_new = ids_old_to_new[range_old]
        intersection_new = np.intersect1d(range_new, input_indices_new)
        assert intersection_new.size > 0

        # Re-order the values of `intersection_new` to reflect the order in
        # `input_indices_new`
        intersection_new = input_indices_new[
            np.in1d(input_indices_new, intersection_new)
        ]
        intersection_new_ = intersection_new - min_new
        intersection_old = ids_new_to_old[intersection_new]
        intersection_old_ = intersection_old - min_old

        # Get the respective neuron positions within the new topological batches
        pos_new = [new_network._neuron_to_topo_batch_idx[id][1] for id in tb_new]
        pos_new = np.array(pos_new, dtype=int)

        # Copy parameters
        old_weights_and_biases = old_network._weights_and_biases[i][
            :, np.append(intersection_old_, -1)
        ]
        new_weights_and_biases[tb_index][
            pos_new.reshape(-1, 1), np.append(intersection_new_, -1)
        ] = old_weights_and_biases

        if new_network._use_neuron_self_attention:
            old_attention_params_neuron = old_network._attention_params_neuron[i][
                :, :, intersection_old_.reshape(-1, 1), np.append(intersection_old_, -1)
            ]
            new_attention_params_neuron[tb_index][
                np.ix_(
                    pos_new,
                    np.arange(3),
                    intersection_new_,
                    np.append(intersection_new_, -1),
                )
            ] = old_attention_params_neuron

        if new_network._use_topo_self_attention:
            old_attention_params_topo = old_network._attention_params_topo[i][
                :, intersection_old_.reshape(-1, 1), np.append(intersection_old_, -1)
            ]
            new_attention_params_topo[tb_index][
                :, intersection_new_.reshape(-1, 1), np.append(intersection_new_, -1)
            ] = old_attention_params_topo

        if new_network._use_topo_norm:
            old_topo_norm_params = old_network._topo_norm_params[i][intersection_old_]
            new_topo_norm_params[tb_index][intersection_new_] = old_topo_norm_params

        if new_network._use_adaptive_activations:
            old_adaptive_activation_params = old_network._adaptive_activation_params[i]
            new_adaptive_activation_params[tb_index][
                pos_new
            ] = old_adaptive_activation_params

    new_weights_and_biases = [jnp.array(w) for w in new_weights_and_biases]
    new_attention_params_neuron = (
        [jnp.array(w) for w in new_attention_params_neuron]
        if new_network._use_neuron_self_attention
        else [jnp.nan] * new_network._num_topo_batches
    )
    new_attention_params_topo = (
        [jnp.array(w) for w in new_attention_params_topo]
        if new_network._use_topo_self_attention
        else [jnp.nan] * new_network._num_topo_batches
    )
    new_topo_norm_params = (
        [jnp.array(w) for w in new_topo_norm_params]
        if new_network._use_topo_norm
        else [jnp.nan] * new_network._num_topo_batches
    )
    new_adaptive_activation_params = (
        [jnp.array(w) for w in new_adaptive_activation_params]
        if new_network._use_adaptive_activations
        else [jnp.nan] * new_network._num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network._weights_and_biases,
            network._attention_params_neuron,
            network._attention_params_topo,
            network._topo_norm_params,
            network._adaptive_activation_params,
        ),
        new_network,
        (
            new_weights_and_biases,
            new_attention_params_neuron,
            new_attention_params_topo,
            new_topo_norm_params,
            new_adaptive_activation_params,
        ),
    )


def add_hidden_neurons(
    network: NeuralNetwork,
    new_hidden_neurons: Sequence[Any],
    *,
    key: Optional[jr.PRNGKey] = None,
) -> NeuralNetwork:
    """Add hidden neurons to the network. Note that this function only adds neurons
    themselves, not any connections associated with the new neurons, effectively
    adding them as isolated nodes in the graph. Use [`connex.add_connections`][]
    after this function has been called to add the desired connections.

    **Arguments:**

    - `network`: The `NeuralNetwork` to add neurons to
    - `new_hidden_neurons`: A sequence of new hidden neurons (more specifically, their
        identifiers/names) to add to the network. These must be unique, i.e. cannot
        already exist in the  These must also specifically be hidden neurons.
        To add input or output neurons, use [`connex.add_input_neurons`][] or
        [`connex.add_output_neurons`][].
    - `key`: The `jax.random.PRNGKey` used for new parameter initialization.
        Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.

    **Returns:**

    A `NeuralNetwork` with the new hidden neurons added and parameters from the
    original network retained.
    """
    # If there is nothing to change, return the network as given
    if len(new_hidden_neurons) == 0:
        return deepcopy(network)

    # Check that none of the new neurons already exist in the network
    existing_neurons = network._graph.nodes()
    for neuron in new_hidden_neurons:
        if neuron in existing_neurons:
            raise ValueError(f"Neuron {neuron} already exists in the network.")

    # Update graph information
    new_graph = nx.DiGraph(network._graph)
    new_graph.add_nodes_from(new_hidden_neurons)

    # Update topological sort
    topo_sort = network._topo_sort
    num_output_neurons = len(network._output_neurons)

    # It doesn't really matter where in a topological sort new isolated nodes are
    # added, it still remains a valid topological sort. We add them right before the
    # output neurons to make it easier to keep track of indices when copying
    # parameters over.
    topo_sort = (
        topo_sort[:-num_output_neurons]
        + list(new_hidden_neurons)
        + network._output_neurons
    )

    # Random key
    key = key if key is not None else jr.PRNGKey(0)

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        network._input_neurons,
        network._output_neurons,
        network._hidden_activation,
        network._output_transformation,
        network._dropout_dict,
        network._use_topo_norm,
        network._use_topo_self_attention,
        network._use_neuron_self_attention,
        network._use_adaptive_activations,
        topo_sort=topo_sort,
        key=key,
    )

    old_network = network
    num_new_hidden_neurons = len(new_hidden_neurons)

    # Sanity check parameter shapes
    assert old_network._num_topo_batches == new_network._num_topo_batches
    for i in range(old_network._num_topo_batches):
        if i != old_network._num_topo_batches - 2:
            assert (
                old_network._weights_and_biases[i].shape
                == new_network._weights_and_biases[i].shape
            )
            assert (
                old_network._attention_params_neuron[i].shape
                == new_network._attention_params_neuron[i].shape
            )
            assert (
                old_network._attention_params_topo[i].shape
                == new_network._attention_params_topo[i].shape
            )
            assert (
                old_network._topo_norm_params[i].shape
                == new_network._topo_norm_params[i].shape
            )

    # Copy weights and biases
    assert (
        old_network._weights_and_biases[-2].shape
        == new_network._weights_and_biases[-2][:-num_new_hidden_neurons].shape
    )
    new_weights_and_biases = (
        old_network._weights_and_biases[:-2]
        + [
            new_network._weights_and_biases[-2]
            .at[:-num_new_hidden_neurons]
            .set(old_network._weights_and_biases[-2])
        ]
        + [old_network._weights_and_biases[-1]]
    )

    # Copy neuron-level attention parameters
    assert (
        old_network._attention_params_neuron[-2].shape
        == new_network._attention_params_neuron[-2][:-num_new_hidden_neurons].shape
    )
    new_attention_params_neuron = (
        old_network._attention_params_neuron[:-2]
        + [
            new_network._attention_params_neuron[-2]
            .at[:-num_new_hidden_neurons]
            .set(old_network._attention_params_neuron[-2])
        ]
        + [old_network._attention_params_neuron[-1]]
        if new_network._use_neuron_self_attention
        else [jnp.nan] * new_network._num_topo_batches
    )

    # Copy topo-level attention parameters
    assert (
        old_network._attention_params_topo[-2].shape
        == new_network._attention_params_topo[-2].shape
    )
    new_attention_params_topo = deepcopy(old_network._attention_params_topo)

    # Copy topo norm parameters
    assert (
        old_network._topo_norm_params[-2].shape
        == new_network._topo_norm_params[-2].shape
    )
    new_topo_norm_params = deepcopy(old_network._topo_norm_params)

    # Copy adaptive activation parameters
    assert (
        old_network._adaptive_activation_params[-2].shape
        == new_network._adaptive_activation_params[-2][:-num_new_hidden_neurons].shape
    )
    new_adaptive_activation_params = (
        old_network._adaptive_activation_params[:-2]
        + [
            new_network._adaptive_activation_params[-2]
            .at[:-num_new_hidden_neurons]
            .set(old_network._adaptive_activation_params[-2])
        ]
        + [old_network._adaptive_activation_params[-1]]
        if new_network._use_adaptive_activations
        else [jnp.nan] * new_network._num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network._weights_and_biases,
            network._attention_params_neuron,
            network._attention_params_topo,
            network._topo_norm_params,
            network._adaptive_activation_params,
        ),
        new_network,
        (
            new_weights_and_biases,
            new_attention_params_neuron,
            new_attention_params_topo,
            new_topo_norm_params,
            new_adaptive_activation_params,
        ),
    )


def add_output_neurons(
    network: NeuralNetwork,
    new_output_neurons: Sequence[Any],
    *,
    key: Optional[jr.PRNGKey] = None,
) -> NeuralNetwork:
    """Add output neurons to the network. Note that this function only adds neurons
    themselves, not any connections associated with the new neurons, effectively
    adding them as isolated nodes in the graph. Use [`connex.add_connections`][]
    after this function has been called to add any desired connections.

    **Arguments:**

    - `network`: The `NeuralNetwork` to add neurons to
    - `new_output_neurons`: A sequence of new output neurons (more specifically, their
        identifiers/names) to add to the network. These must be unique, i.e. cannot
        already exist in the network. These must also specifically be output neurons.
        To add input or output neurons, use [`connex.add_input_neurons`][] or
        [`connex.add_output_neurons`][].
    - `key`: The `jax.random.PRNGKey` used for new parameter initialization.
        Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.

    **Returns:**

    A `NeuralNetwork` with the new output neurons added and parameters from the
    original network retained.
    """
    # If there is nothing to change, return the network as given
    if len(new_output_neurons) == 0:
        return network

    # Update output neurons
    output_neurons = network._output_neurons + list(new_output_neurons)
    num_output_neurons = len(output_neurons)

    # Check that none of the new neurons already exist in the network
    existing_neurons = network._graph.nodes
    for neuron in new_output_neurons:
        if neuron in existing_neurons:
            raise ValueError(f"Neuron {neuron} already exists in the network.")

    # Update graph information
    new_graph = nx.DiGraph(network._graph)
    new_graph.add_nodes_from(new_output_neurons)

    # Update topological sort, appending the new output neurons to the end
    # of the output neuron list
    topo_sort = network._topo_sort + list(new_output_neurons)

    # Random key
    key = key if key is not None else jr.PRNGKey(0)

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        network._input_neurons,
        output_neurons,
        network._hidden_activation,
        network._output_transformation,
        network._dropout_dict,
        network._use_topo_norm,
        network._use_topo_self_attention,
        network._use_neuron_self_attention,
        network._use_adaptive_activations,
        topo_sort=topo_sort,
        key=key,
    )

    old_network = network
    num_new_output_neurons = len(new_output_neurons)

    # Sanity check parameter shapes
    assert old_network._num_topo_batches == new_network._num_topo_batches
    for i in range(old_network._num_topo_batches - 1):
        assert (
            old_network._weights_and_biases[i].shape
            == new_network._weights_and_biases[i].shape
        )
        assert (
            old_network._attention_params_neuron[i].shape
            == new_network._attention_params_neuron[i].shape
        )
        assert (
            old_network._attention_params_topo[i].shape
            == new_network._attention_params_topo[i].shape
        )
        assert (
            old_network._topo_norm_params[i].shape
            == new_network._topo_norm_params[i].shape
        )

    # Copy weights and biases
    assert (
        old_network._weights_and_biases[-1].shape
        == new_network._weights_and_biases[-1][:-num_new_output_neurons].shape
    )
    new_weights_and_biases = old_network._weights_and_biases[:-1] + [
        new_network._weights_and_biases[-1]
        .at[-num_output_neurons:-num_new_output_neurons]
        .set(old_network._weights_and_biases[-1])
    ]

    # Copy neuron-level attention parameters
    assert (
        old_network._attention_params_neuron[-1].shape
        == new_network._attention_params_neuron[-1][:-num_new_output_neurons].shape
    )
    new_attention_params_neuron = (
        (
            old_network._attention_params_neuron[:-1]
            + [
                new_network._attention_params_neuron[-1]
                .at[-num_output_neurons:-num_new_output_neurons]
                .set(old_network._attention_params_neuron[-1])
            ]
        )
        if new_network._use_neuron_self_attention
        else [jnp.nan] * new_network._num_topo_batches
    )

    # Copy topo-level attention parameters
    assert (
        old_network._attention_params_topo[-1].shape
        == new_network._attention_params_topo[-1].shape
    )
    new_attention_params_topo = deepcopy(
        (old_network._attention_params_topo)
        if new_network._use_topo_self_attention
        else [jnp.nan] * new_network._num_topo_batches
    )

    # Copy topo norm parameters
    assert (
        old_network._topo_norm_params[-1].shape
        == new_network._topo_norm_params[-1].shape
    )
    new_topo_norm_params = deepcopy(
        (old_network._topo_norm_params)
        if new_network._use_topo_norm
        else [jnp.nan] * new_network._num_topo_batches
    )

    # Copy adaptive activation parameters
    assert (
        old_network._adaptive_activation_params[-1].shape
        == new_network._adaptive_activation_params[-1][:-num_new_output_neurons].shape
    )
    new_adaptive_activation_params = (
        (
            old_network._adaptive_activation_params[:-1]
            + [
                new_network._adaptive_activation_params[-1]
                .at[-num_output_neurons:-num_new_output_neurons]
                .set(old_network._adaptive_activation_params[-1])
            ]
        )
        if new_network._use_adaptive_activations
        else [jnp.nan] * new_network._num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network._weights_and_biases,
            network._attention_params_neuron,
            network._attention_params_topo,
            network._topo_norm_params,
            network._adaptive_activation_params,
        ),
        new_network,
        (
            new_weights_and_biases,
            new_attention_params_neuron,
            new_attention_params_topo,
            new_topo_norm_params,
            new_adaptive_activation_params,
        ),
    )


def add_input_neurons(
    network: NeuralNetwork,
    new_input_neurons: Sequence[Any],
    *,
    key: Optional[jr.PRNGKey] = None,
) -> NeuralNetwork:
    """Add input neurons to the network. Note that this function only adds neurons
    themselves, not any connections associated with the new neurons, effectively adding
    them as isolated nodes in the graph. Use [`connex.add_connections`][] after this
    function has been called to add the desired connections.

    **Arguments:**

    - `network`: The `NeuralNetwork` to add neurons to
    - `new_input_neurons`: A sequence of new input neurons (more specifically, their
        identifiers/names) to add to the network. These must be unique, i.e. cannot
        already exist in the network. These must also specifically be input neurons.
        To add hidden or output neurons, use [`connex.add_hidden_neurons`][] or
        [`connex.add_output_neurons`][].
    - `key`: The `jax.random.PRNGKey` used for new parameter initialization.
        Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.

    **Returns:**

    A `NeuralNetwork` with the new input neurons added and parameters from the
    original network retained. The new input neurons are added _before_ the
    existing input neurons. For example, if the previous input neurons were
    `[0, 1]` and new input neurons `[2, 3]` were added, the new input neurons
    would be `[2, 3, 0, 1]`.
    """
    # If there is nothing to change, return the network as given
    if len(new_input_neurons) == 0:
        return deepcopy(network)

    # Update input neurons
    input_neurons = list(new_input_neurons) + network._input_neurons

    # Check that none of the new neurons already exist in the network
    existing_neurons = network._graph.nodes()
    for neuron in new_input_neurons:
        if neuron in existing_neurons:
            raise ValueError(f"Neuron {neuron} already exists in the network.")

    # Update graph information
    new_graph = nx.DiGraph(network._graph)
    new_graph.add_nodes_from(new_input_neurons)

    # Update topological sort, appending the new input neurons to the beginning
    # of the input neuron list
    topo_sort = list(new_input_neurons) + network._topo_sort

    # Random key
    key = key if key is not None else jr.PRNGKey(0)

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        input_neurons,
        network._output_neurons,
        network._hidden_activation,
        network._output_transformation,
        network._dropout_dict,
        network._use_topo_norm,
        network._use_topo_self_attention,
        network._use_neuron_self_attention,
        network._use_adaptive_activations,
        topo_sort=topo_sort,
        key=key,
    )

    old_network = network

    # Sanity check parameter shapes
    assert old_network._num_topo_batches == new_network._num_topo_batches
    for i in range(old_network._num_topo_batches):
        assert (
            old_network._weights_and_biases[i].shape
            == new_network._weights_and_biases[i].shape
        )
        assert (
            old_network._attention_params_neuron[i].shape
            == new_network._attention_params_neuron[i].shape
        )
        assert (
            old_network._attention_params_topo[i].shape
            == new_network._attention_params_topo[i].shape
        )
        assert (
            old_network._topo_norm_params[i].shape
            == new_network._topo_norm_params[i].shape
        )

    # Copy weights and biases
    new_weights_and_biases = deepcopy(old_network._weights_and_biases)

    # Copy neuron-level attention parameters
    new_attention_params_neuron = deepcopy(old_network._attention_params_neuron)

    # Copy topo-level attention parameters
    new_attention_params_topo = deepcopy(old_network._attention_params_topo)

    # Copy topo norm parameters
    new_topo_norm_params = deepcopy(old_network._topo_norm_params)

    # Copy adaptive activation parameters
    new_adaptive_activation_params = deepcopy(old_network._adaptive_activation_params)

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network._weights_and_biases,
            network._attention_params_neuron,
            network._attention_params_topo,
            network._topo_norm_params,
            network._adaptive_activation_params,
        ),
        new_network,
        (
            new_weights_and_biases,
            new_attention_params_neuron,
            new_attention_params_topo,
            new_topo_norm_params,
            new_adaptive_activation_params,
        ),
    )


def remove_neurons(
    network: NeuralNetwork,
    neurons: Sequence[Any],
) -> NeuralNetwork:
    """Remove neurons and any of their incoming/outgoing connections from the network.

    **Arguments:**

    - `network`: The `NeuralNetwork` to add neurons to
    - `neurons`: A sequence of neurons (more specifically, their identifiers/names) to
        remove from the network. These can be input, hidden, or output neurons.

    **Returns:**

    A `NeuralNetwork` with the specified neurons removed and parameters from the
    original network retained.
    """
    # If there is nothing to change, return the network as given
    if len(neurons) == 0:
        return deepcopy(network)

    # Check that all of the new neurons already exist in the network
    existing_neurons = network._graph.nodes()
    for neuron in neurons:
        if neuron not in existing_neurons:
            raise ValueError(f"Neuron {neuron} does not exist in the network.")

    # Set input and output neurons
    input_neurons = deepcopy(network._input_neurons)
    output_neurons = deepcopy(network._output_neurons)
    for neuron in neurons:
        if neuron in input_neurons:
            input_neurons.remove(neuron)
        elif neuron in output_neurons:
            output_neurons.remove(neuron)

    # Update graph information
    new_graph = nx.DiGraph(network._graph)
    new_graph.remove_nodes_from(neurons)

    # Update topological sort
    topo_sort = deepcopy(network._topo_sort)
    for neuron in neurons:
        topo_sort.remove(neuron)

    # Update dropout dict
    dropout_dict = deepcopy(network._dropout_dict)
    for neuron in neurons:
        del dropout_dict[neuron]

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        input_neurons,
        output_neurons,
        network._hidden_activation,
        network._output_transformation,
        dropout_dict,
        network._use_topo_norm,
        network._use_topo_self_attention,
        network._use_neuron_self_attention,
        network._use_adaptive_activations,
        topo_sort=topo_sort,
    )

    old_network = network
    ids_old_to_new, ids_new_to_old = _get_id_mappings_old_new(old_network, new_network)

    # Copy parameters
    new_weights_and_biases = [np.array(w) for w in new_network._weights_and_biases]
    new_attention_params_neuron = [
        np.array(w) for w in new_network._attention_params_neuron
    ]
    new_attention_params_topo = [
        np.array(w) for w in new_network._attention_params_topo
    ]
    new_topo_norm_params = [np.array(w) for w in new_network._topo_norm_params]
    new_adaptive_activation_params = [
        np.array(w) for w in new_network._adaptive_activation_params
    ]

    # Loop through each topo batch in the new network and copy the corresponding
    # parameters present in the old network to the new network
    for i, tb_new in enumerate(new_network._topo_batches):
        # Get the indices of the topological batch in the old network
        tb_old = ids_new_to_old[tb_new]
        # Get the index of the old network topo batch `tb_old` is a subset of.
        # Note that each topo batch in the new network is a subset of a topo batch
        # in the old network.
        tb_index, _ = old_network._neuron_to_topo_batch_idx[tb_old[0]]

        # Make sure it is actually a subset
        intersection = np.intersect1d(
            tb_old, np.array(old_network._topo_batches[tb_index])
        )
        assert intersection.size == tb_old.size

        # For the current topological batch in the new network, get the minimum and
        # maximum index of the topo batch's collective inputs, and get the indices
        # of the corresponding contiguous range in terms of old network ids.
        min_new, max_new = new_network._min_index[i], new_network._max_index[i]
        range_new = np.arange(min_new, max_new + 1)
        min_old, max_old = (
            old_network._min_index[tb_index],
            old_network._max_index[tb_index],
        )
        range_old = np.arange(min_old, max_old + 1)
        input_indices_old = ids_new_to_old[range_new]
        intersection_old = np.intersect1d(range_old, input_indices_old)
        assert intersection_old.size > 0

        # Re-order the values of `intersection_old` to reflect the order in
        # `input_indices_old`
        intersection_old = input_indices_old[
            np.in1d(input_indices_old, intersection_old)
        ]
        intersection_new = ids_old_to_new[intersection_old]
        intersection_old_ = intersection_old - min_old
        intersection_new_ = intersection_new - min_new

        # Get the respective neuron positions within the old topological batches
        pos_old = [old_network._neuron_to_topo_batch_idx[id][1] for id in tb_old]
        pos_old = np.array(pos_old, dtype=int)

        # Copy parameters
        old_weights_and_biases = old_network._weights_and_biases[tb_index][pos_old][
            :, np.append(intersection_old_, -1)
        ]
        new_weights_and_biases[i][
            :, np.append(intersection_new_, -1)
        ] = old_weights_and_biases

        if new_network._use_neuron_self_attention:
            old_attention_params_neuron = old_network._attention_params_neuron[
                tb_index
            ][pos_old][
                :, :, intersection_old_.reshape(-1, 1), np.append(intersection_old_, -1)
            ]

            new_attention_params_neuron[i][
                :, :, intersection_new_.reshape(-1, 1), np.append(intersection_new_, -1)
            ] = old_attention_params_neuron

        if new_network._use_topo_self_attention:
            old_attention_params_topo = old_network._attention_params_topo[tb_index][
                :, intersection_old_.reshape(-1, 1), np.append(intersection_old_, -1)
            ]
            new_attention_params_topo[i][
                :, intersection_new_.reshape(-1, 1), np.append(intersection_new_, -1)
            ] = old_attention_params_topo

        if new_network._use_topo_norm:
            old_topo_norm_params = old_network._topo_norm_params[tb_index][
                intersection_old_
            ]
            new_topo_norm_params[i][intersection_new_] = old_topo_norm_params

        if new_network._use_adaptive_activations:
            old_adaptive_activation_params = old_network._adaptive_activation_params[
                tb_index
            ][pos_old]
            new_adaptive_activation_params[i] = old_adaptive_activation_params

    new_weights_and_biases = [jnp.array(w) for w in new_weights_and_biases]
    new_attention_params_neuron = (
        [jnp.array(w) for w in new_attention_params_neuron]
        if new_network._use_neuron_self_attention
        else [jnp.nan] * new_network._num_topo_batches
    )
    new_attention_params_topo = (
        [jnp.array(w) for w in new_attention_params_topo]
        if new_network._use_topo_self_attention
        else [jnp.nan] * new_network._num_topo_batches
    )
    new_topo_norm_params = (
        [jnp.array(w) for w in new_topo_norm_params]
        if new_network._use_topo_norm
        else [jnp.nan] * new_network._num_topo_batches
    )
    new_adaptive_activation_params = (
        [jnp.array(w) for w in new_adaptive_activation_params]
        if new_network._use_adaptive_activations
        else [jnp.nan] * new_network._num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network._weights_and_biases,
            network._attention_params_neuron,
            network._attention_params_topo,
            network._topo_norm_params,
            network._adaptive_activation_params,
        ),
        new_network,
        (
            new_weights_and_biases,
            new_attention_params_neuron,
            new_attention_params_topo,
            new_topo_norm_params,
            new_adaptive_activation_params,
        ),
    )


def set_dropout_p(
    network: NeuralNetwork, dropout_p: Union[float, Mapping[Any, float]]
) -> NeuralNetwork:
    """Set the per-neuron dropout probabilities.

    **Arguments:**

    - `network`: The `NeuralNetwork` whose dropout probabilities will be modified.
    - `dropout_p`: Either a float or mapping from neuron (`Any`) to float. If a
        single float, all hidden neurons will have that dropout probability, and
        all input and output neurons will have dropout probability 0 by default.
        If a `Mapping`, it is assumed that `dropout_p` maps a neuron to its dropout
        probability, and all unspecified neurons will retain their current dropout
        probability.

    **Returns:**

    A copy of the network with dropout probabilities as specified.
    The original network (including unspecified dropout probabilities) is left
    unchanged.
    """

    def update_dropout_probabilities():
        if isinstance(dropout_p, float):
            hidden_dropout = {neuron: dropout_p for neuron in network._hidden_neurons}
            input_output_dropout = {
                neuron: 0.0
                for neuron in network._input_neurons + network._output_neurons
            }
            return {**hidden_dropout, **input_output_dropout}
        else:
            assert isinstance(dropout_p, Mapping)
            for n, d in dropout_p.items():
                if n not in network._graph.nodes:
                    raise ValueError(f"'{n}' is not present in the network.")
                if not isinstance(d, float):
                    raise TypeError(f"Invalid dropout value of {d} for neuron {n}.")
                return {**network._dropout_dict, **dropout_p}

    dropout_dict = update_dropout_probabilities()
    dropout_array = jnp.array(
        [dropout_dict[neuron] for neuron in network._topo_sort], dtype=float
    )

    assert jnp.all(jnp.greater_equal(dropout_array, 0))
    assert jnp.all(jnp.less_equal(dropout_array, 1))

    return eqx.tree_at(
        lambda network: (network._dropout_dict, network._dropout_array),
        network,
        (dropout_dict, dropout_array),
    )
