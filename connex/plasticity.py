from typing import Any, Mapping, Optional, Sequence, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np

from .network import NeuralNetwork


def _get_id_mappings_old_new(
    old_network: NeuralNetwork, new_network: NeuralNetwork
) -> Tuple[np.ndarray, np.ndarray]:
    old_neurons, new_neurons = old_network.graph.nodes, new_network.graph.nodes
    ids_old_to_new = np.empty((old_network.num_neurons,), dtype=int)
    for neuron in old_neurons:
        old_id = old_network.neuron_to_id[neuron]
        new_id = new_network.neuron_to_id[neuron] if neuron in new_neurons else -1
        ids_old_to_new[old_id] = new_id
    ids_new_to_old = np.empty((new_network.num_neurons,), dtype=int)
    for neuron in new_neurons:
        new_id = new_network.neuron_to_id[neuron]
        old_id = old_network.neuron_to_id[neuron] if neuron in old_neurons else -1
        ids_new_to_old[new_id] = old_id
    return ids_old_to_new, ids_new_to_old


def add_connections(
    network: NeuralNetwork,
    connections: Mapping[Any, Sequence[Any]],
    *,
    key: Optional[jr.PRNGKey] = None,
) -> NeuralNetwork:
    """Add connections to the network.

    **Arguments:**

    - `network`: A `NeuralNetwork` object.
    - `connections`: An adjacency dict mapping an existing neuron (by its
        NetworkX id) to its new outgoing connections. Connections that already
        exist are ignored.
    - `key`: The `jax.random.PRNGKey` used for new weight initialization.
        Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.

    **Returns:**

    A `NeuralNetwork` object with the specified connections added and original
    parameters retained.
    """
    # If there is nothing to change, return the network as given
    if len(connections) == 0:
        return network

    # Check that all new connections are between neurons actually in the network
    # and that no neuron outputs to an input neuron
    existing_neurons = network.graph.nodes
    for input, outputs in connections.items():
        if input not in existing_neurons:
            raise ValueError(f"Neuron {input} does not exist in the network.")
        for output in outputs:
            if output not in existing_neurons:
                raise ValueError(f"Neuron {output} does not exist in the network.")
            if output in network.input_neurons:
                raise ValueError(
                    f"""
                    Input neurons cannot receive output from other neurons.
                    The neuron in this case was neuron {input} attempting to
                    add a connection to input neuron {output}.
                    """
                )

    # Update connectivity information
    new_graph = nx.DiGraph(network.graph)
    for (input, outputs) in connections.items():
        new_edges = [(input, output) for output in outputs]
        new_graph.add_edges_from(new_edges)

    # Update topological sort (reference: https://stackoverflow.com/a/24764451)
    def _add_edge_rec(topo_sort, input, output, visited={}):
        input_index = topo_sort.index(input)
        output_index = topo_sort.index(output)
        assert input_index != output_index
        if input_index < output_index:
            return topo_sort, visited
        if output in visited:
            raise ValueError(f"Edge ({input}, {output}) creates a cycle.")
        topo_sort.remove(output)
        topo_sort.insert(input_index, output)
        visited.add(output)
        new_edges = network.graph.out_edges(output)
        for (input_, output_) in new_edges:
            topo_sort, visited = _add_edge_rec(topo_sort, input_, output_, visited)

    topo_sort = network.topo_sort
    for input, outputs in connections.items():
        for output in outputs:
            topo_sort, _ = _add_edge_rec(topo_sort, input, output)

    # Random key
    key = key if key is not None else jr.PRNGKey(0)

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        network.input_neurons,
        network.output_neurons,
        network.hidden_activation,
        network.output_transformation,
        network.dropout_p,
        network.use_topo_norm,
        network.use_topo_self_attention,
        network.use_neuron_self_attention,
        network.use_adaptive_activations,
        topo_sort=topo_sort,
        key=key,
    )

    old_network = network
    ids_old_to_new, ids_new_to_old = _get_id_mappings_old_new(old_network, new_network)
    assert np.all(ids_old_to_new >= 0)
    assert np.app(ids_new_to_old >= 0)

    # Copy parameters
    new_weights_and_biases = [np.array(w) for w in new_network.weights_and_biases]
    new_attention_params_neuron = [
        np.array(w) for w in new_network.attention_params_neuron
    ]
    new_attention_params_topo = [np.array(w) for w in new_network.attention_params_topo]
    new_topo_norm_params = [np.array(w) for w in new_network.topo_norm_params]
    new_adaptive_activation_params = [
        np.array(w) for w in new_network.adaptive_activation_params
    ]

    # Loop through each topo batch in the new network and copy the corresponding
    # parameters present in the old network to the new network
    for i, tb_new in enumerate(new_network.topo_batches):
        tb_old = ids_new_to_old[tb_new]

        # Get the index of the topo batch `tb_old` is a subset of
        tb_index, _ = old_network.neuron_to_topo_batch_idx[tb_old[0]]

        # Make sure it is actually a subset
        intersection = np.intersect1d(
            tb_old, np.array(old_network.topo_batches[tb_index])
        )
        assert intersection.size == tb_old.size

        min_new, max_new = new_network.min_index[i], new_network.max_index[i]
        range_new = np.arange(min_new, max_new + 1)
        min_old, max_old = (
            old_network.min_index[tb_index],
            old_network.max_index[tb_index],
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
        intersection_old_ = intersection_old - min_old
        intersection_new = ids_old_to_new[intersection_old]
        intersection_new_ = intersection_new - min_new

        # Get the respective neuron positions within the old topological batches
        pos_old = [old_network.neuron_to_topo_batch_idx[id][1] for id in tb_old]
        pos_old = np.array(pos_old, dtype=int)

        # Copy parameters
        old_weights_and_biases = old_network.weights_and_biases[tb_index][
            pos_old, np.append(intersection_old_, -1)
        ]
        new_weights_and_biases[i][
            :, np.append(intersection_new_, -1)
        ] = old_weights_and_biases

        if new_network.use_neuron_self_attention:
            old_attention_params_neuron = old_network.attention_params_neuron[tb_index][
                pos_old, :, intersection_old_, np.append(intersection_old_, -1)
            ]
            new_attention_params_neuron[i][
                :, :, intersection_new_, np.append(intersection_new_, -1)
            ] = old_attention_params_neuron

        if new_network.use_topo_self_attention:
            old_attention_params_topo = old_network.attention_params_topo[tb_index][
                pos_old, intersection_old_
            ]
            new_attention_params_topo[i][
                :, intersection_new_
            ] = old_attention_params_topo

        if new_network.use_topo_norm:
            old_topo_norm_params = old_network.topo_norm_params[tb_index][
                pos_old, intersection_old_
            ]
            new_topo_norm_params[i][:, intersection_new_] = old_topo_norm_params

        if new_network.use_adaptive_activations:
            old_adaptive_activation_params = old_network.adaptive_activation_params[
                tb_index
            ][pos_old]
            new_adaptive_activation_params[i] = old_adaptive_activation_params

    new_weights_and_biases = [jnp.array(w) for w in new_weights_and_biases]
    new_attention_params_neuron = (
        [jnp.array(w) for w in new_attention_params_neuron]
        if new_network.use_neuron_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )
    new_attention_params_topo = (
        [jnp.array(w) for w in new_attention_params_topo]
        if new_network.use_topo_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )
    new_topo_norm_params = (
        [jnp.array(w) for w in new_topo_norm_params]
        if new_network.use_topo_norm
        else [jnp.nan] * new_network.num_topo_batches
    )
    new_adaptive_activation_params = (
        [jnp.array(w) for w in new_adaptive_activation_params]
        if new_network.use_adaptive_activations
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network.weights_and_biases,
            network.attention_params_neuron,
            network.attention_params_topo,
            network.topo_norm_params,
            network.adaptive_activation_params,
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
    connections: Mapping[Any, Sequence[Any]],
) -> NeuralNetwork:
    """Remove connections from the network.

    **Arguments:**

    - `network`: A `NeuralNetwork` object.
    - `connections`: An adjacency dict mapping an existing neuron (by its
        NetworkX id) to its current outgoing connections to remove. Connections that
        do not exist are ignored.

    **Returns:**

    A `NeuralNetwork` object with the specified connections removed and original
    parameters retained.
    """
    # If there is nothing to change, return the network as given
    if len(connections) == 0:
        return network

    # Check that all new connections are between neurons actually in the network
    existing_neurons = network.graph.nodes
    for input, outputs in connections.items():
        if input not in existing_neurons:
            raise ValueError(f"Neuron {input} does not exist in the network.")
        for output in outputs:
            if output not in existing_neurons:
                raise ValueError(f"Neuron {output} does not exist in the network.")

    # Update connectivity information
    new_graph = nx.DiGraph(network.graph)
    for (input, outputs) in connections.items():
        edges_to_remove = [(input, output) for output in outputs]
        new_graph.remove_edges_from(edges_to_remove)

    # Update topological sort
    topo_sort = network.topo_sort
    neuron_inputs = network.adjacency_dict_inv
    unique_outputs = {}
    for input, outputs in connections.items():
        for output in outputs:
            unique_outputs.add(output)
            neuron_inputs[output].remove(input)

    # Remove all the output neurons of connections to remove from the topo sort
    for output in unique_outputs:
        topo_sort.remove(output)

    # Add each of them back in immediately after the neuron with the greatest
    # topological index that exists as one of the removed neuron's inputs
    for output in unique_outputs:
        neuron = neuron_inputs[output][-1]
        neuron_idx = network.neuron_to_id[neuron]
        topo_sort.insert(neuron_idx + 1, output)

    # Get the current network key
    network_key = network._get_key()

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        network.input_neurons,
        network.output_neurons,
        network.hidden_activation,
        network.output_transformation,
        network.dropout_p,
        network.use_topo_norm,
        network.use_topo_self_attention,
        network.use_neuron_self_attention,
        network.use_adaptive_activations,
        topo_sort=topo_sort,
        key=network_key,
    )

    old_network = network
    ids_old_to_new, ids_new_to_old = _get_id_mappings_old_new(old_network, new_network)
    assert np.all(ids_old_to_new) >= 0
    assert np.all(ids_new_to_old) >= 0

    # Copy parameters
    new_weights_and_biases = [np.array(w) for w in new_network.weights_and_biases]
    new_attention_params_neuron = [
        np.array(w) for w in new_network.attention_params_neuron
    ]
    new_attention_params_topo = [np.array(w) for w in new_network.attention_params_topo]
    new_topo_norm_params = [np.array(w) for w in new_network.topo_norm_params]
    new_adaptive_activation_params = [
        np.array(w) for w in new_network.adaptive_activation_params
    ]

    # Loop through each topo batch in the old network and copy the corresponding
    # parameters present in the new network from the old network
    for i, tb_old in enumerate(old_network.topo_batches):
        tb_new = ids_old_to_new[tb_old]

        # Get the index in the old network of the topo batch `tb_new` is a subset of
        tb_index, _ = new_network.neuron_to_topo_batch_idx[tb_new[0]]

        # Make sure it is actually a subset
        intersection = np.intersect1d(
            tb_new, np.array(new_network.topo_batches[tb_index])
        )
        assert intersection.size == tb_new.size

        min_old, max_old = old_network.min_index[i], new_network.max_index[i]
        range_old = np.arange(min_old, max_old + 1)
        min_new, max_new = (
            new_network.min_index[tb_index],
            new_network.max_index[tb_index],
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
        pos_new = [new_network.neuron_to_topo_batch_idx[id][1] for id in tb_new]
        pos_new = np.array(pos_new, dtype=int)

        # Copy parameters
        old_weights_and_biases = old_network.weights_and_biases[i][
            :, np.append(intersection_old_, -1)
        ]
        new_network.weights_and_biases[tb_index][
            pos_new, np.append(intersection_new_, -1)
        ] = old_weights_and_biases

        if new_network.use_neuron_self_attention:
            old_attention_params_neuron = old_network.attention_params_neuron[i][
                :, :, intersection_new_, np.append(intersection_new_, -1)
            ]
            new_attention_params_neuron[tb_index][
                pos_new, :, intersection_old_, np.append(intersection_new_, -1)
            ] = old_attention_params_neuron

        if new_network.use_topo_self_attention:
            old_attention_params_topo = old_network.attention_params_topo[i][
                :, intersection_new_
            ]
            new_attention_params_topo[tb_index][
                pos_new, intersection_old_
            ] = old_attention_params_topo

        if new_network.use_topo_norm:
            old_topo_norm_params = old_network.topo_norm_params[i][:, intersection_new_]
            new_topo_norm_params[tb_index][
                pos_new, intersection_old_
            ] = old_topo_norm_params

        if new_network.use_adaptive_activations:
            old_adaptive_activation_params = old_network.adaptive_activation_params[i]
            new_adaptive_activation_params[tb_index][
                pos_new
            ] = old_adaptive_activation_params

    new_weights_and_biases = [jnp.array(w) for w in new_weights_and_biases]
    new_attention_params_neuron = (
        [jnp.array(w) for w in new_attention_params_neuron]
        if new_network.use_neuron_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )
    new_attention_params_topo = (
        [jnp.array(w) for w in new_attention_params_topo]
        if new_network.use_topo_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )
    new_topo_norm_params = (
        [jnp.array(w) for w in new_topo_norm_params]
        if new_network.use_topo_norm
        else [jnp.nan] * new_network.num_topo_batches
    )
    new_adaptive_activation_params = (
        [jnp.array(w) for w in new_adaptive_activation_params]
        if new_network.use_adaptive_activations
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network.weights_and_biases,
            network.attention_params_neuron,
            network.attention_params_topo,
            network.topo_norm_params,
            network.adaptive_activation_params,
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
    adding them as isolated nodes in the graph. Use `cnx.add_connections` after this
    function has been called to add the desired connections.

    **Arguments:**

    - `network`: The `NeuralNetwork` to add neurons to
    - `new_hidden_neurons`: A sequence of new hidden neurons (more specifically, their
        identifiers/names) to add to the network. These must be unique, i.e. cannot
        already exist in the network. These must also specifically be hidden neurons.
        To add input or output neurons, use `cnx.add_input_neurons` or
        `cnx.add_output_neurons`.
    - `key`: The `jax.random.PRNGKey` used for new parameter initialization.
        Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.

    **Returns:**

    A `NeuralNetwork` with the new hidden neurons added and parameters from the
    original network retained.
    """
    # If there is nothing to change, return the network as given
    if len(new_hidden_neurons) == 0:
        return network

    # Check that none of the new neurons already exist in the network
    existing_neurons = network.graph.nodes
    for neuron in new_hidden_neurons:
        if neuron in existing_neurons:
            raise ValueError(f"Neuron {neuron} already exists in the network.")

    # Update graph information
    new_graph = nx.DiGraph(network.graph)
    new_graph.add_nodes_from(new_hidden_neurons)

    # Update topological sort
    topo_sort = network.topo_sort
    num_output_neurons = len(network.output_neurons)

    # It doesn't really matter where in a topological sort new isolated nodes are
    # added, it still remains a valid topological sort. We add them right before the
    # output neurons to make it easier to keep track of indices when copying
    # parameters over.
    topo_sort = (
        topo_sort[:-num_output_neurons]
        + list(new_hidden_neurons)
        + network.output_neurons
    )

    # Random key
    key = key if key is not None else jr.PRNGKey(0)

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        network.input_neurons,
        network.output_neurons,
        network.hidden_activation,
        network.output_transformation,
        network.dropout_p,
        network.use_topo_norm,
        network.use_topo_self_attention,
        network.use_neuron_self_attention,
        network.use_adaptive_activations,
        topo_sort=topo_sort,
        key=key,
    )

    old_network = network
    num_new_hidden_neurons = len(new_hidden_neurons)
    assert old_network.num_topo_batches == new_network.num_topo_batches
    for i in range(old_network.num_topo_batches - 1):
        assert (
            old_network.weights_and_biases[i].shape
            == new_network.weights_and_biases[i].shape
        )
        assert (
            old_network.attention_params_neuron[i].shape
            == new_network.attention_params_neuron[i].shape
        )
        assert (
            old_network.attention_params_topo[i].shape
            == new_network.attention_params_topo[i].shape
        )
        assert (
            old_network.topo_norm_params[i].shape
            == new_network.topo_norm_params[i].shape
        )

    # Copy weights and biases
    assert (
        old_network.weights_and_biases[-1].shape
        == new_network.weights_and_biases[-1][:-num_new_hidden_neurons].shape
    )
    new_weights_and_biases = old_network.weights_and_biases[:-1] + [
        new_network.weights_and_biases[-1]
        .at[-num_output_neurons:]
        .set(old_network.weights_and_biases[-1])
    ]

    # Copy neuron-level attention parameters
    assert (
        old_network.attention_params_neuron[-1].shape
        == new_network.attention_params_neuron[-1][:-num_new_hidden_neurons].shape
    )
    new_attention_params_neuron = (
        old_network.attention_params_neuron[:-1]
        + [
            new_network.attention_params_neuron[-1]
            .at[-num_output_neurons:]
            .set(old_network.attention_params_neuron[-1])
        ]
        if new_network.use_neuron_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Copy topo-level attention parameters
    assert (
        old_network.attention_params_topo[-1].shape
        == new_network.attention_params_topo[-1][:-num_new_hidden_neurons].shape
    )
    new_attention_params_topo = (
        old_network.attention_params_topo[:-1]
        + [
            new_network.attention_params_topo[-1]
            .at[-num_output_neurons:]
            .set(old_network.attention_params_topo[-1])
        ]
        if new_network.use_topo_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Copy topo norm parameters
    assert (
        old_network.topo_norm_params[-1].shape
        == new_network.topo_norm_params[-1][:-num_new_hidden_neurons].shape
    )
    new_topo_norm_params = (
        old_network.topo_norm_params[:-1]
        + [
            new_network.topo_norm_params[-1]
            .at[-num_output_neurons:]
            .set(old_network.topo_norm_params[-1])
        ]
        if new_network.use_topo_norm
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Copy adaptive activation parameters
    assert (
        old_network.adaptive_activation_params.size
        == new_network.adaptive_activation_params.size - num_new_hidden_neurons
    )
    indices = jnp.arange(old_network.num_neurons - num_output_neurons).append(
        jnp.arange(
            new_network.num_neurons - num_output_neurons, new_network.num_neurons
        )
    )
    new_adaptive_activation_params = (
        new_network.adaptive_activation_params.at[indices].set(
            old_network.adaptive_activation_params
        )
        if new_network.use_adaptive_activations
        else jnp.nan
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network.weights_and_biases,
            network.attention_params_neuron,
            network.attention_params_topo,
            network.topo_norm_params,
            network.adaptive_activation_params,
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
    adding them as isolated nodes in the graph. Use `cnx.add_connections` after this
    function has been called to add any desired connections.

    **Arguments:**

    - `network`: The `NeuralNetwork` to add neurons to
    - `new_output_neurons`: A sequence of new output neurons (more specifically, their
        identifiers/names) to add to the network. These must be unique, i.e. cannot
        already exist in the network. These must also specifically be output neurons.
        To add input or output neurons, use `cnx.add_input_neurons` or
        `cnx.add_output_neurons`.
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
    output_neurons = network.output_neurons + list(new_output_neurons)
    num_output_neurons = len(output_neurons)

    # Check that none of the new neurons already exist in the network
    existing_neurons = network.graph.nodes
    for neuron in new_output_neurons:
        if neuron in existing_neurons:
            raise ValueError(f"Neuron {neuron} already exists in the network.")

    # Update graph information
    new_graph = nx.DiGraph(network.graph)
    new_graph.add_nodes_from(new_output_neurons)

    # Update topological sort, appending the new output neurons to the end
    # of the output neuron list
    topo_sort = network.topo_sort + list(new_output_neurons)

    # Random key
    key = key if key is not None else jr.PRNGKey(0)

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        network.input_neurons,
        output_neurons,
        network.hidden_activation,
        network.output_transformation,
        network.dropout_p,
        network.use_topo_norm,
        network.use_topo_self_attention,
        network.use_neuron_self_attention,
        network.use_adaptive_activations,
        topo_sort=topo_sort,
        key=key,
    )

    old_network = network
    num_new_output_neurons = len(new_output_neurons)
    assert old_network.num_topo_batches == new_network.num_topo_batches
    for i in range(old_network.num_topo_batches - 1):
        assert (
            old_network.weights_and_biases[i].shape
            == new_network.weights_and_biases[i].shape
        )
        assert (
            old_network.attention_params_neuron[i].shape
            == new_network.attention_params_neuron[i].shape
        )
        assert (
            old_network.attention_params_topo[i].shape
            == new_network.attention_params_topo[i].shape
        )
        assert (
            old_network.topo_norm_params[i].shape
            == new_network.topo_norm_params[i].shape
        )

    # Copy weights and biases
    assert (
        old_network.weights_and_biases[-1].shape
        == new_network.weights_and_biases[-1][:-num_new_output_neurons].shape
    )
    new_weights_and_biases = old_network.weights_and_biases[:-1] + [
        new_network.weights_and_biases[-1]
        .at[-num_output_neurons:-num_new_output_neurons]
        .set(old_network.weights_and_biases[-1])
    ]

    # Copy neuron-level attention parameters
    assert (
        old_network.attention_params_neuron[-1].shape
        == new_network.attention_params_neuron[-1][:-num_new_output_neurons].shape
    )
    new_attention_params_neuron = (
        (
            old_network.attention_params_neuron[:-1]
            + [
                new_network.attention_params_neuron[-1]
                .at[-num_output_neurons:-num_new_output_neurons]
                .set(old_network.attention_params_neuron[-1])
            ]
        )
        if new_network.use_neuron_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Copy topo-level attention parameters
    assert (
        old_network.attention_params_topo[-1].shape
        == new_network.attention_params_topo[-1][:-num_new_output_neurons].shape
    )
    new_attention_params_topo = (
        (
            old_network.attention_params_topo[:-1]
            + [
                new_network.attention_params_topo[-1]
                .at[-num_output_neurons:-num_new_output_neurons]
                .set(old_network.attention_params_topo[-1])
            ]
        )
        if new_network.use_topo_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Copy topo norm parameters
    assert (
        old_network.topo_norm_params[-1].shape
        == new_network.topo_norm_params[-1][:-num_new_output_neurons].shape
    )
    new_topo_norm_params = (
        (
            old_network.topo_norm_params[:-1]
            + [
                new_network.topo_norm_params[-1]
                .at[-num_output_neurons:-num_new_output_neurons]
                .set(old_network.topo_norm_params[-1])
            ]
        )
        if new_network.use_topo_norm
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Copy adaptive activation parameters
    assert (
        old_network.adaptive_activation_params[-1].shape
        == new_network.adaptive_activation_params[-1][:-num_new_output_neurons].shape
    )
    new_adaptive_activation_params = (
        (
            old_network.adaptive_activation_params[:-1]
            + [
                new_network.adaptive_activation_params[-1]
                .at[-num_output_neurons:-num_new_output_neurons]
                .set(old_network.adaptive_activation_params[-1])
            ]
        )
        if new_network.use_adaptive_activations
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network.weights_and_biases,
            network.attention_params_neuron,
            network.attention_params_topo,
            network.topo_norm_params,
            network.adaptive_activation_params,
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
    them as isolated nodes in the graph. Use `cnx.add_connections` after this function
    has been called to add the desired connections.

    **Arguments:**

    - `network`: The `NeuralNetwork` to add neurons to
    - `new_input_neurons`: A sequence of new input neurons (more specifically, their
        identifiers/names) to add to the network. These must be unique, i.e. cannot
        already exist in the network. These must also specifically be input neurons.
        To add hidden or output neurons, use `cnx.add_hidden_neurons` or
        `cnx.add_output_neurons`.
    - `key`: The `jax.random.PRNGKey` used for new parameter initialization.
        Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.

    **Returns:**

    A `NeuralNetwork` with the new input neurons added and parameters from the
    original network retained.
    """
    # If there is nothing to change, return the network as given
    if len(new_input_neurons) == 0:
        return network

    # Update input neurons
    input_neurons = network.input_neurons + list(new_input_neurons)

    # Check that none of the new neurons already exist in the network
    existing_neurons = network.graph.nodes
    for neuron in new_input_neurons:
        if neuron in existing_neurons:
            raise ValueError(f"Neuron {neuron} already exists in the network.")

    # Update graph information
    new_graph = nx.DiGraph(network.graph)
    new_graph.add_nodes_from(new_input_neurons)

    # Update topological sort, appending the new input neurons to the end of the
    # input neuron list
    topo_sort = network.topo_sort
    num_input_neurons = len(network.input_neurons)
    first, rest = topo_sort[:num_input_neurons], topo_sort[num_input_neurons:]
    topo_sort = first + list(new_input_neurons) + rest

    # Random key
    key = key if key is not None else jr.PRNGKey(0)

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        input_neurons,
        network.output_neurons,
        network.hidden_activation,
        network.output_transformation,
        network.dropout_p,
        network.use_topo_norm,
        network.use_topo_self_attention,
        network.use_neuron_self_attention,
        network.use_adaptive_activations,
        topo_sort=topo_sort,
        key=key,
    )

    old_network = network
    num_new_input_neurons = len(new_input_neurons)
    num_old_input_neurons = len(network.input_neurons)
    assert old_network.num_topo_batches == new_network.num_topo_batches
    for i in range(1, old_network.num_topo_batches):
        assert (
            old_network.weights_and_biases[i].shape
            == new_network.weights_and_biases[i].shape
        )
        assert (
            old_network.attention_params_neuron[i].shape
            == new_network.attention_params_neuron[i].shape
        )
        assert (
            old_network.attention_params_topo[i].shape
            == new_network.attention_params_topo[i].shape
        )
        assert (
            old_network.topo_norm_params[i].shape
            == new_network.topo_norm_params[i].shape
        )

    # Copy weights and biases
    assert (
        old_network.weights_and_biases[0].shape
        == new_network.weights_and_biases[0][num_new_input_neurons:].shape
    )
    new_weights_and_biases = [
        new_network.weights_and_biases[0]
        .at[:num_old_input_neurons]
        .set(old_network.weights_and_biases[0])
    ] + old_network.weights_and_biases[1:]

    # Copy neuron-level attention parameters
    assert (
        old_network.attention_params_neuron[0].shape
        == new_network.attention_params_neuron[0][num_new_input_neurons:].shape
    )
    new_attention_params_neuron = (
        [
            new_network.attention_params_neuron[0]
            .at[:num_old_input_neurons]
            .set(old_network.attention_params_neuron[0])
        ]
        + old_network.attention_params_neuron[1:]
        if new_network.use_neuron_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Copy topo-level attention parameters
    assert (
        old_network.attention_params_topo[0].shape
        == new_network.attention_params_topo[0][num_new_input_neurons:].shape
    )
    new_attention_params_topo = (
        [
            new_network.attention_params_topo[0]
            .at[:num_old_input_neurons]
            .set(old_network.attention_params_topo[0])
        ]
        + old_network.attention_params_topo[1:]
        if new_network.use_topo_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Copy topo norm parameters
    assert (
        old_network.topo_norm_params[0].shape
        == new_network.topo_norm_params[0][num_new_input_neurons:].shape
    )
    new_topo_norm_params = (
        [
            new_network.topo_norm_params[0]
            .at[:num_old_input_neurons]
            .set(old_network.topo_norm_params[0])
        ]
        + old_network.topo_norm_params[1:]
        if new_network.use_topo_norm
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Copy adaptive activation parameters
    assert (
        old_network.adaptive_activation_params[0].shape
        == new_network.adaptive_activation_params[0][num_new_input_neurons:].shape
    )
    new_adaptive_activation_params = (
        [
            new_network.adaptive_activation_params[0]
            .at[:num_old_input_neurons]
            .set(old_network.adaptive_activation_params[0])
        ]
        + old_network.adaptive_activation_params[1:]
        if new_network.use_adaptive_activations
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network.weights_and_biases,
            network.attention_params_neuron,
            network.attention_params_topo,
            network.topo_norm_params,
            network.adaptive_activation_params,
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
        return network

    # Check that all of the new neurons already exist in the network
    existing_neurons = network.graph.nodes
    for neuron in neurons:
        if neuron not in existing_neurons:
            raise ValueError(f"Neuron {neuron} does not exist in the network.")

    # Remove all incoming and outgoing connections of all neurons to be removed
    edges_to_remove = {}
    for neuron in neurons:
        edges_to_remove[neuron] = network.adjacency_dict[neuron]
        incoming_neurons = network.adjacency_dict_inv[neuron]
        for in_neuron in incoming_neurons:
            edges_to_remove[in_neuron] = [neuron]

    network = remove_connections(network, edges_to_remove)

    # Set input and output neurons
    input_neurons = network.input_neurons
    output_neurons = network.output_neurons
    for neuron in neurons:
        if neuron in input_neurons:
            input_neurons.remove(neuron)
        elif neuron in output_neurons:
            output_neurons.remove(neuron)

    # Update graph information
    new_graph = nx.DiGraph(network.graph)
    new_graph.remove_nodes_from(neurons)

    # Update topological sort
    topo_sort = network.topo_sort
    for neuron in neurons:
        topo_sort.remove(neuron)

    # Random key
    key = network._get_key()

    # Create new network
    new_network = NeuralNetwork(
        new_graph,
        input_neurons,
        output_neurons,
        network.hidden_activation,
        network.output_transformation,
        network.dropout_p,
        network.use_topo_norm,
        network.use_topo_self_attention,
        network.use_neuron_self_attention,
        network.use_adaptive_activations,
        topo_sort=topo_sort,
        key=key,
    )

    old_network = network
    ids_old_to_new, ids_new_to_old = _get_id_mappings_old_new(old_network, new_network)

    # Copy neuron parameters and attention parameters
    new_weights_and_biases = [np.array(w) for w in new_network.weights_and_biases]
    new_attention_params_neuron = [
        np.array(w) for w in new_network.attention_params_neuron
    ]
    new_attention_params_topo = [np.array(w) for w in new_network.attention_params_topo]
    new_topo_norm_params = [np.array(w) for w in new_network.topo_norm_params]
    new_adaptive_activation_params = [
        np.array(w) for w in new_network.adaptive_activation_params
    ]

    # Loop through each topo batch in the new network and copy the corresponding
    # parameters present in the old network to the new network
    for i, tb_new in enumerate(new_network.topo_batches):
        tb_old = ids_new_to_old[tb_new]
        assert np.all(tb_old >= 0)

        # Get the index of the topo batch `tb_old` is a subset of
        tb_index, _ = old_network.neuron_to_topo_batch_idx[tb_old[0]]

        # Make sure it is actually a subset
        intersection = np.intersect1d(
            tb_old, np.array(old_network.topo_batches[tb_index])
        )
        assert intersection.size == tb_old.size

        min_new, max_new = new_network.min_index[i], new_network.max_index[i]
        range_new = np.arange(min_new, max_new + 1)
        min_old, max_old = (
            old_network.min_index[tb_index],
            old_network.max_index[tb_index],
        )
        range_old = np.arange(min_old, max_old + 1)
        input_indices_old = ids_new_to_old[range_new]
        assert np.all(input_indices_old >= 0)
        intersection_old = np.intersect1d(range_old, input_indices_old)
        assert intersection_old.size > 0

        # Re-order the values of `intersection_old` to reflect the order in
        # `input_indices_old`
        intersection_old = input_indices_old[
            np.in1d(input_indices_old, intersection_old)
        ]
        intersection_old_ = intersection_old - min_old
        intersection_new = ids_old_to_new[intersection_old]
        assert np.all(intersection_new >= 0)
        intersection_new_ = intersection_new - min_new

        # Get the respective neuron positions within the old topological batches
        pos_old = [old_network.neuron_to_topo_batch_idx[id][1] for id in tb_old]
        pos_old = np.array(pos_old, dtype=int)

        # Copy parameters
        old_weights_and_biases = old_network.weights_and_biases[tb_index][
            pos_old, np.append(intersection_old_, -1)
        ]
        new_weights_and_biases[i][
            :, np.append(intersection_new_, -1)
        ] = old_weights_and_biases

        if new_network.use_neuron_self_attention:
            old_attention_params_neuron = old_network.attention_params_neuron[
                tb_index
            ][  # noqa: E501
                pos_old, :, intersection_old_, np.append(intersection_old_, -1)
            ]
            new_attention_params_neuron[i][
                :, :, intersection_new_, np.append(intersection_new_, -1)
            ] = old_attention_params_neuron

        if new_network.use_topo_self_attention:
            old_attention_params_topo = old_network.attention_params_topo[tb_index][
                pos_old, intersection_old_
            ]
            new_attention_params_topo[i][
                :, intersection_new_
            ] = old_attention_params_topo

        if new_network.use_topo_norm:
            old_topo_norm_params = old_network.topo_norm_params[tb_index][
                pos_old, intersection_old_
            ]
            new_topo_norm_params[i][:, intersection_new_] = old_topo_norm_params

        if new_network.use_adaptive_activations:
            old_adaptive_activation_params = old_network.adaptive_activation_params[
                tb_index
            ][pos_old]
            new_adaptive_activation_params[i] = old_adaptive_activation_params

    new_weights_and_biases = [jnp.array(w) for w in new_weights_and_biases]
    new_attention_params_neuron = (
        [jnp.array(w) for w in new_attention_params_neuron]
        if new_network.use_neuron_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )
    new_attention_params_topo = (
        [jnp.array(w) for w in new_attention_params_topo]
        if new_network.use_topo_self_attention
        else [jnp.nan] * new_network.num_topo_batches
    )
    new_topo_norm_params = (
        [jnp.array(w) for w in new_topo_norm_params]
        if new_network.use_topo_norm
        else [jnp.nan] * new_network.num_topo_batches
    )
    new_adaptive_activation_params = (
        [jnp.array(w) for w in new_adaptive_activation_params]
        if new_network.use_adaptive_activations
        else [jnp.nan] * new_network.num_topo_batches
    )

    # Transfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: (
            network.weights_and_biases,
            network.attention_params_neuron,
            network.attention_params_topo,
            network.topo_norm_params,
            network.adaptive_activation_params,
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
