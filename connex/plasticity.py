import functools as ft
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import equinox as eqx
import equinox.experimental as eqxe

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

import networkx as nx
import numpy as np

from .network import NeuralNetwork
from .utils import _identity


def _get_id_mappings_old_new(
    old_network: NeuralNetwork, new_network: NeuralNetwork
) -> Tuple[np.ndarray, np.ndarray]:
    assert old_network.num_neurons == new_network.num_neurons
    assert set(old_network.graph.nodes) == set(new_network.graph.nodes)
    ids_old_to_new = np.empty((old_network.num_neurons,), dtype=int)
    ids_new_to_old = np.empty((old_network.num_neurons,), dtype=int)
    for neuron in old_network.graph.nodes:
        old_id = old_network.neuron_to_id[neuron]
        new_id = new_network.neuron_to_id[neuron]
        ids_old_to_new[old_id] = new_id
        ids_new_to_old[new_id] = old_id
    return ids_old_to_new, ids_new_to_old


def add_connections(
    network: NeuralNetwork,
    connections: Mapping[Any, Sequence[Any]],
    *,
    key: Optional[jr.PRNGKey] = None
) -> NeuralNetwork:
    """Add connections to the network.
    
    **Arguments:**

    - `network`: A `NeuralNetwork` object.
    - `connections`: An adjacency dict mapping an existing neuron (by its 
        NetworkX id) to its new outgoing connections. Connections that already 
        exist are ignored.
    - `key`: The `jax.random.PRNGKey` used for new weight initialization. 
        Optional, keyword-only argument. Defaults to the key stored in `network.key_state`.

    **Returns:**

    A `NeuralNetwork` object with the specified connections added and original
    parameters retained.
    """
    # Set input and output neurons
    input_neurons = [network.topo_sort[id] for id in network.input_neurons]
    output_neurons = [network.topo_sort[id] for id in network.output_neurons]

    # Check that all new connections are between neurons actually in the network
    # and that no connection have an input neurons as outputs
    existing_neurons = network.graph.nodes
    for input, outputs in connections.items():
        assert input in existing_neurons, input
        for output in outputs:
            assert output in existing_neurons, f"Neuron {output} does not exist in the network."
            assert output not in input_neurons, f"Cannot add connection to input neuron {output}."

    # Set element-wise activations
    hidden_activation = network.hidden_activation \
        if isinstance(network.hidden_activation, eqx.Module) \
        else network._hidden_activation

    # Update connectivity information
    new_graph = nx.DiGraph(network.graph)
    for (input, outputs) in connections.items():
        new_edges = [(input, output) for output in outputs]
        new_graph.add_edges_from(new_edges)

    # Copy dropout info to dict
    dropout_p = {}
    for id in network.topo_sort:
        dropout_p[network.topo_sort[id]] = network.dropout_p[id]

    # Update topological sort (reference: https://stackoverflow.com/a/24764451)
    def _add_edge_rec(topo_sort, input, output, visited={}):
        input_index = topo_sort.index(input)
        output_index = topo_sort.index(output)
        assert input_index != output_index, "Input and output cannot be the same"
        if input_index < output_index:
            return topo_sort, visited
        assert output not in visited, f"Edge ({input}, {output}) creates a cycle"
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
    key = key if key is not None else network._get_current_key()

    # Create new network
    new_network = NeuralNetwork(
        graph=new_graph,
        input_neurons=input_neurons,
        output_neurons=output_neurons,
        hidden_activation=hidden_activation,
        output_activation=network.output_transformation,
        dropout_p=dropout_p,
        use_topo_norm=network.use_topo_norm,
        use_topo_self_attention=network.use_topo_self_attention,
        use_neuron_self_attention=network.use_neuron_self_attention,
        use_adaptive_activations=network.use_adaptive_activations,
        topo_sort=topo_sort,
        key=key
    )

    ids_old_to_new, ids_new_to_old = _get_id_mappings_old_new(old_network, new_network)
    # Copy neuron parameters and attention parameters
    new_weights_and_biases = [np.array(w) for w in new_network.weights_and_biases]
    new_attention_params_neuron = [np.array(w) for w in new_network.attention_params_neuron]
    new_attention_params_topo = [np.array(w) for w in new_network.attention_params_topo]
    new_topo_norm_params = [np.array(w) for w in new_network.topo_norm_params]
    new_adaptive_activation_params = [np.array(w) for w in new_network.adaptive_activation_params]
    old_network = network
    # Loop through each topo batch in the new network and copy the corresponding parameters
    # present in the old network to the new network
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
        min_old, max_old = old_network.min_index[tb_index], old_network.max_index[tb_index]
        range_old = np.arange(min_old, max_old + 1)
        input_indices_old = ids_new_to_old[range_new]
        intersection_old = np.intersect1d(range_old, input_indices_old)
        assert intersection_old.size > 0

        # Re-order the values of `intersection_old` to reflect the order in `input_indices_old`
        intersection_old = input_indices_old[np.in1d(input_indices_old, intersection_old)]
        intersection_old_ = intersection_old - min_old
        intersection_new = ids_old_to_new[intersection_old]
        intersection_new_ = intersection_new - min_new

        # Get the respective neuron positions within the old topological batches
        pos_old = [old_network.neuron_to_topo_batch_idx[id][1] for id in tb_old]
        pos_old = np.array(pos_old, dtype=int)

        # Copy parameters
        old_weights_and_biases = old_network.weights_and_biases[tb_index][pos_old, np.append(intersection_old_, -1)]
        new_weights_and_biases[i][:, np.append(intersection_new_, -1)] = old_weights_and_biases

        if new_network.use_neuron_self_attention:
            old_attention_params_neuron = old_network.attention_params_neuron[tb_index][
                pos_old, :, intersection_old_, np.append(intersection_old_, -1)
            ]
            new_attention_params_neuron[i][
                :, :, intersection_new_, np.append(intersection_new_, -1)
            ] = old_attention_params_neuron

        if new_network.use_topo_self_attention:
            old_attention_params_topo = old_network.attention_params_topo[tb_index][pos_old, intersection_old_]
            new_attention_params_topo[i][:, intersection_new_] = old_attention_params_topo

        if new_network.use_topo_norm:
            old_topo_norm_params = old_network.topo_norm_params[tb_index][pos_old, intersection_old_]
            new_topo_norm_params[i][:, intersection_new_] = old_topo_norm_params

        if new_network.use_adaptive_activations:
            old_adaptive_activation_params = old_network.adaptive_activation_params[tb_index][pos_old]
            new_adaptive_activation_params[i] = old_adaptive_activation_params

    new_weights_and_biases = [jnp.array(w) for w in new_weights_and_biases]
    new_attention_params_neuron = [jnp.array(w) for w in new_attention_params_neuron] \
        if new_network.use_neuron_self_attention else [jnp.nan]
    new_attention_params_topo = [jnp.array(w) for w in new_attention_params_topo] \
        if new_network.use_topo_self_attention else [jnp.nan]
    new_topo_norm_params = [jnp.array(w) for w in new_topo_norm_params] \
        if new_network.use_topo_norm else [jnp.nan]
    new_adaptive_activation_params = [jnp.array(w) for w in new_adaptive_activation_params] \
        if new_network.use_adaptive_activations else [jnp.nan]

    # Trasfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: 
        (
            network.weights_and_biases, 
            network.attention_params_neuron, 
            network.attention_params_topo, 
            network.topo_norm_params,
            network.adaptive_activation_params
        ),
        new_network,
        (
            new_weights_and_biases,
            new_attention_params_neuron,
            new_attention_params_topo,
            new_topo_norm_params,
            new_adaptive_activation_params
        )
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
    # Set input and output neurons
    input_neurons = [network.topo_sort[id] for id in network.input_neurons]
    output_neurons = [network.topo_sort[id] for id in network.output_neurons]

    # Check that all new connections are between neurons actually in the network
    existing_neurons = network.graph.nodes
    for input, outputs in connections.items():
        assert input in existing_neurons, input
        for output in outputs:
            assert output in existing_neurons, f"Neuron {output} does not exist in the network."

    # Set element-wise activations
    hidden_activation = network.hidden_activation \
        if isinstance(network.hidden_activation, eqx.Module) \
        else network._hidden_activation

    # Update connectivity information
    new_graph = nx.DiGraph(network.graph)
    for (input, outputs) in connections.items():
        edges_to_remove = [(input, output) for output in outputs]
        new_graph.remove_edges_from(edges_to_remove)

    # Copy dropout info to dict
    dropout_p = {}
    for id in network.topo_sort:
        dropout_p[network.topo_sort[id]] = network.dropout_p[id]

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
        # TODO: topo_sort this info within the dict in network?
        max_input_idx = max(map(network.neuron_to_id.get, neuron_inputs[output]))
        neuron = network.topo_sort[max_input_idx]
        neuron_idx = topo_sort.index(neuron) # TODO: this seems inefficient
        topo_sort.insert(neuron_idx + 1, output)

    # Get the current network key
    network_key = network._get_current_key()

    # Create new network
    new_network = NeuralNetwork(
        graph=new_graph,
        input_neurons=input_neurons,
        output_neurons=output_neurons,
        hidden_activation=hidden_activation,
        output_activation=network.output_transformation,
        dropout_p=dropout_p,
        use_topo_norm=network.use_topo_norm,
        use_topo_self_attention=network.use_topo_self_attention,
        use_neuron_self_attention=network.use_neuron_self_attention,
        use_adaptive_activations=network.use_adaptive_activations,
        topo_sort=topo_sort,
        key=network_key
    )

    ids_old_to_new, ids_new_to_old = _get_id_mappings_old_new(old_network, new_network)
    # Copy neuron parameters and attention parameters
    _jnp_to_np = lambda lst: list(map(np.array, lst))
    new_weights_and_biases = _jnp_to_np(new_network.weights_and_biases) # TODO: idk which I like more
    new_attention_params_neuron = [np.array(w) for w in new_network.attention_params_neuron]
    new_attention_params_topo = [np.array(w) for w in new_network.attention_params_topo]
    new_topo_norm_params = [np.array(w) for w in new_network.topo_norm_params]
    new_adaptive_activation_params = [np.array(w) for w in new_network.adaptive_activation_params]
    old_network = network
    # Loop through each topo batch in the old network and copy the corresponding parameters
    # present in the new network from the old network
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
        min_new, max_new = new_network.min_index[tb_index], new_network.max_index[tb_index]
        range_new = np.arange(min_new, max_new + 1)
        input_indices_new = ids_old_to_new[range_old]
        intersection_new = np.intersect1d(range_new, input_indices_new)
        assert intersection_new.size > 0

        # Re-order the values of `intersection_new` to reflect the order in `input_indices_new`
        intersection_new = input_indices_new[np.in1d(input_indices_new, intersection_new)]
        intersection_new_ = intersection_new - min_new
        intersection_old = ids_new_to_old[intersection_new]
        intersection_old_ = intersection_old - min_old

        # Get the respective neuron positions within the new topological batches
        pos_new = [new_network.neuron_to_topo_batch_idx[id][1] for id in tb_new]
        pos_new = np.array(pos_new, dtype=int)

        # Copy parameters
        old_weights_and_biases = old_network.weights_and_biases[i][:, np.append(intersection_old_, -1)]
        new_network.weights_and_biases[tb_index][pos_new, np.append(intersection_new_, -1)] = old_weights_and_biases

        if new_network.use_neuron_self_attention:
            old_attention_params_neuron = old_network.attention_params_neuron[i][
                :, :, intersection_new_, np.append(intersection_new_, -1)
            ]
            new_attention_params_neuron[tb_index][
                pos_new, :, intersection_old_, np.append(intersection_new_, -1)
            ] = old_attention_params_neuron

        if new_network.use_topo_self_attention:
            old_attention_params_topo = old_network.attention_params_topo[i][:, intersection_new_]
            new_attention_params_topo[tb_index][pos_new, intersection_old_] = old_attention_params_topo

        if new_network.use_topo_norm:
            old_topo_norm_params = old_network.topo_norm_params[i][:, intersection_new_]
            new_topo_norm_params[tb_index][pos_new, intersection_old_] = old_topo_norm_params

        if new_network.use_adaptive_activations:
            old_adaptive_activation_params = old_network.adaptive_activation_params[i]
            new_adaptive_activation_params[tb_index][pos_new] = old_adaptive_activation_params

    new_weights_and_biases = [jnp.array(w) for w in new_weights_and_biases]
    new_attention_params_neuron = [jnp.array(w) for w in new_attention_params_neuron] \
        if new_network.use_neuron_self_attention else [jnp.nan]
    new_attention_params_topo = [jnp.array(w) for w in new_attention_params_topo] \
        if new_network.use_topo_self_attention else [jnp.nan]
    new_topo_norm_params = [jnp.array(w) for w in new_topo_norm_params] \
        if new_network.use_topo_norm else [jnp.nan]
    new_adaptive_activation_params = [jnp.array(w) for w in new_adaptive_activation_params] \
        if new_network.use_adaptive_activations else [jnp.nan]

    # Trasfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: 
        (
            network.weights_and_biases, 
            network.attention_params_neuron, 
            network.attention_params_topo, 
            network.topo_norm_params,
            network.adaptive_activation_params
        ),
        new_network,
        (
            new_weights_and_biases,
            new_attention_params_neuron,
            new_attention_params_topo,
            new_topo_norm_params,
            new_adaptive_activation_params
        )
    )


def add_hidden_neurons(
    network: NeuralNetwork,
    new_hidden_neurons: Sequence[Any],
    *,
    key: Optional[jr.PRNGKey] = None
) -> NeuralNetwork:
    """Add hidden neurons to the network. Note that this function only adds neurons themselves,
    not any connections associated with the new neurons, effectively adding them as isolated nodes 
    in the graph. Use `cnx.add_connections` after this function has been called to add the desired 
    connections.

    **Arguments:**

    - `network`: The `NeuralNetwork` to add neurons to
    - `new_hidden_neurons`: A sequence of new hidden neurons (more specifically, their 
        identifiers/names) to add to the network. These must be unique, i.e. cannot already
        exist in the network. These must also specifically be hidden neurons. To add input or 
        output neurons, use `cnx.add_input_neurons` or `cnx.add_output_neurons`.
    - `key`: The `jax.random.PRNGKey` used for new parameter initialization. 
        Optional, keyword-only argument. Defaults to the key saved in `network.key_state`.

    **Returns:**

    A `NeuralNetwork` with the new neurons and respective connections added and original
    parameters present in the original network retained.
    """
    # Set input and output neurons (TODO: idk I don't like this?)
    input_neurons = [network.topo_sort[id] for id in network.input_neurons]
    output_neurons = [network.topo_sort[id] for id in network.output_neurons]

    # Check that none of the new neurons already exist in the network
    existing_neurons = network.graph.nodes
    for neuron in new_hidden_neurons:
        assert neuron not in existing_neurons, neuron

    # Set element-wise activations
    hidden_activation = network.hidden_activation \
        if isinstance(network.hidden_activation, eqx.Module) \
        else network._hidden_activation

    # Update graph information
    new_graph = nx.DiGraph(network.graph)
    new_graph.add_nodes_from(new_hidden_neurons)

    # Copy dropout info to dict (TODO: idk I don't like this?)
    dropout_p = {}
    for id in network.topo_sort:
        dropout_p[network.topo_sort[id]] = network.dropout_p[id]

    # Update topological sort
    topo_sort = network.topo_sort
    num_output_neurons = len(output_neurons)
    # It doesn't really matter where in a topological sort new isolated nodes are added, 
    # it still remains a valid topological sort. We add them right before the output neurons
    # to make it easier to keep track of indices when copying parameters over.
    topo_sort = topo_sort[:-num_output_neurons] + list(new_hidden_neurons) + output_neurons

    # Random key
    key = key if key is not None else network._get_current_key()

    # Create new network
    new_network = NeuralNetwork(
        graph=new_graph,
        input_neurons=input_neurons,
        output_neurons=output_neurons,
        hidden_activation=hidden_activation,
        output_activation=network.output_transformation,
        dropout_p=dropout_p,
        use_topo_norm=network.use_topo_norm,
        use_topo_self_attention=network.use_topo_self_attention,
        use_neuron_self_attention=network.use_neuron_self_attention,
        use_adaptive_activations=network.use_adaptive_activations,
        topo_sort=topo_sort,
        key=key
    )

    num_new_hidden_neurons = len(new_hidden_neurons)
    old_network = network
    assert old_network.num_topo_batches == new_network.num_topo_batches
    for i in range(old_network.num_topo_batches - 1):
        assert old_network.weights_and_biases[i].shape == new_network.weights_and_biases[i].shape
        assert old_network.attention_params_neuron[i].shape == new_network.attention_params_neuron[i].shape
        assert old_network.attention_params_topo[i].shape == new_network.attention_params_topo[i].shape
        assert old_network.topo_norm_params[i].shape == new_network.topo_norm_params[i].shape
    
    # Copy weights and biases
    assert old_network.weights_and_biases[-1].shape == new_network.weights_and_biases[-1][:-num_new_hidden_neurons].shape
    new_weights_and_biases = old_network.weights_and_biases[:-1] + \
        [new_network.weights_and_biases[-1].at[-num_output_neurons:].set(old_network.weights_and_biases[-1])]
    # Copy neuron-level attention parameters
    assert old_network.attention_params_neuron[-1].shape == new_network.attention_params_neuron[-1][:-num_new_hidden_neurons].shape
    new_attention_params_neuron = old_network.attention_params_neuron[:-1] + \
        [new_network.attention_params_neuron[-1].at[-num_output_neurons:].set(old_network.attention_params_neuron[-1])] \
        if new_network.use_neuron_self_attention else [jnp.nan]
    # Copy topo-level attention parameters
    assert old_network.attention_params_topo[-1].shape == new_network.attention_params_topo[-1][:-num_new_hidden_neurons].shape
    new_attention_params_topo = old_network.attention_params_topo[:-1] + \
        [new_network.attention_params_topo[-1].at[-num_output_neurons:].set(old_network.attention_params_topo[-1])] \
        if new_network.use_topo_self_attention else [jnp.nan]
    # Copy topo norm parameters
    assert old_network.topo_norm_params[-1].shape == new_network.topo_norm_params[-1][:-num_new_hidden_neurons].shape
    new_topo_norm_params = old_network.topo_norm_params[:-1] + \
        [new_network.topo_norm_params[-1].at[-num_output_neurons:].set(old_network.topo_norm_params[-1])] \
        if new_network.use_topo_norm else [jnp.nan]
    # Copy adaptive activation parameters
    assert old_network.adaptive_activation_params.size == new_network.adaptive_activation_params.size - num_new_hidden_neurons
    indices = jnp.arange(old_network.num_neurons - num_output_neurons).append(
        jnp.arange(new_network.num_neurons - num_output_neurons, new_network.num_neurons)
    )
    new_adaptive_activation_params = new_network.adaptive_activation_params.at[indices].set(old_network.adaptive_activation_params)

    # Trasfer all copied parameters to new network and return
    return eqx.tree_at(
        lambda network: 
        (
            network.weights_and_biases, 
            network.attention_params_neuron, 
            network.attention_params_topo, 
            network.topo_norm_params,
            network.adaptive_activation_params
        ),
        new_network,
        (
            new_weights_and_biases,
            new_attention_params_neuron,
            new_attention_params_topo,
            new_topo_norm_params,
            new_adaptive_activation_params
        )
    )



# def contract_cluster(
#     network: NeuralNetwork, 
#     neurons: Sequence[Hashable], 
#     contract_fn: Callable = jnp.max,
# ) -> NeuralNetwork:
#     pass
