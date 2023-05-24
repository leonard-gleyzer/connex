import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest

from connex import (
    add_connections,
    add_hidden_neurons,
    add_input_neurons,
    add_output_neurons,
    NeuralNetwork,
    remove_connections,
    remove_neurons,
)
from connex._plasticity import _get_id_mappings_old_new


def test_get_id_mappings_old_new():
    # Create a test graph
    graph1 = nx.DiGraph()
    graph1.add_edges_from([(1, 2), (2, 3), (1, 4), (4, 3), (1, 5)])

    # Create a NeuralNetwork instance for the test graph
    input_neurons1 = [1]
    output_neurons1 = [3]
    old_network = NeuralNetwork(graph1, input_neurons1, output_neurons1)

    # Create a second test graph
    graph2 = nx.DiGraph()
    graph2.add_edges_from([(1, 2), (2, 3), (1, 4)])

    # Create a NeuralNetwork instance for the second test graph
    input_neurons2 = [1]
    output_neurons2 = [3]
    new_network = NeuralNetwork(graph2, input_neurons2, output_neurons2)

    # Test the _get_id_mappings_old_new function
    ids_old_to_new, ids_new_to_old = _get_id_mappings_old_new(old_network, new_network)

    # Check if the ids_old_to_new mapping is correct
    assert np.array_equal(ids_old_to_new, np.array([0, 1, 2, -1, 3]))

    # Check if the ids_new_to_old mapping is correct
    assert np.array_equal(ids_new_to_old, np.array([0, 1, 2, 4]))


def test_add_connections_invalid():
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
    network = NeuralNetwork(graph, [1], [4])

    with pytest.raises(ValueError, match=r"does not exist in the network"):
        add_connections(network, {0: [1]})

    with pytest.raises(ValueError, match=r"does not exist in the network"):
        add_connections(network, {1: ["1"]})

    with pytest.raises(ValueError, match=r"Input neurons cannot receive"):
        add_connections(network, {2: [1]})

    with pytest.raises(ValueError):
        add_connections(network, {3: [2]})

    with pytest.raises(ValueError):
        add_connections(network, {3: [3]})


def test_remove_connections_invalid():
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
    network = NeuralNetwork(graph, [1], [4])

    with pytest.raises(ValueError, match=r"does not exist in the network"):
        remove_connections(network, {0: [1]})

    with pytest.raises(ValueError, match=r"does not exist in the network"):
        remove_connections(network, {1: ["1"]})


def _parameter_equivalence(network1: NeuralNetwork, network2: NeuralNetwork):
    # This function assumes that network1 is a subset of network2
    assert network1._use_adaptive_activations == network2._use_adaptive_activations
    assert network1._use_topo_norm == network2._use_topo_norm
    assert network1._use_topo_self_attention == network2._use_topo_self_attention
    assert network1._use_neuron_self_attention == network2._use_neuron_self_attention
    for neuron in network1._graph.nodes():
        if neuron in network1._input_neurons:
            continue
        assert neuron in network2._graph.nodes()
        neurons1_in = network1._adjacency_dict_inv[neuron]
        tb1, idx1 = network1._neuron_to_topo_batch_idx[network1._neuron_to_id[neuron]]
        tb1_inputs = np.arange(network1._min_index[tb1], network1._max_index[tb1] + 1)
        tb2, idx2 = network2._neuron_to_topo_batch_idx[network2._neuron_to_id[neuron]]
        tb2_inputs = np.arange(network2._min_index[tb2], network2._max_index[tb2] + 1)
        bias1 = network1._weights_and_biases[tb1][idx1, -1]
        bias2 = network2._weights_and_biases[tb2][idx2, -1]
        assert jnp.isclose(bias1, bias2)
        for n in neurons1_in:
            id1_in = network1._neuron_to_id[n]
            weight1 = network1._weights_and_biases[tb1][
                idx1, id1_in - network1._min_index[tb1]
            ]
            id2_in = network2._neuron_to_id[n]
            weight2 = network2._weights_and_biases[tb2][
                idx2, id2_in - network2._min_index[tb2]
            ]
            assert jnp.isclose(weight1, weight2)

        if network1._use_neuron_self_attention:
            query1, key1, value1 = network1._attention_params_neuron[tb1][idx1][:]
            query1w, query1b = query1[:, :-1], query1[:, -1]
            key1w, key1b = key1[:, :-1], key1[:, -1]
            value1w, value1b = value1[:, :-1], value1[:, -1]

            query2, key2, value2 = network2._attention_params_neuron[tb2][idx2][:]
            query2w, query2b = query2[:, :-1], query2[:, -1]
            key2w, key2b = key2[:, :-1], key2[:, -1]
            value2w, value2b = value2[:, :-1], value2[:, -1]

            for id1 in tb1_inputs:
                _neuron1 = network1._topo_sort[id1]
                _id1 = id1 - network1._min_index[tb1]
                _query1b = query1b[_id1]
                _key1b = key1b[_id1]
                _value1b = value1b[_id1]

                id2 = network2._neuron_to_id[_neuron1]
                if id2 not in tb2_inputs:
                    continue
                _id2 = network2._neuron_to_id[_neuron1] - network2._min_index[tb2]
                _query2b = query2b[_id2]
                _key2b = key2b[_id2]
                _value2b = value2b[_id2]

                assert jnp.isclose(_query1b, _query2b)
                assert jnp.isclose(_key1b, _key2b)
                assert jnp.isclose(_value1b, _value2b)

                for id11 in tb1_inputs:
                    _neuron11 = network1._topo_sort[id11]
                    _id11 = id11 - network1._min_index[tb1]
                    _query1w = query1w[_id1, _id11]
                    _key1w = key1w[_id1, _id11]
                    _value1w = value1w[_id1, _id11]

                    id22 = network2._neuron_to_id[_neuron11]
                    if id22 not in tb2_inputs:
                        continue
                    _id22 = network2._neuron_to_id[_neuron11] - network2._min_index[tb2]
                    _query2w = query2w[_id2, _id22]
                    _key2w = key2w[_id2, _id22]
                    _value2w = value2w[_id2, _id22]

                    assert jnp.isclose(_query1w, _query2w)
                    assert jnp.isclose(_key1w, _key2w)
                    assert jnp.isclose(_value1w, _value2w)

        if network1._use_topo_self_attention:
            query1, key1, value1 = network1._attention_params_topo[tb1][:]
            query1w, query1b = query1[:, :-1], query1[:, -1]
            key1w, key1b = key1[:, :-1], key1[:, -1]
            value1w, value1b = value1[:, :-1], value1[:, -1]

            query2, key2, value2 = network2._attention_params_topo[tb2][:]
            query2w, query2b = query2[:, :-1], query2[:, -1]
            key2w, key2b = key2[:, :-1], key2[:, -1]
            value2w, value2b = value2[:, :-1], value2[:, -1]

            for id1 in tb1_inputs:
                _neuron1 = network1._topo_sort[id1]
                _id1 = id1 - network1._min_index[tb1]
                _query1b = query1b[_id1]
                _key1b = key1b[_id1]
                _value1b = value1b[_id1]

                id2 = network2._neuron_to_id[_neuron1]
                if id2 not in tb2_inputs:
                    continue
                _id2 = network2._neuron_to_id[_neuron1] - network2._min_index[tb2]
                _query2b = query2b[_id2]
                _key2b = key2b[_id2]
                _value2b = value2b[_id2]

                assert jnp.isclose(_query1b, _query2b)
                assert jnp.isclose(_key1b, _key2b)
                assert jnp.isclose(_value1b, _value2b)

                for id11 in tb1_inputs:
                    _neuron11 = network1._topo_sort[id11]
                    _id11 = id11 - network1._min_index[tb1]
                    _query1w = query1w[_id1, _id11]
                    _key1w = key1w[_id1, _id11]
                    _value1w = value1w[_id1, _id11]

                    id22 = network2._neuron_to_id[_neuron11]
                    if id22 not in tb2_inputs:
                        continue
                    _id22 = network2._neuron_to_id[_neuron11] - network2._min_index[tb2]
                    _query2w = query2w[_id2, _id22]
                    _key2w = key2w[_id2, _id22]
                    _value2w = value2w[_id2, _id22]

                    assert jnp.isclose(_query1w, _query2w)
                    assert jnp.isclose(_key1w, _key2w)
                    assert jnp.isclose(_value1w, _value2w)

        if network1._use_topo_norm:
            topo1 = network1._topo_norm_params[tb1]
            topo2 = network2._topo_norm_params[tb2]

            for id1 in tb1_inputs:
                _neuron1 = network1._topo_sort[id1]
                _id1 = id1 - network1._min_index[tb1]
                gamma1, beta1 = topo1[_id1][:]

                id2 = network2._neuron_to_id[_neuron1]
                if id2 not in tb2_inputs:
                    continue
                _id2 = id2 - network2._min_index[tb2]
                gamma2, beta2 = topo2[_id2][:]

                assert jnp.isclose(gamma1, gamma2)
                assert jnp.isclose(beta1, beta2)

        if network1._use_adaptive_activations:
            ada1 = network1._adaptive_activation_params[tb1]
            ada2 = network2._adaptive_activation_params[tb2]

            a1, b1 = ada1[idx1][:]
            a2, b2 = ada2[idx2][:]

            assert jnp.isclose(a1, a2)
            assert jnp.isclose(b1, b2)


def test_add_connections_valid():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
    old_network = NeuralNetwork(
        graph,
        [0, 1],
        [4],
        use_topo_norm=True,
        use_adaptive_activations=True,
        use_topo_self_attention=True,
        use_neuron_self_attention=True,
    )

    new_connections = {0: [4], 1: [3]}
    new_network = add_connections(old_network, new_connections)
    _parameter_equivalence(old_network, new_network)

    new_connections = {3: [2]}
    new_network = add_connections(old_network, new_connections)
    _parameter_equivalence(old_network, new_network)

    new_connections = {0: [4], 1: [3], 3: [2]}
    new_network = add_connections(old_network, new_connections)
    _parameter_equivalence(old_network, new_network)


def test_remove_connections_valid():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (0, 3), (1, 2), (2, 4), (3, 4), (0, 4), (1, 3)])
    old_network = NeuralNetwork(
        graph,
        [0, 1],
        [4],
        use_topo_norm=True,
        use_adaptive_activations=True,
        use_topo_self_attention=True,
        use_neuron_self_attention=True,
    )

    connections = {0: [4], 1: [3]}
    new_network = remove_connections(old_network, connections)

    _parameter_equivalence(new_network, old_network)


def test_add_neurons_invalid():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (0, 3), (1, 2), (2, 4), (3, 4), (0, 4), (1, 3)])
    old_network = NeuralNetwork(
        graph,
        [0, 1],
        [4],
        use_topo_norm=True,
        use_adaptive_activations=True,
        use_topo_self_attention=True,
        use_neuron_self_attention=True,
    )
    with pytest.raises(ValueError, match="already exists"):
        add_hidden_neurons(old_network, [4])

    with pytest.raises(ValueError, match="already exists"):
        add_input_neurons(old_network, [4])

    with pytest.raises(ValueError, match="already exists"):
        add_output_neurons(old_network, [4])


def test_add_neurons_valid():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (0, 3), (1, 2), (2, 4), (3, 4), (0, 4), (1, 3)])
    old_network = NeuralNetwork(
        graph,
        [0, 1],
        [4],
        use_topo_norm=True,
        use_adaptive_activations=True,
        use_topo_self_attention=True,
        use_neuron_self_attention=True,
    )
    new_network = add_hidden_neurons(old_network, [6])
    _parameter_equivalence(old_network, new_network)
    assert 6 in new_network._graph.nodes()
    assert 6 in new_network._topo_sort

    new_network = add_input_neurons(old_network, [6])
    _parameter_equivalence(old_network, new_network)
    assert 6 in new_network._graph.nodes()
    assert 6 in new_network._topo_sort
    assert 6 in new_network._input_neurons

    new_network = add_output_neurons(old_network, [6])
    _parameter_equivalence(old_network, new_network)
    assert 6 in new_network._graph.nodes()
    assert 6 in new_network._topo_sort
    assert 6 in new_network._output_neurons


def test_remove_neurons():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (4, 1), (0, 2), (0, 3), (1, 2), (1, 5)])
    old_network = NeuralNetwork(
        graph,
        [0, 4],
        [2, 5],
        use_topo_norm=True,
        use_adaptive_activations=True,
        use_topo_self_attention=True,
        use_neuron_self_attention=True,
    )

    with pytest.raises(ValueError, match="does not exist"):
        remove_neurons(old_network, [8])

    new_network = remove_neurons(old_network, [0])
    assert 0 not in new_network._input_neurons
    _parameter_equivalence(new_network, old_network)

    new_network = remove_neurons(old_network, [5])
    assert 5 not in new_network._output_neurons
    _parameter_equivalence(new_network, old_network)

    new_network = remove_neurons(old_network, [1, 3])
    _parameter_equivalence(new_network, old_network)
