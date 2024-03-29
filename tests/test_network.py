import itertools as it
from copy import deepcopy

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np
import pytest

from connex import NeuralNetwork


def test_simple_dag():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    graph_copy = deepcopy(graph)
    input_neurons = [0]
    output_neurons = [4]

    nn = NeuralNetwork(graph, input_neurons, output_neurons)

    assert nn._input_neurons == input_neurons
    assert nn._output_neurons == output_neurons
    assert nn._hidden_neurons == [1, 2, 3]
    assert nn._graph.graph == graph_copy.graph
    assert nn._graph.nodes == graph_copy.nodes
    assert nn._graph.edges == graph_copy.edges
    assert nn._graph.adj == graph_copy.adj


def test_cycle_raises_value_error():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 1)])

    with pytest.raises(ValueError, match=r"`graph` contains cycles"):
        NeuralNetwork(graph, [0], [4])


def test_non_disjoint_input_output_neurons():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

    with pytest.raises(ValueError, match=r"appear in both input and output neurons"):
        NeuralNetwork(graph, [0, 1], [2, 1])


def test_invalid_topo_sort():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    with pytest.raises(ValueError, match=r"Invalid `topo_sort` at neuron"):
        NeuralNetwork(graph, [0], [4], topo_sort=[1, 0, 2, 3, 4])


def test_graph_unchanged():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    graph_copy = nx.DiGraph()
    graph_copy.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    nn = NeuralNetwork(graph, [0], [4])

    assert list(nn._graph.nodes) == list(graph_copy.nodes)
    assert list(nn._graph.edges) == list(graph_copy.edges)


def test_input_neurons_with_inputs():
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 0), (0, 2), (2, 3), (3, 4)])

    with pytest.raises(ValueError, match=r"Input neuron"):
        NeuralNetwork(graph, [0], [4])


def test_output_neurons_with_outputs():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    with pytest.raises(ValueError, match=r"Output neuron"):
        NeuralNetwork(graph, [0], [3])


def test_topo_batches():
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("A", "C"),
            ("B", "C"),
            ("B", "D"),
            ("A", "D"),
            ("C", "E"),
            ("C", "F"),
            ("D", "F"),
            ("E", "G"),
            ("F", "G"),
            ("G", "H"),
            ("G", "I"),
        ]
    )
    input_neurons = ["A", "B"]
    output_neurons = ["H", "I"]

    nn = NeuralNetwork(graph, input_neurons, output_neurons)

    expected_topo_batches = [
        jnp.array([2, 3], dtype=int),
        jnp.array([4, 5], dtype=int),
        jnp.array([6], dtype=int),
        jnp.array([7, 8], dtype=int),
    ]
    for tb1, tb2 in zip(nn._topo_batches, expected_topo_batches):
        assert jnp.array_equal(tb1, tb2)


def test_neuron_to_topo_batch_idx():
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("A", "C"),
            ("B", "C"),
            ("B", "D"),
            ("C", "E"),
            ("C", "F"),
            ("D", "F"),
            ("E", "G"),
            ("F", "G"),
            ("G", "H"),
            ("G", "I"),
        ]
    )
    input_neurons = ["A", "B"]
    output_neurons = ["H", "I"]

    nn = NeuralNetwork(graph, input_neurons, output_neurons)

    expected_neuron_to_topo_batch_idx = {
        2: (0, 0),
        3: (0, 1),
        4: (1, 0),
        5: (1, 1),
        6: (2, 0),
        7: (3, 0),
        8: (3, 1),
    }

    assert nn._neuron_to_topo_batch_idx == expected_neuron_to_topo_batch_idx


def test_valid_activations():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    NeuralNetwork(
        graph, [0], [4], hidden_activation=jnp.tanh, output_transformation=jnp.exp
    )


def test_invalid_hidden_activation():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    def invalid_hidden_activation(x):
        return jnp.array([1, 2, 3])

    with pytest.raises(ValueError, match=r"Activation function output must have shape"):
        NeuralNetwork(
            graph,
            [0],
            [4],
            hidden_activation=invalid_hidden_activation,
            output_transformation=jnp.exp,
        )


def test_invalid_output_transformation():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    def invalid_output_transformation(x):
        return jnp.array([1, 2, 3])

    with pytest.raises(ValueError, match=r"Activation function output must have shape"):
        NeuralNetwork(
            graph, [0], [4], output_transformation=invalid_output_transformation
        )


def test_hidden_activation_with_exception():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    def hidden_activation_with_exception(x):
        raise ValueError("Something went wrong!")

    with pytest.raises(
        Exception, match=r"Exception caught when checking activation function"
    ):
        NeuralNetwork(
            graph, [0], [4], hidden_activation=hidden_activation_with_exception
        )


def test_output_transformation_with_exception():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    def output_transformation_with_exception(x):
        raise ValueError("Something went wrong!")

    with pytest.raises(
        Exception, match=r"Exception caught when checking activation function"
    ):
        NeuralNetwork(
            graph,
            [0],
            [4],
            output_transformation=output_transformation_with_exception,
        )


def test_set_parameters_default():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    nn = NeuralNetwork(graph, [0], [4])

    assert len(nn._weights_and_biases) == len(nn._topo_batches)
    for wb, tb, tl in zip(nn._weights_and_biases, nn._topo_batches, nn._topo_lengths):
        assert wb.shape == (len(tb), tl + 1)
    assert len(nn._masks) == len(nn._topo_batches)

    for mask, tb, tl, mi in zip(
        nn._masks, nn._topo_batches, nn._topo_lengths, nn._min_index
    ):
        assert mask.shape == (len(tb), tl)
        for k, i in enumerate(map(int, tb)):
            neuron_inputs = nn._adjacency_dict_inv[nn._topo_sort[i]]
            neuron_inputs = [nn._neuron_to_id[n] for n in neuron_inputs]
            neuron_inputs = np.array(neuron_inputs, dtype=int) - mi
            for ni in neuron_inputs:
                assert mask[k, ni] == 1
            neuron_non_inputs = np.setdiff1d(jnp.arange(tl), neuron_inputs)
            for nni in neuron_non_inputs:
                assert mask[k, nni] == 0

    assert nn._use_topo_norm is False
    assert nn._use_topo_self_attention is False
    assert nn._use_neuron_self_attention is False
    assert nn._use_adaptive_activations is False


def test_set_parameters_topo_norm():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    nn = NeuralNetwork(graph, [0], [4], use_topo_norm=True)

    assert nn._use_topo_norm is True
    assert len(nn._topo_norm_params) == len(nn._topo_batches)
    for tn_params, tb in zip(nn._topo_norm_params, nn._topo_batches):
        assert tn_params.shape == (len(tb), 2)


def test_set_parameters_topo_self_attention():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (1, 4), (3, 4)])
    nn = NeuralNetwork(graph, [0], [4], use_topo_self_attention=True)

    assert nn._use_topo_self_attention is True
    assert len(nn._attention_params_neuron) == len(nn._topo_batches)
    for att_params, tl in zip(nn._attention_params_topo, nn._topo_lengths):
        assert att_params.shape == (3, tl, tl + 1)


def test_set_parameters_neuron_self_attention():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (1, 4), (3, 4)])
    nn = NeuralNetwork(graph, [0], [4], use_neuron_self_attention=True)

    assert nn._use_neuron_self_attention is True
    assert len(nn._attention_params_neuron) == len(nn._topo_batches)
    for att_params, tb, tl in zip(
        nn._attention_params_neuron, nn._topo_batches, nn._topo_lengths
    ):
        assert att_params.shape == (len(tb), 3, tl, tl + 1)
    assert len(nn._attention_masks_neuron) == len(nn._topo_batches)
    for att_mask, tb, tl, mi in zip(
        nn._attention_masks_neuron,
        nn._topo_batches,
        nn._topo_lengths,
        nn._min_index,
    ):
        assert att_mask.shape == (len(tb), tl, tl)
        for k, i in enumerate(map(int, tb)):
            neuron_inputs = nn._adjacency_dict_inv[nn._topo_sort[i]]
            neuron_inputs = [nn._neuron_to_id[n] for n in neuron_inputs]
            neuron_inputs = np.array(neuron_inputs, dtype=int) - mi
            for ni1, ni2 in it.product(neuron_inputs, neuron_inputs):
                assert att_mask[k, ni1, ni2] != jnp.inf
            neuron_non_inputs = np.setdiff1d(jnp.arange(tl), neuron_inputs)
            for nni in neuron_non_inputs:
                assert jnp.all(att_mask[k, nni, :] == jnp.inf) and jnp.all(
                    att_mask[k, :, nni] == jnp.inf
                )


def test_set_parameters_adaptive_activations():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    nn = NeuralNetwork(graph, [0], [4], use_adaptive_activations=True)

    assert nn._use_adaptive_activations is True
    assert len(nn._adaptive_activation_params) == len(nn._topo_batches)
    for aa_params, tb in zip(nn._adaptive_activation_params, nn._topo_batches):
        assert aa_params.shape == (len(tb), 2)


def test_set_dropout_p_initial():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    input_neurons = [0]
    output_neurons = [4]

    # Test float dropout probability
    nn = NeuralNetwork(graph, input_neurons, output_neurons, dropout_p=0.5)
    expected_dropout_array = jnp.array([0.0, 0.5, 0.5, 0.5, 0.0], dtype=float)
    assert jnp.allclose(nn._dropout_array, expected_dropout_array)

    # Test mapping dropout probability
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    nn = NeuralNetwork(
        graph, input_neurons, output_neurons, dropout_p={1: 0.1, 2: 0.2, 3: 0.3}
    )
    expected_dropout_array = jnp.array([0.0, 0.1, 0.2, 0.3, 0.0], dtype=float)
    assert jnp.allclose(nn._dropout_array, expected_dropout_array)

    # Test invalid dropout probabilities
    with pytest.raises(ValueError):
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        NeuralNetwork(graph, input_neurons, output_neurons, dropout_p={"1": 0.0})

    with pytest.raises(TypeError):
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        NeuralNetwork(graph, input_neurons, output_neurons, dropout_p={1: "invalid"})


def test_set_dropout_p_single_float():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (1, 2), (2, 3), (2, 4)])
    input_neurons = [0, 1]
    output_neurons = [3, 4]
    network = NeuralNetwork(graph, input_neurons, output_neurons, dropout_p=0.5)

    for neuron in network._hidden_neurons:
        assert network._dropout_dict[neuron] == 0.5

    for neuron in network._input_neurons + network._output_neurons:
        assert network._dropout_dict[neuron] == 0.0


def test_set_dropout_p_mapping():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (1, 2), (2, 3), (2, 4)])
    input_neurons = [0, 1]
    output_neurons = [3, 4]
    dropout_mapping = {2: 0.4, 3: 0.6}
    network = NeuralNetwork(
        graph, input_neurons, output_neurons, dropout_p=dropout_mapping
    )

    for neuron, dropout_prob in dropout_mapping.items():
        assert network._dropout_dict[neuron] == dropout_prob


def test_to_networkx_weighted_digraph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (1, 2), (2, 3), (2, 4)])
    input_neurons = [0, 1]
    output_neurons = [3, 4]
    network = NeuralNetwork(graph, input_neurons, output_neurons)

    digraph = network.to_networkx_weighted_digraph()

    assert isinstance(digraph, nx.DiGraph)
    assert digraph.number_of_nodes() == network._graph.number_of_nodes()
    assert digraph.number_of_edges() == network._graph.number_of_edges()

    for neuron, inputs in network._adjacency_dict_inv.items():
        if neuron in network._input_neurons:
            continue
        topo_batch_idx, pos_idx = network._neuron_to_topo_batch_idx[neuron]
        for input in inputs:
            assert digraph.has_edge(input, neuron)
            assert "weight" in digraph[input][neuron]
            weight = digraph[input][neuron]["weight"]
            col_idx = network._neuron_to_id[input] - network._min_index[topo_batch_idx]
            assert network._masks[topo_batch_idx][pos_idx, col_idx]
            weight_net = network._weights_and_biases[topo_batch_idx][pos_idx, col_idx]
            assert weight == weight_net


def test_call_forward_pass():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    input_neurons = [0]
    output_neurons = [3]
    network = NeuralNetwork(graph, input_neurons, output_neurons)

    input_array = jnp.array([1.0])
    output_array = network(input_array, key=jr.PRNGKey(0))

    assert output_array.shape == (1,)


def test_call_dropout():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    input_neurons = [0]
    output_neurons = [3]
    network = NeuralNetwork(graph, input_neurons, output_neurons, dropout_p={3: 1.0})

    input_array = jnp.array([1.0])
    output_array = network(input_array, key=jr.PRNGKey(0))

    assert jnp.all(output_array == 0.0)


def test_apply_topo_norm():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    input_neurons = [0]
    output_neurons = [3]
    network = NeuralNetwork(graph, input_neurons, output_neurons, use_topo_norm=True)

    norm_params = jnp.array([[1.0, 0.0]])
    vals = jnp.array([1.0])

    normalized_vals = network._apply_topo_norm(norm_params, vals)
    expected_normalized_vals = jnp.array([1.0])

    assert jnp.allclose(normalized_vals, expected_normalized_vals)
    norm_params = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    vals = jnp.array([1.0, 2.0])

    normalized_vals = network._apply_topo_norm(norm_params, vals)
    expected_normalized_vals = jnp.array([-1.0, 1.0])

    assert jnp.allclose(normalized_vals, expected_normalized_vals, rtol=1e-4)


def test_apply_topo_self_attention():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    input_neurons = [0]
    output_neurons = [3]
    network = NeuralNetwork(
        graph, input_neurons, output_neurons, use_topo_self_attention=True
    )

    attn_params = jnp.array(
        [[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]],
        dtype=float,
    )
    vals = jnp.array([1.0, 2.0])

    attn_output = network._apply_topo_self_attention(attn_params, vals)
    expected_attn_output = (
        jnn.softmax(jnp.array([[1.0, 2.0], [2.0, 4.0]]) / jnp.sqrt(2.0)) @ vals + vals
    )

    assert jnp.allclose(attn_output, expected_attn_output)


def test_apply_neuron_self_attention():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    input_neurons = [0]
    output_neurons = [3]
    network = NeuralNetwork(
        graph, input_neurons, output_neurons, use_neuron_self_attention=True
    )

    attn_params = jnp.array(
        [[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]],
        dtype=float,
    )

    attn_mask = jnp.zeros((2, 2))
    attn_mask = attn_mask.at[1, 1].set(jnp.inf)
    vals = jnp.array([1.0, 2.0])
    id = jnp.array(1)
    attn_output = network._apply_neuron_self_attention(id, attn_params, attn_mask, vals)
    expected_attn_output = (
        jnn.softmax(jnp.array([[1.0, 2.0], [2.0, -jnp.inf]])) @ vals + vals
    )

    assert jnp.allclose(attn_output, expected_attn_output)


def test_apply_activation():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    input_neurons = [0]
    output_neurons = [3]
    network = NeuralNetwork(
        graph, input_neurons, output_neurons, use_adaptive_activations=True
    )

    id = jnp.array(1)
    affine = jnp.array([1.0])
    ada_params = jnp.array([1.0, 1.0])

    activated_output = network._apply_activation(id, affine, ada_params)
    expected_activated_output = jnn.gelu(jnp.array([1.0]))

    assert jnp.allclose(activated_output, expected_activated_output)
