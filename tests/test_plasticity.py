import connex as cnx
import jax.numpy as jnp
import numpy as np

import pytest


def test_add_connections():
    mlp = cnx.nn.MLP(1, 1, 2, 2)
    adj_mat = mlp.adjacency_matrix
    adj_copy = jnp.copy(adj_mat)
    new_connections = [(0, 3), (1, 5)]
    net = cnx.add_connections(mlp, new_connections)
    adj_mat = adj_mat.at[0, 3].set(1)
    adj_mat = adj_mat.at[1, 5].set(1)
    assert jnp.array_equal(adj_mat, net.adjacency_matrix)
    assert jnp.array_equal(mlp.parameter_matrix, net.parameter_matrix)
    assert jnp.array_equal(mlp.adjacency_matrix, adj_copy)
    assert len(net.topo_batches) == 4
    topo_batches = [[0], [1, 2], [3, 4], [5]]
    topo_batches = [jnp.array(tb) for tb in topo_batches]
    for i in range(len(net.topo_batches)):
        assert jnp.array_equal(topo_batches[i], net.topo_batches[i])


def test_remove_connections():
    mlp = cnx.nn.MLP(1, 1, 2, 2)
    new_connections = [(0, 3), (1, 5)]
    net = cnx.add_connections(mlp, new_connections)
    net = cnx.remove_connections(net, new_connections)
    assert jnp.array_equal(mlp.adjacency_matrix, net.adjacency_matrix)


def test_add_neurons():
    mlp = cnx.nn.MLP(1, 1, 2, 2)
    new_neuron_data = []
    new_neuron_data.append(
        {'in_neurons': None, 'out_neurons': [3, 5], 'type': 'input', 'dropout_p': 0.}
    )
    new_neuron_data.append(
        {'in_neurons': [0], 'out_neurons': [1, 2], 'type': 'hidden', 'dropout_p': None}
    )
    new_neuron_data.append(
        {'in_neurons': [1], 'out_neurons': None, 'type': 'output', 'dropout_p': 1}
    )
    new_net, new_ids = cnx.add_neurons(mlp, new_neuron_data)
    assert jnp.array_equal(new_net.input_neurons, jnp.append(mlp.input_neurons, 6))
    assert jnp.array_equal(new_net.output_neurons, jnp.append(mlp.output_neurons, 8))
    assert new_ids == [6, 7, 8]
    adj_mat = jnp.zeros((9, 9))
    adj_mat = adj_mat.at[:6, :6].set(mlp.adjacency_matrix)
    adj_mat = adj_mat.at[6, 3].set(1)
    adj_mat = adj_mat.at[6, 5].set(1)
    adj_mat = adj_mat.at[0, 7].set(1)
    adj_mat = adj_mat.at[7, 1].set(1)
    adj_mat = adj_mat.at[7, 2].set(1)
    adj_mat = adj_mat.at[1, 8].set(1)
    assert jnp.array_equal(new_net.adjacency_matrix, adj_mat)
    assert jnp.array_equal(mlp.parameter_matrix[:, :-1], new_net.parameter_matrix[:6, :6])
    assert jnp.array_equal(mlp.parameter_matrix[:, -1], new_net.parameter_matrix[:6, -1])
    topo_batches = [[0, 6], [7], [1, 2], [8], [3, 4], [5]]
    topo_batches = [jnp.array(tb) for tb in topo_batches]
    assert len(new_net.topo_batches) == 6
    for i in range(6):
        assert jnp.array_equal(topo_batches[i], new_net.topo_batches[i])
        

def test_remove_neurons():
    mlp = cnx.nn.MLP(1, 1, 2, 2)
    new_neuron_data = []
    new_neuron_data.append(
        {'in_neurons': None, 'out_neurons': [3, 5], 'type': 'input', 'dropout_p': 0.}
    )
    new_neuron_data.append(
        {'in_neurons': [0], 'out_neurons': [1, 2], 'type': 'hidden', 'dropout_p': None}
    )
    new_neuron_data.append(
        {'in_neurons': [1], 'out_neurons': None, 'type': 'output', 'dropout_p': 1}
    )
    net, _ = cnx.add_neurons(mlp, new_neuron_data)
    _remove_neurons = [2, 3, 4, 5, 6, 7]
    new_net, id_map = cnx.remove_neurons(net, _remove_neurons)
    assert id_map == {0: 0, 1: 1, 8: 2}
    adj_mat = np.zeros((3, 3))
    adj_mat[0, 1] = 1
    adj_mat[1, 2] = 1
    adj_mat = jnp.array(adj_mat, dtype=jnp.int32)
    assert jnp.array_equal(new_net.adjacency_matrix, adj_mat)
    assert new_net.input_neurons.tolist() == [0]
    assert new_net.output_neurons.tolist() == [2]
    _remove_neurons = jnp.array(_remove_neurons)
    arr = jnp.delete(net.parameter_matrix, _remove_neurons, 0)
    arr = jnp.delete(arr, _remove_neurons, 1)
    assert jnp.array_equal(new_net.parameter_matrix, arr)
        

def test_connect_networks():
    mlp1 = cnx.nn.MLP(1, 1, 2, 2)
    mlp2 = cnx.nn.MLP(1, 1, 2, 2)
    net, id_map = cnx.connect_networks(mlp1, mlp2, {0: 5}, {0: 5})
    assert id_map == {i: i + 6 for i in range(6)}
    adj_mat = np.zeros((12, 12))
    adj_mat[0, 1:3] = 1
    adj_mat[1:3, 3:5] = 1
    adj_mat[3:5, 5] = 1
    adj_mat[6, 7:9] = 1
    adj_mat[7:9, 9:11] = 1
    adj_mat[9:11, 11] = 1
    adj_mat[0, 11] = 1
    adj_mat[6, 5] = 1
    adj_mat = jnp.array(adj_mat, dtype=jnp.int32)
    assert jnp.array_equal(net.adjacency_matrix, adj_mat)
    assert net.input_neurons.tolist() == [0, 6]
    assert net.output_neurons.tolist() == [5, 11]
    pm = jnp.delete(net.parameter_matrix, jnp.arange(6, 12), 0)
    pm = jnp.delete(pm, jnp.arange(6, 12), 1)
    assert jnp.array_equal(mlp1.parameter_matrix, pm)
    pm = jnp.delete(net.parameter_matrix, jnp.arange(6), 0)
    pm = jnp.delete(pm, jnp.arange(6), 1)
    assert jnp.array_equal(mlp2.parameter_matrix, pm)