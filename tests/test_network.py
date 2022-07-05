import connex as cnx
import equinox as eqx
from jax import vmap
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import optax

import pytest


def test_topological_batching():
    mlp = cnx.nn.MLP(1, 1, 2, 2)
    mcmlp = cnx.nn.MCMLP(1, 1, 2, 2)
    topo_batches = [[0], [1, 2], [3, 4], [5]]
    topo_batches = [jnp.array(tb) for tb in topo_batches]
    assert len(mlp.topo_batches) == 4
    assert len(mcmlp.topo_batches) == 4
    for i in range(4):
        assert jnp.array_equal(mlp.topo_batches[i], topo_batches[i])
        assert jnp.array_equal(mcmlp.topo_batches[i], topo_batches[i])
    adj_dict = {0: [1, 2], 1: [3, 4], 2: [3, 4], 3: [5], 4: [2, 5]}
    with pytest.raises(AssertionError):
        cnx.NeuralNetwork(6, adj_dict, [0], [2])
    

def test_training():
    cnx_mlp = cnx.nn.MLP(1, 1, 16, 1, jnn.relu)
    eqx_mlp = eqx.nn.MLP(1, 1, 16, 1, jnn.relu, key=jr.PRNGKey(0))

    cnx_optim = optax.adam(1e-3)
    cnx_opt_state = cnx_optim.init(eqx.filter(cnx_mlp, eqx.is_array))

    eqx_optim = optax.adam(1e-3)
    eqx_opt_state = eqx_optim.init(eqx.filter(eqx_mlp, eqx.is_array))

    def loss_fn(model, x, y):
        preds = vmap(model)(x)
        return jnp.mean((preds - y) ** 2)

    @eqx.filter_jit
    def step(model, optim, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    X = jnp.expand_dims(jnp.linspace(0, 2 * jnp.pi, 250), 1)
    y = 2 * X

    n_epochs = 5000

    for _ in range(n_epochs):
        cnx_mlp, cnx_opt_state, _ = step(cnx_mlp, cnx_optim, cnx_opt_state, X, y)
        eqx_mlp, eqx_opt_state, _ = step(eqx_mlp, eqx_optim, eqx_opt_state, X, y)

    cnx_y = vmap(cnx_mlp)(X)
    eqx_y = vmap(eqx_mlp)(X)
    max_abs_error = jnp.max(jnp.abs(jnp.squeeze(cnx_y - eqx_y)))
    assert max_abs_error < 1e-1

    
def test_nn_activation_functions():
    eqx_mlp = eqx.nn.MLP(1, 1, 16, 2, jnn.silu, key=jr.PRNGKey(0))
    cnx_mlp = cnx.nn.MLP(1, 1, 16, 2, eqx_mlp, eqx_mlp, eqx_mlp)

    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(cnx_mlp, eqx.is_array))

    def loss_fn(model, x, y):
        preds = vmap(model)(x)
        return jnp.mean((preds - y) ** 2)

    @eqx.filter_jit
    def step(model, optimizer, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    X = jnp.expand_dims(jnp.linspace(0, 2 * jnp.pi, 250), 1)
    y = jnp.sin(X)

    n_epochs = 10

    for _ in range(n_epochs):
        cnx_mlp, opt_state, _ = step(cnx_mlp, optim, opt_state, X, y)

    for i in range(len(eqx_mlp.layers)):
        assert not jnp.array_equal(
            eqx_mlp.layers[i].weight, 
            cnx_mlp.hidden_activation.layers[i].weight
        )
        assert not jnp.array_equal(
            eqx_mlp.layers[i].bias, 
            cnx_mlp.hidden_activation.layers[i].bias
        )
        assert not jnp.array_equal(
            eqx_mlp.layers[i].weight, 
            cnx_mlp.output_activation_elem.layers[i].weight
        )
        assert not jnp.array_equal(
            eqx_mlp.layers[i].bias, 
            cnx_mlp.output_activation_elem.layers[i].bias
        )
        assert not jnp.array_equal(
            cnx_mlp.hidden_activation.layers[i].weight,
            cnx_mlp.output_activation_elem.layers[i].weight
        )
        assert not jnp.array_equal(
            cnx_mlp.hidden_activation.layers[i].bias,
            cnx_mlp.output_activation_elem.layers[i].bias
        )
        assert not jnp.array_equal(
            eqx_mlp.layers[i].weight, 
            cnx_mlp.output_activation_group.layers[i].weight
        )
        assert not jnp.array_equal(
            eqx_mlp.layers[i].bias, 
            cnx_mlp.output_activation_group.layers[i].bias
        )
        assert not jnp.array_equal(
            cnx_mlp.hidden_activation.layers[i].weight,
            cnx_mlp.output_activation_group.layers[i].weight
        )
        assert not jnp.array_equal(
            cnx_mlp.hidden_activation.layers[i].bias,
            cnx_mlp.output_activation_group.layers[i].bias
        )


def test_dropout():
    mlp = cnx.nn.MLP(1, 1, 4, 2)
    mlp.set_dropout_p(0.5)
    arr = jnp.ones((mlp.num_neurons,)) * 0.5
    arr = arr.at[jnp.array([0, 9])].set(0.)
    assert jnp.array_equal(mlp.get_dropout_p(), arr)
    zeros = jnp.zeros((mlp.num_neurons,))
    mlp.set_dropout_p(zeros)
    assert jnp.array_equal(mlp.get_dropout_p(), zeros)


def test_networkx():
    mlp = cnx.nn.MLP(1, 1, 2, 1)
    mlp_digraph = mlp.to_networkx_graph()
    nodes = list(mlp_digraph.nodes.data())
    true_nodes = [
        (0, {'bias': None, 'group': 'input'}),
        (1, {'bias': mlp.parameter_matrix[1, -1], 'group': 'hidden'}),
        (2, {'bias': mlp.parameter_matrix[2, -1], 'group': 'hidden'}),
        (3, {'bias': mlp.parameter_matrix[3, -1], 'group': 'output'}),
    ]
    assert eqx.tree_equal(nodes, true_nodes)
    edges = list(mlp_digraph.edges.data())
    true_edges = [
        (0, 1, {'weight': mlp.parameter_matrix[1, 0]}),
        (0, 2, {'weight': mlp.parameter_matrix[2, 0]}),
        (1, 3, {'weight': mlp.parameter_matrix[3, 1]}),
        (2, 3, {'weight': mlp.parameter_matrix[3, 2]}),
    ]
    assert eqx.tree_equal(edges, true_edges)