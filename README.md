<h1 align='center'>Connex</h1>


Connex is a small [JAX](https://github.com/google/jax) library built on [Equinox](https://github.com/patrick-kidger/equinox) whose aim is to incorporate artificial analogues of biological neural network attributes into deep learning research and architecture design. Currently, this includes:

- **Complex Connectivity**: Turn any directed acyclic graph (DAG) into a trainable neural network.
- **Plasticity**: Add and remove both connections and neurons at the individual level.
- **Firing Modulation**: Set and modify dropout probabilities for all neurons individually.

## Installation

```bash
pip install connex
```

## Documentation

Available at [https://leonard-gleyzer.github.io/connex/](https://leonard-gleyzer.github.io/connex/).

## Usage

As a small example, let's create a trainable neural network from the following DAG 

![dag](https://www.mdpi.com/algorithms/algorithms-13-00256/article_deploy/html/images/algorithms-13-00256-g001.png)

with input neuron 0 and output neurons 3 and 11 (in that order), with a ReLU activation function for the hidden neurons:

```python
import connex as cnx
import jax.nn as jnn

# Specify number of neurons
num_neurons = 12

# Build the adjacency dict
adjacency_dict = {
    0: [1, 2, 3],
    1: [4],
    2: [4, 5],
    4: [6],
    5: [7],
    6: [8, 9],
    7: [10],
    8: [11],
    9: [11],
    10: [11]
}

# Specify the input and output neurons
input_neurons = [0]
output_neurons = [3, 11]

# Create the network
network = cnx.NeuralNetwork(
    num_neurons,
    adjacency_dict, 
    input_neurons, 
    output_neurons,
    jnn.relu
)
```

That's it! A `connex.NeuralNetwork` is a subclass of `equinox.Module`, so it can be trained in the same fashion:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

# Initialize the optimizer
optim = optax.adam(1e-3)
opt_state = optim.init(eqx.filter(network, eqx.is_array))

# Define the loss function
@eqx.filter_value_and_grad
def loss_fn(model, x, y):
    preds = jax.vmap(model)(x)
    return jnp.mean((preds - y) ** 2)

# Define a single training step
@eqx.filter_jit
def step(model, opt_state, x, y):
    loss, grads = loss_fn(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# Toy data
x = jnp.expand_dims(jnp.linspace(0, 2 * jnp.pi, 250), 1)
y = jnp.hstack((jnp.cos(x), jnp.sin(x)))

# Training loop
n_epochs = 1000
for _ in range(n_epochs):
    network, opt_state, loss = step(network, opt_state, x, y)
```

Now suppose we wish to add connections 1 &rarr; 6 and 2 &rarr; 11, remove neuron 9, and set the dropout probability of all hidden neurons to 0.1:

```python
# Add connections
network = cnx.add_connections(network, [(1, 6), (2, 11)])

# Remove neuron
network, _ = cnx.remove_neurons(network, [9])

# Set dropout probability
network.set_dropout_p(0.1)
```

That's all there is to it.  The new connections have been initialized with untrained parameters, and the neurons in the original network that have not been removed (along with their respective incoming and outgoing connections) have retained their trained parameters. Furthermore, since a `connex.NeuralNetwork` is an `equinox.Module`, it can seamlessly be used as a submodule inside other Equinox Modules.

For more information about manipulating connectivity structure and the `NeuralNetwork` base class, please see the API section of the documentation. For examples of subclassing `NeuralNetwork`, please see `connex.nn`.

Feedback is greatly appeciated!