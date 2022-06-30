import jax.numpy as jnp
from networkx import DiGraph
from typing import Callable, Optional, Tuple

from .network import NeuralNetwork


def _get_group_and_activation(
    network: NeuralNetwork, 
    id: int
) -> Tuple[str, Optional[Callable]]:
    assert 0 <= id < network.num_neurons
    
    if jnp.isin(id, network.input_neurons):
        group = 'input'
        activation = None
    elif jnp.isin(id, network.output_neurons):
        group = 'output'
        activation = network.output_activation
    else:
        group = 'hidden'
        activation = network.activation

    return group, activation


def to_networkx_graph(network: NeuralNetwork) -> DiGraph:
    """Get the equivalent `networkx.DiGraph` from a connex.NeuralNetwork`.

    **Arguments**:

    - `network`: A `NeuralNetwork` object.

    **Returns**:

    The equivalent `network.DiGraph` object, where "equivalent" here means that
    """
    assert isinstance(network, NeuralNetwork)

    graph = DiGraph(
        activation=network.activation, 
        output_activation=network.output_activation,
        input_neurons=network.input_neurons,
        output_neurons=network.output_neurons,
        dropout_p=network.get_dropout_p(),
        key=network.key
    )

    for id in range(network.num_neurons):
        neuron_bias = network.parameter_matrix[id, -1]
        group, activation = _get_group_and_activation(network, id)
        dropout_p = network.get_dropout_p()
        graph.add_node(
            id,
            group=group, 
            bias=neuron_bias, 
            activation=activation,
            dropout_p=dropout_p[id]
        )
        
    for (neuron, outputs) in network.adjacency_dict.items():
        for output in outputs:
            weight = network.parameter_matrix[output, neuron]
            graph.add_edge(neuron, output, weight=weight)

    return graph
    

def from_networkx_graph(graph: DiGraph) -> NeuralNetwork:
    """Get the equivalent `connex.NeuralNetwork` from a `networkx.DiGraph`.

    **Arguments**:

    - `graph`: A `networkx.DiGraph` object.

    **Returns**:

    The equivalent `connex.NeuralNetwork` object, where "equivalent" here means
    that
    """
    assert isinstance(graph, DiGraph)