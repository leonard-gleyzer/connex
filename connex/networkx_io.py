import networkx as nx

from .network import NeuralNetwork


def to_networkx_graph(network: NeuralNetwork) -> nx.Graph:
    """Get the equivalent `networkx.Graph` from a connex.NeuralNetwork`.

    **Arguments**:

    - `network`: A `NeuralNetwork` object.

    **Returns**:

    The equivalent `network.Graph` object, where "equivalent" here means that
    """
    assert isinstance(network, NeuralNetwork)


def from_networkx_graph(graph: nx.Graph) -> NeuralNetwork:
    """Get the equivalent `connex.NeuralNetwork` from a `networkx.Graph`.

    **Arguments**:

    - `graph`: A `networkx.Graph` object.

    **Returns**:

    The equivalent `connex.NeuralNetwork` object, where "equivalent" here means
    that
    """
    assert isinstance(graph, nx.Graph)