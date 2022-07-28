import typing
from typing import Mapping, Sequence
import jax.nn as jnn
import networkx as nx


# Documentation helpers.

def _identity(x):
    return x


if getattr(typing, "GENERATING_DOCUMENTATION", False):

    def silu(_):
        pass

    jnn.silu = silu
    _identity.__qualname__ = "identity"

###############################################################


def _invert_dict(
    _dict: Mapping[int, Sequence[int]]
) -> Mapping[int, Sequence[int]]:
    _dict_inv = {}
    for (key, vals) in _dict.items():
        for val in vals:
            if val in _dict_inv:
                _dict_inv[val].append(key)
            else:
                _dict_inv[val] = [key]
    for n in range(len(_dict)):
        if n not in _dict_inv:
            _dict_inv[n] = []
    return _dict_inv


def _nx_digraph_to_adjacency_dict(
    graph: nx.DiGraph
) -> Mapping[int, Sequence[int]]:
    assert isinstance(graph, nx.DiGraph)