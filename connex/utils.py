import typing
from typing import Mapping, Sequence
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


# Documentation helpers.

def _identity(x):
    return x


if getattr(typing, "GENERATING_DOCUMENTATION", False):

    def silu(_):
        pass

    jnn.silu = silu
    _identity.__qualname__ = "identity"


# The following two functions are used to switch between adjacency dicts and
# adjacency matrices. The reason is that, internally, connex uses adjacency matrices.
# However, the end-user works with adjacency dicts, since they are cleaner to work
# with and reason about/debug.

def _adjacency_dict_to_matrix(
    num_neurons: int, 
    adjacency_dict: Mapping[int, Sequence[int]]
) -> jnp.array:
    adjacency_matrix = np.zeros((num_neurons, num_neurons))
    for input, outputs in adjacency_dict.items():
        if outputs:
            adjacency_matrix[input, outputs] = 1
    return jnp.array(adjacency_matrix, dtype=int)


def _adjacency_matrix_to_dict(adjacency_matrix: jnp.array
) -> Mapping[int, Sequence[int]]:
    num_neurons = adjacency_matrix.shape[0]
    adjacency_dict = {}
    for i in range(num_neurons):
        out = jnp.ravel(jnp.argwhere(adjacency_matrix[i])).tolist()
        if out:
            adjacency_dict[i] = out
    return adjacency_dict