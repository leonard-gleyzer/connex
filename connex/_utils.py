import time
import typing
from collections import defaultdict
from typing import Any

import jax.nn as jnn
import jax.random as jr


# Documentation helpers.


def _identity(x):
    return x


if getattr(typing, "GENERATING_DOCUMENTATION", False):

    def gelu(_):
        pass

    jnn.gelu = gelu
    _identity.__qualname__ = "identity"

DiGraphLike = Any

###############################################################


def _invert_dict(_dict):
    _dict_inv = {}
    for (key, vals) in _dict.items():
        for val in vals:
            if val in _dict_inv:
                _dict_inv[val].append(key)
            else:
                _dict_inv[val] = [key]
    for k in _dict.keys():
        if k not in _dict_inv:
            _dict_inv[k] = []
    return _dict_inv


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper


def _edges_to_adjacency_dict(edges):
    adjacency_dict = defaultdict(list)
    for u, v in edges:
        adjacency_dict[u].append(v)
    return dict(adjacency_dict)


def _keygen():
    curr_time = time.time()
    curr_time = str(curr_time).replace(".", "")
    seed = int(curr_time)
    key = jr.PRNGKey(seed)
    return key
