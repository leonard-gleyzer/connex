import typing
from typing import Any, Dict, List

import jax.nn as jnn


# Documentation helpers.


def _identity(x):
    return x


if getattr(typing, "GENERATING_DOCUMENTATION", False):

    def silu(_):
        pass

    jnn.silu = silu
    _identity.__qualname__ = "identity"

###############################################################


def _invert_dict(_dict: Dict[Any, List[Any]]) -> Dict[Any, List[Any]]:
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
