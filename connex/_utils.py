import typing

import jax.nn as jnn


# Documentation helpers.


def _identity(x):
    return x


if getattr(typing, "GENERATING_DOCUMENTATION", False):

    def gelu(_):
        pass

    jnn.gelu = gelu
    _identity.__qualname__ = "identity"
