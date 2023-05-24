import importlib.metadata

from . import nn as nn
from ._network import NeuralNetwork as NeuralNetwork
from ._plasticity import (
    add_connections,
    add_hidden_neurons,
    add_input_neurons,
    add_output_neurons,
    remove_connections,
    remove_neurons,
)


__version__ = importlib.metadata.version("connex")
