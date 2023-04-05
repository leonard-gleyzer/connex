# from . import nn
from .network import NeuralNetwork
from .plasticity import (
    _get_id_mappings_old_new,
    add_connections,
    add_hidden_neurons,
    add_input_neurons,
    add_output_neurons,
    remove_connections,
    remove_neurons,
)


__version__ = "0.1.4"
