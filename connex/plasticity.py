from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import equinox as eqx
import equinox.experimental as eqxe

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from .network import NeuralNetwork
from .utils import _adjacency_matrix_to_dict, _identity


def add_connections(
    network: NeuralNetwork,
    connections: Sequence[Tuple[int, int]],
    input_neurons: Optional[Sequence[int]] = None,
    output_neurons: Optional[Sequence[int]] = None,
    dropout_p:  Optional[Union[float, Sequence[float]]] = None,
) -> NeuralNetwork:
    """Add connections to the network.
    
    **Arguments**:

    - `network`: A `NeuralNetwork` object
    - `connections`: A sequence of pairs `(i, j)` to add to the network
        as directed edges.
    - `input_neurons`: A sequence of `int` indicating the ids of the 
        input neurons. The order here matters, as the input data will be
        passed into the input neurons in the order passed in here. Optional
        argument. If `None`, the input neurons of the original network will
        be retained.
    - `output_neurons`: A sequence of `int` indicating the ids of the 
        output neurons. The order here matters, as the output data will be
        read from the output neurons in the order passed in here. Optional 
        argument. If `None`, the output neurons of the original network will
        be retained.
    - `dropout_p`: Dropout probability. If a single `float`, the same dropout
        probability will be applied to all hidden neurons. If a `Sequence[float]`,
        the sequence must have length `num_neurons`, where `dropout_p[i]` is the
        dropout probability for neuron `i`. Note that this allows dropout to be 
        applied to input and output neurons as well. Optional argument. If `None`, 
        the dropout probabilities of the original network will be retained.

    **Returns**:

    A `NeuralNetwork` object with the input connections added and original
    parameters retained.
    """
    adjacency_matrix = jnp.copy(network.adjacency_matrix)
    for (from_neuron, to_neuron) in connections:
        assert 0 <= from_neuron < network.num_neurons
        assert 0 <= to_neuron < network.num_neurons
        assert from_neuron != to_neuron
        adjacency_matrix = adjacency_matrix.at[from_neuron, to_neuron].set(1)
    if input_neurons is None:
        input_neurons = network.input_neurons
    if output_neurons is None:
        output_neurons = network.output_neurons
    if dropout_p is None:
        dropout_p = network.get_dropout_p()

    adjacency_dict = _adjacency_matrix_to_dict(adjacency_matrix)
    hidden_activation = network.hidden_activation \
        if isinstance(network.hidden_activation, eqx.Module) \
        else network.hidden_activation_
    output_activation_elem = network.output_activation_elem \
        if isinstance(network.output_activation_elem, eqx.Module) \
        else network.output_activation_elem_

    return NeuralNetwork(
        network.num_neurons,
        adjacency_dict,
        input_neurons,
        output_neurons,
        hidden_activation,
        output_activation_elem,
        network.output_activation_group,
        dropout_p,
        key=eqxe.get_state(network.key, jr.PRNGKey(0)),
        parameter_matrix=network.parameter_matrix
    )


def remove_connections(
    network: NeuralNetwork,
    connections: Sequence[Tuple[int, int]],
    input_neurons: Optional[Sequence[int]] = None,
    output_neurons: Optional[Sequence[int]] = None,
    dropout_p:  Optional[Union[float, Sequence[float]]] = None,
) -> NeuralNetwork:
    """Remove connections from the network.
    
    **Arguments**:

    - `network`: A `NeuralNetwork` object
    - `connections`: A sequence of pairs `(i, j)` to remove from the network
        as directed edges.
    - `input_neurons`: A sequence of `int` indicating the ids of the 
        input neurons. The order here matters, as the input data will be
        passed into the input neurons in the order passed in here. Optional
        argument. If `None`, the input neurons of the original network will
        be retained.
    - `output_neurons`: A sequence of `int` indicating the ids of the 
        output neurons. The order here matters, as the output data will be
        read from the output neurons in the order passed in here. Optional 
        argument. If `None`, the output neurons of the original network will
        be retained.
    - `dropout_p`: Dropout probability. If a single `float`, the same dropout
        probability will be applied to all hidden neurons. If a `Sequence[float]`,
        the sequence must have length `num_neurons`, where `dropout_p[i]` is the
        dropout probability for neuron `i`. Note that this allows dropout to be 
        applied to input and output neurons as well. Optional argument. If `None`, 
        the dropout probabilities of the original network will be retained.

    **Returns**:

    A `NeuralNetwork` object with the desired connections removed and original
    parameters retained.
    """
    adjacency_matrix = jnp.copy(network.adjacency_matrix)
    for (from_neuron, to_neuron) in connections:
        assert 0 <= from_neuron < network.num_neurons
        assert 0 <= to_neuron < network.num_neurons
        assert from_neuron != to_neuron
        adjacency_matrix = adjacency_matrix.at[from_neuron, to_neuron].set(0)
    if input_neurons is None:
        input_neurons = network.input_neurons
    if output_neurons is None:
        output_neurons = network.output_neurons
    if dropout_p is None:
        dropout_p = network.get_dropout_p()

    adjacency_dict = _adjacency_matrix_to_dict(adjacency_matrix)
    hidden_activation = network.hidden_activation \
        if isinstance(network.hidden_activation, eqx.Module) \
        else network.hidden_activation_
    output_activation_elem = network.output_activation_elem \
        if isinstance(network.output_activation_elem, eqx.Module) \
        else network.output_activation_elem_

    return NeuralNetwork(
        network.num_neurons,
        adjacency_dict,
        input_neurons,
        output_neurons,
        hidden_activation,
        output_activation_elem,
        network.output_activation_group,
        dropout_p,
        key=eqxe.get_state(network.key, jr.PRNGKey(0)),
        parameter_matrix=network.parameter_matrix
    )


def add_neurons(
    network: NeuralNetwork,
    new_neuron_data: Sequence[Mapping],
) -> Tuple[NeuralNetwork, Sequence[int]]:
    """Add neurons to the network. These can be input, hidden, or output neurons.
    
    **Arguments**:
    
    - `network`: A `NeuralNetwork` object
    - `new_neuron_data`: A sequence of dictionaries, where each dictionary 
        represents a new neuron to add to the network. Each dictionary must 
        have 4 `str` fields:
        * `'in_neurons'`: An `Optional[Sequence[int]]` indexing the neurons from the 
            original network that feed into the new neuron.
        * `'out_neurons'`: An `Optional[Sequence[int]]` indexing the neurons from the 
            original network which the new neuron feeds into.
        * `'group'`: One of {`'input'`, `'hidden'`, `'output'`}. A `str` representing
            which group the new neuron belongs to.
        * `'dropout_p'`: An `Optional[float]`, the dropout probability for the new neuron. 
            Defaults to 0.

    **Returns**:

    A 2-tuple where the first element is the new `NeuralNetwork` with the new neurons
    added and parameters from original neurons retained, and the second element 
    is the sequence of the ids assigned to the added neurons in the order they 
    were passed in through the input argument `new_neuron_data`.
    """
    num_new_neurons = len(new_neuron_data)
    total_num_neurons = network.num_neurons + num_new_neurons
    adjacency_matrix = jnp.zeros((total_num_neurons, total_num_neurons))
    adjacency_matrix = adjacency_matrix \
        .at[:-num_new_neurons, :-num_new_neurons] \
        .set(network.adjacency_matrix)

    input_neurons = network.input_neurons
    output_neurons = network.output_neurons
    dropout_p = network.get_dropout_p()
    id = network.num_neurons

    for neuron_datum in new_neuron_data:
        in_neurons = neuron_datum['in_neurons']
        if in_neurons is not None:
            in_neurons = jnp.array(in_neurons, dtype=int)
            adjacency_matrix = adjacency_matrix.at[in_neurons, id].set(1)
        out_neurons = neuron_datum['out_neurons']
        if out_neurons is not None:
            out_neurons = jnp.array(out_neurons, dtype=int)
            adjacency_matrix = adjacency_matrix.at[id, out_neurons].set(1)

        group = neuron_datum['group']
        assert group in {'input', 'hidden', 'output'}
        if group == 'input':
            input_neurons = jnp.append(input_neurons, id)
        elif group == 'output':
            output_neurons = jnp.append(output_neurons, id)

        _dropout_p = neuron_datum['dropout_p']
        if _dropout_p is None:
            _dropout_p = 0.
        dropout_p = jnp.append(dropout_p, _dropout_p)
        
        id += 1

    key = eqxe.get_state(network.key, jr.PRNGKey(0))
    parameter_matrix = jr.normal(
        key, (total_num_neurons, total_num_neurons + 1)
    ) * 0.1
    parameter_matrix = parameter_matrix \
        .at[:network.num_neurons, :network.num_neurons] \
        .set(network.parameter_matrix[:, :-1])
    parameter_matrix = parameter_matrix \
        .at[:network.num_neurons, -1] \
        .set(network.parameter_matrix[:, -1])

    adjacency_dict = _adjacency_matrix_to_dict(adjacency_matrix)
    hidden_activation = network.hidden_activation \
        if isinstance(network.hidden_activation, eqx.Module) \
        else network.hidden_activation_
    output_activation_elem = network.output_activation_elem \
        if isinstance(network.output_activation_elem, eqx.Module) \
        else network.output_activation_elem_

    _network = NeuralNetwork(
        total_num_neurons,
        adjacency_dict,
        input_neurons,
        output_neurons,
        hidden_activation,
        output_activation_elem,
        network.output_activation_group,
        dropout_p,
        key=key,
        parameter_matrix=parameter_matrix
    )

    new_neuron_ids = jnp.arange(num_new_neurons) + network.num_neurons
    return _network, new_neuron_ids.tolist()


def remove_neurons(network: NeuralNetwork, ids: Sequence[int],
) -> Tuple[NeuralNetwork, Dict[int, int]]:
    """Remove neurons from the network. These can be input, hidden, or output neurons.
    
    **Arguments**:
    
    - `network`: A `NeuralNetwork` object.
    - `ids`: A sequence of `int` ids corresponding to the neurons to remove
        from the network.

    **Returns**:

    A 2-tuple where the first element is the new `NeuralNetwork` with the desired neurons
    removed (along with all respective incoming and outgoing connections)
    and parameters from original neurons retained, and the second element is
    a dictionary mapping neuron ids from the original network to their respective 
    ids in the new network.
    """
    for id in ids:
        assert 0 <= id < network.num_neurons, id
    ids = jnp.array(ids)

    id_map = {}
    sub = 0
    for id in range(network.num_neurons):
        if id in ids:
            sub += 1
        else:
            id_map[id] = id - sub

    # Adjust input and output neurons.
    input_neurons = jnp.setdiff1d(network.input_neurons, ids)
    output_neurons = jnp.setdiff1d(network.output_neurons, ids)
    input_neurons = [id_map[n] for n in input_neurons.tolist()]
    output_neurons = [id_map[n] for n in output_neurons.tolist()]
    
    # Adjust adjacency matrix.
    adjacency_matrix = network.adjacency_matrix
    adjacency_matrix = jnp.delete(adjacency_matrix, ids, 0)
    adjacency_matrix = jnp.delete(adjacency_matrix, ids, 1)

    # Adjust dropout.
    keep_original_idx = jnp.array(list(sorted(id_map.keys())), dtype=int)
    dropout_p = network.get_dropout_p()
    dropout_p = dropout_p[keep_original_idx]

    # Adjust parameter matrix.
    parameter_matrix = network.parameter_matrix
    parameter_matrix = jnp.delete(parameter_matrix, ids, 0)
    parameter_matrix = jnp.delete(parameter_matrix, ids, 1)

    adjacency_dict = _adjacency_matrix_to_dict(adjacency_matrix)
    hidden_activation = network.hidden_activation \
        if isinstance(network.hidden_activation, eqx.Module) \
        else network.hidden_activation_
    output_activation_elem = network.output_activation_elem \
        if isinstance(network.output_activation_elem, eqx.Module) \
        else network.output_activation_elem_

    network = NeuralNetwork(
        network.num_neurons - len(ids),
        adjacency_dict,
        input_neurons,
        output_neurons,
        hidden_activation,
        output_activation_elem,
        network.output_activation_group,
        dropout_p,
        key=eqxe.get_state(network.key, jr.PRNGKey(0)),
        parameter_matrix=parameter_matrix
    )

    return network, id_map


def connect_networks(
    network1: NeuralNetwork,
    network2: NeuralNetwork,
    connection_map_1_to_2: Mapping[int, Sequence[int]] = {},
    connection_map_2_to_1: Mapping[int, Sequence[int]] = {},
    input_neurons: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    output_neurons: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    activation: Callable = jnn.silu,
    output_activation_elem: Callable = _identity,
    output_activation_group: Callable = _identity,
    dropout_p: Optional[Union[float, Sequence[float]]] = None,
    keep_parameters: bool = True,
    *,
    key: Optional[jr.PRNGKey] = None,
) -> Tuple[NeuralNetwork, Dict[int, int]]:
    """Connect two networks together in a specified manner.
    
    **Arguments**:

    - `network1`: A `NeuralNetwork` object.
    - `network2`: A `NeuralNetwork` object.
    - `connection_map_1_to_2` A dictionary that maps an `int` id representing the
        corresponding neuron in `network1` to a sequence of `int` ids representing
        the corresponding neurons in `network2` to which to connect the `network1` neuron.
    - `connection_map_2_to_1` A dictionary that maps an `int` id representing the
        corresponding neuron in `network2` to a sequence of int ids representing
        the corresponding neurons in `network1` to which to connect the `network2` neuron.
    - `input_neurons`: A 2-tuple of `int` sequences, where the first sequence is ids
        of `network1` neurons and the second sequence is ids of `network2` neurons. The two 
        sequences will be concatenated (and appropriately re-numbered) to form the
        input neurons of the new network. Optional argument. If `None`, the input neurons
        of the new network will be the concatenation of the input neurons of `network1`
        and `network2`.
    - `output_neurons`: A 2-tuple of `int` sequences, where the first sequence is ids
        of `network1` neurons and the second sequence is ids of `network2` neurons. The two 
        sequences will be concatenated (and appropriately re-numbered) to form the
        output neurons of the new network. Optional argument. If `None`, the output neurons
        of the new network will be the concatenation of the output neurons of `network1` 
        and `network2`.
    - `activation`: The activation function applied element-wise to the 
        hidden (i.e. non-input, non-output) neurons. It can itself be a 
        trainable `equinox.Module`.
    - `output_activation`: The activation function applied element-wise to 
        the  output neurons. It can itself be a trainable `equinox.Module`.
    - `dropout_p`: Dropout probability. If a single `float`, the same dropout
        probability will be applied to all hidden neurons. If a `Sequence[float]`,
        the sequence must have length `num_neurons`, where `dropout_p[i]` is the
        dropout probability for neuron `i`. Note that this allows dropout to be 
        applied to input and output neurons as well. Optional argument. If `None`, 
        defaults to the concatenation of `network1.get_dropout_p()` and 
        `network2.get_dropout_p()`.
    - `keep_parameters`: If `True`, copies the parameters of `network1` and `network2`
        to the appropriate parameter entries of the new network.
    - `key`: The `PRNGKey` used to initialize parameters. Optional, keyword-only argument. 
        Defaults to `jax.random.PRNGKey(0)`.

    **Returns**:

    A 2-tuple where the first element is the new `NeuralNetwork`, and the second element is
    A dictionary mapping neuron ids from `network2` to their respective 
    ids in the new network. The `network1` ids are left unchanged.
    """
    # Set key. Done this way for nicer docs.
    key = key if key is not None else jr.PRNGKey(0)

    # Set input and output neurons.
    if input_neurons is None:
        input_neurons = [network1.input_neurons, network2.input_neurons]
    input_neurons = jnp.append(
        jnp.array(input_neurons[0]), 
        jnp.array(input_neurons[1]) + network1.num_neurons
    )

    if output_neurons is None:
        output_neurons = [network1.output_neurons, network2.output_neurons]
    output_neurons = jnp.append(
        jnp.array(output_neurons[0]), 
        jnp.array(output_neurons[1]) + network1.num_neurons
    )

    # Set adjacency matrix.
    num_neurons = network1.num_neurons + network2.num_neurons
    adjacency_matrix = jnp.zeros((num_neurons, num_neurons))
    adjacency_matrix = adjacency_matrix \
        .at[:network1.num_neurons, :network1.num_neurons] \
        .set(network1.adjacency_matrix)
    adjacency_matrix = adjacency_matrix \
        .at[network1.num_neurons:, network1.num_neurons:] \
        .set(network2.adjacency_matrix)   

    for (from_neuron, to_neurons) in connection_map_1_to_2.items():
        _to_neurons = jnp.array(to_neurons) + network1.num_neurons
        adjacency_matrix = adjacency_matrix.at[from_neuron, _to_neurons].set(1)

    for (from_neuron, to_neurons) in connection_map_2_to_1.items():
        _from_neuron = from_neuron + network1.num_neurons
        adjacency_matrix = adjacency_matrix.at[_from_neuron, to_neurons].set(1)

    # Set dropout probabilities.
    if dropout_p is None:
        dropout_p = jnp.append(
            network1.get_dropout_p(), 
            network2.get_dropout_p()
        )

    # Initialize parameters iid ~ N(0, 0.01).
    parameter_matrix = jr.normal(
        key, (num_neurons, num_neurons + 1)
    ) * 0.1

    if keep_parameters:
        # Copy parameters from input networks to corresponding neurons in new network.
        parameter_matrix = parameter_matrix \
            .at[:network1.num_neurons, :network1.num_neurons] \
            .set(network1.parameter_matrix[:, :-1])
        parameter_matrix = parameter_matrix \
            .at[:network1.num_neurons, -1] \
            .set(network1.parameter_matrix[:, -1])
        parameter_matrix = parameter_matrix \
            .at[network1.num_neurons:, network1.num_neurons:-1] \
            .set(network2.parameter_matrix[:, :-1])
        parameter_matrix = parameter_matrix \
            .at[network1.num_neurons:, -1] \
            .set(network2.parameter_matrix[:, -1])

    num_neurons = network1.num_neurons + network2.num_neurons
    adjacency_dict = _adjacency_matrix_to_dict(adjacency_matrix)
    
    network = NeuralNetwork(
        num_neurons,
        adjacency_dict,
        input_neurons,
        output_neurons,
        activation,
        output_activation_elem,
        output_activation_group,
        dropout_p,
        key=key,
        parameter_matrix=parameter_matrix
    )

    neuron_ids = jnp.arange(network2.num_neurons) + network1.num_neurons
    return network, dict(enumerate(neuron_ids.tolist()))