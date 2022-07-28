from typing import Callable, Hashable, Mapping, Optional, Sequence, Tuple, Union

import equinox as eqx
import equinox.experimental as eqxe

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

import numpy as np

from .network import NeuralNetwork
from .utils import _identity


def add_connections(
    network: NeuralNetwork,
    connections: Mapping[int, Sequence[int]],
    input_neurons: Optional[Sequence[int]] = None,
    output_neurons: Optional[Sequence[int]] = None,
    *,
    key: Optional[jr.PRNGKey] = None
) -> NeuralNetwork:
    """Add connections to the network.
    
    **Arguments**:

    - `network`: A `NeuralNetwork` object.
    - `connections`: An adjacency dict mapping an existing neuron id to
        its new outgoing connections. Connections that already exist are
        ignored.
    - `input_neurons`: A sequence of `int` indicating the ids of the input 
        neurons. The order here matters, as the input data is passed into 
        the input neurons in the order passed in here. Optional argument. 
        If `None`, the input neurons of the original network are retained.
    - `output_neurons`: A sequence of `int` indicating the ids of the output 
        neurons. The order here matters, as the output values are read from 
        the output neurons in the order passed in here. Optional argument. 
        If `None`, the output neurons of the original network are retained.
    - `key`: The `jax.random.PRNGKey` used for new weight initialization. 
        Optional, keyword-only argument. Defaults to `jax.random.PRNGKey(0)`.

    **Returns**:

    A `NeuralNetwork` object with the specified connections added and original
    parameters retained.
    """
    # Set input and output neurons.
    if input_neurons is None:
        input_neurons = network.input_neurons
    if output_neurons is None:
        output_neurons = network.output_neurons

    # Set element-wise activations.
    hidden_activation = network.hidden_activation \
        if isinstance(network.hidden_activation, eqx.Module) \
        else network._hidden_activation
    output_activation_elem = network.output_activation_elem \
        if isinstance(network.output_activation_elem, eqx.Module) \
        else network._output_activation_elem

    # Update connectivity information.
    adjacency_dict = network.adjacency_dict
    for (input, outputs) in connections.items():
        for output in outputs:
            if output not in adjacency_dict[input]:
                adjacency_dict[input].append(output)

    # Create new network.
    new_network = NeuralNetwork(
        network.num_neurons,
        adjacency_dict,
        input_neurons,
        output_neurons,
        hidden_activation,
        output_activation_elem,
        network.output_activation_group,
        network.get_dropout_p(),
        key=key
    )

    # Transfer relevant parameters from original network.
    old_masks = [np.array(m) for m in network.masks]
    new_weights = [np.array(w) for w in new_network.weights]
    new_bias = np.array(new_network.bias)
    old_min_max_diffs = network.maxs - network.mins
    for n in network.topo_sort[network.num_input_neurons:]:
        old_batch_idx, old_pos_idx = network.neuron_to_topo_batch_idx[n]
        new_batch_idx, new_pos_idx = new_network.neuron_to_topo_batch_idx[n]
        for i in range(old_min_max_diffs[old_batch_idx]):
            if old_masks[old_batch_idx][old_pos_idx, i]:
                in_neuron = network.topo_sort[i + network.mins[old_batch_idx]]
                weight = network.weights[old_batch_idx][old_pos_idx, i]
                new_weights[new_batch_idx][
                    new_pos_idx, 
                    new_network.topo_sort_inv[in_neuron] - new_network.mins[new_batch_idx]
                ] = weight
        bias = network.bias[network.topo_sort_inv[n] - network.num_input_neurons]
        new_bias[new_network.topo_sort_inv[n] - new_network.num_input_neurons] = bias
    new_weights = [jnp.array(w) for w in new_weights]
    new_bias = jnp.array(new_bias)

    return eqx.tree_at(
        lambda net: (net.weights, net.bias), new_network, (new_weights, new_bias)
    )


def remove_connections(
    network: NeuralNetwork,
    connections: Mapping[int, Sequence[int]],
    input_neurons: Optional[Sequence[int]] = None,
    output_neurons: Optional[Sequence[int]] = None,
) -> NeuralNetwork:
    """Remove connections from the network.
    
    **Arguments**:

    - `network`: A `NeuralNetwork` object.
    - `connections`: An adjacency dict mapping an existing neuron id to
        its outgoing connections to remove. Connections that do not exist 
        are ignored.
    - `input_neurons`: A sequence of `int` indicating the ids of the input 
        neurons. The order here matters, as the input data is passed into 
        the input neurons in the order passed in here. Optional argument. 
        If `None`, the input neurons of the original network are retained.
    - `output_neurons`: A sequence of `int` indicating the ids of the output 
        neurons. The order here matters, as the output values are read from 
        the output neurons in the order passed in here. Optional argument. 
        If `None`, the output neurons of the original network are retained.

    **Returns**:

    A `NeuralNetwork` object with the specified connections removed and original
    parameters retained.
    """
    # Set input and output neurons.
    if input_neurons is None:
        input_neurons = network.input_neurons
    if output_neurons is None:
        output_neurons = network.output_neurons

    # Set element-wise activations.
    hidden_activation = network.hidden_activation \
        if isinstance(network.hidden_activation, eqx.Module) \
        else network._hidden_activation
    output_activation_elem = network.output_activation_elem \
        if isinstance(network.output_activation_elem, eqx.Module) \
        else network._output_activation_elem

    # Update connectivity information.
    adjacency_dict = network.adjacency_dict
    for (input, outputs) in connections.items():
        for output in outputs:
            if output in adjacency_dict[input]:
                adjacency_dict[input].remove(output)

    # Create new network.
    new_network = NeuralNetwork(
        network.num_neurons,
        adjacency_dict,
        input_neurons,
        output_neurons,
        hidden_activation,
        output_activation_elem,
        network.output_activation_group,
        network.get_dropout_p(),
        key=network._dropout_key()
    )

    # Transfer relevant parameters from original network.
    new_masks = [np.array(m) for m in new_network.masks]
    new_weights = [np.array(w) for w in new_network.weights]
    new_bias = np.array(new_network.bias)
    new_min_max_diffs = new_network.maxs - new_network.mins
    for n in new_network.topo_sort[new_network.num_input_neurons:]:
        old_batch_idx, old_pos_idx = network.neuron_to_topo_batch_idx[n]
        new_batch_idx, new_pos_idx = new_network.neuron_to_topo_batch_idx[n]
        for i in range(new_min_max_diffs[new_batch_idx]):
            if new_masks[new_batch_idx][new_pos_idx, i]:
                in_neuron = new_network.topo_sort[i + new_network.mins[new_batch_idx]]
                weight = network.weights[old_batch_idx][
                    old_pos_idx, 
                    network.topo_sort_inv[in_neuron] - network.mins[old_batch_idx]
                ]
                new_weights[new_batch_idx][
                    new_pos_idx, 
                    new_network.topo_sort_inv[in_neuron] - new_network.mins[new_batch_idx]
                ] = weight
        bias = network.bias[network.topo_sort_inv[n] - network.num_input_neurons]
        new_bias[new_network.topo_sort_inv[n] - new_network.num_input_neurons] = bias
    new_weights = [jnp.array(w) for w in new_weights]
    new_bias = jnp.array(new_bias)

    return eqx.tree_at(
        lambda net: (net.weights, net.bias), new_network, (new_weights, new_bias)
    )


def add_neurons(
    network: NeuralNetwork,
    new_neuron_data: Sequence[Mapping],
) -> Tuple[NeuralNetwork, Sequence[int]]:
    """Add neurons to the network. These can be input, hidden, or output neurons.
    
    **Arguments**:
    
    - `network`: A `NeuralNetwork` object.
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


def remove_neurons(
    network: NeuralNetwork, ids: Sequence[int],
) -> Tuple[NeuralNetwork, Mapping[int, int]]:
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


def contract_cluster(
    network: NeuralNetwork, 
    neurons: Sequence[Hashable], 
    contract_fn: Callable = jnp.max,
) -> NeuralNetwork:
    pass