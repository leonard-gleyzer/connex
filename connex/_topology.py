from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from ._spec import GraphSpec


@dataclass(frozen=True)
class BatchTopology:
    target_ids: tuple[int, ...]
    target_is_output: tuple[bool, ...]
    input_ids: tuple[tuple[int, ...], ...]
    input_mask: tuple[tuple[bool, ...], ...]
    unique_input_ids: tuple[int, ...]
    unique_inverse: tuple[tuple[int, ...], ...]
    unique_input_mask: tuple[tuple[bool, ...], ...]
    edge_input_ids: tuple[int, ...]
    edge_target_positions: tuple[int, ...]
    edge_input_positions: tuple[int, ...]
    edge_unique_inverse: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.target_ids)

    @property
    def max_inputs(self) -> int:
        return len(self.input_ids[0]) if self.input_ids else 0

    @property
    def unique_size(self) -> int:
        return len(self.unique_input_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_input_ids)

    @property
    def has_output_targets(self) -> bool:
        return any(self.target_is_output)

    @property
    def all_targets_output(self) -> bool:
        return all(self.target_is_output)


@dataclass(frozen=True)
class CompiledTopology:
    spec: GraphSpec
    node_to_id: dict[Any, int]
    id_to_node: tuple[Any, ...]
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    hidden_ids: tuple[int, ...]
    batches: tuple[BatchTopology, ...]
    node_position: dict[Any, tuple[int, int]]
    edge_position: dict[tuple[Any, Any], tuple[int, int, int]]
    edge_linear_position: dict[tuple[Any, Any], tuple[int, int]]
    dropout_probs: tuple[float, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.id_to_node)

    @property
    def output_size(self) -> int:
        return len(self.output_ids)

    def node_label(self, node_id: int) -> Any:
        return self.id_to_node[int(node_id)]

    def node_id(self, node: Any) -> int:
        return self.node_to_id[node]


def compile_topology(spec: GraphSpec) -> CompiledTopology:
    node_to_id = {node: i for i, node in enumerate(spec.topo_sort)}
    id_to_node = tuple(spec.topo_sort)
    input_ids = tuple(node_to_id[node] for node in spec.inputs)
    output_ids = tuple(node_to_id[node] for node in spec.outputs)
    hidden_ids = tuple(
        node_to_id[node]
        for node in spec.topo_sort
        if node not in spec.inputs and node not in spec.outputs
    )

    topo_batches = _compute_batches(spec.graph, spec.topo_sort, node_to_id, spec.inputs)
    batches = tuple(_compile_batch(spec, target_ids, node_to_id) for target_ids in topo_batches)

    node_position: dict[Any, tuple[int, int]] = {}
    edge_position: dict[tuple[Any, Any], tuple[int, int, int]] = {}
    edge_linear_position: dict[tuple[Any, Any], tuple[int, int]] = {}
    for batch_index, batch in enumerate(batches):
        for target_pos, target_id in enumerate(batch.target_ids):
            target = id_to_node[target_id]
            node_position[target] = (batch_index, target_pos)
            for input_pos, has_edge in enumerate(batch.input_mask[target_pos]):
                if has_edge:
                    source = id_to_node[batch.input_ids[target_pos][input_pos]]
                    edge_position[(source, target)] = (
                        batch_index,
                        target_pos,
                        input_pos,
                    )
        for edge_pos, (target_pos, input_pos) in enumerate(
            zip(batch.edge_target_positions, batch.edge_input_positions)
        ):
            target = id_to_node[batch.target_ids[target_pos]]
            source = id_to_node[batch.input_ids[target_pos][input_pos]]
            edge_linear_position[(source, target)] = (batch_index, edge_pos)

    dropout_by_node = spec.dropout_by_node()
    dropout_probs = tuple(float(dropout_by_node[node]) for node in spec.topo_sort)

    return CompiledTopology(
        spec=spec,
        node_to_id=node_to_id,
        id_to_node=id_to_node,
        input_ids=input_ids,
        output_ids=output_ids,
        hidden_ids=hidden_ids,
        batches=batches,
        node_position=node_position,
        edge_position=edge_position,
        edge_linear_position=edge_linear_position,
        dropout_probs=dropout_probs,
    )


def _compute_batches(
    graph: nx.DiGraph,
    topo_sort: tuple[Any, ...],
    node_to_id: dict[Any, int],
    inputs: tuple[Any, ...],
) -> list[list[int]]:
    processed = set(inputs)
    remaining = [node for node in topo_sort if node not in processed]
    topo_batches: list[list[int]] = []

    while remaining:
        batch = [
            node
            for node in remaining
            if all(predecessor in processed for predecessor in graph.predecessors(node))
        ]
        if not batch:
            raise ValueError("Unable to compile topological batches.")
        topo_batches.append([node_to_id[node] for node in batch])
        processed.update(batch)
        batch_set = set(batch)
        remaining = [node for node in remaining if node not in batch_set]

    return topo_batches


def _compile_batch(
    spec: GraphSpec, target_ids: list[int], node_to_id: dict[Any, int]
) -> BatchTopology:
    id_to_node = tuple(spec.topo_sort)
    incoming: list[list[int]] = []
    unique_inputs: list[int] = []
    seen: set[int] = set()

    for target_id in target_ids:
        target = id_to_node[target_id]
        inputs = sorted(
            (node_to_id[source] for source in spec.graph.predecessors(target)),
            key=int,
        )
        incoming.append(inputs)
        for input_id in inputs:
            if input_id not in seen:
                seen.add(input_id)
                unique_inputs.append(input_id)

    max_inputs = max((len(inputs) for inputs in incoming), default=0)
    pad_id = 0
    input_ids: list[tuple[int, ...]] = []
    input_mask: list[tuple[bool, ...]] = []
    unique_inverse: list[tuple[int, ...]] = []
    unique_input_mask: list[tuple[bool, ...]] = []
    edge_input_ids: list[int] = []
    edge_target_positions: list[int] = []
    edge_input_positions: list[int] = []
    edge_unique_inverse: list[int] = []
    unique_positions = {node_id: i for i, node_id in enumerate(unique_inputs)}

    for target_pos, inputs in enumerate(incoming):
        row_ids = inputs + [pad_id] * (max_inputs - len(inputs))
        row_mask = [True] * len(inputs) + [False] * (max_inputs - len(inputs))
        row_inverse = [unique_positions[input_id] for input_id in inputs]
        row_inverse += [0] * (max_inputs - len(inputs))
        input_ids.append(tuple(row_ids))
        input_mask.append(tuple(row_mask))
        unique_inverse.append(tuple(row_inverse))
        unique_row_mask = [False] * len(unique_inputs)
        for input_id in inputs:
            unique_row_mask[unique_positions[input_id]] = True
        unique_input_mask.append(tuple(unique_row_mask))
        for input_pos, input_id in enumerate(inputs):
            edge_input_ids.append(input_id)
            edge_target_positions.append(target_pos)
            edge_input_positions.append(input_pos)
            edge_unique_inverse.append(unique_positions[input_id])

    return BatchTopology(
        target_ids=tuple(target_ids),
        target_is_output=tuple(
            id_to_node[target_id] in spec.outputs for target_id in target_ids
        ),
        input_ids=tuple(input_ids),
        input_mask=tuple(input_mask),
        unique_input_ids=tuple(unique_inputs),
        unique_inverse=tuple(unique_inverse),
        unique_input_mask=tuple(unique_input_mask),
        edge_input_ids=tuple(edge_input_ids),
        edge_target_positions=tuple(edge_target_positions),
        edge_input_positions=tuple(edge_input_positions),
        edge_unique_inverse=tuple(edge_unique_inverse),
    )
