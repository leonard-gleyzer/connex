from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
from jax import Array

from . import ops as cnx_ops
from ._model import NeuralDAG
from ._spec import DropoutLike, GraphSpec


class TopologyEditor:
    """Fluent editor for structural changes to a `NeuralDAG`."""

    def __init__(self, model: NeuralDAG):
        self._model = model
        self._graph = nx.DiGraph(model.spec.graph)
        self._inputs = list(model.spec.inputs)
        self._outputs = list(model.spec.outputs)
        self._topo_sort = list(model.spec.topo_sort)
        self._dropout = model.spec.dropout
        self._ops = list(model.ops)

    def add_edges(
        self,
        edges: Sequence[tuple[Any, Any]] | Mapping[Any, Sequence[Any]],
    ) -> TopologyEditor:
        if isinstance(edges, Mapping):
            edge_list = [
                (source, target)
                for source, targets in edges.items()
                for target in targets
            ]
        else:
            edge_list = list(edges)
        for source, target in edge_list:
            if source not in self._graph:
                raise ValueError(f"Node {source!r} does not exist.")
            if target not in self._graph:
                raise ValueError(f"Node {target!r} does not exist.")
        self._graph.add_edges_from(edge_list)
        return self

    def remove_edges(
        self,
        edges: Sequence[tuple[Any, Any]] | Mapping[Any, Sequence[Any]],
    ) -> TopologyEditor:
        if isinstance(edges, Mapping):
            edge_list = [
                (source, target)
                for source, targets in edges.items()
                for target in targets
            ]
        else:
            edge_list = list(edges)
        self._graph.remove_edges_from(edge_list)
        return self

    def add_hidden_nodes(self, nodes: Sequence[Any]) -> TopologyEditor:
        nodes = list(nodes)
        self._validate_new_nodes(nodes)
        self._graph.add_nodes_from(nodes)
        insert_at = len(self._topo_sort) - len(self._outputs)
        self._topo_sort[insert_at:insert_at] = nodes
        return self

    def add_input_nodes(self, nodes: Sequence[Any]) -> TopologyEditor:
        nodes = list(nodes)
        self._validate_new_nodes(nodes)
        self._graph.add_nodes_from(nodes)
        self._inputs = nodes + self._inputs
        self._topo_sort = nodes + self._topo_sort
        return self

    def add_output_nodes(self, nodes: Sequence[Any]) -> TopologyEditor:
        nodes = list(nodes)
        self._validate_new_nodes(nodes)
        self._graph.add_nodes_from(nodes)
        self._outputs = self._outputs + nodes
        self._topo_sort = self._topo_sort + nodes
        return self

    def remove_nodes(self, nodes: Sequence[Any]) -> TopologyEditor:
        for node in nodes:
            if node not in self._graph:
                raise ValueError(f"Node {node!r} does not exist.")
        self._graph.remove_nodes_from(nodes)
        removed = set(nodes)
        self._inputs = [node for node in self._inputs if node not in removed]
        self._outputs = [node for node in self._outputs if node not in removed]
        self._topo_sort = [node for node in self._topo_sort if node not in removed]
        if isinstance(self._dropout, Mapping):
            self._dropout = {
                node: probability
                for node, probability in self._dropout.items()
                if node not in removed
            }
        return self

    def set_dropout(self, dropout: DropoutLike) -> TopologyEditor:
        self._dropout = dropout
        self._ops = [
            op.with_dropout(dropout) if isinstance(op, cnx_ops.Dropout) else op
            for op in self._ops
        ]
        return self

    def build(self, *, key: Array | None = None) -> NeuralDAG:
        spec = GraphSpec(
            self._graph,
            inputs=self._inputs,
            outputs=self._outputs,
            topo_sort=self._topo_sort,
            dropout=self._dropout,
        )
        return self._model.rebuild(spec, key=key, ops=tuple(self._ops))

    def _validate_new_nodes(self, nodes: Sequence[Any]) -> None:
        for node in nodes:
            if node in self._graph:
                raise ValueError(f"Node {node!r} already exists.")


def edit(model: NeuralDAG) -> TopologyEditor:
    return TopologyEditor(model)
