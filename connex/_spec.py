from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import networkx as nx


DropoutLike = float | Mapping[Any, float]


@dataclass(frozen=True, init=False)
class GraphSpec:
    """Validated, immutable graph definition for a Connex model."""

    graph: nx.DiGraph
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    topo_sort: tuple[Any, ...]
    dropout: DropoutLike

    def __init__(
        self,
        graph: nx.DiGraph | Any,
        inputs: Sequence[Any],
        outputs: Sequence[Any],
        *,
        topo_sort: Sequence[Any] | None = None,
        dropout: DropoutLike = 0.0,
    ):
        graph = nx.DiGraph(graph)
        inputs = tuple(inputs)
        outputs = tuple(outputs)

        if len(inputs) == 0 or len(outputs) == 0:
            raise ValueError("`inputs` and `outputs` must be nonempty.")

        missing_inputs = [node for node in inputs if node not in graph]
        if missing_inputs:
            raise ValueError(f"`inputs` contains node(s) not in the graph: {missing_inputs}.")

        missing_outputs = [node for node in outputs if node not in graph]
        if missing_outputs:
            raise ValueError(
                f"`outputs` contains node(s) not in the graph: {missing_outputs}."
            )

        overlap = set(inputs) & set(outputs)
        if overlap:
            raise ValueError(f"Node(s) appear in both inputs and outputs: {list(overlap)}.")

        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            raise ValueError(f"`graph` contains cycles: {cycles}.")

        for node in inputs:
            predecessors = list(graph.predecessors(node))
            if predecessors:
                raise ValueError(f"Input node {node!r} has incoming edge(s): {predecessors}.")

        for node in outputs:
            successors = list(graph.successors(node))
            if successors:
                raise ValueError(f"Output node {node!r} has outgoing edge(s): {successors}.")

        ordered = self._canonical_topo_sort(graph, inputs, outputs, topo_sort)
        dropout = self._validate_dropout(graph, dropout)

        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "topo_sort", ordered)
        object.__setattr__(self, "dropout", dropout)

    @staticmethod
    def _canonical_topo_sort(
        graph: nx.DiGraph,
        inputs: tuple[Any, ...],
        outputs: tuple[Any, ...],
        topo_sort: Sequence[Any] | None,
    ) -> tuple[Any, ...]:
        if topo_sort is None:
            topo_sort = tuple(nx.topological_sort(graph))
        else:
            topo_sort = tuple(topo_sort)
            if set(topo_sort) != set(graph.nodes):
                raise ValueError("`topo_sort` must contain exactly the graph nodes.")
            positions = {node: i for i, node in enumerate(topo_sort)}
            for source, target in graph.edges:
                if positions[source] >= positions[target]:
                    raise ValueError(
                        f"`topo_sort` orders edge ({source!r}, {target!r}) incorrectly."
                    )

        middle = [node for node in topo_sort if node not in inputs and node not in outputs]
        return tuple(inputs + tuple(middle) + outputs)

    @staticmethod
    def _validate_dropout(graph: nx.DiGraph, dropout: DropoutLike) -> DropoutLike:
        if isinstance(dropout, Real):
            dropout = float(dropout)
            if not 0 <= dropout <= 1:
                raise ValueError("Dropout probabilities must be in [0, 1].")
            return dropout

        if not isinstance(dropout, Mapping):
            raise TypeError("`dropout` must be a float or a mapping.")

        validated: dict[Any, float] = {}
        for node, probability in dropout.items():
            if node not in graph:
                raise ValueError(f"Dropout specified for node not in graph: {node!r}.")
            if not isinstance(probability, Real):
                raise TypeError(f"Invalid dropout probability for {node!r}: {probability!r}.")
            probability = float(probability)
            if not 0 <= probability <= 1:
                raise ValueError("Dropout probabilities must be in [0, 1].")
            validated[node] = probability
        return validated

    def dropout_by_node(self) -> dict[Any, float]:
        if isinstance(self.dropout, float):
            hidden = set(self.topo_sort) - set(self.inputs) - set(self.outputs)
            return {
                node: self.dropout if node in hidden else 0.0 for node in self.topo_sort
            }
        return {node: float(self.dropout.get(node, 0.0)) for node in self.topo_sort}

    def with_graph(
        self,
        graph: nx.DiGraph,
        *,
        inputs: Sequence[Any] | None = None,
        outputs: Sequence[Any] | None = None,
        topo_sort: Sequence[Any] | None = None,
        dropout: DropoutLike | None = None,
    ) -> GraphSpec:
        return GraphSpec(
            graph,
            inputs=self.inputs if inputs is None else inputs,
            outputs=self.outputs if outputs is None else outputs,
            topo_sort=self.topo_sort if topo_sort is None else topo_sort,
            dropout=self.dropout if dropout is None else dropout,
        )

    def with_dropout(self, dropout: DropoutLike) -> GraphSpec:
        return GraphSpec(
            self.graph,
            inputs=self.inputs,
            outputs=self.outputs,
            topo_sort=self.topo_sort,
            dropout=dropout,
        )
