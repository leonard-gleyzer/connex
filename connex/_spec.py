from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import networkx as nx


DropoutLike = float | Mapping[Any, float]


@dataclass(frozen=True, init=False)
class GraphSpec:
    """Validated, immutable graph definition for a Connex model.

    A `GraphSpec` is the non-trainable description of a model: the graph, the
    ordered input and output nodes, the canonical topological order, and the
    dropout configuration. `NeuralDAG` compiles this object into static topology
    metadata and initializes trainable operation parameters from it.

    `graph` may be a `networkx.DiGraph` or any object accepted by
    `networkx.DiGraph(graph)`, including adjacency dictionaries and edge lists.
    Connex validates that the graph is a DAG, that input nodes have no incoming
    edges, that output nodes have no outgoing edges, and that no node appears in
    both `inputs` and `outputs`.

    The order of `inputs` and `outputs` is semantically meaningful. Forward
    calls read `x[i]` into `inputs[i]` and return output values in `outputs`
    order.
    """

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
        """Create a validated graph specification.

        **Arguments:**

        - `graph`: A directed graph or graph-like object accepted by
          `networkx.DiGraph`. The resulting graph must be acyclic.
        - `inputs`: Ordered input node labels. These nodes must exist in the
          graph and must not receive incoming edges.
        - `outputs`: Ordered output node labels. These nodes must exist in the
          graph and must not have outgoing edges.
        - `topo_sort`: Optional topological order. Supplying this avoids a
          NetworkX topological sort during construction and gives stable
          ordering for isolated hidden nodes. The sequence must contain exactly
          the graph nodes and must respect every edge.
        - `dropout`: Either a scalar probability or a mapping from node label to
          probability. Scalar dropout applies to hidden nodes only. Mapping
          dropout defaults unspecified nodes to zero and can target inputs,
          hidden nodes, and outputs.

        **Raises:**

        - `ValueError`: If the graph is cyclic, missing required nodes, has
          invalid input/output edge structure, has overlapping inputs/outputs,
          has an invalid topological order, or contains dropout probabilities
          outside `[0, 1]`.
        - `TypeError`: If `dropout` is neither a scalar nor a mapping.
        """
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
        """Return dropout probabilities keyed by graph node label.

        Scalar dropout expands to every hidden node and leaves inputs and
        outputs at probability zero. Mapping dropout is returned with missing
        nodes filled in as zero. The output dictionary follows `topo_sort`
        ordering.
        """
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
        """Return a new `GraphSpec` with replacement graph metadata.

        Any argument left as `None` is copied from the current specification.
        The returned object is fully revalidated, so this method is useful for
        small immutable graph transformations outside the editor API.
        """
        return GraphSpec(
            graph,
            inputs=self.inputs if inputs is None else inputs,
            outputs=self.outputs if outputs is None else outputs,
            topo_sort=self.topo_sort if topo_sort is None else topo_sort,
            dropout=self.dropout if dropout is None else dropout,
        )

    def with_dropout(self, dropout: DropoutLike) -> GraphSpec:
        """Return a new `GraphSpec` with the same graph and new dropout."""
        return GraphSpec(
            self.graph,
            inputs=self.inputs,
            outputs=self.outputs,
            topo_sort=self.topo_sort,
            dropout=dropout,
        )
