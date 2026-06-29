from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
from jax import Array

from . import ops as cnx_ops
from ._model import NeuralDAG
from ._spec import DropoutLike, GraphSpec


class TopologyEditor:
    """Fluent editor for structural changes to a `NeuralDAG`.

    Editors accumulate changes on a pending graph description. Calling
    `build(...)` validates the result, recompiles topology, initializes any new
    parameters, and transfers compatible parameters from the original model.

    Methods mutate the editor object and return `self`, so edits can be chained:

    ```python
    model = (
        connex.edit(model)
        .add_edges([(1, 3)])
        .remove_nodes([2])
        .build(key=key)
    )
    ```
    """

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
        """Add directed edges between existing nodes.

        `edges` can be a sequence of `(source, target)` tuples or an adjacency
        mapping. Nodes must already exist. Duplicate edges are ignored by
        NetworkX. The final graph is validated when `build()` is called, so
        cycles and invalid input/output structure fail there.
        """
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
        """Remove directed edges.

        Missing edges are ignored, matching NetworkX behavior. Removing edges
        never removes nodes; isolated hidden nodes remain in the graph unless
        explicitly removed with `remove_nodes`.
        """
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
        """Add hidden nodes with no edges.

        New hidden nodes are inserted before output nodes in the stored
        topological order. Add edges in the same editor chain if the nodes
        should participate in computation.
        """
        nodes = list(nodes)
        self._validate_new_nodes(nodes)
        self._graph.add_nodes_from(nodes)
        insert_at = len(self._topo_sort) - len(self._outputs)
        self._topo_sort[insert_at:insert_at] = nodes
        return self

    def add_input_nodes(self, nodes: Sequence[Any]) -> TopologyEditor:
        """Add input nodes with no incoming edges.

        New input nodes are prepended to the input order, so they correspond to
        the leading entries of future input arrays. Add outgoing edges in the
        same editor chain if they should feed existing graph structure.
        """
        nodes = list(nodes)
        self._validate_new_nodes(nodes)
        self._graph.add_nodes_from(nodes)
        self._inputs = nodes + self._inputs
        self._topo_sort = nodes + self._topo_sort
        return self

    def add_output_nodes(self, nodes: Sequence[Any]) -> TopologyEditor:
        """Add output nodes with no outgoing edges.

        New output nodes are appended to the output order, so they appear at the
        end of future model outputs. Add incoming edges before `build()` if they
        should compute nonzero values.
        """
        nodes = list(nodes)
        self._validate_new_nodes(nodes)
        self._graph.add_nodes_from(nodes)
        self._outputs = self._outputs + nodes
        self._topo_sort = self._topo_sort + nodes
        return self

    def remove_nodes(self, nodes: Sequence[Any]) -> TopologyEditor:
        """Remove nodes and all incident edges.

        Removed nodes are also removed from input/output ordering and from
        mapping-style dropout configuration.
        """
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
        """Update dropout for the rebuilt model.

        Scalar dropout applies to hidden nodes. Mapping dropout can target any
        node. Built-in dropout operations in the current op pipeline are updated
        to use the same configuration.
        """
        self._dropout = dropout
        self._ops = [
            op.with_dropout(dropout) if isinstance(op, cnx_ops.Dropout) else op
            for op in self._ops
        ]
        return self

    def build(self, *, key: Array | None = None) -> NeuralDAG:
        """Build the edited model.

        **Arguments:**

        - `key`: Random key used to initialize parameters for new nodes or
          edges. Existing compatible parameters are transferred where each
          operation supports transfer.

        **Returns:**

        A new `NeuralDAG`. The original model and earlier editor snapshots are
        unchanged.
        """
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
    """Start a fluent topology edit for `model`.

    This is the preferred public entry point for graph mutation. It returns a
    `TopologyEditor`; call `build(...)` on that editor to obtain the new model.
    """
    return TopologyEditor(model)
