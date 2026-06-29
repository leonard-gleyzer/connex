from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import optax
import pytest

import connex as cnx


def test_no_legacy_public_api():
    assert not hasattr(cnx, "NeuralNetwork")
    assert not hasattr(cnx, "add_connections")
    assert not hasattr(cnx, "remove_neurons")


def test_graph_spec_validation():
    with pytest.raises(ValueError, match="cycles"):
        cnx.GraphSpec(nx.DiGraph([(0, 1), (1, 0)]), inputs=[0], outputs=[1])

    with pytest.raises(ValueError, match="incoming"):
        cnx.GraphSpec(nx.DiGraph([(0, 1), (1, 2)]), inputs=[1], outputs=[2])

    with pytest.raises(ValueError, match="outgoing"):
        cnx.GraphSpec(nx.DiGraph([(0, 1), (1, 2)]), inputs=[0], outputs=[1])

    with pytest.raises(ValueError, match="both"):
        cnx.GraphSpec(nx.DiGraph([(0, 1)]), inputs=[0], outputs=[0])


def test_topology_compiles_labels_and_isolated_hidden_nodes():
    graph = nx.DiGraph()
    graph.add_edges_from([("x", "h"), ("h", "y")])
    graph.add_node(("isolated", 0))
    spec = cnx.GraphSpec(
        graph,
        inputs=["x"],
        outputs=["y"],
        topo_sort=["x", ("isolated", 0), "h", "y"],
    )
    model = cnx.NeuralDAG(spec, key=jr.key(0))

    assert model.topology.node_id("x") == 0
    assert ("isolated", 0) in model.topology.node_position
    assert model(jnp.array([1.0])).shape == (1,)


def test_default_forward_and_networkx_export():
    graph = nx.DiGraph([(0, 1), (1, 2)])
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[2])
    model = cnx.NeuralDAG(spec, key=jr.key(0))

    y = model(jnp.array([1.0]))
    assert y.shape == (1,)

    weighted = model.to_networkx_weighted_digraph()
    assert weighted.number_of_edges() == 2
    assert "weight" in weighted[0][1]
    assert "weight" in weighted[1][2]


def test_dropout_requires_explicit_key_and_is_not_trainable_leaf():
    graph = nx.DiGraph([(0, 1), (1, 2)])
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[2], dropout={1: 0.5})
    model = cnx.NeuralDAG(spec, key=jr.key(0))

    with pytest.raises(ValueError, match="Dropout requires"):
        model(jnp.array([1.0]))

    assert model(jnp.array([1.0]), key=jr.key(1)).shape == (1,)
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_array))
    assert all(leaf.shape != (model.topology.num_nodes,) for leaf in leaves)


def test_all_builtin_feature_ops_forward():
    graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[3], dropout=0.1)
    model = cnx.NeuralDAG(
        spec,
        ops=cnx.ops.default_ops(
            topo_norm=True,
            topo_self_attention=True,
            neuron_self_attention=True,
            adaptive_activation=True,
        ),
        key=jr.key(0),
    )

    assert model(jnp.array([1.0]), key=jr.key(1)).shape == (1,)


def test_sparse_default_ops_forward_and_validation():
    graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[3], dropout=0.1)
    model = cnx.NeuralDAG(
        spec,
        ops=cnx.ops.default_ops(
            affine="sparse",
            topo_norm=True,
            topo_self_attention=True,
            neuron_self_attention=True,
            adaptive_activation=True,
        ),
        key=jr.key(0),
    )

    assert model(jnp.array([1.0]), key=jr.key(1)).shape == (1,)
    with pytest.raises(ValueError, match="affine"):
        cnx.ops.default_ops(affine="dense")


def _hybrid_spec():
    graph = nx.DiGraph(
        [
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (0, 5),
            (1, 6),
            (2, 7),
            (4, 8),
            (5, 8),
            (6, 8),
            (7, 8),
        ]
    )
    return cnx.GraphSpec(graph, inputs=[0, 1, 2, 3], outputs=[8])


def test_default_ops_use_hybrid_affine_with_mixed_batch_selection():
    model = cnx.NeuralDAG(_hybrid_spec(), key=jr.key(0))
    op = next(op for op in model.ops if isinstance(op, cnx.ops.HybridEdgeAffine))

    assert op.sparse_batches == (True, False)
    assert model(jnp.ones((4,))).shape == (1,)

    weighted = model.to_networkx_weighted_digraph()
    assert weighted.number_of_edges() == 11
    assert "weight" in weighted[0][4]
    assert "weight" in weighted[4][8]


def _edge_weight(model, edge):
    op = next(op for op in model.ops if isinstance(op, cnx.ops.EdgeAffine))
    batch, target, source = model.topology.edge_position[edge]
    return op.weights[batch][target, source]


def _sparse_edge_weight(model, edge):
    op = next(op for op in model.ops if isinstance(op, cnx.ops.SparseEdgeAffine))
    batch, edge_pos = model.topology.edge_linear_position[edge]
    return op.weights[batch][edge_pos]


def _node_bias(model, node):
    op = next(op for op in model.ops if isinstance(op, cnx.ops.EdgeAffine))
    batch, target = model.topology.node_position[node]
    return op.biases[batch][target]


def _sparse_node_bias(model, node):
    op = next(op for op in model.ops if isinstance(op, cnx.ops.SparseEdgeAffine))
    batch, target = model.topology.node_position[node]
    return op.biases[batch][target]


def _hybrid_edge_weight(model, edge):
    op = next(op for op in model.ops if isinstance(op, cnx.ops.HybridEdgeAffine))
    batch, target, source = model.topology.edge_position[edge]
    if op.sparse_batches[batch]:
        _, edge_pos = model.topology.edge_linear_position[edge]
        return op.weights[batch][edge_pos]
    return op.weights[batch][target, source]


def _hybrid_node_bias(model, node):
    op = next(op for op in model.ops if isinstance(op, cnx.ops.HybridEdgeAffine))
    batch, target = model.topology.node_position[node]
    return op.biases[batch][target]


def _copy_padded_affine_to_sparse(padded, sparse):
    padded_op = next(op for op in padded.ops if isinstance(op, cnx.ops.EdgeAffine))
    sparse_op = next(op for op in sparse.ops if isinstance(op, cnx.ops.SparseEdgeAffine))

    weights = list(sparse_op.weights)
    biases = list(padded_op.biases)
    for edge, (batch, target_pos, input_pos) in padded.topology.edge_position.items():
        _, edge_pos = sparse.topology.edge_linear_position[edge]
        weights[batch] = weights[batch].at[edge_pos].set(
            padded_op.weights[batch][target_pos, input_pos]
        )

    copied = cnx.ops.SparseEdgeAffine(
        weights=tuple(weights),
        biases=tuple(biases),
        weight_scale=sparse_op.weight_scale,
    )
    ops = tuple(
        copied if isinstance(op, cnx.ops.SparseEdgeAffine) else op for op in sparse.ops
    )
    return cnx.NeuralDAG.from_parts(sparse.spec, sparse.topology, ops)


def _copy_padded_affine_to_hybrid(padded, hybrid):
    padded_op = next(op for op in padded.ops if isinstance(op, cnx.ops.EdgeAffine))
    hybrid_op = next(op for op in hybrid.ops if isinstance(op, cnx.ops.HybridEdgeAffine))

    weights = list(hybrid_op.weights)
    biases = list(padded_op.biases)
    for edge, (batch, target_pos, input_pos) in padded.topology.edge_position.items():
        weight = padded_op.weights[batch][target_pos, input_pos]
        if hybrid_op.sparse_batches[batch]:
            _, edge_pos = hybrid.topology.edge_linear_position[edge]
            weights[batch] = weights[batch].at[edge_pos].set(weight)
        else:
            weights[batch] = weights[batch].at[target_pos, input_pos].set(weight)

    copied = cnx.ops.HybridEdgeAffine(
        weights=tuple(weights),
        biases=tuple(biases),
        sparse_batches=hybrid_op.sparse_batches,
        matmul_batches=hybrid_op.matmul_batches,
        padding_ratio_threshold=hybrid_op.padding_ratio_threshold,
        weight_scale=hybrid_op.weight_scale,
    )
    ops = tuple(
        copied if isinstance(op, cnx.ops.HybridEdgeAffine) else op for op in hybrid.ops
    )
    return cnx.NeuralDAG.from_parts(hybrid.spec, hybrid.topology, ops)


def _copy_padded_affine_to_matmul(padded, matmul):
    padded_op = next(op for op in padded.ops if isinstance(op, cnx.ops.EdgeAffine))
    matmul_op = next(op for op in matmul.ops if isinstance(op, cnx.ops.DenseMatmulAffine))

    unique_positions = []
    for batch in matmul.topology.batches:
        unique_positions.append(
            {
                matmul.topology.node_label(node_id): input_pos
                for input_pos, node_id in enumerate(batch.unique_input_ids)
            }
        )

    weights = list(matmul_op.weights)
    biases = list(padded_op.biases)
    for edge, (batch, target_pos, input_pos) in padded.topology.edge_position.items():
        source, _ = edge
        unique_pos = unique_positions[batch][source]
        weights[batch] = weights[batch].at[target_pos, unique_pos].set(
            padded_op.weights[batch][target_pos, input_pos]
        )

    copied = cnx.ops.DenseMatmulAffine(
        weights=tuple(weights),
        biases=tuple(biases),
        weight_scale=matmul_op.weight_scale,
    )
    ops = tuple(
        copied if isinstance(op, cnx.ops.DenseMatmulAffine) else op for op in matmul.ops
    )
    return cnx.NeuralDAG.from_parts(matmul.spec, matmul.topology, ops)


def test_sparse_affine_matches_padded_affine_and_networkx_export():
    graph = nx.DiGraph(
        [
            (0, 2),
            (1, 2),
            (0, 3),
            (2, 4),
            (3, 4),
            (1, 4),
        ]
    )
    spec = cnx.GraphSpec(graph, inputs=[0, 1], outputs=[4])
    padded = cnx.NeuralDAG(
        spec,
        ops=[cnx.ops.EdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )
    sparse = cnx.NeuralDAG(
        spec,
        ops=[cnx.ops.SparseEdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(1),
    )
    sparse = _copy_padded_affine_to_sparse(padded, sparse)

    x = jnp.array([0.25, -1.5])
    assert jnp.allclose(sparse(x), padded(x))

    padded_graph = padded.to_networkx_weighted_digraph()
    sparse_graph = sparse.to_networkx_weighted_digraph()
    for edge in graph.edges:
        assert jnp.isclose(
            padded_graph.edges[edge]["weight"], sparse_graph.edges[edge]["weight"]
        )


def test_sparse_affine_matches_padded_after_feature_ops():
    graph = nx.DiGraph(
        [
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 4),
            (3, 4),
            (1, 4),
        ]
    )
    spec = cnx.GraphSpec(graph, inputs=[0, 1], outputs=[4])
    common_ops = [
        cnx.ops.TopoNorm(),
        cnx.ops.TopoSelfAttention(),
        cnx.ops.NeuronSelfAttention(),
    ]
    padded = cnx.NeuralDAG(
        spec,
        ops=[*common_ops, cnx.ops.EdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )
    sparse = cnx.NeuralDAG(
        spec,
        ops=[*common_ops, cnx.ops.SparseEdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )
    sparse = _copy_padded_affine_to_sparse(padded, sparse)

    x = jnp.array([0.25, -1.5])
    assert jnp.allclose(sparse(x), padded(x))


def test_hybrid_affine_matches_padded_affine_and_networkx_export():
    spec = _hybrid_spec()
    padded = cnx.NeuralDAG(
        spec,
        ops=[cnx.ops.EdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )
    hybrid = cnx.NeuralDAG(
        spec,
        ops=[cnx.ops.HybridEdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(1),
    )
    hybrid = _copy_padded_affine_to_hybrid(padded, hybrid)

    x = jnp.array([0.25, -1.5, 0.75, 2.0])
    assert jnp.allclose(hybrid(x), padded(x))

    padded_graph = padded.to_networkx_weighted_digraph()
    hybrid_graph = hybrid.to_networkx_weighted_digraph()
    for edge in spec.graph.edges:
        assert jnp.isclose(
            padded_graph.edges[edge]["weight"], hybrid_graph.edges[edge]["weight"]
        )


def test_dense_matmul_affine_matches_padded_affine_and_export():
    graph = nx.DiGraph(
        [
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 4),
            (3, 4),
        ]
    )
    spec = cnx.GraphSpec(graph, inputs=[0, 1], outputs=[4])
    padded = cnx.NeuralDAG(
        spec,
        ops=[cnx.ops.EdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )
    matmul = cnx.NeuralDAG(
        spec,
        ops=[cnx.ops.DenseMatmulAffine(), cnx.ops.OutputTransform()],
        key=jr.key(1),
    )
    matmul = _copy_padded_affine_to_matmul(padded, matmul)

    x = jnp.array([0.25, -1.5])
    assert jnp.allclose(matmul(x), padded(x))

    padded_graph = padded.to_networkx_weighted_digraph()
    matmul_graph = matmul.to_networkx_weighted_digraph()
    for edge in spec.graph.edges:
        assert jnp.isclose(
            padded_graph.edges[edge]["weight"], matmul_graph.edges[edge]["weight"]
        )


def test_hybrid_selects_matmul_for_dense_shared_predecessor_batches():
    graph = nx.DiGraph((source, target) for source in range(96) for target in range(96, 192))
    spec = cnx.GraphSpec(graph, inputs=list(range(96)), outputs=list(range(96, 192)))
    model = cnx.NeuralDAG(
        spec,
        ops=[cnx.ops.HybridEdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )
    op = next(op for op in model.ops if isinstance(op, cnx.ops.HybridEdgeAffine))

    assert op.sparse_batches == (False,)
    assert op.matmul_batches == (True,)
    assert model(jnp.ones((96,))).shape == (96,)


def test_native_batched_forward_matches_vmap():
    graph = nx.DiGraph(
        [
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 4),
            (3, 4),
        ]
    )
    spec = cnx.GraphSpec(graph, inputs=[0, 1], outputs=[4])
    model = cnx.NeuralDAG(spec, key=jr.key(0))
    x = jnp.stack(
        [
            jnp.array([0.25, -1.5]),
            jnp.array([1.0, 2.0]),
            jnp.array([-0.75, 0.5]),
        ]
    )

    assert jnp.allclose(model.batched(x), jax.vmap(model)(x))


def test_fused_default_matches_unfused_default_and_batches():
    graph = nx.DiGraph(
        [
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 4),
            (3, 4),
        ]
    )
    spec = cnx.GraphSpec(graph, inputs=[0, 1], outputs=[4])
    unfused = cnx.NeuralDAG(
        spec,
        ops=cnx.ops.default_ops(affine="hybrid"),
        key=jr.key(0),
    )
    affine = next(op for op in unfused.ops if isinstance(op, cnx.ops.HybridEdgeAffine))
    fused = cnx.NeuralDAG(
        spec,
        ops=[
            cnx.ops.FusedDefaultOp(affine=affine, activation=jax.nn.gelu),
            cnx.ops.OutputTransform(),
        ],
        key=jr.key(1),
    )
    fused_op = next(op for op in fused.ops if isinstance(op, cnx.ops.FusedDefaultOp))
    fused = cnx.NeuralDAG.from_parts(
        fused.spec,
        fused.topology,
        (cnx.ops.FusedDefaultOp(affine=affine, activation=jax.nn.gelu), fused.ops[-1]),
    )

    x = jnp.array([0.25, -1.5])
    xs = jnp.stack([x, jnp.array([1.0, 2.0]), jnp.array([-0.75, 0.5])])
    assert isinstance(fused_op, cnx.ops.FusedDefaultOp)
    assert jnp.allclose(fused(x), unfused(x))
    assert jnp.allclose(fused.batched(xs), jax.vmap(fused)(xs))


def test_editor_add_edges_remove_nodes_and_transfer_parameters():
    graph = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 3)])
    model = cnx.NeuralDAG(
        cnx.GraphSpec(graph, inputs=[0], outputs=[3]),
        ops=[cnx.ops.EdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )
    original_weight = _edge_weight(model, (0, 1))
    original_bias = _node_bias(model, 1)

    edited = cnx.edit(model).add_edges([(1, 2)]).build(key=jr.key(1))
    assert (1, 2) in edited.topology.edge_position
    assert jnp.isclose(_edge_weight(edited, (0, 1)), original_weight)
    assert jnp.isclose(_node_bias(edited, 1), original_bias)

    removed = cnx.edit(edited).remove_nodes([2]).build(key=jr.key(2))
    assert 2 not in removed.spec.graph
    assert (0, 1) in removed.topology.edge_position
    assert jnp.isclose(_edge_weight(removed, (0, 1)), original_weight)


def test_editor_transfers_sparse_affine_parameters():
    graph = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 3)])
    model = cnx.NeuralDAG(
        cnx.GraphSpec(graph, inputs=[0], outputs=[3]),
        ops=[cnx.ops.SparseEdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )
    original_weight = _sparse_edge_weight(model, (0, 1))
    original_bias = _sparse_node_bias(model, 1)

    edited = cnx.edit(model).add_edges([(1, 2)]).build(key=jr.key(1))
    assert (1, 2) in edited.topology.edge_linear_position
    assert jnp.isclose(_sparse_edge_weight(edited, (0, 1)), original_weight)
    assert jnp.isclose(_sparse_node_bias(edited, 1), original_bias)

    removed = cnx.edit(edited).remove_nodes([2]).build(key=jr.key(2))
    assert 2 not in removed.spec.graph
    assert (0, 1) in removed.topology.edge_linear_position
    assert jnp.isclose(_sparse_edge_weight(removed, (0, 1)), original_weight)


def test_editor_transfers_hybrid_affine_parameters_across_backend_change():
    model = cnx.NeuralDAG(
        _hybrid_spec(),
        ops=[cnx.ops.HybridEdgeAffine(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )
    original_sparse_weight = _hybrid_edge_weight(model, (0, 4))
    original_padded_weight = _hybrid_edge_weight(model, (4, 8))
    original_sparse_bias = _hybrid_node_bias(model, 4)
    original_padded_bias = _hybrid_node_bias(model, 8)

    edited = (
        cnx.edit(model)
        .add_edges(
            [
                (1, 5),
                (2, 5),
                (3, 5),
                (0, 6),
                (2, 6),
                (3, 6),
                (0, 7),
                (1, 7),
                (3, 7),
            ]
        )
        .build(key=jr.key(1))
    )
    edited_op = next(
        op for op in edited.ops if isinstance(op, cnx.ops.HybridEdgeAffine)
    )

    assert edited_op.sparse_batches == (False, False)
    assert jnp.isclose(_hybrid_edge_weight(edited, (0, 4)), original_sparse_weight)
    assert jnp.isclose(_hybrid_edge_weight(edited, (4, 8)), original_padded_weight)
    assert jnp.isclose(_hybrid_node_bias(edited, 4), original_sparse_bias)
    assert jnp.isclose(_hybrid_node_bias(edited, 8), original_padded_bias)


def test_editor_add_edges_requires_existing_nodes():
    graph = nx.DiGraph([(0, 1)])
    model = cnx.NeuralDAG(cnx.GraphSpec(graph, inputs=[0], outputs=[1]), key=jr.key(0))

    with pytest.raises(ValueError, match="does not exist"):
        cnx.edit(model).add_edges([(0, 2)])


def test_editor_set_dropout_updates_new_model_only():
    graph = nx.DiGraph([(0, 1), (1, 2)])
    model = cnx.NeuralDAG(cnx.GraphSpec(graph, inputs=[0], outputs=[2]), key=jr.key(0))
    updated = cnx.edit(model).set_dropout(0.25).build(key=jr.key(1))

    assert model.spec.dropout == 0.0
    assert updated.spec.dropout == 0.25
    with pytest.raises(ValueError, match="Dropout requires"):
        updated(jnp.array([1.0]))


def test_op_level_dropout_survives_rebuild():
    graph = nx.DiGraph([(0, 1), (1, 2)])
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[2])
    model = cnx.NeuralDAG(
        spec,
        ops=cnx.ops.default_ops(dropout=0.25),
        key=jr.key(0),
    )
    rebuilt = cnx.edit(model).add_hidden_nodes([3]).build(key=jr.key(1))

    with pytest.raises(ValueError, match="Dropout requires"):
        rebuilt(jnp.array([1.0]))


class ScaleOutputs(cnx.ops.Op):
    scale: jax.Array = eqx.field(default_factory=lambda: jnp.array(2.0))
    name: str = eqx.field(static=True, default="scale_outputs")

    def init(self, topology, *, key):
        return ScaleOutputs(scale=jnp.array(2.0))

    def apply(self, ctx, *, state=None, key=None):
        assert ctx.outputs is not None
        return replace(ctx, outputs=ctx.outputs * self.scale)


class NoOp(cnx.ops.Op):
    name: str = eqx.field(static=True, default="noop")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)


def _without_scan(model):
    return cnx.NeuralDAG.from_parts(
        model.spec,
        model.topology,
        (*model.ops[:-1], NoOp(), model.ops[-1]),
    )


def _scan_entries(model):
    return [entry for entry in model.execution_plan if entry.kind == "scan"]


def test_custom_user_op_participates_in_pipeline():
    graph = nx.DiGraph([(0, 1)])
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[1])
    base = cnx.NeuralDAG(spec, key=jr.key(0))
    custom = cnx.NeuralDAG(
        spec,
        ops=[cnx.ops.EdgeAffine(), ScaleOutputs(), cnx.ops.OutputTransform()],
        key=jr.key(0),
    )

    assert jnp.allclose(custom(jnp.array([1.0])), base(jnp.array([1.0])) * 2)


@pytest.mark.parametrize("affine", ["padded", "sparse", "matmul", "hybrid"])
def test_scan_execution_plan_matches_generic_chain(affine):
    graph = nx.DiGraph((i, i + 1) for i in range(12))
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[12])
    scanned = cnx.NeuralDAG(
        spec,
        ops=cnx.ops.default_ops(affine=affine),
        key=jr.key(0),
    )
    generic = _without_scan(scanned)
    scan_entries = _scan_entries(scanned)

    assert len(scan_entries) == 1
    assert scan_entries[0].live_target_ids == (scanned.topology.output_ids[0],)
    assert not _scan_entries(generic)

    x = jnp.array([0.5])
    xs = jnp.asarray([[0.5], [1.0], [-0.25]], dtype=jnp.float32)
    assert jnp.allclose(scanned(x), generic(x))
    assert jnp.allclose(scanned.batched(xs), generic.batched(xs))
    assert jnp.allclose(scanned.batched(xs), jax.vmap(scanned)(xs))


def test_scan_execution_plan_matches_fused_default_with_dropout():
    graph = nx.DiGraph((i, i + 1) for i in range(10))
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[10], dropout=0.2)
    scanned = cnx.NeuralDAG(
        spec,
        ops=cnx.ops.default_ops(fused=True),
        key=jr.key(0),
    )
    generic = _without_scan(scanned)

    assert _scan_entries(scanned)
    assert not _scan_entries(generic)

    key = jr.key(1)
    x = jnp.array([0.5])
    xs = jnp.asarray([[0.5], [1.0], [-0.25]], dtype=jnp.float32)
    assert jnp.allclose(scanned(x, key=key), generic(x, key=key))
    assert jnp.allclose(scanned.batched(xs, key=key), generic.batched(xs, key=key))


def test_scan_live_out_keeps_branch_inputs_available():
    graph = nx.DiGraph(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (2, 7),
            (5, 7),
        ]
    )
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[6, 7])
    scanned = cnx.NeuralDAG(spec, key=jr.key(0))
    generic = _without_scan(scanned)
    scan_plan = _scan_entries(scanned)[0]

    assert scan_plan.live_target_ids == (
        scanned.topology.node_id(2),
        scanned.topology.node_id(5),
    )
    assert jnp.allclose(scanned(jnp.array([0.5])), generic(jnp.array([0.5])))


def test_nn_builders_smoke():
    assert cnx.nn.MLP(2, 1, 4, 2, key=jr.key(0))(jnp.ones((2,))).shape == (1,)
    assert cnx.nn.DenseMLP(2, 1, 4, 2, key=jr.key(0))(jnp.ones((2,))).shape == (1,)


def test_training_with_explicit_batched_keys():
    graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    spec = cnx.GraphSpec(graph, inputs=[0], outputs=[3], dropout=0.2)
    model = cnx.NeuralDAG(spec, key=jr.key(0))
    optim = optax.adam(1e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    x = jnp.expand_dims(jnp.linspace(0, 1, 32), 1)
    y = jnp.sin(x)

    @eqx.filter_value_and_grad
    def loss_fn(model, x, y, key):
        keys = jr.split(key, x.shape[0])
        preds = jax.vmap(lambda x_i, key_i: model(x_i, key=key_i))(x, keys)
        return jnp.mean((preds - y) ** 2)

    @eqx.filter_jit
    def step(model, opt_state, x, y, key):
        loss, grads = loss_fn(model, x, y, key)
        updates, opt_state = optim.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss

    key = jr.key(1)
    initial = float(loss_fn(model, x, y, key)[0])
    for _ in range(5):
        key, step_key = jr.split(key)
        model, opt_state, loss = step(model, opt_state, x, y, step_key)

    assert float(loss) < initial
