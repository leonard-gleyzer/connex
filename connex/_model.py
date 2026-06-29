from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
from jax import Array

from . import ops as cnx_ops
from ._spec import GraphSpec
from ._topology import BatchTopology, CompiledTopology, compile_topology


@dataclass(frozen=True)
class BatchPlan:
    batch_index: int
    batch: BatchTopology
    needs_batch_inputs: bool
    needs_padded_inputs: bool
    needs_edge_inputs: bool
    target_ids: tuple[int, ...]
    target_is_output: tuple[bool, ...]
    input_ids: tuple[tuple[int, ...], ...]
    input_mask: tuple[tuple[bool, ...], ...]
    unique_input_ids: tuple[int, ...]
    unique_inverse: tuple[tuple[int, ...], ...]
    edge_input_ids: tuple[int, ...]
    edge_target_positions: tuple[int, ...]
    edge_input_positions: tuple[int, ...]
    edge_unique_inverse: tuple[int, ...]
    target_ids_array: Array
    target_is_output_array: Array
    input_ids_array: Array
    input_mask_array: Array
    unique_input_ids_array: Array
    unique_inverse_array: Array
    edge_input_ids_array: Array
    edge_target_positions_array: Array
    edge_input_positions_array: Array
    edge_unique_inverse_array: Array
    kind: str = "batch"


@dataclass(frozen=True)
class ScanComponents:
    affine_index: int
    activation: Any
    dropout_index: int | None
    fused: bool


@dataclass(frozen=True)
class ScanPlan:
    start: int
    stop: int
    input_id: int
    target_ids: tuple[int, ...]
    target_is_output: tuple[bool, ...]
    live_target_ids: tuple[int, ...]
    live_positions: tuple[int, ...]
    target_ids_array: Array
    target_is_output_array: Array
    live_target_ids_array: Array
    live_positions_array: Array
    components: ScanComponents
    kind: str = "scan"


ExecutionPlanEntry = BatchPlan | ScanPlan


class NeuralDAG(eqx.Module):
    """A trainable neural network runtime compiled from a DAG specification.

    `NeuralDAG` is the core Connex model type. It is an `equinox.Module` whose
    trainable leaves live inside operation objects, while graph structure and
    compiled topology metadata are static. This makes models compatible with
    `eqx.filter_jit`, `eqx.filter_value_and_grad`, `eqx.apply_updates`, and the
    usual JAX transformations.

    Forward evaluation proceeds in compiled topological order. Connex groups
    nodes into batches whose predecessors are already available, gathers the
    data requested by each operation, and writes newly computed node values into
    a value buffer. Compatible one-input chain regions are represented as
    `jax.lax.scan` execution-plan segments; unsupported operation stacks and
    custom operations use the generic topological batch path.
    """

    spec: GraphSpec = eqx.field(static=True)
    topology: CompiledTopology = eqx.field(static=True)
    batch_needs: tuple[tuple[bool, bool, bool], ...] = eqx.field(static=True)
    execution_plan: tuple[ExecutionPlanEntry, ...] = eqx.field(static=True)
    ops: tuple[Any, ...]

    def __init__(
        self,
        spec: GraphSpec,
        *,
        ops: Sequence[Any] | None = None,
        key: Array | None = None,
    ):
        """Initialize a model from a graph specification.

        **Arguments:**

        - `spec`: Validated graph, input/output ordering, topological order, and
          dropout configuration.
        - `ops`: Operation pipeline. If `None`, Connex uses
          `connex.ops.default_ops()`, which is a hybrid affine op, activation,
          dropout, and output transform.
        - `key`: JAX random key used to initialize operation parameters. If
          omitted, `jax.random.key(0)` is used.

        The operation sequence is initialized once against the compiled
        topology. Each operation may create trainable arrays, static metadata,
        or no state at all.
        """
        topology = compile_topology(spec)
        ops = cnx_ops.default_ops() if ops is None else tuple(ops)
        key = jr.key(0) if key is None else key
        keys = jr.split(key, max(1, len(ops)))
        initialized_ops = tuple(op.init(topology, key=keys[i]) for i, op in enumerate(ops))

        self.spec = spec
        self.topology = topology
        self.batch_needs = _batch_needs(initialized_ops, topology)
        self.execution_plan = _execution_plan(
            initialized_ops, topology, self.batch_needs
        )
        self.ops = initialized_ops

    @classmethod
    def from_parts(
        cls,
        spec: GraphSpec,
        topology: CompiledTopology,
        ops: Sequence[Any],
    ) -> NeuralDAG:
        """Construct a model from already compiled components.

        This is mostly useful for internal rebuilds, tests, and advanced users
        who are deliberately reusing an existing `CompiledTopology` and
        operation tuple. Most code should call `NeuralDAG(spec, ...)` or use the
        topology editor instead.
        """
        model = object.__new__(cls)
        ops = tuple(ops)
        object.__setattr__(model, "spec", spec)
        object.__setattr__(model, "topology", topology)
        batch_needs = _batch_needs(ops, topology)
        object.__setattr__(model, "batch_needs", batch_needs)
        object.__setattr__(
            model, "execution_plan", _execution_plan(ops, topology, batch_needs)
        )
        object.__setattr__(model, "ops", ops)
        return model

    def rebuild(
        self,
        spec: GraphSpec,
        *,
        key: Array | None = None,
        ops: Sequence[Any] | None = None,
    ) -> NeuralDAG:
        """Recompile a model against a new graph specification.

        The new operation pipeline is initialized for `spec`, then each old
        operation is asked to transfer compatible parameters into the initialized
        replacement. Built-in affine operations preserve parameters for nodes and
        edges that still exist by label.

        **Arguments:**

        - `spec`: New graph specification.
        - `key`: Random key for initializing newly created parameters.
        - `ops`: Optional replacement operation sequence. If omitted, the
          current operations are reused.

        **Returns:**

        A new `NeuralDAG`; the original model is unchanged.
        """
        new_topology = compile_topology(spec)
        old_ops = self.ops if ops is None else tuple(ops)
        key = jr.key(0) if key is None else key
        keys = jr.split(key, max(1, len(old_ops)))
        initialized = tuple(
            op.init(new_topology, key=keys[i]) for i, op in enumerate(old_ops)
        )
        transferred = tuple(
            old.transfer(self.topology, new_topology, new)
            for old, new in zip(old_ops, initialized)
        )
        return NeuralDAG.from_parts(spec, new_topology, transferred)

    @eqx.filter_jit
    def __call__(self, x: Array, *, key: Array | None = None) -> Array:
        """Evaluate one input example.

        **Arguments:**

        - `x`: Array whose trailing dimension matches `len(spec.inputs)`.
          Values are assigned to input nodes in specification order.
        - `key`: Optional JAX random key used by runtime stochastic operations
          such as dropout. If any active dropout operation is present, a key is
          required.

        **Returns:**

        Output values in `spec.outputs` order after the output-stage operations
        have been applied.
        """
        topology = self.topology
        values = jnp.zeros((topology.num_nodes,), dtype=jnp.asarray(x).dtype)
        input_ids = jnp.asarray(topology.input_ids, dtype=int)
        output_ids = jnp.asarray(topology.output_ids, dtype=int)
        values = values.at[input_ids].set(x)

        op_keys: list[Array | None]
        if key is None:
            op_keys = [None] * len(self.ops)
        else:
            split = jr.split(key, max(1, len(self.ops)))
            op_keys = [split[i] for i in range(len(self.ops))]

        runtime_states = tuple(
            op.init_runtime(topology, key=op_keys[i]) for i, op in enumerate(self.ops)
        )

        for op, state, op_key in zip(self.ops, runtime_states, op_keys):
            values = op.apply_inputs(topology, values, state=state, key=op_key)

        for entry in self.execution_plan:
            if isinstance(entry, ScanPlan):
                values = _apply_scan_segment(self, values, entry, runtime_states)
            else:
                values = _apply_batch_plan(self, values, entry, runtime_states, op_keys)

        y = values[output_ids]
        for op, state, op_key in zip(self.ops, runtime_states, op_keys):
            if op.stage == "output":
                y = op.apply_output(topology, y, state=state, key=op_key)
        return y

    @eqx.filter_jit
    def batched(self, x: Array, *, key: Array | None = None) -> Array:
        """Evaluate a batch of examples with one shared runtime state.

        This is the native batched path and avoids compiling `jax.vmap(model)`
        over the whole DAG. It is appropriate for deterministic evaluation or
        when a shared dropout mask is desired. For independent stochastic state
        per example, split keys and use `jax.vmap(lambda x_i, key_i:
        model(x_i, key=key_i))(x, keys)`.

        **Arguments:**

        - `x`: Array of shape `(batch, len(spec.inputs))`.
        - `key`: Optional runtime key shared by the whole batch.

        **Returns:**

        Array of shape `(batch, len(spec.outputs))`.
        """
        topology = self.topology
        x = jnp.asarray(x)
        values = jnp.zeros((x.shape[0], topology.num_nodes), dtype=x.dtype)
        input_ids = jnp.asarray(topology.input_ids, dtype=int)
        output_ids = jnp.asarray(topology.output_ids, dtype=int)
        values = values.at[:, input_ids].set(x)

        op_keys: list[Array | None]
        if key is None:
            op_keys = [None] * len(self.ops)
        else:
            split = jr.split(key, max(1, len(self.ops)))
            op_keys = [split[i] for i in range(len(self.ops))]

        runtime_states = tuple(
            op.init_runtime(topology, key=op_keys[i]) for i, op in enumerate(self.ops)
        )

        for op, state, op_key in zip(self.ops, runtime_states, op_keys):
            values = op.apply_inputs(topology, values, state=state, key=op_key)

        for entry in self.execution_plan:
            if isinstance(entry, ScanPlan):
                values = _apply_scan_segment(self, values, entry, runtime_states)
            else:
                values = _apply_batch_plan(self, values, entry, runtime_states, op_keys)

        y = values[:, output_ids]
        for op, state, op_key in zip(self.ops, runtime_states, op_keys):
            if op.stage == "output":
                y = op.apply_output(topology, y, state=state, key=op_key)
        return y

    def to_networkx_weighted_digraph(self) -> nx.DiGraph:
        """Export the model graph with learned edge weights.

        The returned `networkx.DiGraph` copies `self.spec.graph` and annotates
        each edge with a `"weight"` attribute when a built-in affine operation is
        present. Sparse, matmul, padded, and hybrid affine layouts are all
        mapped back to the original graph edge labels.
        """
        graph = nx.DiGraph(self.spec.graph)
        padded_affine = next(
            (op for op in self.ops if isinstance(op, cnx_ops.EdgeAffine)), None
        )
        sparse_affine = next(
            (op for op in self.ops if isinstance(op, cnx_ops.SparseEdgeAffine)), None
        )
        matmul_affine = next(
            (op for op in self.ops if isinstance(op, cnx_ops.DenseMatmulAffine)), None
        )
        hybrid_affine = next(
            (op for op in self.ops if isinstance(op, cnx_ops.HybridEdgeAffine)), None
        )
        if (
            padded_affine is None
            and sparse_affine is None
            and matmul_affine is None
            and hybrid_affine is None
        ):
            return graph

        edge_attrs = {}
        if padded_affine is not None:
            for edge, (
                batch,
                target_pos,
                input_pos,
            ) in self.topology.edge_position.items():
                edge_attrs[edge] = {
                    "weight": padded_affine.weights[batch][target_pos, input_pos]
                }
        else:
            if sparse_affine is not None:
                for edge, (batch, edge_pos) in self.topology.edge_linear_position.items():
                    edge_attrs[edge] = {"weight": sparse_affine.weights[batch][edge_pos]}
            elif matmul_affine is not None:
                unique_positions = _unique_positions_by_batch(self.topology)
                for edge, (
                    batch,
                    target_pos,
                    _,
                ) in self.topology.edge_position.items():
                    source, _ = edge
                    input_pos = unique_positions[batch][source]
                    edge_attrs[edge] = {
                        "weight": matmul_affine.weights[batch][target_pos, input_pos]
                    }
            else:
                assert hybrid_affine is not None
                unique_positions = _unique_positions_by_batch(self.topology)
                for edge, (
                    batch,
                    target_pos,
                    input_pos,
                ) in self.topology.edge_position.items():
                    if hybrid_affine.sparse_batches[batch]:
                        _, edge_pos = self.topology.edge_linear_position[edge]
                        weight = hybrid_affine.weights[batch][edge_pos]
                    elif hybrid_affine.matmul_batches[batch]:
                        source, _ = edge
                        unique_input_pos = unique_positions[batch][source]
                        weight = hybrid_affine.weights[batch][
                            target_pos, unique_input_pos
                        ]
                    else:
                        weight = hybrid_affine.weights[batch][target_pos, input_pos]
                    edge_attrs[edge] = {"weight": weight}
        nx.set_edge_attributes(graph, edge_attrs)
        return graph


def _batch_needs(
    ops: Sequence[Any],
    topology: CompiledTopology,
) -> tuple[tuple[bool, bool, bool], ...]:
    batch_ops = tuple(op for op in ops if op.stage != "output")
    return tuple(
        (
            any(op.needs_batch_inputs_for_batch(i) for op in batch_ops),
            any(op.needs_padded_inputs_for_batch(i) for op in batch_ops),
            any(op.needs_edge_inputs_for_batch(i) for op in batch_ops),
        )
        for i in range(len(topology.batches))
    )


def _execution_plan(
    ops: Sequence[Any],
    topology: CompiledTopology,
    batch_needs: tuple[tuple[bool, bool, bool], ...],
) -> tuple[ExecutionPlanEntry, ...]:
    scan_components = _scan_components(ops)
    if scan_components is None:
        return tuple(_batch_plan(topology, batch_needs, i) for i in range(len(topology.batches)))

    plan: list[ExecutionPlanEntry] = []
    i = 0
    while i < len(topology.batches):
        if not _scan_batch_eligible(topology, i):
            plan.append(_batch_plan(topology, batch_needs, i))
            i += 1
            continue

        start = i
        i += 1
        while i < len(topology.batches) and _scan_batch_eligible(topology, i):
            previous_target = topology.batches[i - 1].target_ids[0]
            next_input = topology.batches[i].input_ids[0][0]
            if next_input != previous_target:
                break
            i += 1

        if i - start >= 4:
            plan.append(_scan_plan(topology, start, i, scan_components))
        else:
            plan.extend(_batch_plan(topology, batch_needs, j) for j in range(start, i))
    return tuple(plan)


def _batch_plan(
    topology: CompiledTopology,
    batch_needs: tuple[tuple[bool, bool, bool], ...],
    batch_index: int,
) -> BatchPlan:
    needs_batch_inputs, needs_padded_inputs, needs_edge_inputs = batch_needs[batch_index]
    batch = topology.batches[batch_index]
    return BatchPlan(
        batch_index=batch_index,
        batch=batch,
        needs_batch_inputs=needs_batch_inputs,
        needs_padded_inputs=needs_padded_inputs,
        needs_edge_inputs=needs_edge_inputs,
        target_ids=batch.target_ids,
        target_is_output=batch.target_is_output,
        input_ids=batch.input_ids,
        input_mask=batch.input_mask,
        unique_input_ids=batch.unique_input_ids,
        unique_inverse=batch.unique_inverse,
        edge_input_ids=batch.edge_input_ids,
        edge_target_positions=batch.edge_target_positions,
        edge_input_positions=batch.edge_input_positions,
        edge_unique_inverse=batch.edge_unique_inverse,
        target_ids_array=jnp.asarray(batch.target_ids, dtype=int),
        target_is_output_array=jnp.asarray(batch.target_is_output, dtype=bool),
        input_ids_array=jnp.asarray(batch.input_ids, dtype=int),
        input_mask_array=jnp.asarray(batch.input_mask, dtype=bool),
        unique_input_ids_array=jnp.asarray(batch.unique_input_ids, dtype=int),
        unique_inverse_array=jnp.asarray(batch.unique_inverse, dtype=int),
        edge_input_ids_array=jnp.asarray(batch.edge_input_ids, dtype=int),
        edge_target_positions_array=jnp.asarray(
            batch.edge_target_positions, dtype=int
        ),
        edge_input_positions_array=jnp.asarray(batch.edge_input_positions, dtype=int),
        edge_unique_inverse_array=jnp.asarray(batch.edge_unique_inverse, dtype=int),
    )


def _scan_batch_eligible(topology: CompiledTopology, batch_index: int) -> bool:
    batch = topology.batches[batch_index]
    if batch.size != 1 or batch.max_inputs != 1:
        return False
    if batch.unique_size != 1 or batch.num_edges != 1:
        return False
    return bool(batch.input_mask[0][0])


def _scan_plan(
    topology: CompiledTopology,
    start: int,
    stop: int,
    components: ScanComponents,
) -> ScanPlan:
    batches = topology.batches[start:stop]
    target_ids = tuple(batch.target_ids[0] for batch in batches)
    live_positions = _scan_live_out_positions(topology, target_ids)
    live_target_ids = tuple(target_ids[position] for position in live_positions)
    return ScanPlan(
        start=start,
        stop=stop,
        input_id=batches[0].input_ids[0][0],
        target_ids=target_ids,
        target_is_output=tuple(batch.target_is_output[0] for batch in batches),
        live_target_ids=live_target_ids,
        live_positions=live_positions,
        target_ids_array=jnp.asarray(target_ids, dtype=int),
        target_is_output_array=jnp.asarray(
            tuple(batch.target_is_output[0] for batch in batches), dtype=bool
        ),
        live_target_ids_array=jnp.asarray(live_target_ids, dtype=int),
        live_positions_array=jnp.asarray(live_positions, dtype=int),
        components=components,
    )


def _scan_live_out_positions(
    topology: CompiledTopology,
    target_ids: tuple[int, ...],
) -> tuple[int, ...]:
    output_ids = set(topology.output_ids)
    live_positions = []
    for position, target_id in enumerate(target_ids):
        if target_id in output_ids:
            live_positions.append(position)
            continue

        node = topology.node_label(target_id)
        internal_successor = (
            target_ids[position + 1] if position + 1 < len(target_ids) else None
        )
        for successor in topology.spec.graph.successors(node):
            if topology.node_id(successor) != internal_successor:
                live_positions.append(position)
                break
    return tuple(live_positions)


def _scan_components(ops: Sequence[Any]) -> ScanComponents | None:
    indexed = tuple((i, op) for i, op in enumerate(ops) if op.stage != "output")
    if len(indexed) == 1 and isinstance(indexed[0][1], cnx_ops.FusedDefaultOp):
        op_index, fused = indexed[0]
        if not _is_scan_affine(fused.affine):
            return None
        return ScanComponents(
            affine_index=op_index,
            activation=fused.activation,
            dropout_index=op_index if fused.active else None,
            fused=True,
        )
    if not indexed:
        return None

    affine_index, affine = indexed[0]
    if not _is_scan_affine(affine):
        return None

    activation = None
    dropout_index = None
    position = 1
    if position < len(indexed) and isinstance(indexed[position][1], cnx_ops.Activation):
        activation = indexed[position][1].activation
        position += 1
    if position < len(indexed) and isinstance(indexed[position][1], cnx_ops.Dropout):
        dropout_index = indexed[position][0] if indexed[position][1].active else None
        position += 1
    if position != len(indexed):
        return None
    return ScanComponents(
        affine_index=affine_index,
        activation=activation,
        dropout_index=dropout_index,
        fused=False,
    )


def _is_scan_affine(op: Any) -> bool:
    return isinstance(
        op,
        (
            cnx_ops.EdgeAffine,
            cnx_ops.SparseEdgeAffine,
            cnx_ops.DenseMatmulAffine,
            cnx_ops.HybridEdgeAffine,
        ),
    )


def _apply_batch_plan(
    model: NeuralDAG,
    values: Array,
    plan: BatchPlan,
    runtime_states: tuple[Any, ...],
    op_keys: list[Array | None],
) -> Array:
    topology = model.topology
    batch = plan.batch
    batch_size = values.shape[0] if values.ndim == 2 else None
    target_ids = plan.target_ids_array
    target_is_output = plan.target_is_output_array

    if plan.needs_batch_inputs:
        unique_ids = plan.unique_input_ids_array
        batch_inputs = values[:, unique_ids] if values.ndim == 2 else values[unique_ids]
    else:
        unique_ids = jnp.zeros((0,), dtype=int)
        if values.ndim == 2:
            batch_inputs = jnp.zeros((batch_size, 0), dtype=values.dtype)
        else:
            batch_inputs = jnp.zeros((0,), dtype=values.dtype)

    if plan.needs_padded_inputs:
        input_ids = plan.input_ids_array
        input_mask = plan.input_mask_array.astype(values.dtype)
        unique_inverse = plan.unique_inverse_array
        if values.ndim == 2:
            input_values = values[:, input_ids] * input_mask
        else:
            input_values = values[input_ids] * input_mask
    else:
        input_ids = jnp.zeros((batch.size, 0), dtype=int)
        input_mask = jnp.zeros((batch.size, 0), dtype=values.dtype)
        unique_inverse = jnp.zeros((batch.size, 0), dtype=int)
        if values.ndim == 2:
            input_values = jnp.zeros((batch_size, batch.size, 0), dtype=values.dtype)
        else:
            input_values = jnp.zeros((batch.size, 0), dtype=values.dtype)

    if plan.needs_edge_inputs:
        edge_input_ids = plan.edge_input_ids_array
        edge_target_positions = plan.edge_target_positions_array
        edge_input_positions = plan.edge_input_positions_array
        edge_unique_inverse = plan.edge_unique_inverse_array
        edge_input_values = (
            values[:, edge_input_ids] if values.ndim == 2 else values[edge_input_ids]
        )
    else:
        edge_input_ids = jnp.zeros((0,), dtype=int)
        edge_target_positions = jnp.zeros((0,), dtype=int)
        edge_input_positions = jnp.zeros((0,), dtype=int)
        edge_unique_inverse = jnp.zeros((0,), dtype=int)
        if values.ndim == 2:
            edge_input_values = jnp.zeros((batch_size, 0), dtype=values.dtype)
        else:
            edge_input_values = jnp.zeros((0,), dtype=values.dtype)

    ctx = cnx_ops.ForwardContext(
        topology=topology,
        batch_index=plan.batch_index,
        batch=batch,
        values=values,
        target_ids=target_ids,
        target_is_output=target_is_output,
        input_ids=input_ids,
        input_mask=input_mask,
        unique_input_ids=unique_ids,
        unique_inverse=unique_inverse,
        edge_input_ids=edge_input_ids,
        edge_target_positions=edge_target_positions,
        edge_input_positions=edge_input_positions,
        edge_unique_inverse=edge_unique_inverse,
        batch_inputs=batch_inputs,
        input_values=input_values,
        edge_input_values=edge_input_values,
    )

    for op, state, op_key in zip(model.ops, runtime_states, op_keys):
        if op.stage != "output":
            ctx = op.apply(ctx, state=state, key=op_key)

    if ctx.outputs is None:
        raise RuntimeError("No op produced neuron outputs; include an affine op.")
    if values.ndim == 2:
        return values.at[:, target_ids].set(ctx.outputs)
    return values.at[target_ids].set(ctx.outputs)


def _apply_scan_segment(
    model: NeuralDAG,
    values: Array,
    plan: ScanPlan,
    runtime_states: tuple[Any, ...],
) -> Array:
    affine = _scan_affine(model.ops, plan.components)
    activation = plan.components.activation
    dropout_index = plan.components.dropout_index
    weights, biases = _scan_affine_params(affine, plan.start, plan.stop)
    target_ids = plan.target_ids_array
    target_is_output = plan.target_is_output_array
    if dropout_index is None:
        keep = jnp.ones((plan.stop - plan.start,), dtype=values.dtype)
    else:
        dropout_state = runtime_states[dropout_index]
        keep = jnp.asarray(dropout_state[target_ids], dtype=values.dtype)

    carry = values[plan.input_id] if values.ndim == 1 else values[:, plan.input_id]

    def step(carry, params):
        weight, bias, is_output, keep_value = params
        raw = carry * weight + bias
        if activation is None:
            output = raw
        else:
            output = jnp.where(is_output, raw, activation(raw))
        output = output * keep_value
        return output, output

    _, outputs = jax.lax.scan(step, carry, (weights, biases, target_is_output, keep))
    if not plan.live_target_ids:
        return values

    live_positions = plan.live_positions_array
    live_target_ids = plan.live_target_ids_array
    live_outputs = outputs[live_positions]
    if values.ndim == 1:
        return values.at[live_target_ids].set(live_outputs)
    return values.at[:, live_target_ids].set(jnp.swapaxes(live_outputs, 0, 1))


def _scan_affine(ops: tuple[Any, ...], components: ScanComponents) -> Any:
    op = ops[components.affine_index]
    return op.affine if components.fused else op


def _scan_affine_params(affine: Any, start: int, stop: int) -> tuple[Array, Array]:
    weights = []
    biases = []
    for batch_index in range(start, stop):
        biases.append(affine.biases[batch_index][0])
        if isinstance(affine, cnx_ops.SparseEdgeAffine):
            weights.append(affine.weights[batch_index][0])
        elif isinstance(affine, cnx_ops.HybridEdgeAffine):
            if affine.sparse_batches[batch_index]:
                weights.append(affine.weights[batch_index][0])
            else:
                weights.append(affine.weights[batch_index][0, 0])
        else:
            weights.append(affine.weights[batch_index][0, 0])
    return jnp.stack(weights), jnp.stack(biases)


def _unique_positions_by_batch(
    topology: CompiledTopology,
) -> tuple[dict[Any, int], ...]:
    return tuple(
        {
            topology.node_label(node_id): input_pos
            for input_pos, node_id in enumerate(batch.unique_input_ids)
        }
        for batch in topology.batches
    )
