from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum, auto
from numbers import Real
from typing import Any

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, vmap

from ._spec import DropoutLike
from ._topology import BatchTopology, CompiledTopology
from ._utils import _identity


class ParamLayout(Enum):
    GLOBAL = auto()
    NODE = auto()
    EDGE = auto()
    TOPO_BATCH = auto()
    TOPO_INPUT = auto()
    CUSTOM = auto()


@dataclass(frozen=True)
class ForwardContext:
    topology: CompiledTopology
    batch_index: int
    batch: BatchTopology
    values: Array
    target_ids: Array
    target_is_output: Array
    input_ids: Array
    input_mask: Array
    unique_input_ids: Array
    unique_inverse: Array
    edge_input_ids: Array
    edge_target_positions: Array
    edge_input_positions: Array
    edge_unique_inverse: Array
    batch_inputs: Array
    input_values: Array
    edge_input_values: Array
    outputs: Array | None = None


class Op(eqx.Module):
    """Base class for user-definable Connex operations."""

    name: str = eqx.field(static=True, default="op")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.GLOBAL)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=True)
    needs_padded_inputs: bool = eqx.field(static=True, default=True)
    needs_edge_inputs: bool = eqx.field(static=True, default=True)

    def init(self, topology: CompiledTopology, *, key: Array) -> Op:
        return self

    def init_runtime(self, topology: CompiledTopology, *, key: Array | None) -> Any:
        return None

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        return ctx

    def apply_inputs(
        self,
        topology: CompiledTopology,
        values: Array,
        *,
        state: Any = None,
        key: Array | None = None,
    ) -> Array:
        return values

    def apply_output(
        self,
        topology: CompiledTopology,
        y: Array,
        *,
        state: Any = None,
        key: Array | None = None,
    ) -> Array:
        return y

    def needs_batch_inputs_for_batch(self, batch_index: int) -> bool:
        return self.needs_batch_inputs

    def needs_padded_inputs_for_batch(self, batch_index: int) -> bool:
        return self.needs_padded_inputs

    def needs_edge_inputs_for_batch(self, batch_index: int) -> bool:
        return self.needs_edge_inputs

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: Op,
    ) -> Op:
        return initialized


class EdgeAffine(Op):
    weights: tuple[Array, ...] = ()
    biases: tuple[Array, ...] = ()
    weight_scale: float = eqx.field(static=True, default=0.1)
    name: str = eqx.field(static=True, default="edge_affine")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.EDGE)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=True)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def init(self, topology: CompiledTopology, *, key: Array) -> EdgeAffine:
        keys = jr.split(key, max(1, len(topology.batches) * 2))
        weights: list[Array] = []
        biases: list[Array] = []
        for i, batch in enumerate(topology.batches):
            weights.append(
                jr.normal(keys[2 * i], (batch.size, batch.max_inputs)) * self.weight_scale
            )
            biases.append(jr.normal(keys[2 * i + 1], (batch.size,)) * self.weight_scale)
        return EdgeAffine(
            weights=tuple(weights),
            biases=tuple(biases),
            weight_scale=self.weight_scale,
        )

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        weights = self.weights[ctx.batch_index]
        biases = self.biases[ctx.batch_index]
        outputs = jnp.sum(weights * ctx.input_values, axis=-1) + biases
        return replace(ctx, outputs=outputs)

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: EdgeAffine,
    ) -> EdgeAffine:
        weights = list(initialized.weights)
        biases = list(
            _transfer_node_params(
                self.biases, initialized.biases, old_topology, new_topology
            )
        )

        edge_updates: dict[int, list[tuple[int, int, Array]]] = {}
        for edge, old_position in old_topology.edge_position.items():
            if edge in new_topology.edge_position:
                old_batch, old_target, old_input = old_position
                new_batch, new_target, new_input = new_topology.edge_position[edge]
                edge_updates.setdefault(new_batch, []).append(
                    (
                        new_target,
                        new_input,
                        self.weights[old_batch][old_target, old_input],
                    )
                )
        for batch_index, updates in edge_updates.items():
            targets, inputs, values = zip(*updates)
            weights[batch_index] = weights[batch_index].at[
                jnp.asarray(targets, dtype=int), jnp.asarray(inputs, dtype=int)
            ].set(jnp.stack(values))

        return EdgeAffine(
            weights=tuple(weights),
            biases=tuple(biases),
            weight_scale=self.weight_scale,
        )


class SparseEdgeAffine(Op):
    weights: tuple[Array, ...] = ()
    biases: tuple[Array, ...] = ()
    weight_scale: float = eqx.field(static=True, default=0.1)
    name: str = eqx.field(static=True, default="sparse_edge_affine")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.EDGE)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=True)

    def init(self, topology: CompiledTopology, *, key: Array) -> SparseEdgeAffine:
        keys = jr.split(key, max(1, len(topology.batches) * 2))
        weights: list[Array] = []
        biases: list[Array] = []
        for i, batch in enumerate(topology.batches):
            weights.append(jr.normal(keys[2 * i], (batch.num_edges,)) * self.weight_scale)
            biases.append(jr.normal(keys[2 * i + 1], (batch.size,)) * self.weight_scale)
        return SparseEdgeAffine(
            weights=tuple(weights),
            biases=tuple(biases),
            weight_scale=self.weight_scale,
        )

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        weights = self.weights[ctx.batch_index]
        biases = self.biases[ctx.batch_index]
        if ctx.edge_input_values.ndim == 1:
            contributions = weights * ctx.edge_input_values
            outputs = biases.at[ctx.edge_target_positions].add(contributions)
        else:
            outputs = vmap(
                lambda inputs: biases.at[ctx.edge_target_positions].add(
                    weights * inputs
                )
            )(ctx.edge_input_values)
        return replace(ctx, outputs=outputs)

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: SparseEdgeAffine,
    ) -> SparseEdgeAffine:
        weights = list(initialized.weights)
        biases = list(
            _transfer_node_params(
                self.biases, initialized.biases, old_topology, new_topology
            )
        )

        edge_updates: dict[int, list[tuple[int, Array]]] = {}
        for edge, (old_batch, old_edge_pos) in old_topology.edge_linear_position.items():
            if edge in new_topology.edge_linear_position:
                new_batch, new_edge_pos = new_topology.edge_linear_position[edge]
                edge_updates.setdefault(new_batch, []).append(
                    (new_edge_pos, self.weights[old_batch][old_edge_pos])
                )
        for batch_index, updates in edge_updates.items():
            edge_positions, values = zip(*updates)
            weights[batch_index] = weights[batch_index].at[
                jnp.asarray(edge_positions, dtype=int)
            ].set(jnp.stack(values))

        return SparseEdgeAffine(
            weights=tuple(weights),
            biases=tuple(biases),
            weight_scale=self.weight_scale,
        )


class DenseMatmulAffine(Op):
    weights: tuple[Array, ...] = ()
    biases: tuple[Array, ...] = ()
    weight_scale: float = eqx.field(static=True, default=0.1)
    name: str = eqx.field(static=True, default="dense_matmul_affine")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.EDGE)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=True)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def init(self, topology: CompiledTopology, *, key: Array) -> DenseMatmulAffine:
        keys = jr.split(key, max(1, len(topology.batches) * 2))
        weights: list[Array] = []
        biases: list[Array] = []
        for i, batch in enumerate(topology.batches):
            mask = jnp.asarray(batch.unique_input_mask, dtype=float)
            weights.append(
                jr.normal(keys[2 * i], (batch.size, batch.unique_size))
                * mask
                * self.weight_scale
            )
            biases.append(jr.normal(keys[2 * i + 1], (batch.size,)) * self.weight_scale)
        return DenseMatmulAffine(
            weights=tuple(weights),
            biases=tuple(biases),
            weight_scale=self.weight_scale,
        )

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        weights = self.weights[ctx.batch_index]
        biases = self.biases[ctx.batch_index]
        mask = jnp.asarray(ctx.batch.unique_input_mask, dtype=weights.dtype)
        outputs = jnp.matmul(ctx.batch_inputs, (weights * mask).T) + biases
        return replace(ctx, outputs=outputs)

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: DenseMatmulAffine,
    ) -> DenseMatmulAffine:
        weights = list(initialized.weights)
        biases = _transfer_node_params(
            self.biases, initialized.biases, old_topology, new_topology
        )
        old_unique_positions = _unique_positions_by_batch(old_topology)
        new_unique_positions = _unique_positions_by_batch(new_topology)

        edge_updates: dict[int, list[tuple[int, int, Array]]] = {}
        for edge, (old_batch, old_target, _) in old_topology.edge_position.items():
            if edge not in new_topology.edge_position:
                continue
            source, _ = edge
            old_input = old_unique_positions[old_batch][source]
            new_batch, new_target, _ = new_topology.edge_position[edge]
            new_input = new_unique_positions[new_batch][source]
            edge_updates.setdefault(new_batch, []).append(
                (
                    new_target,
                    new_input,
                    self.weights[old_batch][old_target, old_input],
                )
            )
        for batch_index, updates in edge_updates.items():
            targets, inputs, values = zip(*updates)
            weights[batch_index] = weights[batch_index].at[
                jnp.asarray(targets, dtype=int), jnp.asarray(inputs, dtype=int)
            ].set(jnp.stack(values))

        return DenseMatmulAffine(
            weights=tuple(weights),
            biases=biases,
            weight_scale=self.weight_scale,
        )


class HybridEdgeAffine(Op):
    weights: tuple[Array, ...] = ()
    biases: tuple[Array, ...] = ()
    sparse_batches: tuple[bool, ...] = eqx.field(static=True, default=())
    matmul_batches: tuple[bool, ...] = eqx.field(static=True, default=())
    padding_ratio_threshold: float = eqx.field(static=True, default=2.0)
    weight_scale: float = eqx.field(static=True, default=0.1)
    name: str = eqx.field(static=True, default="hybrid_edge_affine")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.EDGE)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=True)
    needs_edge_inputs: bool = eqx.field(static=True, default=True)

    def init(self, topology: CompiledTopology, *, key: Array) -> HybridEdgeAffine:
        keys = jr.split(key, max(1, len(topology.batches) * 2))
        weights: list[Array] = []
        biases: list[Array] = []
        sparse_batches: list[bool] = []
        matmul_batches: list[bool] = []

        for i, batch in enumerate(topology.batches):
            mode = _hybrid_affine_mode(batch, self.padding_ratio_threshold)
            use_sparse = mode == "sparse"
            use_matmul = mode == "matmul"
            sparse_batches.append(use_sparse)
            matmul_batches.append(use_matmul)
            if use_sparse:
                shape = (batch.num_edges,)
                weight = jr.normal(keys[2 * i], shape) * self.weight_scale
            elif use_matmul:
                shape = (batch.size, batch.unique_size)
                mask = jnp.asarray(batch.unique_input_mask, dtype=float)
                weight = jr.normal(keys[2 * i], shape) * mask * self.weight_scale
            else:
                shape = (batch.size, batch.max_inputs)
                weight = jr.normal(keys[2 * i], shape) * self.weight_scale
            weights.append(weight)
            biases.append(jr.normal(keys[2 * i + 1], (batch.size,)) * self.weight_scale)

        return HybridEdgeAffine(
            weights=tuple(weights),
            biases=tuple(biases),
            sparse_batches=tuple(sparse_batches),
            matmul_batches=tuple(matmul_batches),
            padding_ratio_threshold=self.padding_ratio_threshold,
            weight_scale=self.weight_scale,
        )

    def needs_batch_inputs_for_batch(self, batch_index: int) -> bool:
        return bool(self.matmul_batches and self.matmul_batches[batch_index])

    def needs_padded_inputs_for_batch(self, batch_index: int) -> bool:
        use_matmul = bool(self.matmul_batches and self.matmul_batches[batch_index])
        return not self.sparse_batches[batch_index] and not use_matmul

    def needs_edge_inputs_for_batch(self, batch_index: int) -> bool:
        return self.sparse_batches[batch_index]

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        biases = self.biases[ctx.batch_index]
        use_matmul = bool(self.matmul_batches and self.matmul_batches[ctx.batch_index])
        if self.sparse_batches[ctx.batch_index]:
            weights = self.weights[ctx.batch_index]
            if ctx.edge_input_values.ndim == 1:
                contributions = weights * ctx.edge_input_values
                outputs = biases.at[ctx.edge_target_positions].add(contributions)
            else:
                outputs = vmap(
                    lambda inputs: biases.at[ctx.edge_target_positions].add(
                        weights * inputs
                    )
                )(ctx.edge_input_values)
        elif use_matmul:
            weights = self.weights[ctx.batch_index]
            mask = jnp.asarray(ctx.batch.unique_input_mask, dtype=weights.dtype)
            outputs = jnp.matmul(ctx.batch_inputs, (weights * mask).T) + biases
        else:
            weights = self.weights[ctx.batch_index]
            outputs = jnp.sum(weights * ctx.input_values, axis=-1) + biases
        return replace(ctx, outputs=outputs)

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: HybridEdgeAffine,
    ) -> HybridEdgeAffine:
        weights = list(initialized.weights)
        biases = list(
            _transfer_node_params(
                self.biases, initialized.biases, old_topology, new_topology
            )
        )
        old_unique_positions = _unique_positions_by_batch(old_topology)
        new_unique_positions = _unique_positions_by_batch(new_topology)

        sparse_updates: dict[int, list[tuple[int, Array]]] = {}
        matrix_updates: dict[int, list[tuple[int, int, Array]]] = {}
        for edge, old_position in old_topology.edge_position.items():
            if edge not in new_topology.edge_position:
                continue
            old_batch, old_target, old_input = old_position
            if self.sparse_batches[old_batch]:
                _, old_edge_pos = old_topology.edge_linear_position[edge]
                weight = self.weights[old_batch][old_edge_pos]
            elif self.matmul_batches and self.matmul_batches[old_batch]:
                source, _ = edge
                old_unique = old_unique_positions[old_batch][source]
                weight = self.weights[old_batch][old_target, old_unique]
            else:
                weight = self.weights[old_batch][old_target, old_input]

            new_batch, new_target, new_input = new_topology.edge_position[edge]
            if initialized.sparse_batches[new_batch]:
                _, new_edge_pos = new_topology.edge_linear_position[edge]
                sparse_updates.setdefault(new_batch, []).append((new_edge_pos, weight))
            elif initialized.matmul_batches and initialized.matmul_batches[new_batch]:
                source, _ = edge
                new_unique = new_unique_positions[new_batch][source]
                matrix_updates.setdefault(new_batch, []).append(
                    (new_target, new_unique, weight)
                )
            else:
                matrix_updates.setdefault(new_batch, []).append(
                    (new_target, new_input, weight)
                )
        for batch_index, updates in sparse_updates.items():
            edge_positions, values = zip(*updates)
            weights[batch_index] = weights[batch_index].at[
                jnp.asarray(edge_positions, dtype=int)
            ].set(jnp.stack(values))
        for batch_index, updates in matrix_updates.items():
            targets, inputs, values = zip(*updates)
            weights[batch_index] = weights[batch_index].at[
                jnp.asarray(targets, dtype=int), jnp.asarray(inputs, dtype=int)
            ].set(jnp.stack(values))

        return HybridEdgeAffine(
            weights=tuple(weights),
            biases=tuple(biases),
            sparse_batches=initialized.sparse_batches,
            matmul_batches=initialized.matmul_batches,
            padding_ratio_threshold=self.padding_ratio_threshold,
            weight_scale=self.weight_scale,
        )


class TopoNorm(Op):
    params: tuple[Array, ...] = ()
    name: str = eqx.field(static=True, default="topo_norm")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.TOPO_INPUT)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=True)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def init(self, topology: CompiledTopology, *, key: Array) -> TopoNorm:
        keys = jr.split(key, max(1, len(topology.batches)))
        params = []
        for i, batch in enumerate(topology.batches):
            gamma_beta = jnp.ones((batch.unique_size, 2))
            if batch.unique_size:
                gamma_beta = gamma_beta + jr.normal(keys[i], gamma_beta.shape) * 0.1
            params.append(gamma_beta)
        return TopoNorm(params=tuple(params))

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        if ctx.batch.unique_size <= 1:
            return ctx
        params = self.params[ctx.batch_index]
        gamma, beta = params[:, 0], params[:, 1]
        batch_inputs = jnn.standardize(ctx.batch_inputs) * gamma + beta
        input_values = ctx.input_values
        if ctx.input_values.size:
            input_values = batch_inputs[ctx.unique_inverse] * ctx.input_mask
        edge_input_values = ctx.edge_input_values
        if ctx.edge_input_values.size:
            edge_input_values = batch_inputs[ctx.edge_unique_inverse]
        return replace(
            ctx,
            batch_inputs=batch_inputs,
            input_values=input_values,
            edge_input_values=edge_input_values,
        )

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: TopoNorm,
    ) -> TopoNorm:
        return TopoNorm(
            params=_transfer_topo_input_params(
                self.params, initialized.params, old_topology, new_topology
            )
        )


class TopoSelfAttention(Op):
    params: tuple[Array, ...] = ()
    weight_scale: float = eqx.field(static=True, default=0.1)
    name: str = eqx.field(static=True, default="topo_self_attention")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.TOPO_INPUT)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=True)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def init(self, topology: CompiledTopology, *, key: Array) -> TopoSelfAttention:
        keys = jr.split(key, max(1, len(topology.batches)))
        params = []
        for i, batch in enumerate(topology.batches):
            shape = (3, batch.unique_size, batch.unique_size + 1)
            params.append(jr.normal(keys[i], shape) * self.weight_scale)
        return TopoSelfAttention(params=tuple(params), weight_scale=self.weight_scale)

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        if ctx.batch.unique_size == 0:
            return ctx
        params = self.params[ctx.batch_index]
        query_params, key_params, value_params = params
        query = query_params[:, :-1] @ ctx.batch_inputs + query_params[:, -1]
        key_values = key_params[:, :-1] @ ctx.batch_inputs + key_params[:, -1]
        value = value_params[:, :-1] @ ctx.batch_inputs + value_params[:, -1]
        scale = jax_rsqrt(float(ctx.batch.unique_size))
        attention = jnn.softmax(jnp.outer(query, key_values) * scale)
        batch_inputs = attention @ value + ctx.batch_inputs
        input_values = ctx.input_values
        if ctx.input_values.size:
            input_values = batch_inputs[ctx.unique_inverse] * ctx.input_mask
        edge_input_values = ctx.edge_input_values
        if ctx.edge_input_values.size:
            edge_input_values = batch_inputs[ctx.edge_unique_inverse]
        return replace(
            ctx,
            batch_inputs=batch_inputs,
            input_values=input_values,
            edge_input_values=edge_input_values,
        )

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: TopoSelfAttention,
    ) -> TopoSelfAttention:
        params = list(initialized.params)
        old_signatures = _batch_signatures(old_topology)
        new_signatures = _batch_signatures(new_topology)
        old_by_signature = {signature: i for i, signature in enumerate(old_signatures)}

        for new_batch, signature in enumerate(new_signatures):
            old_batch = old_by_signature.get(signature)
            if old_batch is None:
                continue
            old_inputs = _unique_input_labels(old_topology, old_batch)
            new_inputs = _unique_input_labels(new_topology, new_batch)
            old_positions = {node: i for i, node in enumerate(old_inputs)}
            new_positions = {node: i for i, node in enumerate(new_inputs)}
            for node, old_i in old_positions.items():
                if node not in new_positions:
                    continue
                new_i = new_positions[node]
                params[new_batch] = params[new_batch].at[:, new_i, -1].set(
                    self.params[old_batch][:, old_i, -1]
                )
                for other, old_j in old_positions.items():
                    if other in new_positions:
                        new_j = new_positions[other]
                        params[new_batch] = params[new_batch].at[:, new_i, new_j].set(
                            self.params[old_batch][:, old_i, old_j]
                        )
        return TopoSelfAttention(params=tuple(params), weight_scale=self.weight_scale)


class NeuronSelfAttention(Op):
    params: tuple[Array, ...] = ()
    weight_scale: float = eqx.field(static=True, default=0.1)
    name: str = eqx.field(static=True, default="neuron_self_attention")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.NODE)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=True)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def init(self, topology: CompiledTopology, *, key: Array) -> NeuronSelfAttention:
        keys = jr.split(key, max(1, len(topology.batches)))
        params = []
        for i, batch in enumerate(topology.batches):
            shape = (batch.size, 3, batch.max_inputs, batch.max_inputs + 1)
            params.append(jr.normal(keys[i], shape) * self.weight_scale)
        return NeuronSelfAttention(params=tuple(params), weight_scale=self.weight_scale)

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        if ctx.batch.max_inputs == 0:
            return ctx
        params = self.params[ctx.batch_index]
        counts = jnp.maximum(jnp.sum(ctx.input_mask, axis=1), 1.0)

        def apply_one(row_params, row_inputs, row_mask, count):
            query_params, key_params, value_params = row_params
            query = query_params[:, :-1] @ row_inputs + query_params[:, -1]
            key_values = key_params[:, :-1] @ row_inputs + key_params[:, -1]
            value = value_params[:, :-1] @ row_inputs + value_params[:, -1]
            invalid = ~jnp.outer(row_mask.astype(bool), row_mask.astype(bool))
            logits = jnp.outer(query, key_values) * jax_rsqrt(count)
            attention = jnn.softmax(jnp.where(invalid, -jnp.inf, logits))
            return (attention @ value + row_inputs) * row_mask

        input_values = vmap(apply_one)(params, ctx.input_values, ctx.input_mask, counts)
        edge_input_values = ctx.edge_input_values
        if ctx.edge_input_values.size:
            edge_input_values = input_values[
                ctx.edge_target_positions, ctx.edge_input_positions
            ]
        return replace(ctx, input_values=input_values, edge_input_values=edge_input_values)

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: NeuronSelfAttention,
    ) -> NeuronSelfAttention:
        params = list(initialized.params)
        for node, old_node_pos in old_topology.node_position.items():
            if node not in new_topology.node_position:
                continue
            old_batch, old_target = old_node_pos
            new_batch, new_target = new_topology.node_position[node]
            old_inputs = _row_input_labels(old_topology, old_batch, old_target)
            new_inputs = _row_input_labels(new_topology, new_batch, new_target)
            old_positions = {input_node: i for i, input_node in enumerate(old_inputs)}
            new_positions = {input_node: i for i, input_node in enumerate(new_inputs)}
            for input_node, old_i in old_positions.items():
                if input_node not in new_positions:
                    continue
                new_i = new_positions[input_node]
                params[new_batch] = params[new_batch].at[
                    new_target, :, new_i, -1
                ].set(self.params[old_batch][old_target, :, old_i, -1])
                for other_node, old_j in old_positions.items():
                    if other_node in new_positions:
                        new_j = new_positions[other_node]
                        params[new_batch] = params[new_batch].at[
                            new_target, :, new_i, new_j
                        ].set(self.params[old_batch][old_target, :, old_i, old_j])
        return NeuronSelfAttention(params=tuple(params), weight_scale=self.weight_scale)


class Activation(Op):
    activation: Callable = _identity
    name: str = eqx.field(static=True, default="activation")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.GLOBAL)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        assert ctx.outputs is not None
        if ctx.batch.all_targets_output:
            return ctx
        activated = self.activation(ctx.outputs)
        if ctx.batch.has_output_targets:
            outputs = jnp.where(ctx.target_is_output, ctx.outputs, activated)
        else:
            outputs = activated
        return replace(ctx, outputs=outputs)

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: Activation,
    ) -> Activation:
        return self


class AdaptiveActivation(Op):
    activation: Callable = _identity
    params: tuple[Array, ...] = ()
    name: str = eqx.field(static=True, default="adaptive_activation")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.NODE)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def init(self, topology: CompiledTopology, *, key: Array) -> AdaptiveActivation:
        keys = jr.split(key, max(1, len(topology.batches)))
        params = []
        for i, batch in enumerate(topology.batches):
            values = jnp.ones((batch.size, 2))
            if batch.size:
                values = values + jr.normal(keys[i], values.shape) * 0.1
            params.append(values)
        return AdaptiveActivation(activation=self.activation, params=tuple(params))

    def apply(
        self, ctx: ForwardContext, *, state: Any = None, key: Array | None = None
    ) -> ForwardContext:
        assert ctx.outputs is not None
        if ctx.batch.all_targets_output:
            return ctx
        params = self.params[ctx.batch_index]
        a, b = params[:, 0], params[:, 1]
        activated = self.activation(ctx.outputs * b) * a
        if ctx.batch.has_output_targets:
            outputs = jnp.where(ctx.target_is_output, ctx.outputs, activated)
        else:
            outputs = activated
        return replace(ctx, outputs=outputs)

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: AdaptiveActivation,
    ) -> AdaptiveActivation:
        params = _transfer_node_params(
            self.params, initialized.params, old_topology, new_topology
        )
        return AdaptiveActivation(activation=self.activation, params=params)


class Dropout(Op):
    dropout: Any = eqx.field(static=True, default=None)
    dropout_probs: tuple[float, ...] = eqx.field(static=True, default=())
    active: bool = eqx.field(static=True, default=False)
    name: str = eqx.field(static=True, default="dropout")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.GLOBAL)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def init(self, topology: CompiledTopology, *, key: Array) -> Dropout:
        probs = (
            _dropout_probs(topology, self.dropout)
            if self.dropout is not None
            else topology.dropout_probs
        )
        probs = tuple(float(p) for p in probs)
        return Dropout(
            dropout=self.dropout,
            dropout_probs=probs,
            active=any(p > 0 for p in probs),
        )

    def init_runtime(self, topology: CompiledTopology, *, key: Array | None) -> Array | None:
        if not self.active:
            return None
        probs = jnp.asarray(self.dropout_probs, dtype=float)
        if key is None:
            raise ValueError("Dropout requires an explicit `key`.")
        return jr.uniform(key, probs.shape) > probs

    def apply_inputs(
        self,
        topology: CompiledTopology,
        values: Array,
        *,
        state: Array | None = None,
        key: Array | None = None,
    ) -> Array:
        if not self.active:
            return values
        assert state is not None
        input_ids = jnp.asarray(topology.input_ids, dtype=int)
        if values.ndim == 1:
            return values.at[input_ids].set(values[input_ids] * state[input_ids])
        return values.at[:, input_ids].set(values[:, input_ids] * state[input_ids])

    def apply(
        self, ctx: ForwardContext, *, state: Array | None = None, key: Array | None = None
    ) -> ForwardContext:
        if not self.active:
            return ctx
        assert ctx.outputs is not None
        assert state is not None
        if ctx.outputs.ndim == 1:
            return replace(ctx, outputs=ctx.outputs * state[ctx.target_ids])
        return replace(ctx, outputs=ctx.outputs * state[ctx.target_ids])

    def with_dropout(self, dropout: DropoutLike) -> Dropout:
        return Dropout(dropout=_freeze_dropout(dropout))


class FusedDefaultOp(Op):
    affine: Op = eqx.field(default_factory=HybridEdgeAffine)
    activation: Callable = _identity
    dropout: Any = eqx.field(static=True, default=None)
    dropout_probs: tuple[float, ...] = eqx.field(static=True, default=())
    active: bool = eqx.field(static=True, default=False)
    name: str = eqx.field(static=True, default="fused_default")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.CUSTOM)
    stage: str = eqx.field(static=True, default="batch")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def init(self, topology: CompiledTopology, *, key: Array) -> FusedDefaultOp:
        affine_key, _ = jr.split(key)
        affine = self.affine.init(topology, key=affine_key)
        probs = (
            _dropout_probs(topology, self.dropout)
            if self.dropout is not None
            else topology.dropout_probs
        )
        probs = tuple(float(p) for p in probs)
        return FusedDefaultOp(
            affine=affine,
            activation=self.activation,
            dropout=self.dropout,
            dropout_probs=probs,
            active=any(p > 0 for p in probs),
        )

    def init_runtime(self, topology: CompiledTopology, *, key: Array | None) -> Array | None:
        if not self.active:
            return None
        probs = jnp.asarray(self.dropout_probs, dtype=float)
        if key is None:
            raise ValueError("Dropout requires an explicit `key`.")
        return jr.uniform(key, probs.shape) > probs

    def apply_inputs(
        self,
        topology: CompiledTopology,
        values: Array,
        *,
        state: Array | None = None,
        key: Array | None = None,
    ) -> Array:
        if not self.active:
            return values
        assert state is not None
        input_ids = jnp.asarray(topology.input_ids, dtype=int)
        if values.ndim == 1:
            return values.at[input_ids].set(values[input_ids] * state[input_ids])
        return values.at[:, input_ids].set(values[:, input_ids] * state[input_ids])

    def needs_batch_inputs_for_batch(self, batch_index: int) -> bool:
        return self.affine.needs_batch_inputs_for_batch(batch_index)

    def needs_padded_inputs_for_batch(self, batch_index: int) -> bool:
        return self.affine.needs_padded_inputs_for_batch(batch_index)

    def needs_edge_inputs_for_batch(self, batch_index: int) -> bool:
        return self.affine.needs_edge_inputs_for_batch(batch_index)

    def apply(
        self, ctx: ForwardContext, *, state: Array | None = None, key: Array | None = None
    ) -> ForwardContext:
        ctx = self.affine.apply(ctx, state=state, key=key)
        assert ctx.outputs is not None
        if not ctx.batch.all_targets_output:
            activated = self.activation(ctx.outputs)
            if ctx.batch.has_output_targets:
                outputs = jnp.where(ctx.target_is_output, ctx.outputs, activated)
            else:
                outputs = activated
            ctx = replace(ctx, outputs=outputs)
        if self.active:
            assert state is not None
            ctx = replace(ctx, outputs=ctx.outputs * state[ctx.target_ids])
        return ctx

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: FusedDefaultOp,
    ) -> FusedDefaultOp:
        affine = self.affine.transfer(
            old_topology, new_topology, initialized.affine
        )
        return FusedDefaultOp(
            affine=affine,
            activation=self.activation,
            dropout=initialized.dropout,
            dropout_probs=initialized.dropout_probs,
            active=initialized.active,
        )

    def with_dropout(self, dropout: DropoutLike) -> FusedDefaultOp:
        return FusedDefaultOp(
            affine=self.affine,
            activation=self.activation,
            dropout=_freeze_dropout(dropout),
        )


class OutputTransform(Op):
    transform: Callable = _identity
    name: str = eqx.field(static=True, default="output_transform")
    layout: ParamLayout = eqx.field(static=True, default=ParamLayout.GLOBAL)
    stage: str = eqx.field(static=True, default="output")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def apply_output(
        self,
        topology: CompiledTopology,
        y: Array,
        *,
        state: Any = None,
        key: Array | None = None,
    ) -> Array:
        return self.transform(y)

    def transfer(
        self,
        old_topology: CompiledTopology,
        new_topology: CompiledTopology,
        initialized: OutputTransform,
    ) -> OutputTransform:
        return self


def default_ops(
    *,
    affine: str = "hybrid",
    activation: Callable = jnn.gelu,
    output_transform: Callable = _identity,
    dropout: DropoutLike | None = None,
    fused: bool = False,
    topo_norm: bool = False,
    topo_self_attention: bool = False,
    neuron_self_attention: bool = False,
    adaptive_activation: bool = False,
) -> tuple[Op, ...]:
    ops: list[Op] = []
    if topo_norm:
        ops.append(TopoNorm())
    if topo_self_attention:
        ops.append(TopoSelfAttention())
    if neuron_self_attention:
        ops.append(NeuronSelfAttention())
    affine_op = _affine_op(affine)
    if fused and not ops and not adaptive_activation:
        ops.append(
            FusedDefaultOp(
                affine=affine_op,
                activation=activation,
                dropout=_freeze_dropout(dropout),
            )
        )
        ops.append(OutputTransform(transform=output_transform))
        return tuple(ops)
    ops.append(affine_op)
    if adaptive_activation:
        ops.append(AdaptiveActivation(activation=activation))
    else:
        ops.append(Activation(activation=activation))
    ops.append(Dropout(dropout=_freeze_dropout(dropout)))
    ops.append(OutputTransform(transform=output_transform))
    return tuple(ops)


def _affine_op(affine: str) -> Op:
    if affine == "padded":
        return EdgeAffine()
    if affine == "sparse":
        return SparseEdgeAffine()
    if affine == "matmul":
        return DenseMatmulAffine()
    if affine == "hybrid":
        return HybridEdgeAffine()
    raise ValueError("`affine` must be 'padded', 'sparse', 'matmul', or 'hybrid'.")


def jax_rsqrt(x):
    return jnp.asarray(x) ** -0.5


def _use_sparse_batch(batch: BatchTopology, padding_ratio_threshold: float) -> bool:
    padded_slots = batch.size * batch.max_inputs
    if batch.num_edges == 0:
        return False
    return padded_slots / batch.num_edges >= padding_ratio_threshold


def _use_matmul_batch(batch: BatchTopology) -> bool:
    dense_slots = batch.size * batch.unique_size
    if dense_slots == 0:
        return False
    if batch.num_edges != dense_slots:
        return False
    return batch.size > 1 and batch.unique_size >= 96


def _hybrid_affine_mode(
    batch: BatchTopology, padding_ratio_threshold: float
) -> str:
    if _use_sparse_batch(batch, padding_ratio_threshold):
        return "sparse"
    if _use_matmul_batch(batch):
        return "matmul"
    return "padded"


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


def _freeze_dropout(dropout: DropoutLike | None):
    if dropout is None:
        return None
    if isinstance(dropout, Real):
        probability = float(dropout)
        if not 0 <= probability <= 1:
            raise ValueError("Dropout probabilities must be in [0, 1].")
        return probability
    if isinstance(dropout, Mapping):
        items = []
        for node, probability in dropout.items():
            probability = float(probability)
            if not 0 <= probability <= 1:
                raise ValueError("Dropout probabilities must be in [0, 1].")
            items.append((node, probability))
        return tuple(items)
    raise TypeError("`dropout` must be a float, mapping, or None.")


def _dropout_probs(topology: CompiledTopology, dropout: Any) -> tuple[float, ...]:
    if isinstance(dropout, tuple):
        dropout = dict(dropout)
    if isinstance(dropout, Mapping):
        return tuple(float(dropout.get(node, 0.0)) for node in topology.id_to_node)
    hidden = set(topology.hidden_ids)
    probability = float(dropout)
    return tuple(probability if i in hidden else 0.0 for i in range(topology.num_nodes))


def _batch_signatures(topology: CompiledTopology) -> list[frozenset[Any]]:
    return [
        frozenset(topology.node_label(node_id) for node_id in batch.target_ids)
        for batch in topology.batches
    ]


def _unique_input_labels(topology: CompiledTopology, batch_index: int) -> list[Any]:
    return [
        topology.node_label(node_id)
        for node_id in topology.batches[batch_index].unique_input_ids
    ]


def _row_input_labels(
    topology: CompiledTopology, batch_index: int, target_pos: int
) -> list[Any]:
    batch = topology.batches[batch_index]
    labels = []
    for input_id, has_edge in zip(batch.input_ids[target_pos], batch.input_mask[target_pos]):
        if has_edge:
            labels.append(topology.node_label(input_id))
    return labels


def _transfer_node_params(
    old_params: tuple[Array, ...],
    new_params: tuple[Array, ...],
    old_topology: CompiledTopology,
    new_topology: CompiledTopology,
) -> tuple[Array, ...]:
    params = list(new_params)
    updates_by_batch: dict[int, list[tuple[int, Array]]] = {}
    for node, old_position in old_topology.node_position.items():
        if node in new_topology.node_position:
            old_batch, old_pos = old_position
            new_batch, new_pos = new_topology.node_position[node]
            updates_by_batch.setdefault(new_batch, []).append(
                (new_pos, old_params[old_batch][old_pos])
            )
    for batch_index, updates in updates_by_batch.items():
        positions, values = zip(*updates)
        params[batch_index] = params[batch_index].at[
            jnp.asarray(positions, dtype=int)
        ].set(jnp.stack(values))
    return tuple(params)


def _transfer_topo_input_params(
    old_params: tuple[Array, ...],
    new_params: tuple[Array, ...],
    old_topology: CompiledTopology,
    new_topology: CompiledTopology,
) -> tuple[Array, ...]:
    params = list(new_params)
    old_signatures = _batch_signatures(old_topology)
    new_signatures = _batch_signatures(new_topology)
    old_by_signature = {signature: i for i, signature in enumerate(old_signatures)}

    for new_batch, signature in enumerate(new_signatures):
        old_batch = old_by_signature.get(signature)
        if old_batch is None:
            continue
        old_inputs = _unique_input_labels(old_topology, old_batch)
        new_inputs = _unique_input_labels(new_topology, new_batch)
        old_positions = {node: i for i, node in enumerate(old_inputs)}
        for new_pos, node in enumerate(new_inputs):
            if node in old_positions:
                params[new_batch] = params[new_batch].at[new_pos].set(
                    old_params[old_batch][old_positions[node]]
                )
    return tuple(params)
