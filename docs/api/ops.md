# Operations

Connex models are built from operation objects. An operation is an
`equinox.Module` that can own trainable parameters, declare which graph inputs
it needs, transform values during a forward pass, and transfer parameters across
topology edits.

The default pipeline is produced by `connex.ops.default_ops()`:

1. optional topology features, such as normalization or attention;
2. an affine operation;
3. hidden-node activation or adaptive activation;
4. dropout;
5. an output transform.

```python
import connex as cnx
import jax

ops = cnx.ops.default_ops(
    affine="hybrid",
    activation=jax.nn.gelu,
    topo_norm=True,
    dropout=0.1,
)

model = cnx.NeuralDAG(spec, ops=ops, key=key)
```

Operation templates are initialized by `NeuralDAG` against the compiled
topology. You normally create lightweight templates, pass them into the model,
and let the model allocate arrays with the supplied key.

## Forward Context

Batch-stage operations receive a `ForwardContext`. It contains the current
compiled topological batch, the full model value buffer, target ids, predecessor
ids, edge-position metadata, and any outputs produced by earlier operations.

The context exposes three gathered input views:

- `batch_inputs`: one value per unique predecessor in the topological batch;
- `input_values`: padded predecessor rows shaped like target nodes;
- `edge_input_values`: one value per real edge.

Operations declare which views they need through the `needs_batch_inputs`,
`needs_padded_inputs`, and `needs_edge_inputs` flags. Keeping these flags narrow
matters for performance because `NeuralDAG` avoids preparing unused layouts.

## Affine Backends

`default_ops()` uses `affine="hybrid"` unless configured otherwise. Hybrid
selects an affine implementation per topological batch:

- padded rows for compact predecessor layouts;
- sparse edge accumulation when padding would dominate;
- dense matmul for wide complete predecessor batches.

Force `affine="padded"`, `affine="sparse"`, or `affine="matmul"` when
benchmarking a known graph family. The scan execution plan supports all built-in
affine backends when the remaining stack is affine, activation, optional dropout,
and output transform, or the equivalent fused default op.

```python
ops = cnx.ops.default_ops(affine="sparse", activation=jax.nn.relu)
```

`EdgeAffine` stores padded rows and is usually best for compact batches.
`SparseEdgeAffine` stores one weight per real edge and avoids wasted padded
slots. `DenseMatmulAffine` stores a matrix over unique predecessors and is best
when a batch resembles a dense layer. `HybridEdgeAffine` makes that choice per
batch.

## Feature Operations

Feature operations run before the affine transform and modify the gathered
predecessor values:

- `TopoNorm` normalizes the unique predecessor set for each topological batch.
- `TopoSelfAttention` attends over each batch's unique predecessor values.
- `NeuronSelfAttention` attends separately over each target node's input row.

Activation operations run after affine outputs are available:

- `Activation` applies a fixed activation to hidden nodes only.
- `AdaptiveActivation` learns per-node activation scales.
- `Dropout` masks node values with explicit JAX random keys.
- `OutputTransform` transforms the final ordered output array.

## Fused Defaults

`default_ops(fused=True)` uses `FusedDefaultOp` when no feature operations or
adaptive activations are requested. This combines affine, activation, and
dropout into one operation and keeps the model eligible for the optimized scan
execution plan on one-input chain regions.

```python
ops = cnx.ops.default_ops(fused=True, affine="hybrid")
```

## Custom Operations

Subclass `Op` when you need behavior not covered by the built-ins. A custom op
can read the context, return an updated context, own trainable arrays, and
define transfer behavior for topology edits.

```python
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp


class BiasAfterAffine(cnx.ops.Op):
    bias: jax.Array
    name: str = eqx.field(static=True, default="bias_after_affine")
    needs_batch_inputs: bool = eqx.field(static=True, default=False)
    needs_padded_inputs: bool = eqx.field(static=True, default=False)
    needs_edge_inputs: bool = eqx.field(static=True, default=False)

    def apply(self, ctx, *, state=None, key=None):
        assert ctx.outputs is not None
        return replace(ctx, outputs=ctx.outputs + self.bias)


ops = (
    cnx.ops.EdgeAffine(),
    BiasAfterAffine(bias=jnp.asarray(0.1)),
    cnx.ops.Activation(activation=jax.nn.relu),
    cnx.ops.OutputTransform(),
)
```

Custom operations use the generic topological batch execution path. That path is
the right default for arbitrary user code because Connex cannot infer the same
scan and fusion guarantees it has for the built-in affine/activation/dropout
stack.

## Reference

::: connex.ops.Op

---

::: connex.ops.EdgeAffine

---

::: connex.ops.SparseEdgeAffine

---

::: connex.ops.DenseMatmulAffine

---

::: connex.ops.HybridEdgeAffine

---

::: connex.ops.TopoNorm

---

::: connex.ops.TopoSelfAttention

---

::: connex.ops.NeuronSelfAttention

---

::: connex.ops.Activation

---

::: connex.ops.AdaptiveActivation

---

::: connex.ops.Dropout

---

::: connex.ops.FusedDefaultOp

---

::: connex.ops.OutputTransform

---

::: connex.ops.default_ops
