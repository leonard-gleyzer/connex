# Connex

Connex is a JAX library built on Equinox for trainable neural networks whose
topology is defined by a directed acyclic graph.

## Quickstart

```python
import connex as cnx
import jax
import jax.numpy as jnp
import jax.random as jr

graph = {
    0: [1, 2, 3],
    1: [4],
    2: [4, 5],
    4: [6],
    5: [7],
    6: [8, 9],
    7: [10],
    8: [11],
    9: [11],
    10: [11],
}

spec = cnx.GraphSpec(graph, inputs=[0], outputs=[3, 11])
model = cnx.NeuralDAG(
    spec,
    ops=cnx.ops.default_ops(activation=jax.nn.relu),
    key=jr.key(0),
)

y = model(jnp.array([1.0]))
```

The default affine backend is hybrid: each compiled topological batch chooses
padded rows, sparse edge accumulation, or dense matmul based on its structure.
Long one-input chain segments use an automatic `lax.scan` execution plan when
the operation stack supports it. Scan segments only write graph outputs and
nodes consumed outside the segment back into the model value buffer. Use
`cnx.ops.default_ops(affine="padded")`, `cnx.ops.default_ops(affine="sparse")`, or
`cnx.ops.default_ops(affine="matmul")` to force a backend while benchmarking.

## Editing Topology

```python
model = (
    cnx.edit(model)
    .add_edges([(1, 6), (2, 11)])
    .remove_nodes([9])
    .set_dropout(0.1)
    .build(key=jr.key(2))
)
```

When dropout is nonzero, pass an explicit JAX random key to the forward call.
For batched evaluation, split the key and map it alongside the inputs.

## Custom Operations

Connex models are composed from operation objects. Built-in operations live in
`connex.ops`, and user-defined operations can subclass `connex.ops.Op`.

```python
class MyOp(cnx.ops.Op):
    def apply(self, ctx, *, state=None, key=None):
        ...
```
