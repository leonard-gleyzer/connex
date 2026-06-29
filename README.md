<h1 align='center'>Connex</h1>

Connex is a [JAX](https://github.com/jax-ml/jax) library built on
[Equinox](https://github.com/patrick-kidger/equinox) for trainable neural
networks whose topology is defined by a directed acyclic graph.

With Connex, you can:

- compile a DAG into a trainable Equinox module;
- compose built-in or user-defined graph operations;
- add and remove connections or neurons while preserving compatible parameters;
- set scalar or per-node dropout with explicit JAX random keys;
- use padded, sparse, matmul, or hybrid affine backends;
- export trained parameters to a NetworkX weighted digraph.

## Installation

```bash
uv add connex
```

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

`GraphSpec` validates the DAG and preserves input/output ordering. `NeuralDAG`
is an `equinox.Module`, so training uses standard Equinox and Optax patterns:

```python
import equinox as eqx
import optax

optim = optax.adam(1e-3)
opt_state = optim.init(eqx.filter(model, eqx.is_array))


@eqx.filter_value_and_grad
def loss_fn(model, x, y):
    pred = model.batched(x)
    return jnp.mean((pred - y) ** 2)
```

Topology edits go through the editor API. Edits return a new model and leave the
old one untouched:

```python
model = (
    cnx.edit(model)
    .add_edges([(1, 6), (2, 11)])
    .remove_nodes([9])
    .set_dropout(0.1)
    .build(key=jr.key(2))
)
```

The default affine backend is hybrid: each topological batch chooses padded
rows, sparse edge accumulation, or dense matmul based on graph structure. Long
one-input chain segments use an automatic `jax.lax.scan` execution plan when the
operation stack supports it.

Custom operations subclass `connex.ops.Op` and participate in the same pipeline:

```python
class MyOp(cnx.ops.Op):
    def apply(self, ctx, *, state=None, key=None):
        ...
```

Prebuilt graph constructors are available under `connex.nn`:

```python
model = cnx.nn.MLP(2, 1, width=32, depth=3, key=jr.key(0))
```

## Documentation

Full documentation lives at:

https://leonard-gleyzer.github.io/connex

## Citation

```bibtex
@software{gleyzer2023connex,
  author = {Leonard Gleyzer},
  title = {{C}onnex: Fine-grained Control over Neural Network Topology in {JAX}},
  url = {http://github.com/leonard-gleyzer/connex},
  year = {2023},
}
```
