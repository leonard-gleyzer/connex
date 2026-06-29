<h1 align='center'>Connex</h1>

Connex is a [JAX](https://github.com/jax-ml/jax) library built on
[Equinox](https://github.com/patrick-kidger/equinox) for trainable neural
networks whose topology is defined by a directed acyclic graph.

With Connex, you can:

- Compile a DAG into a trainable Equinox module.
- Compose built-in or user-defined graph operations.
- Add and remove connections or neurons while preserving compatible parameters.
- Use explicit JAX random keys for stochastic behavior such as dropout.
- Export a trained model to a NetworkX weighted digraph.

## Installation

```bash
pip install connex
```

## Usage

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
padded predecessor rows, sparse edge accumulation, or dense matmul based on the
batch structure. Long one-input chain segments are collapsed into a `lax.scan`
execution plan automatically when the operation stack supports it. Scan segments
only write values back for graph outputs or nodes consumed outside the segment.
Custom operations and richer feature operations fall back to the generic
topological batch path. You can force a backend when benchmarking a specific
graph family:

```python
model = cnx.NeuralDAG(
    spec,
    ops=cnx.ops.default_ops(affine="sparse", activation=jax.nn.relu),
    key=jr.key(0),
)
```

Dropout is explicit-key only:

```python
model = cnx.NeuralDAG(
    cnx.GraphSpec(graph, inputs=[0], outputs=[3, 11], dropout=0.1),
    key=jr.key(0),
)
y = model(jnp.array([1.0]), key=jr.key(1))
```

Topology edits go through the editor API:

```python
model = (
    cnx.edit(model)
    .add_edges([(1, 6), (2, 11)])
    .remove_nodes([9])
    .set_dropout(0.1)
    .build(key=jr.key(2))
)
```

Custom operations can participate in the same pipeline:

```python
class MyOp(cnx.ops.Op):
    def apply(self, ctx, *, state=None, key=None):
        ...
```

Prebuilt graph constructors are available under `connex.nn`:

```python
model = cnx.nn.MLP(2, 1, width=32, depth=3, key=jr.key(0))
```

## Citation

```bibtex
@software{gleyzer2023connex,
  author = {Leonard Gleyzer},
  title = {{C}onnex: Fine-grained Control over Neural Network Topology in {JAX}},
  url = {http://github.com/leonard-gleyzer/connex},
  year = {2023},
}
```
