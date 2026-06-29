# Model

Connex separates a model into two pieces:

- `GraphSpec` is the immutable graph description: the directed acyclic graph,
  ordered input nodes, ordered output nodes, topological order, and dropout
  configuration.
- `NeuralDAG` is the trainable Equinox module compiled from that specification
  and an operation pipeline.

This split is intentional. Graph structure is static metadata from JAX's point
of view, while trainable arrays live inside operation objects. The result works
with `eqx.filter_jit`, `eqx.filter_value_and_grad`, `eqx.apply_updates`, and
ordinary PyTree filtering.

## Graph Specifications

`GraphSpec` accepts a `networkx.DiGraph` or anything accepted by
`networkx.DiGraph(...)`, including adjacency dictionaries and edge lists.

```python
import connex as cnx
import networkx as nx

graph = nx.DiGraph()
graph.add_edges_from(
    [
        ("x0", "h0"),
        ("x1", "h0"),
        ("h0", "y"),
    ]
)

spec = cnx.GraphSpec(
    graph,
    inputs=["x0", "x1"],
    outputs=["y"],
    topo_sort=["x0", "x1", "h0", "y"],
    dropout={"h0": 0.1},
)
```

The input and output orders are semantic. A forward call places `x[i]` at
`spec.inputs[i]`, and returns outputs in `spec.outputs` order.

Connex validates the structure before parameters are initialized:

- the graph must be acyclic;
- input nodes must exist and have no incoming edges;
- output nodes must exist and have no outgoing edges;
- no node may be both an input and an output;
- an explicit `topo_sort`, if supplied, must contain exactly the graph nodes
  and respect every edge;
- dropout probabilities must be in `[0, 1]`.

Supplying `topo_sort` is optional. It is useful when you want deterministic
placement of isolated hidden nodes or when you already have a topological order
from another graph construction step.

## Runtime Model

`NeuralDAG` compiles the graph into integer node ids, edge-position tables, and
topological batches. A topological batch is a group of target nodes whose
predecessor values are already available. During a forward pass, Connex gathers
only the input layouts requested by the active operations, runs the operation
pipeline for each batch, and writes the new values back into a dense value
buffer.

```python
import jax
import jax.numpy as jnp
import jax.random as jr

model = cnx.NeuralDAG(
    spec,
    ops=cnx.ops.default_ops(activation=jax.nn.relu),
    key=jr.key(0),
)

y = model(jnp.array([1.0, -1.0]))
```

If `ops` is omitted, Connex uses `connex.ops.default_ops()`: a hybrid affine
operation, hidden-node activation, dropout, and output transform. Custom
operation sequences use the same `NeuralDAG` runtime.

## Batching

Use `model.batched(x)` when every item in a batch can share the same runtime
state. This is the native batched path and avoids mapping over the whole model.

```python
x = jnp.ones((32, 2))
y = model.batched(x)
```

For independent stochastic state per example, split keys and map over both
inputs and keys:

```python
keys = jr.split(jr.key(1), x.shape[0])
y = jax.vmap(lambda x_i, key_i: model(x_i, key=key_i))(x, keys)
```

This distinction matters for dropout. `model.batched(x, key=key)` uses one
runtime dropout state for the batch; `jax.vmap(...)` with split keys gives each
example its own mask.

## Execution Planning

The runtime squeezes structure out of the compiled topology before execution:

- ordinary graph regions run one compiled topological batch at a time;
- compatible one-input chain regions run as `jax.lax.scan` segments;
- scan segments scatter only live-out values, meaning graph outputs and nodes
  consumed outside the segment;
- operation input layouts are prepared only when some operation declares that it
  needs them;
- the hybrid affine backend chooses padded rows, sparse edge accumulation, or
  dense matmul per topological batch.

Custom operations, attention features, normalization, and adaptive activation
use the generic batch path. That path is more general because Connex cannot
assume the local semantics required to fuse them into scan segments.

## Exporting Weights

`to_networkx_weighted_digraph()` returns a copy of the original graph with
learned edge weights attached when a built-in affine operation is present.

```python
weighted = model.to_networkx_weighted_digraph()
weight = weighted["x0"]["h0"]["weight"]
```

This is useful for downstream NetworkX analysis, visualization, pruning
experiments, or debugging topology edits.

## Reference

::: connex.GraphSpec
    options:
        members:
            - __init__
            - dropout_by_node
            - with_graph
            - with_dropout

---

::: connex.NeuralDAG
    options:
        members:
            - __init__
            - __call__
            - batched
            - rebuild
            - to_networkx_weighted_digraph
