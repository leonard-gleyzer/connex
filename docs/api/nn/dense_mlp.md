# DenseMLP

`connex.nn.DenseMLP` builds a densely connected layered DAG. Each layer connects
to every later layer, so later hidden nodes and outputs can reuse all earlier
representations instead of only the immediately preceding layer.

```python
import connex as cnx
import jax
import jax.random as jr

model = cnx.nn.DenseMLP(
    input_size=4,
    output_size=2,
    width=16,
    depth=3,
    activation=jax.nn.silu,
    key=jr.key(0),
)
```

Like `MLP`, this is a regular `NeuralDAG` subclass with generated integer node
labels:

- inputs come first;
- hidden layers follow in order;
- outputs are the final nodes;
- every node in a non-final layer connects to all nodes in later layers.

This graph family is useful when you want skip connectivity without writing the
graph by hand. It increases the number of edges compared with `MLP`, so it is a
good candidate for the hybrid affine backend: compact regions use padded rows,
sparse regions use edge accumulation, and wide complete regions can use dense
matmul.

```python
model = cnx.nn.DenseMLP(
    4,
    2,
    width=16,
    depth=3,
    ops=cnx.ops.default_ops(affine="hybrid", fused=True),
    key=jr.key(1),
)
```

You can still edit the generated topology. For example, remove a skip edge,
add a new hidden node, or export the learned weights for graph analysis:

```python
model = cnx.edit(model).remove_edges([(0, model.spec.outputs[0])]).build(key=jr.key(2))
weighted = model.to_networkx_weighted_digraph()
```

## Reference

::: connex.nn.DenseMLP
    options:
        members:
            - __init__
