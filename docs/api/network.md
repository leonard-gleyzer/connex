# Model

## Execution Notes

`NeuralDAG` compiles the `GraphSpec` into topological batches. The default
operation stack uses a structured execution plan:

- ordinary graph regions run one compiled topological batch at a time;
- compatible one-input chain regions run as `jax.lax.scan` segments;
- scan segments scatter only live-out values, meaning graph outputs and nodes
  consumed outside the segment;
- custom operations, attention features, normalization, and adaptive activation
  use the generic batch path.

Use `NeuralDAG.batched(x)` for native batched evaluation when every item in the
batch can share the same runtime state. For stochastic per-example dropout,
split keys and use `jax.vmap(model)` over `(x_i, key_i)`.

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
