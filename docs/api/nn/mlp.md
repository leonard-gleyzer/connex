# MLP

`connex.nn.MLP` is a convenience constructor for a standard fully connected
layered graph. It is a subclass of `NeuralDAG`, not a separate runtime, so the
result can be trained, edited, exported, and customized with the same APIs as a
hand-written `GraphSpec`.

```python
import connex as cnx
import jax
import jax.random as jr

model = cnx.nn.MLP(
    input_size=2,
    output_size=1,
    width=32,
    depth=3,
    activation=jax.nn.gelu,
    dropout=0.05,
    key=jr.key(0),
)
```

The generated graph uses integer node labels:

- input nodes are `0 .. input_size - 1`;
- hidden nodes are placed layer by layer after the inputs;
- output nodes are the final `output_size` labels;
- edges connect each layer only to the next layer.

The output order is the generated output-node order. The input shape is
`(input_size,)` for one example and `(batch, input_size)` for `model.batched`.

```python
y = model(x)
ys = model.batched(xs)
```

Because the model is still a `NeuralDAG`, you can keep the initial MLP as a
starting topology and then specialize it:

```python
model = (
    cnx.edit(model)
    .add_edges([(0, model.spec.outputs[0])])
    .set_dropout(0.1)
    .build(key=jr.key(1))
)
```

Pass `ops=...` to replace the default operation pipeline entirely. When `ops`
is supplied, the `activation` and `output_transform` arguments are only used if
your custom operations use them.

## Reference

::: connex.nn.MLP
    options:
        members:
            - __init__
