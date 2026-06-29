# Topology Editing

Connex topology editing is the library's artificial-plasticity layer. The graph
can grow new edges, prune existing edges, add or remove neurons, and change
node-wise dropout without mutating the original model.

Use `connex.edit(model)` to start an edit session. The editor accumulates
changes on a pending graph description. Calling `build(...)` validates the new
`GraphSpec`, recompiles topology metadata, initializes any new parameters, and
transfers compatible parameters from the original model.

```python
import connex as cnx
import jax.random as jr

new_model = (
    cnx.edit(model)
    .add_edges([("h0", "h2")])
    .remove_edges([("h1", "y")])
    .set_dropout({"h2": 0.1})
    .build(key=jr.key(1))
)
```

The original `model` is unchanged. The returned model is a regular
`NeuralDAG`, so it can be trained, edited again, exported to NetworkX, or passed
through Equinox transformations.

## Parameter Transfer

Built-in operations preserve parameters by graph label:

- edge parameters are retained for edges that still exist;
- node parameters and biases are retained for nodes that still exist;
- new edges and nodes are initialized from the key passed to `build(...)`;
- removed structure drops its parameters;
- custom operations can opt into their own transfer behavior by overriding
  `Op.transfer(...)`.

Labels are therefore important. If you remove a node and add a new node with a
different label, Connex treats it as new structure. If you preserve the label
across an edit, compatible built-in state follows it.

## Adding And Removing Edges

Edges can be supplied as `(source, target)` pairs or as an adjacency mapping.
Nodes must already exist. Cycles are rejected when `build(...)` constructs the
new `GraphSpec`.

```python
model = (
    cnx.edit(model)
    .add_edges([("x0", "h3"), ("h3", "y")])
    .remove_edges({"h1": ["y"]})
    .build(key=jr.key(2))
)
```

Removing an edge does not remove its incident nodes. Isolated hidden nodes stay
in the graph until you remove them explicitly.

## Adding Nodes

Hidden nodes are inserted before output nodes in the stored topological order.
Input nodes are prepended to the input order, and output nodes are appended to
the output order.

```python
model = (
    cnx.edit(model)
    .add_hidden_nodes(["h_new"])
    .add_edges([("x0", "h_new"), ("h_new", "y")])
    .build(key=jr.key(3))
)
```

Adding input or output nodes changes the shape contract of the model. A new
input node consumes an additional leading input value. A new output node appears
at the end of the returned output array.

```python
model = (
    cnx.edit(model)
    .add_input_nodes(["x_extra"])
    .add_output_nodes(["aux"])
    .add_edges([("x_extra", "h0"), ("h0", "aux")])
    .build(key=jr.key(4))
)
```

## Removing Nodes

Removing nodes also removes all incident edges. If a removed node was an input
or output, the model's input or output order is updated. Mapping-style dropout
entries for removed nodes are discarded.

```python
model = (
    cnx.edit(model)
    .remove_nodes(["h_old"])
    .build(key=jr.key(5))
)
```

## Dropout Editing

`set_dropout(...)` updates the rebuilt model's graph-level dropout
configuration and any built-in dropout operations in the current operation
pipeline.

```python
model = cnx.edit(model).set_dropout(0.05).build(key=jr.key(6))
```

Scalar dropout applies to hidden nodes. Mapping dropout can target any node and
defaults unspecified nodes to zero.

## Reference

::: connex.edit

---

::: connex.TopologyEditor
    options:
        members:
            - add_edges
            - remove_edges
            - add_input_nodes
            - add_hidden_nodes
            - add_output_nodes
            - remove_nodes
            - set_dropout
            - build
