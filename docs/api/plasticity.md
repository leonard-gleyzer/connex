# Topology Editing

Topology edits are performed with `connex.edit(model)`, which returns a
`TopologyEditor`. Calling `build(...)` recompiles the graph and transfers
compatible parameters.

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
