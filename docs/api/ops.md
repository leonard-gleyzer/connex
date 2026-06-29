# Operations

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
