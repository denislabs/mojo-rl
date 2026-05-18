"""nn2/ Phase 1 constants.

DT is pinned to float32 for Phase 1. Phase 3 (AMP) introduces per-layer
`compute_dtype` / `accum_dtype` / `param_dtype` and AMPPolicy on DiffOps.
"""

comptime DT = DType.float32
