"""Constants for nn2/ Phase 1.

DT is pinned to float32 for Phase 1. Phase 3 (AMP) introduces per-layer
`compute_dtype` / `accum_dtype` / `param_dtype` and AMPPolicy on DiffOps.

CPU_SIMD_W is the SIMD lane count for `DT` on the host CPU. Used by
hand-rolled SIMD elementwise paths (ReLU/Tanh/MSE/elementwise) since
Mojo nightly does not autovectorize scalar `ptr[i]` loops — manual
`while i + W <= N: ptr.load[width=W]` gives 3-5x over the scalar form.
See feedback_mojo_cpu_manual_simd_required.
"""

from std.sys import simd_width_of

comptime DT = DType.float32
comptime CPU_SIMD_W = simd_width_of[DT]()
