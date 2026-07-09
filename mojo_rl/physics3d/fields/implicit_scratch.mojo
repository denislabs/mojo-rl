"""Owned scratch tensors for the RNE velocity derivative / implicit integrator
(migration P2 / Stage-I).

`ImplicitScratch` is the stateful replacement for the `ws_implicit_*` region
of the legacy workspace slab (`implicit_extra_workspace_size` in
gpu/constants.mojo): one owned `TensorImpl` per intermediate array of
`compute_rne_vel_derivative`, allocated once by `ImplicitIntegratorFields`
and reused every step.

These are stored as tensors (not per-thread InlineArrays) on PURPOSE: the
big cross-body intermediates are `NBODY*6*NV` / `NV*6*NV` elements — for the
humanoid (NV≈27, NBODY≈14) that is thousands of floats per env, which as
GPU-thread-local InlineArrays would blow local memory (the RK4-ELLIPTIC OOM
lesson). Holding them in device global memory lets one per-env function
serve both the CPU and GPU targets, bit-identically.

Only the persistent cross-body arrays live here; the tiny 6×6 / 6-vector
loop temporaries stay as InlineArrays inside the kernel (GPU-safe).
"""

from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl


@always_inline
def _pos(n: Int) -> Int:
    return n if n > 0 else 1


struct ImplicitScratch[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int = 1,
](Movable):
    """RNE-velocity-derivative scratch: one owned tensor per `ws_implicit_*`
    region (9 tensors)."""

    var cinert: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*10]
    var cdof_sc: TensorImpl[Self.DTYPE]  # [BATCH, NV*6]
    var cvel_sc: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    var cdof_dot: TensorImpl[Self.DTYPE]  # [BATCH, NV*6]
    var dcvel: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6*NV]
    var dcdofdot: TensorImpl[Self.DTYPE]  # [BATCH, NV*6*NV]
    var dcacc: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6*NV]
    var dcfrcbody: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6*NV]
    var qderiv: TensorImpl[Self.DTYPE]  # [BATCH, NV*NV]

    def __init__(out self) raises:
        comptime B = Self.BATCH
        self.cinert = TensorImpl[Self.DTYPE].alloc(_pos(B * Self.NBODY * 10))
        self.cdof_sc = TensorImpl[Self.DTYPE].alloc(_pos(B * Self.NV * 6))
        self.cvel_sc = TensorImpl[Self.DTYPE].alloc(_pos(B * Self.NBODY * 6))
        self.cdof_dot = TensorImpl[Self.DTYPE].alloc(_pos(B * Self.NV * 6))
        self.dcvel = TensorImpl[Self.DTYPE].alloc(
            _pos(B * Self.NBODY * 6 * Self.NV)
        )
        self.dcdofdot = TensorImpl[Self.DTYPE].alloc(
            _pos(B * Self.NV * 6 * Self.NV)
        )
        self.dcacc = TensorImpl[Self.DTYPE].alloc(
            _pos(B * Self.NBODY * 6 * Self.NV)
        )
        self.dcfrcbody = TensorImpl[Self.DTYPE].alloc(
            _pos(B * Self.NBODY * 6 * Self.NV)
        )
        self.qderiv = TensorImpl[Self.DTYPE].alloc(_pos(B * Self.NV * Self.NV))

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Create device buffers for every scratch tensor (once, at setup —
        contents are produced on-device thereafter)."""
        self.cinert.upload(ctx)
        self.cdof_sc.upload(ctx)
        self.cvel_sc.upload(ctx)
        self.cdof_dot.upload(ctx)
        self.dcvel.upload(ctx)
        self.dcdofdot.upload(ctx)
        self.dcacc.upload(ctx)
        self.dcfrcbody.upload(ctx)
        self.qderiv.upload(ctx)
