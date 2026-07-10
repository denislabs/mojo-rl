"""Owned RK4 stage tensors for the per-field tensor pipeline (P2).

`Rk4Scratch` is the stateful replacement for the RK4-extra region of the
flat workspace slab (`rk4_extra_workspace_size` = NQ + 7*NV, addressed via
`ws_rk4_*_offset` in gpu/constants.mojo): one owned `TensorImpl` per
logical region, allocated once by `RK4Integrator` and reused every
step. Region inventory mirrors the legacy layout exactly:

    q0 (NQ)  — qpos saved at stage 0 (all stages integrate FROM q0)
    v0 (NV)  — qvel saved at stage 0 (C[0] = v0)
    A  (4*NV in legacy; A0/A1/A2 here) — per-stage constrained qacc.
              The legacy A[3] slot is allocated but never written: the
              combine kernel reads A[3] straight from qacc_constrained,
              and the combine REUSES the A[0] slot to hold v_combined —
              so A0 doubles as the combine's velocity buffer here too.
    C1 (NV)  — stage-2 velocity intermediate: v0 + dt/2 * A[0]
    C2 (NV)  — stage-3 velocity intermediate: v0 + dt/2 * A[1]
"""

from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl


struct Rk4Scratch[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    BATCH: Int = 1,
](Movable):
    """RK4 stage scratch: one owned tensor per legacy `ws_rk4_*` region
    (the unused legacy A[3] slot is not materialized)."""

    comptime L_Q0 = Layout.row_major(Self.BATCH, Self.NQ)
    comptime L_NV = Layout.row_major(Self.BATCH, Self.NV)

    var q0: TensorImpl[Self.DTYPE]  # [BATCH, NQ]
    var v0: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var A0: TensorImpl[Self.DTYPE]  # [BATCH, NV] (also v_combined in combine)
    var A1: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var A2: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var C1: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var C2: TensorImpl[Self.DTYPE]  # [BATCH, NV]

    def __init__(out self) raises:
        comptime B = Self.BATCH
        self.q0 = TensorImpl[Self.DTYPE].alloc(B * Self.NQ)
        self.v0 = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.A0 = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.A1 = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.A2 = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.C1 = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.C2 = TensorImpl[Self.DTYPE].alloc(B * Self.NV)

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Create device buffers for every scratch tensor (once, at setup —
        contents are produced on-device thereafter)."""
        self.q0.upload(ctx)
        self.v0.upload(ctx)
        self.A0.upload(ctx)
        self.A1.upload(ctx)
        self.A2.upload(ctx)
        self.C1.upload(ctx)
        self.C2.upload(ctx)
