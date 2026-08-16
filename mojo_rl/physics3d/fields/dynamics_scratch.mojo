"""Owned scratch tensors for the dynamics pipeline (migration P2).

`DynamicsScratch` is the stateful replacement for the "integrator temps"
section of the flat workspace slab (`ws_*_offset` in gpu/constants.mojo):
one owned `TensorImpl` per logical scratch array, allocated once by whoever
runs the pipeline (the stateful integrator / a test harness) and reused
every step — no slab, no `ws_*` offsets, no caller-provided workspace
buffer. Host side doubles as the CPU-target storage for the single-source
kernels (they run over `.lt["cpu"]` views of the same tensors).

Solver-specific workspace (M_inv, Newton solver arrays, RK4 stage extras)
is NOT here — it lands with the stateful solver/integrator structs that own
it (later P2 slices).
"""

from max.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl

from .dims import DimsLike


struct DynamicsScratch[
    DTYPE: DType,
    # ⚠ NAMED `DIMS`, NOT `D` LIKE EVERY OTHER CONTAINER, because this struct
    # already has a FIELD called `D` — the LDL diagonal, `[BATCH, NV]` — and
    # the two collide in the struct's own scope ("invalid redefinition of
    # 'D'"). The parameter is passed POSITIONALLY at all 79 call sites, so its
    # name is invisible outside this file; `scratch.D` is read across the
    # whole LDL and solver path and renaming THAT would be the wide change.
    DIMS: DimsLike,
    BATCH: Int = 1,
](Movable):
    """Integrator-temps scratch: one owned tensor per array (12 tensors:
    the `integrator_workspace_size` inventory + m_inv for constraint
    solving)."""

    # Body unchanged — see `Rk4Scratch`.
    comptime NV = Self.DIMS.NV
    comptime NBODY = Self.DIMS.NBODY

    comptime L_CDOF = Layout.row_major(Self.BATCH, Self.NV * 6)
    comptime L_CRB = Layout.row_major(Self.BATCH, Self.NBODY * 10)
    comptime L_M = Layout.row_major(Self.BATCH, Self.NV * Self.NV)
    comptime L_NV = Layout.row_major(Self.BATCH, Self.NV)
    comptime L_B6 = Layout.row_major(Self.BATCH, Self.NBODY * 6)

    var cdof: TensorImpl[Self.DTYPE]  # [BATCH, NV*6]
    var crb: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*10]
    var M: TensorImpl[Self.DTYPE]  # [BATCH, NV*NV]
    var L: TensorImpl[Self.DTYPE]  # [BATCH, NV*NV]
    var D: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var bias: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var fnet: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var qacc_ws: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var qacc_constrained: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var rne_cacc: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    var rne_cfrc: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    var m_inv: TensorImpl[Self.DTYPE]  # [BATCH, NV*NV] (constraint solving)

    def __init__(out self) raises:
        comptime B = Self.BATCH
        self.cdof = TensorImpl[Self.DTYPE].alloc(B * Self.NV * 6)
        self.crb = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 10)
        self.M = TensorImpl[Self.DTYPE].alloc(B * Self.NV * Self.NV)
        self.L = TensorImpl[Self.DTYPE].alloc(B * Self.NV * Self.NV)
        self.D = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.bias = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.fnet = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.qacc_ws = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.qacc_constrained = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.rne_cacc = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 6)
        self.rne_cfrc = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 6)
        self.m_inv = TensorImpl[Self.DTYPE].alloc(B * Self.NV * Self.NV)

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Create device buffers for every scratch tensor (once, at setup —
        contents are produced on-device thereafter)."""
        self.cdof.upload(ctx)
        self.crb.upload(ctx)
        self.M.upload(ctx)
        self.L.upload(ctx)
        self.D.upload(ctx)
        self.bias.upload(ctx)
        self.fnet.upload(ctx)
        self.qacc_ws.upload(ctx)
        self.qacc_constrained.upload(ctx)
        self.rne_cacc.upload(ctx)
        self.rne_cfrc.upload(ctx)
        self.m_inv.upload(ctx)
