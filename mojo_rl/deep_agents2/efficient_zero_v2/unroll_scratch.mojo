"""Persistent device + host scratch for the EZv2 GPU unroll train steps.

Lives in its own module (no cross-package re-imports) so the structs export
cleanly via absolute import paths. Allocated **once** (via ``make``) and reused
every train step — the prior per-step ``enqueue_create_buffer`` in
``blocks.mojo`` / ``blocks_continuous.mojo`` exploded disk on NVIDIA and added
allocation latency. ``PROJ`` is ``PROJM.OUT_DIM`` (passed explicitly since a
struct can't derive it from a Module type param).
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn2.constants import DT


struct EZV2UnrollScratch[
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    PROJ: Int,
](Movable & ImplicitlyDestructible):
    """Discrete EZv2 unroll scratch — one field per per-step GPU buffer."""
    comptime PRED_OUT = Self.ACT + Self.BINS
    comptime DYN_IN = Self.LATENT + Self.ACT
    comptime DYN_OUT = Self.LATENT + Self.BINS

    # H2D host batch slabs
    var d_obs: Optional[DeviceBuffer[DT]]
    var d_act: Optional[DeviceBuffer[DT]]
    var d_pol: Optional[DeviceBuffer[DT]]
    var d_val: Optional[DeviceBuffer[DT]]
    var d_rew: Optional[DeviceBuffer[DT]]
    # device scratch
    var d_zst: Optional[DeviceBuffer[DT]]
    var d_din: Optional[DeviceBuffer[DT]]
    var d_dout: Optional[DeviceBuffer[DT]]
    var d_pout: Optional[DeviceBuffer[DT]]
    var d_gpout: Optional[DeviceBuffer[DT]]
    var d_gdout: Optional[DeviceBuffer[DT]]
    var d_gz: Optional[DeviceBuffer[DT]]
    var d_gpin: Optional[DeviceBuffer[DT]]
    var d_gdin: Optional[DeviceBuffer[DT]]
    var d_gobs: Optional[DeviceBuffer[DT]]
    var d_twv: Optional[DeviceBuffer[DT]]
    var d_twr: Optional[DeviceBuffer[DT]]
    var d_loss: Optional[DeviceBuffer[DT]]
    # consistency scratch
    var d_tstore: Optional[DeviceBuffer[DT]]
    var d_ztmp: Optional[DeviceBuffer[DT]]
    var d_projo: Optional[DeviceBuffer[DT]]
    var d_pk: Optional[DeviceBuffer[DT]]
    var d_gpk: Optional[DeviceBuffer[DT]]
    var d_gproj: Optional[DeviceBuffer[DT]]
    var d_gzcons: Optional[DeviceBuffer[DT]]
    # host loss mirror (zero-fill + D2H reduce)
    var h_loss: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.d_obs = None; self.d_act = None; self.d_pol = None
        self.d_val = None; self.d_rew = None
        self.d_zst = None; self.d_din = None; self.d_dout = None
        self.d_pout = None; self.d_gpout = None; self.d_gdout = None
        self.d_gz = None; self.d_gpin = None; self.d_gdin = None
        self.d_gobs = None; self.d_twv = None; self.d_twr = None
        self.d_loss = None
        self.d_tstore = None; self.d_ztmp = None; self.d_projo = None
        self.d_pk = None; self.d_gpk = None; self.d_gproj = None
        self.d_gzcons = None
        self.h_loss = None

    @staticmethod
    def make(ctx: DeviceContext) raises -> Self:
        comptime b = Self.B
        comptime k = Self.K
        comptime obs = Self.OBS
        comptime act = Self.ACT
        comptime lat = Self.LATENT
        comptime bins = Self.BINS
        comptime proj = Self.PROJ
        comptime din = Self.DYN_IN
        comptime dout = Self.DYN_OUT
        comptime pred = Self.PRED_OUT
        var s = Self()
        s.d_obs = ctx.enqueue_create_buffer[DT]((k + 1) * b * obs)
        s.d_act = ctx.enqueue_create_buffer[DT](k * b)
        s.d_pol = ctx.enqueue_create_buffer[DT]((k + 1) * b * act)
        s.d_val = ctx.enqueue_create_buffer[DT]((k + 1) * b)
        s.d_rew = ctx.enqueue_create_buffer[DT](k * b)
        s.d_zst = ctx.enqueue_create_buffer[DT]((k + 1) * b * lat)
        s.d_din = ctx.enqueue_create_buffer[DT](b * din)
        s.d_dout = ctx.enqueue_create_buffer[DT](b * dout)
        s.d_pout = ctx.enqueue_create_buffer[DT](b * pred)
        s.d_gpout = ctx.enqueue_create_buffer[DT](b * pred)
        s.d_gdout = ctx.enqueue_create_buffer[DT](b * dout)
        s.d_gz = ctx.enqueue_create_buffer[DT](b * lat)
        s.d_gpin = ctx.enqueue_create_buffer[DT](b * lat)
        s.d_gdin = ctx.enqueue_create_buffer[DT](b * din)
        s.d_gobs = ctx.enqueue_create_buffer[DT](b * obs)
        s.d_twv = ctx.enqueue_create_buffer[DT](b * bins)
        s.d_twr = ctx.enqueue_create_buffer[DT](b * bins)
        # 4 contiguous [B] blocks: policy | value | reward | consistency.
        s.d_loss = ctx.enqueue_create_buffer[DT](4 * b)
        s.d_tstore = ctx.enqueue_create_buffer[DT](k * b * proj)
        s.d_ztmp = ctx.enqueue_create_buffer[DT](b * lat)
        s.d_projo = ctx.enqueue_create_buffer[DT](b * proj)
        s.d_pk = ctx.enqueue_create_buffer[DT](b * proj)
        s.d_gpk = ctx.enqueue_create_buffer[DT](b * proj)
        s.d_gproj = ctx.enqueue_create_buffer[DT](b * proj)
        s.d_gzcons = ctx.enqueue_create_buffer[DT](b * lat)
        s.h_loss = ctx.enqueue_create_host_buffer[DT](4 * b)
        ctx.synchronize()
        return s^


struct EZV2UnrollContScratch[
    B: Int,
    K: Int,
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
    PROJ: Int,
](Movable & ImplicitlyDestructible):
    """Continuous EZv2 unroll scratch. The continuous GPU policy loss is a fused
    kernel, so no ``musig``/``ptgt`` scratch is needed (unlike the CPU path).
    Buffer set otherwise mirrors the discrete scratch."""
    comptime MU2 = 2 * Self.ACT_DIM
    comptime PRED_OUT = Self.MU2 + Self.BINS
    comptime DYN_IN = Self.LATENT + Self.ACT_DIM
    comptime DYN_OUT = Self.LATENT + Self.BINS

    # H2D host batch slabs
    var d_obs: Optional[DeviceBuffer[DT]]
    var d_act: Optional[DeviceBuffer[DT]]
    var d_pol: Optional[DeviceBuffer[DT]]
    var d_val: Optional[DeviceBuffer[DT]]
    var d_rew: Optional[DeviceBuffer[DT]]
    # device scratch
    var d_zst: Optional[DeviceBuffer[DT]]
    var d_din: Optional[DeviceBuffer[DT]]
    var d_dout: Optional[DeviceBuffer[DT]]
    var d_pout: Optional[DeviceBuffer[DT]]
    var d_gpout: Optional[DeviceBuffer[DT]]
    var d_gdout: Optional[DeviceBuffer[DT]]
    var d_gz: Optional[DeviceBuffer[DT]]
    var d_gpin: Optional[DeviceBuffer[DT]]
    var d_gdin: Optional[DeviceBuffer[DT]]
    var d_gobs: Optional[DeviceBuffer[DT]]
    var d_twv: Optional[DeviceBuffer[DT]]
    var d_twr: Optional[DeviceBuffer[DT]]
    var d_loss: Optional[DeviceBuffer[DT]]
    # consistency scratch
    var d_tstore: Optional[DeviceBuffer[DT]]
    var d_ztmp: Optional[DeviceBuffer[DT]]
    var d_projo: Optional[DeviceBuffer[DT]]
    var d_pk: Optional[DeviceBuffer[DT]]
    var d_gpk: Optional[DeviceBuffer[DT]]
    var d_gproj: Optional[DeviceBuffer[DT]]
    var d_gzcons: Optional[DeviceBuffer[DT]]
    # host loss mirror (zero-fill + D2H reduce)
    var h_loss: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.d_obs = None; self.d_act = None; self.d_pol = None
        self.d_val = None; self.d_rew = None
        self.d_zst = None; self.d_din = None; self.d_dout = None
        self.d_pout = None; self.d_gpout = None; self.d_gdout = None
        self.d_gz = None; self.d_gpin = None; self.d_gdin = None
        self.d_gobs = None; self.d_twv = None; self.d_twr = None
        self.d_loss = None
        self.d_tstore = None; self.d_ztmp = None; self.d_projo = None
        self.d_pk = None; self.d_gpk = None; self.d_gproj = None
        self.d_gzcons = None
        self.h_loss = None

    @staticmethod
    def make(ctx: DeviceContext) raises -> Self:
        comptime b = Self.B
        comptime k = Self.K
        comptime obs = Self.OBS
        comptime adim = Self.ACT_DIM
        comptime lat = Self.LATENT
        comptime bins = Self.BINS
        comptime proj = Self.PROJ
        comptime din = Self.DYN_IN
        comptime dout = Self.DYN_OUT
        comptime pred = Self.PRED_OUT
        var s = Self()
        s.d_obs = ctx.enqueue_create_buffer[DT]((k + 1) * b * obs)
        s.d_act = ctx.enqueue_create_buffer[DT](k * b * adim)
        s.d_pol = ctx.enqueue_create_buffer[DT]((k + 1) * b * adim)
        s.d_val = ctx.enqueue_create_buffer[DT]((k + 1) * b)
        s.d_rew = ctx.enqueue_create_buffer[DT](k * b)
        s.d_zst = ctx.enqueue_create_buffer[DT]((k + 1) * b * lat)
        s.d_din = ctx.enqueue_create_buffer[DT](b * din)
        s.d_dout = ctx.enqueue_create_buffer[DT](b * dout)
        s.d_pout = ctx.enqueue_create_buffer[DT](b * pred)
        s.d_gpout = ctx.enqueue_create_buffer[DT](b * pred)
        s.d_gdout = ctx.enqueue_create_buffer[DT](b * dout)
        s.d_gz = ctx.enqueue_create_buffer[DT](b * lat)
        s.d_gpin = ctx.enqueue_create_buffer[DT](b * lat)
        s.d_gdin = ctx.enqueue_create_buffer[DT](b * din)
        s.d_gobs = ctx.enqueue_create_buffer[DT](b * obs)
        s.d_twv = ctx.enqueue_create_buffer[DT](b * bins)
        s.d_twr = ctx.enqueue_create_buffer[DT](b * bins)
        # 4 contiguous [B] blocks: policy | value | reward | consistency.
        s.d_loss = ctx.enqueue_create_buffer[DT](4 * b)
        s.d_tstore = ctx.enqueue_create_buffer[DT](k * b * proj)
        s.d_ztmp = ctx.enqueue_create_buffer[DT](b * lat)
        s.d_projo = ctx.enqueue_create_buffer[DT](b * proj)
        s.d_pk = ctx.enqueue_create_buffer[DT](b * proj)
        s.d_gpk = ctx.enqueue_create_buffer[DT](b * proj)
        s.d_gproj = ctx.enqueue_create_buffer[DT](b * proj)
        s.d_gzcons = ctx.enqueue_create_buffer[DT](b * lat)
        s.h_loss = ctx.enqueue_create_host_buffer[DT](4 * b)
        ctx.synchronize()
        return s^
