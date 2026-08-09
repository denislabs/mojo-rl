"""Persistent device + host scratch for the EZv2 GPU unroll train steps.

Lives in its own module (no cross-package re-imports) so the structs export
cleanly via absolute import paths. Allocated **once** (via ``make``) and reused
every train step — the prior per-step ``enqueue_create_buffer`` in
``blocks.mojo`` / ``blocks_continuous.mojo`` exploded disk on NVIDIA and added
allocation latency. ``PROJ`` is ``PROJM.OUT_DIM`` (passed explicitly since a
struct can't derive it from a Module type param).

Device buffers are owned storage ``Tensor`` fields (like muzero's ``MZScratch``)
so the storage ``forward(TensorRefs[Tensor])`` / ``.lt`` device views work; the
``h_*`` host mirrors stay ``HostBuffer`` (host staging for zero-fill + D2H).
"""

from max.gpu.host import DeviceContext, HostBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor


struct EZV2UnrollScratch[
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    PROJ: Int,
](Movable & Deinitable):
    """Discrete EZv2 unroll scratch — one field per per-step GPU buffer."""
    comptime PRED_OUT = Self.ACT + Self.BINS
    comptime DYN_IN = Self.LATENT + Self.ACT
    comptime DYN_OUT = Self.LATENT + Self.BINS

    # H2D host batch slabs
    var d_obs: Tensor
    var d_act: Tensor
    var d_pol: Tensor
    var d_val: Tensor
    var d_rew: Tensor
    # device scratch
    var d_zst: Tensor
    # work tiles for sub-slab forwards (storage forward needs a whole-Tensor
    # input/output; sub-slabs of d_obs/d_zst bridge through these via copy kernels)
    var d_obs_work: Tensor   # [B, OBS] — one obs-seq position at a time
    var z_work: Tensor       # [B, LATENT] — rep/dyn forward output
    var zk_work: Tensor      # [B, LATENT] — reverse-scan forward input
    var d_din: Tensor
    var d_dout: Tensor
    var d_pout: Tensor
    var d_gpout: Tensor
    var d_gdout: Tensor
    var d_gz: Tensor
    var d_gpin: Tensor
    var d_gdin: Tensor
    var d_gobs: Tensor
    var d_twv: Tensor
    var d_twr: Tensor
    var d_loss: Tensor
    # consistency scratch
    var d_tstore: Tensor
    var d_ztmp: Tensor
    var d_projo: Tensor
    var d_pk: Tensor
    var d_gpk: Tensor
    var d_gproj: Tensor
    var d_gzcons: Tensor
    # consistency episode-boundary mask [K, B]
    var d_cmask: Tensor
    var h_cmask_ones: Optional[HostBuffer[DT]]
    # host loss mirror (zero-fill + D2H reduce)
    var h_loss: Optional[HostBuffer[DT]]
    # PER: per-sample IS weights [B] (H2D) + per-sample priority [B] (D2H)
    var d_isw: Tensor
    var d_prio: Tensor
    var h_prio: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.d_obs = Tensor(); self.d_act = Tensor(); self.d_pol = Tensor()
        self.d_val = Tensor(); self.d_rew = Tensor()
        self.d_zst = Tensor()
        self.d_obs_work = Tensor(); self.z_work = Tensor(); self.zk_work = Tensor()
        self.d_din = Tensor(); self.d_dout = Tensor()
        self.d_pout = Tensor(); self.d_gpout = Tensor(); self.d_gdout = Tensor()
        self.d_gz = Tensor(); self.d_gpin = Tensor(); self.d_gdin = Tensor()
        self.d_gobs = Tensor(); self.d_twv = Tensor(); self.d_twr = Tensor()
        self.d_loss = Tensor()
        self.d_tstore = Tensor(); self.d_ztmp = Tensor(); self.d_projo = Tensor()
        self.d_pk = Tensor(); self.d_gpk = Tensor(); self.d_gproj = Tensor()
        self.d_gzcons = Tensor()
        self.d_cmask = Tensor(); self.h_cmask_ones = None
        self.h_loss = None
        self.d_isw = Tensor(); self.d_prio = Tensor(); self.h_prio = None

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
        s.d_obs = Tensor.alloc_gpu(ctx, (k + 1) * b * obs)
        s.d_act = Tensor.alloc_gpu(ctx, k * b)
        s.d_pol = Tensor.alloc_gpu(ctx, (k + 1) * b * act)
        s.d_val = Tensor.alloc_gpu(ctx, (k + 1) * b)
        s.d_rew = Tensor.alloc_gpu(ctx, k * b)
        s.d_zst = Tensor.alloc_gpu(ctx, (k + 1) * b * lat)
        s.d_obs_work = Tensor.alloc_gpu(ctx, b * obs)
        s.z_work = Tensor.alloc_gpu(ctx, b * lat)
        s.zk_work = Tensor.alloc_gpu(ctx, b * lat)
        s.d_din = Tensor.alloc_gpu(ctx, b * din)
        s.d_dout = Tensor.alloc_gpu(ctx, b * dout)
        s.d_pout = Tensor.alloc_gpu(ctx, b * pred)
        s.d_gpout = Tensor.alloc_gpu(ctx, b * pred)
        s.d_gdout = Tensor.alloc_gpu(ctx, b * dout)
        s.d_gz = Tensor.alloc_gpu(ctx, b * lat)
        s.d_gpin = Tensor.alloc_gpu(ctx, b * lat)
        s.d_gdin = Tensor.alloc_gpu(ctx, b * din)
        s.d_gobs = Tensor.alloc_gpu(ctx, b * obs)
        s.d_twv = Tensor.alloc_gpu(ctx, b * bins)
        s.d_twr = Tensor.alloc_gpu(ctx, b * bins)
        # 4 contiguous [B] blocks: policy | value | reward | consistency.
        s.d_loss = Tensor.alloc_gpu(ctx, 4 * b)
        s.d_tstore = Tensor.alloc_gpu(ctx, k * b * proj)
        s.d_ztmp = Tensor.alloc_gpu(ctx, b * lat)
        s.d_projo = Tensor.alloc_gpu(ctx, b * proj)
        s.d_pk = Tensor.alloc_gpu(ctx, b * proj)
        s.d_gpk = Tensor.alloc_gpu(ctx, b * proj)
        s.d_gproj = Tensor.alloc_gpu(ctx, b * proj)
        s.d_gzcons = Tensor.alloc_gpu(ctx, b * lat)
        # consistency boundary mask [K, B]; the host mirror stays all-ones for
        # callers that pass no mask (≡ the unmasked pre-mask behaviour).
        s.d_cmask = Tensor.alloc_gpu(ctx, k * b)
        s.h_cmask_ones = ctx.enqueue_create_host_buffer[DT](k * b)
        s.h_loss = ctx.enqueue_create_host_buffer[DT](4 * b)
        s.d_isw = Tensor.alloc_gpu(ctx, b)
        s.d_prio = Tensor.alloc_gpu(ctx, b)
        s.h_prio = ctx.enqueue_create_host_buffer[DT](b)
        ctx.synchronize()
        for i in range(k * b):
            s.h_cmask_ones.value().unsafe_ptr()[unsafe_offset=i] = Scalar[DT](1.0)
        return s^


struct EZV2UnrollContScratch[
    B: Int,
    K: Int,
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
    PROJ: Int,
](Movable & Deinitable):
    """Continuous EZv2 unroll scratch. The continuous GPU policy loss is a fused
    kernel, so no ``musig``/``ptgt`` scratch is needed (unlike the CPU path).
    Buffer set otherwise mirrors the discrete scratch."""
    comptime MU2 = 2 * Self.ACT_DIM
    comptime PRED_OUT = Self.MU2 + Self.BINS
    comptime DYN_IN = Self.LATENT + Self.ACT_DIM
    comptime DYN_OUT = Self.LATENT + Self.BINS

    # H2D host batch slabs
    var d_obs: Tensor
    var d_act: Tensor
    var d_pol: Tensor
    var d_val: Tensor
    var d_rew: Tensor
    # device scratch
    var d_zst: Tensor
    # work tiles for sub-slab forwards (see discrete scratch).
    var d_obs_work: Tensor   # [B, OBS]
    var z_work: Tensor       # [B, LATENT]
    var zk_work: Tensor      # [B, LATENT]
    var d_din: Tensor
    var d_dout: Tensor
    var d_pout: Tensor
    var d_gpout: Tensor
    var d_gdout: Tensor
    var d_gz: Tensor
    var d_gpin: Tensor
    var d_gdin: Tensor
    var d_gobs: Tensor
    var d_twv: Tensor
    var d_twr: Tensor
    var d_loss: Tensor
    # consistency scratch
    var d_tstore: Tensor
    var d_ztmp: Tensor
    var d_projo: Tensor
    var d_pk: Tensor
    var d_gpk: Tensor
    var d_gproj: Tensor
    var d_gzcons: Tensor
    # consistency episode-boundary mask [K, B] + all-ones host fallback
    var d_cmask: Tensor
    var h_cmask_ones: Optional[HostBuffer[DT]]
    # host loss mirror (zero-fill + D2H reduce)
    var h_loss: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.d_obs = Tensor(); self.d_act = Tensor(); self.d_pol = Tensor()
        self.d_val = Tensor(); self.d_rew = Tensor()
        self.d_zst = Tensor()
        self.d_obs_work = Tensor(); self.z_work = Tensor(); self.zk_work = Tensor()
        self.d_din = Tensor(); self.d_dout = Tensor()
        self.d_pout = Tensor(); self.d_gpout = Tensor(); self.d_gdout = Tensor()
        self.d_gz = Tensor(); self.d_gpin = Tensor(); self.d_gdin = Tensor()
        self.d_gobs = Tensor(); self.d_twv = Tensor(); self.d_twr = Tensor()
        self.d_loss = Tensor()
        self.d_tstore = Tensor(); self.d_ztmp = Tensor(); self.d_projo = Tensor()
        self.d_pk = Tensor(); self.d_gpk = Tensor(); self.d_gproj = Tensor()
        self.d_gzcons = Tensor()
        self.d_cmask = Tensor(); self.h_cmask_ones = None
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
        s.d_obs = Tensor.alloc_gpu(ctx, (k + 1) * b * obs)
        s.d_act = Tensor.alloc_gpu(ctx, k * b * adim)
        s.d_pol = Tensor.alloc_gpu(ctx, (k + 1) * b * adim)
        s.d_val = Tensor.alloc_gpu(ctx, (k + 1) * b)
        s.d_rew = Tensor.alloc_gpu(ctx, k * b)
        s.d_zst = Tensor.alloc_gpu(ctx, (k + 1) * b * lat)
        s.d_obs_work = Tensor.alloc_gpu(ctx, b * obs)
        s.z_work = Tensor.alloc_gpu(ctx, b * lat)
        s.zk_work = Tensor.alloc_gpu(ctx, b * lat)
        s.d_din = Tensor.alloc_gpu(ctx, b * din)
        s.d_dout = Tensor.alloc_gpu(ctx, b * dout)
        s.d_pout = Tensor.alloc_gpu(ctx, b * pred)
        s.d_gpout = Tensor.alloc_gpu(ctx, b * pred)
        s.d_gdout = Tensor.alloc_gpu(ctx, b * dout)
        s.d_gz = Tensor.alloc_gpu(ctx, b * lat)
        s.d_gpin = Tensor.alloc_gpu(ctx, b * lat)
        s.d_gdin = Tensor.alloc_gpu(ctx, b * din)
        s.d_gobs = Tensor.alloc_gpu(ctx, b * obs)
        s.d_twv = Tensor.alloc_gpu(ctx, b * bins)
        s.d_twr = Tensor.alloc_gpu(ctx, b * bins)
        # 4 contiguous [B] blocks: policy | value | reward | consistency.
        s.d_loss = Tensor.alloc_gpu(ctx, 4 * b)
        s.d_tstore = Tensor.alloc_gpu(ctx, k * b * proj)
        s.d_ztmp = Tensor.alloc_gpu(ctx, b * lat)
        s.d_projo = Tensor.alloc_gpu(ctx, b * proj)
        s.d_pk = Tensor.alloc_gpu(ctx, b * proj)
        s.d_gpk = Tensor.alloc_gpu(ctx, b * proj)
        s.d_gproj = Tensor.alloc_gpu(ctx, b * proj)
        s.d_gzcons = Tensor.alloc_gpu(ctx, b * lat)
        # consistency boundary mask [K, B]; the host mirror stays all-ones for
        # callers that pass no mask (≡ the unmasked pre-mask behaviour).
        s.d_cmask = Tensor.alloc_gpu(ctx, k * b)
        s.h_cmask_ones = ctx.enqueue_create_host_buffer[DT](k * b)
        s.h_loss = ctx.enqueue_create_host_buffer[DT](4 * b)
        ctx.synchronize()
        for i in range(k * b):
            s.h_cmask_ones.value().unsafe_ptr()[unsafe_offset=i] = Scalar[DT](1.0)
        return s^
