"""MuZero K-step unroll — the world-model BPTT training step.

One training step over a batch of length-``K`` trajectory windows, structured as
the DreamerV3 ``WMStep`` manual forward-scan / reverse-scan (no monolithic
ComputeGraph — far more tractable through learned dynamics):

  forward scan   z₀ = h(obs₀);  zₖ₊₁ = g(zₖ, aₖ).latent      (store z₀..z_K)
  per position   f(zₖ) → (policy, value);  g(zₖ,aₖ) → reward
  losses         soft-CE(policy, π) + soft-CE(value, twohot(h·v))
                 + soft-CE(reward, twohot(h·r))               (all categorical)
  reverse scan   re-forward each net (to refresh its vjp cache), seed each head's
                 grad slice analytically, run ``Module.vjp``, thread the carry
                 gradient ``∂L/∂zₖ = ∂L_pred/∂zₖ + ½·∂L_dyn/∂zₖ`` back to k−1.

Two MuZero-specific gradients are baked in:
  * **½ scale on the dynamics hidden input** — "scale the gradient at the start
    of the dynamics function by ½" (MuZero appendix). Applied to the latent half
    of ``g``'s input-gradient; compounds naturally across unroll steps.
  * **1/(K+1) per-step loss weight** (legacy parity) folded into ``grad_scale``
    alongside the 1/BATCH mean.

Batch layout is **time-major** so every per-step slice is contiguous (no gather):
``obs0[B,OBS]``, ``actions[K,B]`` (indices), ``policy_tgt[K+1,B,ACT]``,
``value_tgt[K+1,B]`` and ``reward_tgt[K,B]`` (raw scalars; ``h`` + two-hot applied
here). ``v_min/v_max`` are the h-space support shared with the planner + targets.

CPU path first (validated by an overfit test); a GPU branch + CPU↔GPU bit-parity
follows. The min-max latent scaling lives inside the nets (`MZRepNet`/`MZDynNet`
``MinMaxNorm`` tails), so it is already in the autodiff graph — no separate scale.
"""

from std.memory import alloc
from std.math import exp, log, sqrt
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.module import Module, mptr
from mojo_rl.nn2.optimizer.adam import Adam

from .loss_ops import soft_ce_slice_loss_and_grad
from ..zero.twohot_targets import mz_two_hot_target_batch


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


@always_inline
def _dp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(b.unsafe_ptr())


@always_inline
def _lt[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


struct MZScratch[
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
](Movable & ImplicitlyDestructible):
    """Persistent device + host scratch for `mz_unroll_train_step_gpu`.

    Allocated **once** via `make` and reused every training step — per-step
    `enqueue_create_buffer` in the hot loop explodes disk on NVIDIA and adds
    allocation latency. Every buffer is fully overwritten each step before it
    is read (H2D copies of the batch slabs; the forward/reverse scans rewrite
    all scratch; `loss_d` is re-zeroed each step), so reuse is safe.
    """

    comptime PRED_OUT = Self.ACT + Self.BINS
    comptime DYN_IN = Self.LATENT + Self.ACT
    comptime DYN_OUT = Self.LATENT + Self.BINS

    var d_obs0: Optional[DeviceBuffer[DT]]
    var d_act: Optional[DeviceBuffer[DT]]
    var d_pol: Optional[DeviceBuffer[DT]]
    var d_val: Optional[DeviceBuffer[DT]]
    var d_rew: Optional[DeviceBuffer[DT]]
    var zst: Optional[DeviceBuffer[DT]]
    var din: Optional[DeviceBuffer[DT]]
    var dout: Optional[DeviceBuffer[DT]]
    var pout: Optional[DeviceBuffer[DT]]
    var gpout: Optional[DeviceBuffer[DT]]
    var gdout: Optional[DeviceBuffer[DT]]
    var gz: Optional[DeviceBuffer[DT]]
    var gpin: Optional[DeviceBuffer[DT]]
    var gdin: Optional[DeviceBuffer[DT]]
    var gobs: Optional[DeviceBuffer[DT]]
    var twv: Optional[DeviceBuffer[DT]]
    var twr: Optional[DeviceBuffer[DT]]
    var loss_d: Optional[DeviceBuffer[DT]]
    var h_zloss: Optional[HostBuffer[DT]]
    var h_loss: Optional[HostBuffer[DT]]
    # PER scratch: per-sample IS weights (H2D) + value-error priorities (D2H).
    var d_isw: Optional[DeviceBuffer[DT]]
    var d_prio: Optional[DeviceBuffer[DT]]
    var h_prio: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.d_obs0 = None; self.d_act = None; self.d_pol = None
        self.d_val = None; self.d_rew = None; self.zst = None
        self.din = None; self.dout = None; self.pout = None
        self.gpout = None; self.gdout = None; self.gz = None
        self.gpin = None; self.gdin = None; self.gobs = None
        self.twv = None; self.twr = None; self.loss_d = None
        self.h_zloss = None; self.h_loss = None
        self.d_isw = None; self.d_prio = None; self.h_prio = None

    @staticmethod
    def make(ctx: DeviceContext) raises -> Self:
        comptime PO = Self.PRED_OUT
        comptime DI = Self.DYN_IN
        comptime DO = Self.DYN_OUT
        comptime BB = Self.B
        var s = Self()
        s.d_obs0 = ctx.enqueue_create_buffer[DT](BB * Self.OBS)
        s.d_act = ctx.enqueue_create_buffer[DT](Self.K * BB)
        s.d_pol = ctx.enqueue_create_buffer[DT]((Self.K + 1) * BB * Self.ACT)
        s.d_val = ctx.enqueue_create_buffer[DT]((Self.K + 1) * BB)
        s.d_rew = ctx.enqueue_create_buffer[DT](Self.K * BB)
        s.zst = ctx.enqueue_create_buffer[DT]((Self.K + 1) * BB * Self.LATENT)
        s.din = ctx.enqueue_create_buffer[DT](BB * DI)
        s.dout = ctx.enqueue_create_buffer[DT](BB * DO)
        s.pout = ctx.enqueue_create_buffer[DT](BB * PO)
        s.gpout = ctx.enqueue_create_buffer[DT](BB * PO)
        s.gdout = ctx.enqueue_create_buffer[DT](BB * DO)
        s.gz = ctx.enqueue_create_buffer[DT](BB * Self.LATENT)
        s.gpin = ctx.enqueue_create_buffer[DT](BB * Self.LATENT)
        s.gdin = ctx.enqueue_create_buffer[DT](BB * DI)
        s.gobs = ctx.enqueue_create_buffer[DT](BB * Self.OBS)
        s.twv = ctx.enqueue_create_buffer[DT](BB * Self.BINS)
        s.twr = ctx.enqueue_create_buffer[DT](BB * Self.BINS)
        # 3 contiguous [B] blocks: policy | value | reward.
        s.loss_d = ctx.enqueue_create_buffer[DT](3 * BB)
        s.h_zloss = ctx.enqueue_create_host_buffer[DT](3 * BB)
        s.h_loss = ctx.enqueue_create_host_buffer[DT](3 * BB)
        s.d_isw = ctx.enqueue_create_buffer[DT](BB)
        s.d_prio = ctx.enqueue_create_buffer[DT](BB)
        s.h_prio = ctx.enqueue_create_host_buffer[DT](BB)
        ctx.synchronize()
        return s^


def mz_unroll_train_step_cpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
](
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    obs0: UnsafePointer[Scalar[DT], MutAnyOrigin],
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
    policy_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    loss_parts: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """One CPU MuZero unroll training step. Returns the mean total loss
    (policy + value + reward, summed over the K+1 / K positions then averaged
    over batch and unroll length). Mutates all three nets via their optimizers.
    """
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS

    # ── scratch ──
    var zst = _a((K + 1) * B * LATENT)   # stored latents z0..zK
    var din = _a(B * DYN_IN)
    var dout = _a(B * DYN_OUT)
    var pout = _a(B * PRED_OUT)
    var gpout = _a(B * PRED_OUT)          # grad wrt pred output
    var gdout = _a(B * DYN_OUT)           # grad wrt dyn output
    var gz = _a(B * LATENT)               # carry: grad wrt z_{k+1}
    var gpin = _a(B * LATENT)             # working grad wrt z_k
    var gdin = _a(B * DYN_IN)             # grad wrt dyn input
    var gobs = _a(B * OBS)                # grad wrt rep input (discarded)
    var twv = _a(B * BINS)
    var twr = _a(B * BINS)

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)

    # ── forward scan: rep then K dynamics steps, store every z ──
    var obs_t = TileTensor(obs0, row_major[B, OBS]())
    var z0_t = TileTensor(zst, row_major[B, LATENT]())
    rep.forward["cpu", B](obs_t, output=z0_t)

    for k in range(K):
        # build dyn input [z_k | onehot(a_k)]
        var zk = zst + k * B * LATENT
        for b in range(B):
            var dib = din + b * DYN_IN
            var zb = zk + b * LATENT
            for i in range(LATENT):
                dib[i] = zb[i]
            for a in range(ACT):
                dib[LATENT + a] = Scalar[DT](0.0)
            dib[LATENT + Int(actions[k * B + b])] = Scalar[DT](1.0)
        var din_t = TileTensor(din, row_major[B, DYN_IN]())
        var dout_t = TileTensor(dout, row_major[B, DYN_OUT]())
        dyn.forward["cpu", B](din_t, output=dout_t)
        # store next latent z_{k+1} = dyn_out[:, :LATENT]
        var znext = zst + (k + 1) * B * LATENT
        for b in range(B):
            for i in range(LATENT):
                znext[b * LATENT + i] = dout[b * DYN_OUT + i]

    # ── reverse scan: accumulate grads + loss ──
    orep.zero_grad["cpu", REP](rep)
    odyn.zero_grad["cpu", DYN](dyn)
    opred.zero_grad["cpu", PRED](pred)

    var loss = Scalar[DT](0.0)
    # per-component loss accumulators (for the optional loss_parts breakdown)
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    for rk in range(K + 1):
        var k = K - rk
        var zk = zst + k * B * LATENT
        var zk_t = TileTensor(zk, row_major[B, LATENT]())

        # (a) prediction head: re-forward for cache, seed grads, vjp → grad z_k
        var pout_t = TileTensor(pout, row_major[B, PRED_OUT]())
        pred.forward["cpu", B](zk_t, output=pout_t)
        # policy slice [0, ACT)
        var l_pol_k = soft_ce_slice_loss_and_grad[B, PRED_OUT, 0, ACT](
            pout, policy_tgt + k * B * ACT, gscale, gpout
        )
        loss += l_pol_k
        l_pol += l_pol_k
        # value slice [ACT, ACT+BINS)
        mz_two_hot_target_batch[B, BINS](value_tgt + k * B, v_min, v_max, twv)
        var l_val_k = value_coef * soft_ce_slice_loss_and_grad[
            B, PRED_OUT, ACT, BINS
        ](pout, twv, gscale * value_coef, gpout)
        loss += l_val_k
        l_val += l_val_k
        var gpout_t = TileTensor(gpout, row_major[B, PRED_OUT]())
        var gpin_t = TileTensor(gpin, row_major[B, LATENT]())
        pred.vjp["cpu", B](gpout_t, gpin_t)

        # (b) dynamics: carry grad from z_{k+1} + reward head, ½ on hidden input
        if k < K:
            # rebuild dyn input (mirror forward) for cache
            for b in range(B):
                var dib = din + b * DYN_IN
                var zb = zk + b * LATENT
                for i in range(LATENT):
                    dib[i] = zb[i]
                for a in range(ACT):
                    dib[LATENT + a] = Scalar[DT](0.0)
                dib[LATENT + Int(actions[k * B + b])] = Scalar[DT](1.0)
            var din_t = TileTensor(din, row_major[B, DYN_IN]())
            var dout_t = TileTensor(dout, row_major[B, DYN_OUT]())
            dyn.forward["cpu", B](din_t, output=dout_t)
            # grad_dyn_out = [ carry(grad z_{k+1}) | reward grad ]
            for b in range(B):
                for i in range(LATENT):
                    gdout[b * DYN_OUT + i] = gz[b * LATENT + i]
            mz_two_hot_target_batch[B, BINS](
                reward_tgt + k * B, v_min, v_max, twr
            )
            var l_rew_k = soft_ce_slice_loss_and_grad[B, DYN_OUT, LATENT, BINS](
                dout, twr, gscale, gdout
            )
            loss += l_rew_k
            l_rew += l_rew_k
            var gdout_t = TileTensor(gdout, row_major[B, DYN_OUT]())
            var gdin_t = TileTensor(gdin, row_major[B, DYN_IN]())
            dyn.vjp["cpu", B](gdout_t, gdin_t)
            # ∂L/∂z_k += ½ · (grad into dyn's latent input)
            for b in range(B):
                for i in range(LATENT):
                    gpin[b * LATENT + i] += (
                        Scalar[DT](0.5) * gdin[b * DYN_IN + i]
                    )

        # carry ← full grad wrt z_k for the next (k-1) iteration
        for b in range(B):
            for i in range(LATENT):
                gz[b * LATENT + i] = gpin[b * LATENT + i]

    # ── rep: grad wrt z_0 (== carry after the loop) → rep params ──
    var gz0_t = TileTensor(gz, row_major[B, LATENT]())
    var gobs_t = TileTensor(gobs, row_major[B, OBS]())
    rep.vjp["cpu", B](gz0_t, gobs_t)

    opred.step["cpu", PRED](pred)
    odyn.step["cpu", DYN](dyn)
    orep.step["cpu", REP](rep)

    zst.free(); din.free(); dout.free(); pout.free(); gpout.free()
    gdout.free(); gz.free(); gpin.free(); gdin.free(); gobs.free()
    twv.free(); twr.free()
    if loss_parts:
        var lp = loss_parts.value()
        var inv = Scalar[DT](1.0) / Scalar[DT](B)
        lp[0] = l_pol * inv   # policy
        lp[1] = l_val * inv   # value
        lp[2] = l_rew * inv   # reward
    return loss / Scalar[DT](B)


# ──────────────────────────────────────────────────────────────────────
# GPU unroll — device kernels (mirror the host loops of the CPU step).
#
# All flat 1-D over `MutAnyOrigin` device pointers (the validated nn2 GPU
# marshalling idiom — see `dreamerv3/blocks.mojo` `_bcopy`). Concrete `DT`
# everywhere, so no `where dtype.is_floating_point()` guard is needed.
# Bit-for-bit the same arithmetic as `loss_ops` / `twohot_targets` so the
# CPU↔GPU parity test holds within fp32 noise.
# ──────────────────────────────────────────────────────────────────────


def _mz_build_dyn_in_k[
    B_: Int, LATENT_: Int, ACT_: Int, DYN_IN_: Int,
](
    din: LayoutTensor[DT, Layout.row_major(B_ * DYN_IN_), MutAnyOrigin],
    zk: LayoutTensor[DT, Layout.row_major(B_ * LATENT_), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
):
    """Assemble the dynamics input row `[z_k | onehot(a_k)]` per sample.
    `act[b]` is the (float-encoded) action index for this unroll step."""
    var idx = Int(global_idx.x)
    if idx < B_ * DYN_IN_:
        var b = idx // DYN_IN_
        var d = idx % DYN_IN_
        if d < LATENT_:
            din[idx] = rebind[Scalar[DT]](zk[b * LATENT_ + d])
        else:
            var sel = Int(rebind[Scalar[DT]](act[b]))
            din[idx] = (
                Scalar[DT](1.0) if (d - LATENT_) == sel else Scalar[DT](0.0)
            )


def _mz_copy_latent_k[
    B_: Int, LATENT_: Int, DYN_OUT_: Int,
](
    znext: LayoutTensor[DT, Layout.row_major(B_ * LATENT_), MutAnyOrigin],
    dout: LayoutTensor[DT, Layout.row_major(B_ * DYN_OUT_), MutAnyOrigin],
):
    """Copy the latent half `dout[:, :LATENT]` → `znext` (the next z_{k+1})."""
    var idx = Int(global_idx.x)
    if idx < B_ * LATENT_:
        var b = idx // LATENT_
        var i = idx % LATENT_
        znext[idx] = rebind[Scalar[DT]](dout[b * DYN_OUT_ + i])


def _mz_softce_slice_k[
    B_: Int, ROW_: Int, OFF_: Int, NBINS_: Int,
](
    logits: LayoutTensor[DT, Layout.row_major(B_ * ROW_), MutAnyOrigin],
    target: LayoutTensor[DT, Layout.row_major(B_ * NBINS_), MutAnyOrigin],
    grad_out: LayoutTensor[DT, Layout.row_major(B_ * ROW_), MutAnyOrigin],
    loss_buf: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
    grad_scale: Scalar[DT],
    loss_coef: Scalar[DT],
):
    """Soft-CE over the `[OFF, OFF+NBINS)` column slice of a `[B, ROW]` logits
    tile vs a `[B, NBINS]` soft target. Writes `grad_scale·(softmax − q)` into
    the same slice of `grad_out` and **accumulates** `loss_coef·(−Σ q·log sm)`
    into `loss_buf[b]`. One thread per row — mirrors
    `soft_ce_slice_loss_and_grad` exactly."""
    var b = Int(global_idx.x)
    if b < B_:
        var base = b * ROW_ + OFF_
        var m = rebind[Scalar[DT]](logits[base])
        for i in range(1, NBINS_):
            var v = rebind[Scalar[DT]](logits[base + i])
            if v > m:
                m = v
        var s = Scalar[DT](0.0)
        for i in range(NBINS_):
            s += exp(rebind[Scalar[DT]](logits[base + i]) - m)
        var log_s = log(s)
        var tb = b * NBINS_
        var row_loss = Scalar[DT](0.0)
        for i in range(NBINS_):
            var q = rebind[Scalar[DT]](target[tb + i])
            var log_sm = (rebind[Scalar[DT]](logits[base + i]) - m) - log_s
            row_loss += -q * log_sm
            var sm = exp(log_sm)
            grad_out[base + i] = grad_scale * (sm - q)
        loss_buf[b] = rebind[Scalar[DT]](loss_buf[b]) + loss_coef * row_loss


# ── PER kernels (mirror efficient_zero_v2/blocks.mojo) ──────────────────────
def _mz_scale_rows_k[B_: Int, ROW_: Int, OFF_: Int, LEN_: Int](
    grad: LayoutTensor[DT, Layout.row_major(B_ * ROW_), MutAnyOrigin],
    w: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
):
    """Scale the ``[OFF_, OFF_+LEN_)`` column slice of grad row ``b`` by the
    per-sample importance-sampling weight ``w[b]`` (PER gradient weighting).
    One thread per row."""
    var b = Int(global_idx.x)
    if b < B_:
        var wb = rebind[Scalar[DT]](w[b])
        var base = b * ROW_ + OFF_
        for c in range(LEN_):
            grad[base + c] = rebind[Scalar[DT]](grad[base + c]) * wb


def _mz_priority_ce_k[B_: Int, ROW_: Int, OFF_: Int, NBINS_: Int](
    logits: LayoutTensor[DT, Layout.row_major(B_ * ROW_), MutAnyOrigin],
    target: LayoutTensor[DT, Layout.row_major(B_ * NBINS_), MutAnyOrigin],
    out_prio: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
):
    """Per-sample soft-CE of the value-head slice vs the value two-hot target —
    the PER priority signal (root value-prediction error). Writes (does NOT
    accumulate) into ``out_prio[b]``. One thread per row; mirrors the loss half
    of `_mz_softce_slice_k`."""
    var b = Int(global_idx.x)
    if b < B_:
        var base = b * ROW_ + OFF_
        var m = rebind[Scalar[DT]](logits[base])
        for i in range(1, NBINS_):
            var v = rebind[Scalar[DT]](logits[base + i])
            if v > m:
                m = v
        var s = Scalar[DT](0.0)
        for i in range(NBINS_):
            s += exp(rebind[Scalar[DT]](logits[base + i]) - m)
        var log_s = log(s)
        var tb = b * NBINS_
        var row_loss = Scalar[DT](0.0)
        for i in range(NBINS_):
            var q = rebind[Scalar[DT]](target[tb + i])
            row_loss += -q * ((rebind[Scalar[DT]](logits[base + i]) - m) - log_s)
        out_prio[b] = row_loss


def _mz_twohot_k[
    B_: Int, NBINS_: Int,
](
    tgt: LayoutTensor[DT, Layout.row_major(B_ * NBINS_), MutAnyOrigin],
    vals: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
):
    """Two-hot encode raw scalars: apply `h(x)` then two-hot over `NBINS`
    linear bins in `[v_min, v_max]` (h-space support). Mirrors
    `mz_two_hot_target_batch` (`mz_scalar_transform` + `two_hot_encode`)."""
    var b = Int(global_idx.x)
    if b < B_:
        var x = rebind[Scalar[DT]](vals[b])
        # h(x) = sign(x)·(√(|x|+1) − 1) + ε·x   (ε = 0.001)
        var eps = Scalar[DT](0.001)
        var sgn = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
        var ax = x if x >= Scalar[DT](0.0) else -x
        var ht = sgn * (sqrt(ax + Scalar[DT](1.0)) - Scalar[DT](1.0)) + eps * x
        var base = b * NBINS_
        for i in range(NBINS_):
            tgt[base + i] = Scalar[DT](0.0)
        if NBINS_ == 1:
            tgt[base] = Scalar[DT](1.0)
            return
        # clamp into the bin range, then two-hot over linear bins.
        if ht < v_min:
            ht = v_min
        if ht > v_max:
            ht = v_max
        var step = (v_max - v_min) / Scalar[DT](NBINS_ - 1)
        var kf = (ht - v_min) / step
        var k = Int(kf)
        if k >= NBINS_ - 1:
            k = NBINS_ - 2
        if k < 0:
            k = 0
        var bin_low = v_min + step * Scalar[DT](k)
        var bin_high = v_min + step * Scalar[DT](k + 1)
        var width = bin_high - bin_low
        var upper = (bin_high - ht) / width
        tgt[base + k] = upper
        tgt[base + k + 1] = Scalar[DT](1.0) - upper


def _mz_set_carry_latent_k[
    B_: Int, LATENT_: Int, DYN_OUT_: Int,
](
    gdout: LayoutTensor[DT, Layout.row_major(B_ * DYN_OUT_), MutAnyOrigin],
    gz: LayoutTensor[DT, Layout.row_major(B_ * LATENT_), MutAnyOrigin],
):
    """Seed the latent half of the dynamics output-grad with the carry
    `∂L/∂z_{k+1}` (the reward kernel fills the `[LATENT, LATENT+BINS)` half)."""
    var idx = Int(global_idx.x)
    if idx < B_ * LATENT_:
        var b = idx // LATENT_
        var i = idx % LATENT_
        gdout[b * DYN_OUT_ + i] = rebind[Scalar[DT]](gz[idx])


def _mz_accum_half_k[
    B_: Int, LATENT_: Int, DYN_IN_: Int,
](
    gpin: LayoutTensor[DT, Layout.row_major(B_ * LATENT_), MutAnyOrigin],
    gdin: LayoutTensor[DT, Layout.row_major(B_ * DYN_IN_), MutAnyOrigin],
):
    """`∂L/∂z_k += ½·(grad into dyn's latent input)` — the MuZero ½ dynamics
    hidden-input gradient scaling (appendix)."""
    var idx = Int(global_idx.x)
    if idx < B_ * LATENT_:
        var b = idx // LATENT_
        var i = idx % LATENT_
        gpin[idx] = rebind[Scalar[DT]](gpin[idx]) + Scalar[DT](0.5) * rebind[
            Scalar[DT]
        ](gdin[b * DYN_IN_ + i])


def _mz_bcopy_k[N_: Int](
    src: LayoutTensor[DT, Layout.row_major(N_), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N_), MutAnyOrigin],
):
    """Contiguous device→device copy (carry gz ← gpin)."""
    var i = Int(global_idx.x)
    if i < N_:
        dst[i] = rebind[Scalar[DT]](src[i])


def mz_unroll_train_step_gpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    obs_on_device: Bool = False,
](
    ctx: DeviceContext,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    mut scratch: MZScratch[B, K, OBS, ACT, LATENT, BINS],
    obs0: UnsafePointer[Scalar[DT], MutAnyOrigin],
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
    policy_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    loss_parts: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
    is_weights: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
    out_prio: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """GPU MuZero K-step unroll training step — device mirror of
    `mz_unroll_train_step_cpu`.

    PER (optional): when ``is_weights`` is given the per-sample importance
    weights scale the whole prediction-head grad row + the reward slice of the
    dyn grad before each vjp (the latent carry is already weighted, so it is
    NOT re-scaled). When ``out_prio`` is given the root (k=0) value-head soft-CE
    is written per row as the new priority signal. Both are no-ops when ``None``
    → bit-identical to the uniform path.

    Takes the same **host** time-major batch slabs (raw value/reward scalars;
    `h` + two-hot applied on device), H2D-copies them once, runs the forward
    scan (rep + K dynamics) and reverse scan (pred/dyn vjp with the ½ dynamics
    gradient + 1/(K+1) loss weight) entirely on the device, and steps the three
    Adam optimizers in place. Returns the mean total loss (same reduction as
    the CPU path). Device + host scratch is supplied by a persistent
    `MZScratch` (allocated once in `make`, reused every step) — this
    function performs **no** `enqueue_create_buffer` of its own, which would
    explode disk on NVIDIA when called in the per-step hot loop.
    """
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)

    # ── reuse persistent scratch (allocated once in `MZScratch.make`) ──
    var d_obs0 = scratch.d_obs0.value()
    var d_act = scratch.d_act.value()
    var d_pol = scratch.d_pol.value()
    var d_val = scratch.d_val.value()
    var d_rew = scratch.d_rew.value()

    # ── H2D the host batch slabs (fully overwrites the cached buffers) ──
    # ``obs_on_device``: the caller already filled ``scratch.d_obs0`` on-device
    # (e.g. `GPUMCTSSequenceReplay.sample_training_batch_dev` gathered the obs
    # window straight into it), so skip the obs H2D and ignore ``obs0``.
    comptime if not obs_on_device:
        ctx.enqueue_copy(d_obs0, obs0)
    ctx.enqueue_copy(d_act, actions)
    ctx.enqueue_copy(d_pol, policy_tgt)
    ctx.enqueue_copy(d_val, value_tgt)
    ctx.enqueue_copy(d_rew, reward_tgt)

    # ── device scratch (reused) ──
    var zst = scratch.zst.value()
    var din = scratch.din.value()
    var dout = scratch.dout.value()
    var pout = scratch.pout.value()
    var gpout = scratch.gpout.value()
    var gdout = scratch.gdout.value()
    var gz = scratch.gz.value()
    var gpin = scratch.gpin.value()
    var gdin = scratch.gdin.value()
    var gobs = scratch.gobs.value()
    var twv = scratch.twv.value()
    var twr = scratch.twr.value()
    var loss_d = scratch.loss_d.value()

    # ── PER (optional): copy IS weights to device; priorities written at k=0 ──
    var has_isw = Bool(is_weights)
    var d_isw = scratch.d_isw.value()
    var d_prio = scratch.d_prio.value()
    if has_isw:
        ctx.enqueue_copy(d_isw, is_weights.value())

    # zero the 3 loss-component accumulators (policy | value | reward)
    var zloss = scratch.h_zloss.value()
    for i in range(3 * B):
        zloss.unsafe_ptr()[i] = Scalar[DT](0.0)
    ctx.enqueue_copy(loss_d, zloss)

    # raw device pointers (MutAnyOrigin) for TileTensor / LayoutTensor views
    var p_obs0 = _dp(d_obs0)
    var p_act = _dp(d_act)
    var p_pol = _dp(d_pol)
    var p_val = _dp(d_val)
    var p_rew = _dp(d_rew)
    var p_zst = _dp(zst)
    var p_din = _dp(din)
    var p_dout = _dp(dout)
    var p_pout = _dp(pout)
    var p_gpout = _dp(gpout)
    var p_gdout = _dp(gdout)
    var p_gz = _dp(gz)
    var p_gpin = _dp(gpin)
    var p_gdin = _dp(gdin)
    var p_gobs = _dp(gobs)
    var p_twv = _dp(twv)
    var p_twr = _dp(twr)
    var p_loss = _dp(loss_d)
    var p_isw = _dp(d_isw)
    var p_prio = _dp(d_prio)

    comptime nbDIN = (B * DYN_IN + TPB - 1) // TPB
    comptime nbLAT = (B * LATENT + TPB - 1) // TPB
    comptime nbB = (B + TPB - 1) // TPB
    comptime kBuild = _mz_build_dyn_in_k[B, LATENT, ACT, DYN_IN]
    comptime kCopyL = _mz_copy_latent_k[B, LATENT, DYN_OUT]
    comptime kPolCE = _mz_softce_slice_k[B, PRED_OUT, 0, ACT]
    comptime kValCE = _mz_softce_slice_k[B, PRED_OUT, ACT, BINS]
    comptime kRewCE = _mz_softce_slice_k[B, DYN_OUT, LATENT, BINS]
    comptime kTwoHot = _mz_twohot_k[B, BINS]
    comptime kCarry = _mz_set_carry_latent_k[B, LATENT, DYN_OUT]
    comptime kHalf = _mz_accum_half_k[B, LATENT, DYN_IN]
    comptime kBcopy = _mz_bcopy_k[B * LATENT]
    # PER: scale the whole pred-head grad row; scale ONLY the reward slice of
    # the dyn grad; root value-error priority (mirror efficient_zero_v2).
    comptime kScalePred = _mz_scale_rows_k[B, PRED_OUT, 0, PRED_OUT]
    comptime kScaleRew = _mz_scale_rows_k[B, DYN_OUT, LATENT, BINS]
    comptime kPrioCE = _mz_priority_ce_k[B, PRED_OUT, ACT, BINS]

    # ── forward scan: z0 = h(obs0); z_{k+1} = g(z_k, a_k).latent ──
    var z0_t = TileTensor(p_zst, row_major[B, LATENT]())
    rep.forward["gpu", B](
        TileTensor(p_obs0, row_major[B, OBS]()), output=z0_t
    )
    for k in range(K):
        var zk = p_zst + k * B * LATENT
        ctx.enqueue_function[kBuild](
            _lt[B * DYN_IN](p_din),
            _lt[B * LATENT](zk),
            _lt[B](p_act + k * B),
            grid_dim=nbDIN, block_dim=TPB,
        )
        var dout_t = TileTensor(p_dout, row_major[B, DYN_OUT]())
        dyn.forward["gpu", B](
            TileTensor(p_din, row_major[B, DYN_IN]()), output=dout_t
        )
        var znext = p_zst + (k + 1) * B * LATENT
        ctx.enqueue_function[kCopyL](
            _lt[B * LATENT](znext),
            _lt[B * DYN_OUT](p_dout),
            grid_dim=nbLAT, block_dim=TPB,
        )

    # ── reverse scan ──
    orep.zero_grad["gpu", REP](rep)
    odyn.zero_grad["gpu", DYN](dyn)
    opred.zero_grad["gpu", PRED](pred)

    for rk in range(K + 1):
        var k = K - rk
        var zk = p_zst + k * B * LATENT
        var zk_t = TileTensor(zk, row_major[B, LATENT]())

        # (a) prediction head: re-forward (cache), seed grads, vjp → grad z_k
        var pout_t = TileTensor(p_pout, row_major[B, PRED_OUT]())
        pred.forward["gpu", B](zk_t, output=pout_t)
        # policy slice [0, ACT)
        ctx.enqueue_function[kPolCE](
            _lt[B * PRED_OUT](p_pout),
            _lt[B * ACT](p_pol + k * B * ACT),
            _lt[B * PRED_OUT](p_gpout),
            _lt[B](p_loss),
            gscale, Scalar[DT](1.0),
            grid_dim=nbB, block_dim=TPB,
        )
        # value slice [ACT, ACT+BINS): two-hot then soft-CE (scaled by coef)
        ctx.enqueue_function[kTwoHot](
            _lt[B * BINS](p_twv), _lt[B](p_val + k * B), v_min, v_max,
            grid_dim=nbB, block_dim=TPB,
        )
        ctx.enqueue_function[kValCE](
            _lt[B * PRED_OUT](p_pout),
            _lt[B * BINS](p_twv),
            _lt[B * PRED_OUT](p_gpout),
            _lt[B](p_loss + B),                       # value block
            gscale * value_coef, value_coef,
            grid_dim=nbB, block_dim=TPB,
        )
        # PER: value-error priority at the root (k=0), read from value logits +
        # value two-hot (both still intact — reads logits, not grads).
        if out_prio and k == 0:
            ctx.enqueue_function[kPrioCE](
                _lt[B * PRED_OUT](p_pout), _lt[B * BINS](p_twv),
                _lt[B](p_prio), grid_dim=nbB, block_dim=TPB,
            )
        # PER: weight the whole prediction-head grad row by w_b before vjp.
        if has_isw:
            ctx.enqueue_function[kScalePred](
                _lt[B * PRED_OUT](p_gpout), _lt[B](p_isw),
                grid_dim=nbB, block_dim=TPB,
            )
        var gpout_t = TileTensor(p_gpout, row_major[B, PRED_OUT]())
        var gpin_t = TileTensor(p_gpin, row_major[B, LATENT]())
        pred.vjp["gpu", B](gpout_t, gpin_t)

        # (b) dynamics: carry grad from z_{k+1} + reward head, ½ on hidden input
        if k < K:
            ctx.enqueue_function[kBuild](
                _lt[B * DYN_IN](p_din),
                _lt[B * LATENT](zk),
                _lt[B](p_act + k * B),
                grid_dim=nbDIN, block_dim=TPB,
            )
            var dout_t = TileTensor(p_dout, row_major[B, DYN_OUT]())
            dyn.forward["gpu", B](
                TileTensor(p_din, row_major[B, DYN_IN]()), output=dout_t
            )
            # grad_dyn_out = [ carry(grad z_{k+1}) | reward grad ]
            ctx.enqueue_function[kCarry](
                _lt[B * DYN_OUT](p_gdout),
                _lt[B * LATENT](p_gz),
                grid_dim=nbLAT, block_dim=TPB,
            )
            ctx.enqueue_function[kTwoHot](
                _lt[B * BINS](p_twr), _lt[B](p_rew + k * B), v_min, v_max,
                grid_dim=nbB, block_dim=TPB,
            )
            ctx.enqueue_function[kRewCE](
                _lt[B * DYN_OUT](p_dout),
                _lt[B * BINS](p_twr),
                _lt[B * DYN_OUT](p_gdout),
                _lt[B](p_loss + 2 * B),               # reward block
                gscale, Scalar[DT](1.0),
                grid_dim=nbB, block_dim=TPB,
            )
            # PER: weight ONLY the reward slice of the dyn grad by w_b. The
            # latent slice carries the already-weighted gradient from z_{k+1}
            # (kCarry), so scaling it again would double-weight it.
            if has_isw:
                ctx.enqueue_function[kScaleRew](
                    _lt[B * DYN_OUT](p_gdout), _lt[B](p_isw),
                    grid_dim=nbB, block_dim=TPB,
                )
            var gdout_t = TileTensor(p_gdout, row_major[B, DYN_OUT]())
            var gdin_t = TileTensor(p_gdin, row_major[B, DYN_IN]())
            dyn.vjp["gpu", B](gdout_t, gdin_t)
            # ∂L/∂z_k += ½ · (grad into dyn's latent input)
            ctx.enqueue_function[kHalf](
                _lt[B * LATENT](p_gpin),
                _lt[B * DYN_IN](p_gdin),
                grid_dim=nbLAT, block_dim=TPB,
            )

        # carry ← full grad wrt z_k for the next (k-1) iteration
        ctx.enqueue_function[kBcopy](
            _lt[B * LATENT](p_gpin),
            _lt[B * LATENT](p_gz),
            grid_dim=nbLAT, block_dim=TPB,
        )

    # ── rep: grad wrt z_0 (== carry after the loop) → rep params ──
    var gz0_t = TileTensor(p_gz, row_major[B, LATENT]())
    var gobs_t = TileTensor(p_gobs, row_major[B, OBS]())
    rep.vjp["gpu", B](gz0_t, gobs_t)

    opred.step["gpu", PRED](pred)
    odyn.step["gpu", DYN](dyn)
    orep.step["gpu", REP](rep)

    # ── D2H PER priorities + loss with a single sync — 3 [B] blocks:
    #    policy | value | reward (loss); + [B] value-error priorities ──
    ctx.synchronize()
    var hloss = scratch.h_loss.value()
    if out_prio:
        ctx.enqueue_copy(scratch.h_prio.value(), d_prio)
    ctx.enqueue_copy(hloss, loss_d)
    ctx.synchronize()
    if out_prio:
        var op = out_prio.value()
        var hpp = scratch.h_prio.value().unsafe_ptr()
        for b in range(B):
            op[b] = hpp[b]
    var hp = hloss.unsafe_ptr()
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    for b in range(B):
        l_pol += hp[b]
        l_val += hp[B + b]
        l_rew += hp[2 * B + b]
    var inv = Scalar[DT](1.0) / Scalar[DT](B)
    if loss_parts:
        var lp = loss_parts.value()
        lp[0] = l_pol * inv
        lp[1] = l_val * inv
        lp[2] = l_rew * inv
    return (l_pol + l_val + l_rew) * inv
