"""EfficientZeroV2 K-step unroll — MuZero BPTT + SimSiam temporal consistency.

The EZv2 training step is the MuZero unroll (`muzero/blocks.mojo`) with one
addition: a **SimSiam consistency** branch at every dynamics step. For each
unroll position ``k = 1..K`` (no consistency at the root k=0):

    online :  p_k = h_pred(g_proj(z_k))          (z_k = the rolled dynamics latent)
    target :  t_k = sg( g_proj(h(obs_k)) )         (g_proj of the *real* future obs)
    L_G    += −cos(p_k, t_k)                        (stop-grad on t_k)

The consistency gradient flows ``p_k → h_pred → g_proj → z_k`` and is added to
the same per-step latent gradient accumulator ``∂L/∂z_k`` that the policy/value
head feeds — so it then propagates back through the dynamics with the MuZero
½ scaling, exactly like every other contribution to ``z_k``. The target branch
is detached (computed in a pre-pass into ``t_store``, never backpropped).

Cache discipline (the nn re-forward-before-vjp idiom): the target pre-pass runs
``h(obs_k)`` which clobbers the representation net's forward cache, so the final
``rep.vjp`` is preceded by a fresh ``rep.forward(obs0)``. Within the reverse
scan, ``g_proj``/``h_pred`` are re-forwarded on the *online* input immediately
before their ``vjp`` so their caches hold the live (online) activations.

Batch layout is time-major like MuZero, except obs is the **full sequence**
``obs_seq[K+1, B, OBS]`` (``obs_seq[0] == obs0``) so the consistency targets can
encode the real future observations: ``actions[K,B]`` (indices),
``policy_tgt[K+1,B,ACT]``, ``value_tgt[K+1,B]``, ``reward_tgt[K,B]`` (raw).
``cons_mask[K,B]`` (optional) is the reference's ``mask_batch`` applied to the
consistency term only: rows whose ``obs_seq[k]`` is absorbing obs-repeat
padding (the window crossed the episode terminal) are zeroed, so the dynamics
is never trained toward the false "terminal obs is a fixed point" target.

CPU path first (overfit-tested); a GPU branch + CPU↔GPU parity follow.
"""

from std.memory import alloc
from std.math import exp, log
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.module import Module, mptr
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.optimizer import Optimizer

from .loss_ops import consistency_loss_and_grad, consistency_loss_grad_k
from .unroll_scratch import EZV2UnrollScratch
from ..muzero.loss_ops import soft_ce_slice_loss_and_grad
from ..muzero.blocks import (
    _dp,
    _lt,
    _mz_build_dyn_in_k,
    _mz_copy_latent_k,
    _mz_softce_slice_k,
    _mz_twohot_k,
    _mz_set_carry_latent_k,
    _mz_accum_half_k,
    _mz_bcopy_k,
)
from ..zero.twohot_targets import mz_two_hot_target_batch


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


# ── PER kernels (6c-2) ──────────────────────────────────────────────────
def _ez_scale_rows_k[B_: Int, ROW_: Int, OFF_: Int, LEN_: Int](
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


def _ez_priority_ce_k[B_: Int, ROW_: Int, OFF_: Int, NBINS_: Int](
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


def ezv2_unroll_train_step_cpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    O: Optimizer = Adam,
](
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: O,
    mut odyn: O,
    mut opred: O,
    mut oproj: O,
    mut opredh: O,
    obs_seq: UnsafePointer[Scalar[DT], MutAnyOrigin],
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
    policy_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    cons_mask: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
    loss_parts: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """One CPU EZv2 unroll training step (MuZero BPTT + SimSiam consistency).

    Returns the mean total loss (policy + value + reward + consistency). Mutates
    all five nets via their optimizers. ``obs_seq`` is the time-major
    ``[K+1, B, OBS]`` observation sequence (``obs_seq[0]`` is the root obs).
    ``cons_mask`` is the optional ``[K, B]`` episode-boundary mask (reference
    ``mask_batch``): row ``(k-1, b)`` is 0 when ``obs_seq[k]`` is absorbing
    obs-repeat padding rather than a real future observation, zeroing that
    consistency term (``None`` ≡ all ones, the legacy unmasked behaviour).
    """
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime PROJ = PROJM.OUT_DIM

    # ── scratch ──
    var zst = _a((K + 1) * B * LATENT)   # stored latents z0..zK
    var din = _a(B * DYN_IN)
    var dout = _a(B * DYN_OUT)
    var pout = _a(B * PRED_OUT)
    var gpout = _a(B * PRED_OUT)
    var gdout = _a(B * DYN_OUT)
    var gz = _a(B * LATENT)               # carry: grad wrt z_{k+1}
    var gpin = _a(B * LATENT)             # working grad wrt z_k
    var gdin = _a(B * DYN_IN)
    var gobs = _a(B * OBS)                # grad wrt rep input (discarded)
    var twv = _a(B * BINS)
    var twr = _a(B * BINS)
    # consistency scratch
    var tstore = _a(K * B * PROJ)         # detached target projections t_1..t_K
    var ztmp = _a(B * LATENT)             # rep(obs_k) for the target branch
    var projo = _a(B * PROJ)              # online g_proj(z_k)
    var pk = _a(B * PROJ)                 # online h_pred(projo)
    var gpk = _a(B * PROJ)                # grad wrt p_k
    var gproj = _a(B * PROJ)              # grad wrt projector output
    var gzcons = _a(B * LATENT)           # grad wrt z_k from consistency

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    # consistency is summed over K steps (no root term) → 1/K mean.
    var cscale = consistency_coef / Scalar[DT](K * B)

    # ── forward scan: rep then K dynamics steps, store every z ──
    var obs0_t = TileTensor(obs_seq, row_major[B, OBS]())
    var z0_t = TileTensor(zst, row_major[B, LATENT]())
    rep.forward["cpu", B](obs0_t, output=z0_t)

    for k in range(K):
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
        var znext = zst + (k + 1) * B * LATENT
        for b in range(B):
            for i in range(LATENT):
                znext[b * LATENT + i] = dout[b * DYN_OUT + i]

    # ── target pre-pass: t_k = g_proj(h(obs_k)), detached, k = 1..K ──
    # (clobbers rep's cache → rep is re-forwarded before the final rep.vjp)
    for k in range(1, K + 1):
        var obsk_t = TileTensor(obs_seq + k * B * OBS, row_major[B, OBS]())
        var ztmp_t = TileTensor(ztmp, row_major[B, LATENT]())
        rep.forward["cpu", B](obsk_t, output=ztmp_t)
        var tslot = TileTensor(tstore + (k - 1) * B * PROJ, row_major[B, PROJ]())
        proj.forward["cpu", B](ztmp_t, output=tslot)

    # ── reverse scan: accumulate grads + loss ──
    orep.zero_grad["cpu", REP](rep)
    odyn.zero_grad["cpu", DYN](dyn)
    opred.zero_grad["cpu", PRED](pred)
    oproj.zero_grad["cpu", PROJM](proj)
    opredh.zero_grad["cpu", PREDH](predh)

    var loss = Scalar[DT](0.0)
    # per-component loss accumulators (for the optional loss_parts breakdown)
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    var l_cons = Scalar[DT](0.0)
    for rk in range(K + 1):
        var k = K - rk
        var zk = zst + k * B * LATENT
        var zk_t = TileTensor(zk, row_major[B, LATENT]())

        # (a) prediction head: re-forward for cache, seed grads, vjp → grad z_k
        var pout_t = TileTensor(pout, row_major[B, PRED_OUT]())
        pred.forward["cpu", B](zk_t, output=pout_t)
        var l_pol_k = soft_ce_slice_loss_and_grad[B, PRED_OUT, 0, ACT](
            pout, policy_tgt + k * B * ACT, gscale, gpout
        )
        loss += l_pol_k
        l_pol += l_pol_k
        mz_two_hot_target_batch[B, BINS](value_tgt + k * B, v_min, v_max, twv)
        var l_val_k = value_coef * soft_ce_slice_loss_and_grad[
            B, PRED_OUT, ACT, BINS
        ](pout, twv, gscale * value_coef, gpout)
        loss += l_val_k
        l_val += l_val_k
        var gpout_t = TileTensor(gpout, row_major[B, PRED_OUT]())
        var gpin_t = TileTensor(gpin, row_major[B, LATENT]())
        pred.vjp["cpu", B](gpout_t, gpin_t)

        # (b) consistency online branch (k >= 1): p_k = h_pred(g_proj(z_k))
        if k >= 1:
            var projo_t = TileTensor(projo, row_major[B, PROJ]())
            proj.forward["cpu", B](zk_t, output=projo_t)   # refresh proj cache
            var pk_t = TileTensor(pk, row_major[B, PROJ]())
            predh.forward["cpu", B](projo_t, output=pk_t)  # refresh predh cache
            var mk = Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]](None)
            if cons_mask:
                mk = cons_mask.value() + (k - 1) * B
            var l_cons_k = consistency_loss_and_grad[B, PROJ](
                pk, tstore + (k - 1) * B * PROJ, cscale, gpk, mask=mk
            )
            loss += l_cons_k
            l_cons += l_cons_k
            var gpk_t = TileTensor(gpk, row_major[B, PROJ]())
            var gproj_t = TileTensor(gproj, row_major[B, PROJ]())
            predh.vjp["cpu", B](gpk_t, gproj_t)            # → grad proj output
            var gzcons_t = TileTensor(gzcons, row_major[B, LATENT]())
            proj.vjp["cpu", B](gproj_t, gzcons_t)          # → grad z_k
            for b in range(B):
                for i in range(LATENT):
                    gpin[b * LATENT + i] += gzcons[b * LATENT + i]

        # (c) dynamics: carry grad from z_{k+1} + reward head, ½ on hidden input
        if k < K:
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
            for b in range(B):
                for i in range(LATENT):
                    gpin[b * LATENT + i] += (
                        Scalar[DT](0.5) * gdin[b * DYN_IN + i]
                    )

        # carry ← full grad wrt z_k for the next (k-1) iteration
        for b in range(B):
            for i in range(LATENT):
                gz[b * LATENT + i] = gpin[b * LATENT + i]

    # ── rep: re-forward obs0 (cache clobbered by target pre-pass), then vjp ──
    var z0b_t = TileTensor(zst, row_major[B, LATENT]())
    rep.forward["cpu", B](obs0_t, output=z0b_t)
    var gz0_t = TileTensor(gz, row_major[B, LATENT]())
    var gobs_t = TileTensor(gobs, row_major[B, OBS]())
    rep.vjp["cpu", B](gz0_t, gobs_t)

    opred.step["cpu", PRED](pred)
    odyn.step["cpu", DYN](dyn)
    orep.step["cpu", REP](rep)
    oproj.step["cpu", PROJM](proj)
    opredh.step["cpu", PREDH](predh)

    zst.free(); din.free(); dout.free(); pout.free(); gpout.free()
    gdout.free(); gz.free(); gpin.free(); gdin.free(); gobs.free()
    twv.free(); twr.free()
    tstore.free(); ztmp.free(); projo.free(); pk.free(); gpk.free()
    gproj.free(); gzcons.free()
    if loss_parts:
        var lp = loss_parts.value()
        var inv = Scalar[DT](1.0) / Scalar[DT](B)
        lp[0] = l_pol * inv   # policy
        lp[1] = l_val * inv   # value
        lp[2] = l_rew * inv   # reward
        lp[3] = l_cons * inv  # consistency
    return loss / Scalar[DT](B)


# ─────────────────────────────────────────────────────────────────────────
# GPU path
# ─────────────────────────────────────────────────────────────────────────


def _ez_accum_latent_k[N_: Int](
    dst: LayoutTensor[DT, Layout.row_major(N_), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N_), MutAnyOrigin],
):
    """`dst[i] += src[i]` — fold the consistency latent-grad into ``gpin``."""
    var i = Int(global_idx.x)
    if i < N_:
        dst[i] = rebind[Scalar[DT]](dst[i]) + rebind[Scalar[DT]](src[i])


def ezv2_unroll_train_step_gpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    O: Optimizer = Adam,
](
    ctx: DeviceContext,
    mut scratch: EZV2UnrollScratch[B, K, OBS, ACT, LATENT, BINS, PROJM.OUT_DIM],
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: O,
    mut odyn: O,
    mut opred: O,
    mut oproj: O,
    mut opredh: O,
    obs_seq: UnsafePointer[Scalar[DT], MutAnyOrigin],
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
    policy_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    cons_mask: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
    loss_parts: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
    is_weights: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
    out_prio: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
    obs_on_device: Bool = False,
    phase_ns: Optional[UnsafePointer[Float64, MutAnyOrigin]] = None,
    diag_sync: Bool = False,
) raises -> Scalar[DT]:
    """GPU EZv2 K-step unroll training step — device mirror of
    ``ezv2_unroll_train_step_cpu`` (MuZero BPTT + SimSiam consistency,
    ``cons_mask`` = the optional host ``[K, B]`` episode-boundary mask —
    see the CPU docstring).

    Same **host** time-major batch slabs as the CPU path, with ``obs_seq`` the
    full ``[K+1, B, OBS]`` observation sequence (``obs_seq[0]`` is the root obs;
    the rest feed the consistency targets). H2D-copies them once, runs the
    forward scan (rep + K dynamics), the detached target pre-pass
    (``t_k = g_proj(h(obs_k))``), and the reverse scan (pred/dyn/consistency vjp
    with the ½ dynamics gradient + 1/(K+1) loss weight) entirely on device, then
    steps all five Adam optimizers. Returns the mean total loss (same reduction
    as the CPU path). Device + host scratch is supplied by the caller via a
    persistent ``EZV2UnrollScratch`` (allocated **once** in
    ``EZV2UnrollScratch.make`` and reused every step — the old per-step
    ``enqueue_create_buffer`` exploded disk on NVIDIA)."""
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime PROJ = PROJM.OUT_DIM

    var _tp = perf_counter_ns()   # phase timer cursor (host-enqueue profiling)
    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    var cscale = consistency_coef / Scalar[DT](K * B)

    # ── reuse persistent scratch (allocated once in EZV2UnrollScratch.make) ──
    var d_obs = scratch.d_obs.value()
    var d_act = scratch.d_act.value()
    var d_pol = scratch.d_pol.value()
    var d_val = scratch.d_val.value()
    var d_rew = scratch.d_rew.value()
    # ── H2D the host batch slabs (once) ──
    # When obs_on_device, the caller (device-ring replay) has already gathered
    # the [K+1,B,OBS] obs slab straight into scratch.d_obs — skip the ~680 MB
    # host→device copy (the pixel-obs bottleneck). obs_seq is then unused.
    if not obs_on_device:
        ctx.enqueue_copy(d_obs, obs_seq)
    ctx.enqueue_copy(d_act, actions)
    ctx.enqueue_copy(d_pol, policy_tgt)
    ctx.enqueue_copy(d_val, value_tgt)
    ctx.enqueue_copy(d_rew, reward_tgt)
    # consistency boundary mask (all-ones fallback when the caller passes none).
    var d_cmask = scratch.d_cmask.value()
    if cons_mask:
        ctx.enqueue_copy(d_cmask, cons_mask.value())
    else:
        ctx.enqueue_copy(d_cmask, scratch.h_cmask_ones.value())

    var zst = scratch.d_zst.value()
    var din = scratch.d_din.value()
    var dout = scratch.d_dout.value()
    var pout = scratch.d_pout.value()
    var gpout = scratch.d_gpout.value()
    var gdout = scratch.d_gdout.value()
    var gz = scratch.d_gz.value()
    var gpin = scratch.d_gpin.value()
    var gdin = scratch.d_gdin.value()
    var gobs = scratch.d_gobs.value()
    var twv = scratch.d_twv.value()
    var twr = scratch.d_twr.value()
    var loss_d = scratch.d_loss.value()
    var tstore = scratch.d_tstore.value()
    var ztmp = scratch.d_ztmp.value()
    var projo = scratch.d_projo.value()
    var pk = scratch.d_pk.value()
    var gpk = scratch.d_gpk.value()
    var gproj = scratch.d_gproj.value()
    var gzcons = scratch.d_gzcons.value()
    var h_loss = scratch.h_loss.value()
    # PER: H2D the IS weights once (if given); the priority output is D2H'd
    # after the reverse scan. `has_isw` gates all PER work → bit-identical to
    # the unweighted path when `is_weights` is None.
    var has_isw = Bool(is_weights)
    var d_isw = scratch.d_isw.value()
    var d_prio = scratch.d_prio.value()
    if has_isw:
        ctx.enqueue_copy(d_isw, is_weights.value())

    # zero the 4 loss-component accumulators (policy|value|reward|consistency)
    for i in range(4 * B):
        h_loss.unsafe_ptr()[i] = Scalar[DT](0.0)
    ctx.enqueue_copy(loss_d, h_loss)

    var p_obs = _dp(d_obs)
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
    var p_tstore = _dp(tstore)
    var p_ztmp = _dp(ztmp)
    var p_projo = _dp(projo)
    var p_pk = _dp(pk)
    var p_gpk = _dp(gpk)
    var p_gproj = _dp(gproj)
    var p_gzcons = _dp(gzcons)
    var p_cmask = _dp(d_cmask)
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
    comptime kCons = consistency_loss_grad_k[B, PROJ]
    comptime kAccum = _ez_accum_latent_k[B * LATENT]
    # PER row-scaling: pred head (whole row), reward slice only (latent slice is
    # the already-weighted carry), consistency (whole row); + value-error prio.
    comptime kScalePred = _ez_scale_rows_k[B, PRED_OUT, 0, PRED_OUT]
    comptime kScaleRew = _ez_scale_rows_k[B, DYN_OUT, LATENT, BINS]
    comptime kScaleCons = _ez_scale_rows_k[B, PROJ, 0, PROJ]
    comptime kPrioCE = _ez_priority_ce_k[B, PRED_OUT, ACT, BINS]

    if phase_ns:   # [0] setup + H2D + zero-loss
        phase_ns.value()[0] += Float64(perf_counter_ns() - _tp)
        _tp = perf_counter_ns()

    # ── forward scan: z0 = h(obs0); z_{k+1} = g(z_k, a_k).latent ──
    var z0_t = TileTensor(p_zst, row_major[B, LATENT]())
    rep.forward["gpu", B](
        TileTensor(p_obs, row_major[B, OBS]()), output=z0_t
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

    if phase_ns:   # [1] forward scan (K dyn forwards)
        phase_ns.value()[1] += Float64(perf_counter_ns() - _tp)
        if diag_sync:   # [15] forward-scan GPU drain (rep×1 + dyn×K)
            var _rp = perf_counter_ns()
            ctx.synchronize()
            phase_ns.value()[15] += Float64(perf_counter_ns() - _rp)
        _tp = perf_counter_ns()

    # ── target pre-pass: t_k = g_proj(h(obs_k)), detached, k = 1..K ──
    # (clobbers rep's cache → rep is re-forwarded before the final rep.vjp)
    for k in range(1, K + 1):
        var ztmp_t = TileTensor(p_ztmp, row_major[B, LATENT]())
        rep.forward["gpu", B](
            TileTensor(p_obs + k * B * OBS, row_major[B, OBS]()),
            output=ztmp_t,
        )
        var tslot = TileTensor(
            p_tstore + (k - 1) * B * PROJ, row_major[B, PROJ]()
        )
        proj.forward["gpu", B](ztmp_t, output=tslot)

    if phase_ns:   # [2] target pre-pass (K rep forwards + proj)
        phase_ns.value()[2] += Float64(perf_counter_ns() - _tp)
        if diag_sync:   # [16] target-pre-pass GPU drain (rep×K + proj×K)
            var _rp = perf_counter_ns()
            ctx.synchronize()
            phase_ns.value()[16] += Float64(perf_counter_ns() - _rp)
        _tp = perf_counter_ns()

    # ── reverse scan ──
    orep.zero_grad["gpu", REP](rep)
    odyn.zero_grad["gpu", DYN](dyn)
    opred.zero_grad["gpu", PRED](pred)
    oproj.zero_grad["gpu", PROJM](proj)
    opredh.zero_grad["gpu", PREDH](predh)

    for rk in range(K + 1):
        var k = K - rk
        var zk = p_zst + k * B * LATENT
        var zk_t = TileTensor(zk, row_major[B, LATENT]())

        # (a) prediction head: re-forward (cache), seed grads, vjp → grad z_k
        var pout_t = TileTensor(p_pout, row_major[B, PRED_OUT]())
        var _rt = perf_counter_ns()
        pred.forward["gpu", B](zk_t, output=pout_t)
        if phase_ns:   # [6] pred.forward
            phase_ns.value()[6] += Float64(perf_counter_ns() - _rt)
        ctx.enqueue_function[kPolCE](
            _lt[B * PRED_OUT](p_pout),
            _lt[B * ACT](p_pol + k * B * ACT),
            _lt[B * PRED_OUT](p_gpout),
            _lt[B](p_loss),
            gscale, Scalar[DT](1.0),
            grid_dim=nbB, block_dim=TPB,
        )
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
        # value two-hot (both intact — read logits, not grads).
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
        if diag_sync:   # drain → [7] measures pure host enqueue; [11] = the
            var _rp = perf_counter_ns()   # GPU work enqueued *before* pred.vjp
            ctx.synchronize()             # (pred.fwd + kPolCE/kTwoHot/kValCE…)
            if phase_ns:
                phase_ns.value()[11] += Float64(perf_counter_ns() - _rp)
        _rt = perf_counter_ns()
        pred.vjp["gpu", B](gpout_t, gpin_t)
        if phase_ns:   # [7] pred.vjp (pure host enqueue when diag_sync)
            phase_ns.value()[7] += Float64(perf_counter_ns() - _rt)
        if diag_sync and phase_ns:   # [14] pred.vjp GPU drain
            var _rd = perf_counter_ns()
            ctx.synchronize()
            phase_ns.value()[14] += Float64(perf_counter_ns() - _rd)

        # (b) consistency online branch (k >= 1): p_k = h_pred(g_proj(z_k))
        if k >= 1:
            _rt = perf_counter_ns()
            var projo_t = TileTensor(p_projo, row_major[B, PROJ]())
            proj.forward["gpu", B](zk_t, output=projo_t)   # refresh proj cache
            var pk_t = TileTensor(p_pk, row_major[B, PROJ]())
            predh.forward["gpu", B](projo_t, output=pk_t)  # refresh predh cache
            ctx.enqueue_function[kCons](
                _lt[B * PROJ](p_pk),
                _lt[B * PROJ](p_tstore + (k - 1) * B * PROJ),
                _lt[B * PROJ](p_gpk),
                _lt[B](p_loss + 3 * B),               # consistency block
                _lt[B](p_cmask + (k - 1) * B),        # boundary mask row k
                cscale, Scalar[DT](1.0),
                grid_dim=nbB, block_dim=TPB,
            )
            # PER: weight the whole consistency grad row by w_b before vjp.
            if has_isw:
                ctx.enqueue_function[kScaleCons](
                    _lt[B * PROJ](p_gpk), _lt[B](p_isw),
                    grid_dim=nbB, block_dim=TPB,
                )
            var gpk_t = TileTensor(p_gpk, row_major[B, PROJ]())
            var gproj_t = TileTensor(p_gproj, row_major[B, PROJ]())
            predh.vjp["gpu", B](gpk_t, gproj_t)            # → grad proj output
            var gzcons_t = TileTensor(p_gzcons, row_major[B, LATENT]())
            proj.vjp["gpu", B](gproj_t, gzcons_t)          # → grad z_k
            ctx.enqueue_function[kAccum](
                _lt[B * LATENT](p_gpin),
                _lt[B * LATENT](p_gzcons),
                grid_dim=nbLAT, block_dim=TPB,
            )
            if phase_ns:   # [8] consistency branch (proj/predh fwd+vjp)
                phase_ns.value()[8] += Float64(perf_counter_ns() - _rt)

        # (c) dynamics: carry grad from z_{k+1} + reward head, ½ on hidden input
        if k < K:
            ctx.enqueue_function[kBuild](
                _lt[B * DYN_IN](p_din),
                _lt[B * LATENT](zk),
                _lt[B](p_act + k * B),
                grid_dim=nbDIN, block_dim=TPB,
            )
            var dout_t = TileTensor(p_dout, row_major[B, DYN_OUT]())
            _rt = perf_counter_ns()
            dyn.forward["gpu", B](
                TileTensor(p_din, row_major[B, DYN_IN]()), output=dout_t
            )
            if phase_ns:   # [9] dyn.forward
                phase_ns.value()[9] += Float64(perf_counter_ns() - _rt)
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
            if diag_sync:   # drain → [10] measures pure host enqueue; [13] =
                var _rp = perf_counter_ns()   # GPU work enqueued before dyn.vjp
                ctx.synchronize()             # (cons branch + dyn.fwd + kCarry…)
                if phase_ns:
                    phase_ns.value()[13] += Float64(perf_counter_ns() - _rp)
            _rt = perf_counter_ns()
            dyn.vjp["gpu", B](gdout_t, gdin_t)
            if phase_ns:   # [10] dyn.vjp (pure host enqueue when diag_sync)
                phase_ns.value()[10] += Float64(perf_counter_ns() - _rt)
            if diag_sync and phase_ns:   # [12] dyn.vjp GPU drain
                var _rd = perf_counter_ns()
                ctx.synchronize()
                phase_ns.value()[12] += Float64(perf_counter_ns() - _rd)
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

    if phase_ns:   # [3] reverse scan (pred/dyn/cons forward+vjp ×K+1)
        phase_ns.value()[3] += Float64(perf_counter_ns() - _tp)
        _tp = perf_counter_ns()

    # ── rep: re-forward obs0 (cache clobbered by target pre-pass), then vjp ──
    var z0b_t = TileTensor(p_zst, row_major[B, LATENT]())
    rep.forward["gpu", B](
        TileTensor(p_obs, row_major[B, OBS]()), output=z0b_t
    )
    var gz0_t = TileTensor(p_gz, row_major[B, LATENT]())
    var gobs_t = TileTensor(p_gobs, row_major[B, OBS]())
    rep.vjp["gpu", B](gz0_t, gobs_t)

    opred.step["gpu", PRED](pred)
    odyn.step["gpu", DYN](dyn)
    orep.step["gpu", REP](rep)
    oproj.step["gpu", PROJM](proj)
    opredh.step["gpu", PREDH](predh)

    if phase_ns:   # [4] rep re-forward+vjp + 5 optimizer steps (host enqueue)
        phase_ns.value()[4] += Float64(perf_counter_ns() - _tp)
        _tp = perf_counter_ns()

    # ── D2H PER priorities + loss with a SINGLE sync ──
    # Both copies are enqueued after all train kernels; one synchronize drains
    # the stream, then both host mirrors are read. (Was 3 syncs/step — one per
    # D2H plus a redundant pre-copy sync — each a full pipeline drain.)
    if out_prio:
        ctx.enqueue_copy(scratch.h_prio.value(), d_prio)
    ctx.enqueue_copy(h_loss, loss_d)
    ctx.synchronize()
    if out_prio:
        var op = out_prio.value()
        var hpp = scratch.h_prio.value().unsafe_ptr()
        for b in range(B):
            op[b] = hpp[b]

    # ── reduce loss: 4 contiguous [B] blocks policy|value|reward|consistency ──
    var hp = h_loss.unsafe_ptr()
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    var l_cons = Scalar[DT](0.0)
    for b in range(B):
        l_pol += hp[b]
        l_val += hp[B + b]
        l_rew += hp[2 * B + b]
        l_cons += hp[3 * B + b]
    var inv = Scalar[DT](1.0) / Scalar[DT](B)
    if loss_parts:
        var lp = loss_parts.value()
        lp[0] = l_pol * inv
        lp[1] = l_val * inv
        lp[2] = l_rew * inv
        lp[3] = l_cons * inv
    if phase_ns:   # [5] finalize: PER D2H + 2-3 syncs (GPU drain) + loss reduce
        phase_ns.value()[5] += Float64(perf_counter_ns() - _tp)
    return (l_pol + l_val + l_rew + l_cons) * inv
