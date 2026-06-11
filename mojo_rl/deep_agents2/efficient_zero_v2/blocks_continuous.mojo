"""EfficientZeroV2 continuous K-step unroll — MuZero BPTT + SimSiam + Gaussian π.

The continuous twin of `blocks.mojo::ezv2_unroll_train_step_cpu`. Structurally
identical (forward scan h + K dynamics, reverse scan with the ½ dynamics-grad
carry, SimSiam consistency at k=1..K), with two changes for continuous control:

  * **Dynamics input is ``[z | a]`` with a real action *vector*** (``ACT_DIM``
    continuous dims) instead of a one-hot — the build step copies the action
    vector into the ``ACT_DIM`` slots.
  * **Policy head is a squashed Gaussian.** The prediction output row is
    ``[μ_raw | σ_raw | value]`` (``2·ACT_DIM + BINS``). The policy loss is the
    squashed-Gaussian NLL of the search-selected action
    (`loss_ops_continuous.continuous_policy_loss_and_grad`) over the leading
    ``2·ACT_DIM`` slice; value + reward stay categorical soft-CE.

Batch layout (time-major): ``obs_seq[K+1,B,OBS]``, ``actions[K,B,ACT_DIM]`` (the
transition actions), ``policy_act_tgt[K+1,B,ACT_DIM]`` (the per-position target
actions the policy clones), ``value_tgt[K+1,B]``, ``reward_tgt[K,B]`` (raw).
"""

from std.memory import alloc
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.module import Module, mptr
from mojo_rl.nn2.optimizer.adam import Adam

from .loss_ops import consistency_loss_and_grad, consistency_loss_grad_k
from .loss_ops_continuous import (
    continuous_policy_loss_and_grad,
    continuous_policy_loss_grad_k,
)
from .blocks import _ez_accum_latent_k
from .unroll_scratch import EZV2UnrollContScratch
from ..muzero.loss_ops import soft_ce_slice_loss_and_grad
from ..muzero.blocks import (
    _dp,
    _lt,
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


def ezv2_unroll_train_step_continuous_cpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
](
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    obs_seq: UnsafePointer[Scalar[DT], MutAnyOrigin],
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],         # [K, B, ACT_DIM]
    policy_act_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [K+1, B, ACT_DIM]
    value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    policy_coef: Scalar[DT] = Scalar[DT](1.0),
    max_action: Scalar[DT] = Scalar[DT](1.0),
    min_std: Scalar[DT] = Scalar[DT](0.1),
    soft_clamp: Scalar[DT] = Scalar[DT](5.0),
    init_std: Scalar[DT] = Scalar[DT](1.0),
    ent_scale: Scalar[DT] = Scalar[DT](5e-3),
    loss_parts: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """One CPU continuous EZv2 unroll step. Returns the mean total loss. Mutates
    all five nets via their optimizers. ``obs_seq`` is ``[K+1, B, OBS]``."""
    comptime MU2 = 2 * ACT_DIM
    comptime PRED_OUT = MU2 + BINS
    comptime DYN_IN = LATENT + ACT_DIM
    comptime DYN_OUT = LATENT + BINS
    comptime PROJ = PROJM.OUT_DIM

    var zst = _a((K + 1) * B * LATENT)
    var din = _a(B * DYN_IN)
    var dout = _a(B * DYN_OUT)
    var pout = _a(B * PRED_OUT)
    var gpout = _a(B * PRED_OUT)
    var gdout = _a(B * DYN_OUT)
    var gz = _a(B * LATENT)
    var gpin = _a(B * LATENT)
    var gdin = _a(B * DYN_IN)
    var gobs = _a(B * OBS)
    var twv = _a(B * BINS)
    var twr = _a(B * BINS)
    # policy-head slice scratch
    var musig = _a(B * MU2)
    var gmusig = _a(B * MU2)
    var ptgt = _a(B * ACT_DIM)
    # consistency scratch
    var tstore = _a(K * B * PROJ)
    var ztmp = _a(B * LATENT)
    var projo = _a(B * PROJ)
    var pk = _a(B * PROJ)
    var gpk = _a(B * PROJ)
    var gproj = _a(B * PROJ)
    var gzcons = _a(B * LATENT)

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    var cscale = consistency_coef / Scalar[DT](K * B)
    var pscale = policy_coef / Scalar[DT]((K + 1) * B)

    # ── forward scan ──
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
            for a in range(ACT_DIM):
                dib[LATENT + a] = actions[(k * B + b) * ACT_DIM + a]
        var din_t = TileTensor(din, row_major[B, DYN_IN]())
        var dout_t = TileTensor(dout, row_major[B, DYN_OUT]())
        dyn.forward["cpu", B](din_t, output=dout_t)
        var znext = zst + (k + 1) * B * LATENT
        for b in range(B):
            for i in range(LATENT):
                znext[b * LATENT + i] = dout[b * DYN_OUT + i]

    # ── target pre-pass: t_k = g_proj(h(obs_k)), detached, k = 1..K ──
    for k in range(1, K + 1):
        var obsk_t = TileTensor(obs_seq + k * B * OBS, row_major[B, OBS]())
        var ztmp_t = TileTensor(ztmp, row_major[B, LATENT]())
        rep.forward["cpu", B](obsk_t, output=ztmp_t)
        var tslot = TileTensor(tstore + (k - 1) * B * PROJ, row_major[B, PROJ]())
        proj.forward["cpu", B](ztmp_t, output=tslot)

    # ── reverse scan ──
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

        # (a) prediction head: re-forward, seed grads, vjp → grad z_k
        var pout_t = TileTensor(pout, row_major[B, PRED_OUT]())
        pred.forward["cpu", B](zk_t, output=pout_t)
        # zero the policy slice of gpout (continuous loss writes it via scatter)
        for b in range(B):
            for i in range(MU2):
                gpout[b * PRED_OUT + i] = Scalar[DT](0.0)
        # policy: squashed-Gaussian NLL over the [0, 2*ACT_DIM) slice.
        for b in range(B):
            for i in range(MU2):
                musig[b * MU2 + i] = pout[b * PRED_OUT + i]
            for d in range(ACT_DIM):
                ptgt[b * ACT_DIM + d] = policy_act_tgt[
                    (k * B + b) * ACT_DIM + d
                ]
        var l_pol_k = policy_coef * continuous_policy_loss_and_grad[B, ACT_DIM](
            musig, ptgt, pscale, gmusig,
            max_action, min_std, soft_clamp, init_std, ent_scale,
        )
        loss += l_pol_k
        l_pol += l_pol_k
        for b in range(B):
            for i in range(MU2):
                gpout[b * PRED_OUT + i] = gmusig[b * MU2 + i]
        # value: categorical soft-CE over [2*ACT_DIM, 2*ACT_DIM+BINS).
        mz_two_hot_target_batch[B, BINS](value_tgt + k * B, v_min, v_max, twv)
        var l_val_k = value_coef * soft_ce_slice_loss_and_grad[
            B, PRED_OUT, MU2, BINS
        ](pout, twv, gscale * value_coef, gpout)
        loss += l_val_k
        l_val += l_val_k
        var gpout_t = TileTensor(gpout, row_major[B, PRED_OUT]())
        var gpin_t = TileTensor(gpin, row_major[B, LATENT]())
        pred.vjp["cpu", B](gpout_t, gpin_t)

        # (b) consistency online branch (k >= 1)
        if k >= 1:
            var projo_t = TileTensor(projo, row_major[B, PROJ]())
            proj.forward["cpu", B](zk_t, output=projo_t)
            var pk_t = TileTensor(pk, row_major[B, PROJ]())
            predh.forward["cpu", B](projo_t, output=pk_t)
            var l_cons_k = consistency_loss_and_grad[B, PROJ](
                pk, tstore + (k - 1) * B * PROJ, cscale, gpk
            )
            loss += l_cons_k
            l_cons += l_cons_k
            var gpk_t = TileTensor(gpk, row_major[B, PROJ]())
            var gproj_t = TileTensor(gproj, row_major[B, PROJ]())
            predh.vjp["cpu", B](gpk_t, gproj_t)
            var gzcons_t = TileTensor(gzcons, row_major[B, LATENT]())
            proj.vjp["cpu", B](gproj_t, gzcons_t)
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
                for a in range(ACT_DIM):
                    dib[LATENT + a] = actions[(k * B + b) * ACT_DIM + a]
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

        for b in range(B):
            for i in range(LATENT):
                gz[b * LATENT + i] = gpin[b * LATENT + i]

    # ── rep: re-forward obs0, then vjp ──
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
    musig.free(); gmusig.free(); ptgt.free()
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


def _ez_build_dyn_in_cont_k[
    B_: Int, LATENT_: Int, ACT_DIM_: Int, DYN_IN_: Int,
](
    din: LayoutTensor[DT, Layout.row_major(B_ * DYN_IN_), MutAnyOrigin],
    zk: LayoutTensor[DT, Layout.row_major(B_ * LATENT_), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(B_ * ACT_DIM_), MutAnyOrigin],
):
    """Assemble the continuous dynamics input row ``[z_k | a_k]`` per sample —
    the continuous twin of ``_mz_build_dyn_in_k`` (which one-hots a discrete
    index). ``act`` is the action **vector** slab ``[B, ACT_DIM]`` for this
    unroll step; its ``ACT_DIM`` dims drop straight into the trailing slots."""
    var idx = Int(global_idx.x)
    if idx < B_ * DYN_IN_:
        var b = idx // DYN_IN_
        var d = idx % DYN_IN_
        if d < LATENT_:
            din[idx] = rebind[Scalar[DT]](zk[b * LATENT_ + d])
        else:
            din[idx] = rebind[Scalar[DT]](act[b * ACT_DIM_ + (d - LATENT_)])



def ezv2_unroll_train_step_continuous_gpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
](
    ctx: DeviceContext,
    mut scratch: EZV2UnrollContScratch[
        B, K, OBS, ACT_DIM, LATENT, BINS, PROJM.OUT_DIM
    ],
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    obs_seq: UnsafePointer[Scalar[DT], MutAnyOrigin],
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],         # [K, B, ACT_DIM]
    policy_act_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [K+1, B, ACT_DIM]
    value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    policy_coef: Scalar[DT] = Scalar[DT](1.0),
    max_action: Scalar[DT] = Scalar[DT](1.0),
    min_std: Scalar[DT] = Scalar[DT](0.1),
    soft_clamp: Scalar[DT] = Scalar[DT](5.0),
    init_std: Scalar[DT] = Scalar[DT](1.0),
    ent_scale: Scalar[DT] = Scalar[DT](5e-3),
    loss_parts: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """GPU continuous EZv2 K-step unroll — device mirror of
    ``ezv2_unroll_train_step_continuous_cpu`` (MuZero BPTT + SimSiam consistency
    + squashed-Gaussian policy NLL). Same host time-major batch slabs as the CPU
    path: ``obs_seq[K+1,B,OBS]``, ``actions[K,B,ACT_DIM]`` (action **vectors**),
    ``policy_act_tgt[K+1,B,ACT_DIM]`` (search-selected target actions),
    ``value_tgt[K+1,B]``, ``reward_tgt[K,B]``. Device + host scratch is supplied
    by the caller via a persistent ``EZV2UnrollContScratch`` (allocated **once**
    in ``make`` and reused every step — the old per-step
    ``enqueue_create_buffer`` exploded disk on NVIDIA). Returns the mean total
    loss."""
    comptime MU2 = 2 * ACT_DIM
    comptime PRED_OUT = MU2 + BINS
    comptime DYN_IN = LATENT + ACT_DIM
    comptime DYN_OUT = LATENT + BINS
    comptime PROJ = PROJM.OUT_DIM

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    var cscale = consistency_coef / Scalar[DT](K * B)
    var pscale = policy_coef / Scalar[DT]((K + 1) * B)

    # ── reuse persistent scratch (allocated once in make) ──
    var d_obs = scratch.d_obs.value()
    var d_act = scratch.d_act.value()
    var d_pol = scratch.d_pol.value()
    var d_val = scratch.d_val.value()
    var d_rew = scratch.d_rew.value()
    # ── H2D the host batch slabs (once) ──
    ctx.enqueue_copy(d_obs, obs_seq)
    ctx.enqueue_copy(d_act, actions)
    ctx.enqueue_copy(d_pol, policy_act_tgt)
    ctx.enqueue_copy(d_val, value_tgt)
    ctx.enqueue_copy(d_rew, reward_tgt)

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

    comptime nbDIN = (B * DYN_IN + TPB - 1) // TPB
    comptime nbLAT = (B * LATENT + TPB - 1) // TPB
    comptime nbB = (B + TPB - 1) // TPB
    comptime kBuild = _ez_build_dyn_in_cont_k[B, LATENT, ACT_DIM, DYN_IN]
    comptime kCopyL = _mz_copy_latent_k[B, LATENT, DYN_OUT]
    comptime kPol = continuous_policy_loss_grad_k[B, ACT_DIM, PRED_OUT]
    comptime kValCE = _mz_softce_slice_k[B, PRED_OUT, MU2, BINS]
    comptime kRewCE = _mz_softce_slice_k[B, DYN_OUT, LATENT, BINS]
    comptime kTwoHot = _mz_twohot_k[B, BINS]
    comptime kCarry = _mz_set_carry_latent_k[B, LATENT, DYN_OUT]
    comptime kHalf = _mz_accum_half_k[B, LATENT, DYN_IN]
    comptime kBcopy = _mz_bcopy_k[B * LATENT]
    comptime kCons = consistency_loss_grad_k[B, PROJ]
    comptime kAccum = _ez_accum_latent_k[B * LATENT]

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
            _lt[B * ACT_DIM](p_act + k * B * ACT_DIM),
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

    # ── target pre-pass: t_k = g_proj(h(obs_k)), detached, k = 1..K ──
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
        pred.forward["gpu", B](zk_t, output=pout_t)
        # policy: squashed-Gaussian NLL over the [0, 2*ACT_DIM) slice.
        ctx.enqueue_function[kPol](
            _lt[B * PRED_OUT](p_pout),
            _lt[B * ACT_DIM](p_pol + k * B * ACT_DIM),
            _lt[B * PRED_OUT](p_gpout),
            _lt[B](p_loss),
            pscale, policy_coef,
            max_action, min_std, soft_clamp, init_std, ent_scale,
            grid_dim=nbB, block_dim=TPB,
        )
        # value: categorical soft-CE over [2*ACT_DIM, 2*ACT_DIM+BINS).
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
        var gpout_t = TileTensor(p_gpout, row_major[B, PRED_OUT]())
        var gpin_t = TileTensor(p_gpin, row_major[B, LATENT]())
        pred.vjp["gpu", B](gpout_t, gpin_t)

        # (b) consistency online branch (k >= 1): p_k = h_pred(g_proj(z_k))
        if k >= 1:
            var projo_t = TileTensor(p_projo, row_major[B, PROJ]())
            proj.forward["gpu", B](zk_t, output=projo_t)
            var pk_t = TileTensor(p_pk, row_major[B, PROJ]())
            predh.forward["gpu", B](projo_t, output=pk_t)
            ctx.enqueue_function[kCons](
                _lt[B * PROJ](p_pk),
                _lt[B * PROJ](p_tstore + (k - 1) * B * PROJ),
                _lt[B * PROJ](p_gpk),
                _lt[B](p_loss + 3 * B),               # consistency block
                cscale, Scalar[DT](1.0),
                grid_dim=nbB, block_dim=TPB,
            )
            var gpk_t = TileTensor(p_gpk, row_major[B, PROJ]())
            var gproj_t = TileTensor(p_gproj, row_major[B, PROJ]())
            predh.vjp["gpu", B](gpk_t, gproj_t)
            var gzcons_t = TileTensor(p_gzcons, row_major[B, LATENT]())
            proj.vjp["gpu", B](gproj_t, gzcons_t)
            ctx.enqueue_function[kAccum](
                _lt[B * LATENT](p_gpin),
                _lt[B * LATENT](p_gzcons),
                grid_dim=nbLAT, block_dim=TPB,
            )

        # (c) dynamics: carry grad from z_{k+1} + reward head, ½ on hidden input
        if k < K:
            ctx.enqueue_function[kBuild](
                _lt[B * DYN_IN](p_din),
                _lt[B * LATENT](zk),
                _lt[B * ACT_DIM](p_act + k * B * ACT_DIM),
                grid_dim=nbDIN, block_dim=TPB,
            )
            var dout_t = TileTensor(p_dout, row_major[B, DYN_OUT]())
            dyn.forward["gpu", B](
                TileTensor(p_din, row_major[B, DYN_IN]()), output=dout_t
            )
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
            var gdout_t = TileTensor(p_gdout, row_major[B, DYN_OUT]())
            var gdin_t = TileTensor(p_gdin, row_major[B, DYN_IN]())
            dyn.vjp["gpu", B](gdout_t, gdin_t)
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

    # ── reduce loss (D2H once) — 4 [B] blocks: policy|value|reward|consistency ──
    ctx.synchronize()
    ctx.enqueue_copy(h_loss, loss_d)
    ctx.synchronize()
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
    return (l_pol + l_val + l_rew + l_cons) * inv
