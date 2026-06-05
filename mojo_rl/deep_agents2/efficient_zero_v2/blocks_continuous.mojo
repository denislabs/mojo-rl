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

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam

from .loss_ops import consistency_loss_and_grad
from .loss_ops_continuous import continuous_policy_loss_and_grad
from ..muzero.loss_ops import soft_ce_slice_loss_and_grad
from ..zero.twohot_targets import mz_two_hot_target_batch


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


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
        loss += policy_coef * continuous_policy_loss_and_grad[B, ACT_DIM](
            musig, ptgt, pscale, gmusig,
            max_action, min_std, soft_clamp, init_std, ent_scale,
        )
        for b in range(B):
            for i in range(MU2):
                gpout[b * PRED_OUT + i] = gmusig[b * MU2 + i]
        # value: categorical soft-CE over [2*ACT_DIM, 2*ACT_DIM+BINS).
        mz_two_hot_target_batch[B, BINS](value_tgt + k * B, v_min, v_max, twv)
        loss += value_coef * soft_ce_slice_loss_and_grad[
            B, PRED_OUT, MU2, BINS
        ](pout, twv, gscale * value_coef, gpout)
        var gpout_t = TileTensor(gpout, row_major[B, PRED_OUT]())
        var gpin_t = TileTensor(gpin, row_major[B, LATENT]())
        pred.vjp["cpu", B](gpout_t, gpin_t)

        # (b) consistency online branch (k >= 1)
        if k >= 1:
            var projo_t = TileTensor(projo, row_major[B, PROJ]())
            proj.forward["cpu", B](zk_t, output=projo_t)
            var pk_t = TileTensor(pk, row_major[B, PROJ]())
            predh.forward["cpu", B](projo_t, output=pk_t)
            loss += consistency_loss_and_grad[B, PROJ](
                pk, tstore + (k - 1) * B * PROJ, cscale, gpk
            )
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
            loss += soft_ce_slice_loss_and_grad[B, DYN_OUT, LATENT, BINS](
                dout, twr, gscale, gdout
            )
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
    return loss / Scalar[DT](B)
