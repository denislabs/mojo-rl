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

Cache discipline (the nn2 re-forward-before-vjp idiom): the target pre-pass runs
``h(obs_k)`` which clobbers the representation net's forward cache, so the final
``rep.vjp`` is preceded by a fresh ``rep.forward(obs0)``. Within the reverse
scan, ``g_proj``/``h_pred`` are re-forwarded on the *online* input immediately
before their ``vjp`` so their caches hold the live (online) activations.

Batch layout is time-major like MuZero, except obs is the **full sequence**
``obs_seq[K+1, B, OBS]`` (``obs_seq[0] == obs0``) so the consistency targets can
encode the real future observations: ``actions[K,B]`` (indices),
``policy_tgt[K+1,B,ACT]``, ``value_tgt[K+1,B]``, ``reward_tgt[K,B]`` (raw).

CPU path first (overfit-tested); a GPU branch + CPU↔GPU parity follow.
"""

from std.memory import alloc
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam

from .loss_ops import consistency_loss_and_grad
from ..muzero.loss_ops import soft_ce_slice_loss_and_grad
from ..zero.twohot_targets import mz_two_hot_target_batch


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


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
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
    policy_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
) raises -> Scalar[DT]:
    """One CPU EZv2 unroll training step (MuZero BPTT + SimSiam consistency).

    Returns the mean total loss (policy + value + reward + consistency). Mutates
    all five nets via their optimizers. ``obs_seq`` is the time-major
    ``[K+1, B, OBS]`` observation sequence (``obs_seq[0]`` is the root obs).
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
    for rk in range(K + 1):
        var k = K - rk
        var zk = zst + k * B * LATENT
        var zk_t = TileTensor(zk, row_major[B, LATENT]())

        # (a) prediction head: re-forward for cache, seed grads, vjp → grad z_k
        var pout_t = TileTensor(pout, row_major[B, PRED_OUT]())
        pred.forward["cpu", B](zk_t, output=pout_t)
        loss += soft_ce_slice_loss_and_grad[B, PRED_OUT, 0, ACT](
            pout, policy_tgt + k * B * ACT, gscale, gpout
        )
        mz_two_hot_target_batch[B, BINS](value_tgt + k * B, v_min, v_max, twv)
        loss += value_coef * soft_ce_slice_loss_and_grad[
            B, PRED_OUT, ACT, BINS
        ](pout, twv, gscale * value_coef, gpout)
        var gpout_t = TileTensor(gpout, row_major[B, PRED_OUT]())
        var gpin_t = TileTensor(gpin, row_major[B, LATENT]())
        pred.vjp["cpu", B](gpout_t, gpin_t)

        # (b) consistency online branch (k >= 1): p_k = h_pred(g_proj(z_k))
        if k >= 1:
            var projo_t = TileTensor(projo, row_major[B, PROJ]())
            proj.forward["cpu", B](zk_t, output=projo_t)   # refresh proj cache
            var pk_t = TileTensor(pk, row_major[B, PROJ]())
            predh.forward["cpu", B](projo_t, output=pk_t)  # refresh predh cache
            loss += consistency_loss_and_grad[B, PROJ](
                pk, tstore + (k - 1) * B * PROJ, cscale, gpk
            )
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
    return loss / Scalar[DT](B)
