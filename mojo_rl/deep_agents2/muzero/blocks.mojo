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
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam

from .loss_ops import soft_ce_slice_loss_and_grad
from ..zero.twohot_targets import mz_two_hot_target_batch


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


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
    for rk in range(K + 1):
        var k = K - rk
        var zk = zst + k * B * LATENT
        var zk_t = TileTensor(zk, row_major[B, LATENT]())

        # (a) prediction head: re-forward for cache, seed grads, vjp → grad z_k
        var pout_t = TileTensor(pout, row_major[B, PRED_OUT]())
        pred.forward["cpu", B](zk_t, output=pout_t)
        # policy slice [0, ACT)
        loss += soft_ce_slice_loss_and_grad[B, PRED_OUT, 0, ACT](
            pout, policy_tgt + k * B * ACT, gscale, gpout
        )
        # value slice [ACT, ACT+BINS)
        mz_two_hot_target_batch[B, BINS](value_tgt + k * B, v_min, v_max, twv)
        loss += value_coef * soft_ce_slice_loss_and_grad[
            B, PRED_OUT, ACT, BINS
        ](pout, twv, gscale * value_coef, gpout)
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
            loss += soft_ce_slice_loss_and_grad[B, DYN_OUT, LATENT, BINS](
                dout, twr, gscale, gdout
            )
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
    return loss / Scalar[DT](B)
