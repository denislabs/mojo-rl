"""DreamerV3 actor-critic-in-imagination losses (forward).

Ports `dreamerv3/agent.py:imag_loss` for the v1 config: `contdisc=True`
(disc=1), `slowtar=False` (tarval = online value pred), `valnorm`/`advnorm`
= `none` (identity stats), `retnorm` = `perc`. Composes TwoHot pred/loss,
`bounded_normal` logp/entropy, λ-return (PR2), and PercentileNormalize (PR2).

  val      = value.pred()                       # tarval (slowtar False)
  weight   = cumprod(con, axis=T)               # disc=1
  ret      = λ-return(last=0, term=1-con, rew, boot=val, disc=1, lam)  [.,T-1]
  rscale   = retnorm(ret).scale                  # perc EMA
  adv      = (ret - val[:, :-1]) / rscale        # advnorm none → identity
  logpi    = Σ_a logp(act); ent = Σ_a entropy
  policy_loss = weight[:-1] · −(logpi·adv + actent·ent)
  value_loss  = weight[:-1] · (twohot_ce(vlogits, ret) +
                               slowreg·twohot_ce(vlogits, slowval.pred()))

Outputs `policy_loss`, `value_loss`, `ret` are `[BK, T-1]`. Forward only
(PR5a). Validated ≤1e-4 vs the actual reference (extract_pr5.py).
"""

from std.memory import alloc
from std.math import tanh, exp

from mojo_rl.nn.constants import DT
from .twohot import twohot_pred, twohot_loss, twohot_loss_backward
from .dists import bounded_mean, bounded_std, normal_logp, normal_entropy
from .dists_discrete import cat_fwd, cat_bwd, cat_softmax_mix, UNIMIX
from .normalize import PercentileNormalize


@always_inline
def _argmax[
    ACT: Int,
    act_o: Origin[mut=True],
](act: UnsafePointer[Scalar[DT], act_o], base: Int) -> Int:
    """Chosen class = argmax of the one-hot action over ACT lanes."""
    var k = 0
    var best = act[base]
    for a in range(1, ACT):
        if act[base + a] > best:
            best = act[base + a]
            k = a
    return k


def imag_loss_cpu[
    BK: Int, T: Int, ACT: Int, BINS: Int, DISCRETE: Bool = False,
    act_o: Origin[mut=True] = MutAnyOrigin,
    rew_o: Origin[mut=True] = MutAnyOrigin,
    con_o: Origin[mut=True] = MutAnyOrigin,
    vlogits_o: Origin[mut=True] = MutAnyOrigin,
    svlogits_o: Origin[mut=True] = MutAnyOrigin,
    pmean_o: Origin[mut=True] = MutAnyOrigin,
    pstd_raw_o: Origin[mut=True] = MutAnyOrigin,
    bins_o: Origin[mut=True] = MutAnyOrigin,
](
    act: UnsafePointer[Scalar[DT], act_o],  # [BK,T,ACT]
    rew: UnsafePointer[Scalar[DT], rew_o],  # [BK,T]
    con: UnsafePointer[Scalar[DT], con_o],  # [BK,T]
    vlogits: UnsafePointer[Scalar[DT], vlogits_o],  # [BK,T,BINS]
    svlogits: UnsafePointer[Scalar[DT], svlogits_o],  # [BK,T,BINS]
    pmean: UnsafePointer[Scalar[DT], pmean_o],  # [BK,T,ACT] raw
    pstd_raw: UnsafePointer[Scalar[DT], pstd_raw_o],  # [BK,T,ACT]
    bins: UnsafePointer[Scalar[DT], bins_o],  # [BINS]
    minstd: Scalar[DT],
    maxstd: Scalar[DT],
    lam: Scalar[DT],
    actent: Scalar[DT],
    slowreg: Scalar[DT],
    mut retnorm: PercentileNormalize,
    mut out_policy_loss: List[Scalar[DT]],  # [BK,T-1]
    mut out_value_loss: List[Scalar[DT]],  # [BK,T-1]
    mut out_ret: List[Scalar[DT]],  # [BK,T-1]
    slowtar: Bool = False,  # bootstrap λ-return from slowvalue (EMA target)
) raises:
    comptime assert T >= 2, "imag_loss needs T >= 2"
    comptime TM1 = T - 1

    # val[b,t] = TwoHot(vlogits).pred ; slowval[b,t] = TwoHot(svlogits).pred
    var val = alloc[Scalar[DT]](BK * T)
    var slowval = alloc[Scalar[DT]](BK * T)
    for b in range(BK):
        for t in range(T):
            val[b * T + t] = twohot_pred[BINS](
                vlogits, (b * T + t) * BINS, bins
            )
            slowval[b * T + t] = twohot_pred[BINS](
                svlogits, (b * T + t) * BINS, bins
            )

    # weight = cumprod(con) along T
    var weight = alloc[Scalar[DT]](BK * T)
    for b in range(BK):
        var acc = Scalar[DT](1.0)
        for t in range(T):
            acc *= con[b * T + t]
            weight[b * T + t] = acc

    # ret = λ-return(last=0, term=1-con, rew, boot=tarval, disc=1, lam) → [BK,T-1]
    # slowtar=True bootstraps from the EMA slowvalue (target network), breaking
    # the online value→return→value self-feedback loop that runs away at higher
    # learning rates. slowtar=False = bootstrap from the online value (the JAX
    # PR5a fixture convention; keeps the validation spike green).
    for b in range(BK):
        var ret_next = slowval[b * T + (T - 1)] if slowtar else val[
            b * T + (T - 1)
        ]
        var t = T - 2
        while t >= 0:
            var live = con[b * T + t + 1]  # (1-term)*disc
            var cont = lam  # (1-last)*lam
            var vboot = slowval[b * T + t + 1] if slowtar else val[
                b * T + t + 1
            ]
            var interm = (
                rew[b * T + t + 1] + (Scalar[DT](1.0) - cont) * live * vboot
            )
            var cur = interm + live * cont * ret_next
            out_ret[b * TM1 + t] = cur
            ret_next = cur
            t -= 1

    # retnorm (perc) update + stats → rscale
    retnorm.update(out_ret, BK * TM1)
    var rstats = retnorm.stats()
    var rscale = rstats[1]

    for b in range(BK):
        for t in range(TM1):
            var adv = (out_ret[b * TM1 + t] - val[b * T + t]) / rscale
            # policy: logp / entropy of the (b,t) action distribution
            var logpi = Scalar[DT](0.0)
            var ent = Scalar[DT](0.0)
            comptime if DISCRETE:
                # unimix categorical: `pmean` holds logits[BK,T,ACT]; `act` is
                # the sampled one-hot. k = argmax(act). `pstd_raw` unused.
                var base = (b * T + t) * ACT
                var k = _argmax[ACT](act, base)
                var sm = alloc[Scalar[DT]](ACT)
                var pp = alloc[Scalar[DT]](ACT)
                var r = cat_fwd[ACT](pmean, base, UNIMIX, k, sm, pp)
                logpi = r[0]
                ent = r[1]
                sm.free()
                pp.free()
            else:
                # bounded_normal: Σ_a logp / entropy over the action dim
                for a in range(ACT):
                    var idx = (b * T + t) * ACT + a
                    var mean = bounded_mean(pmean[idx])
                    var std = bounded_std(pstd_raw[idx], minstd, maxstd)
                    logpi += normal_logp(act[idx], mean, std)
                    ent += normal_entropy(std)
            out_policy_loss[b * TM1 + t] = weight[b * T + t] * -(
                logpi * adv + actent * ent
            )
            # value: twohot CE vs ret and vs slowval.pred (both at vlogits[b,t])
            var l1 = twohot_loss[BINS](
                vlogits, (b * T + t) * BINS, bins, out_ret[b * TM1 + t]
            )
            var l2 = twohot_loss[BINS](
                vlogits, (b * T + t) * BINS, bins, slowval[b * T + t]
            )
            out_value_loss[b * TM1 + t] = weight[b * T + t] * (
                l1 + slowreg * l2
            )

    val.free()
    slowval.free()
    weight.free()


def imag_loss_backward[
    BK: Int, T: Int, ACT: Int, BINS: Int, DISCRETE: Bool = False,
    act_o: Origin[mut=True] = MutAnyOrigin,
    rew_o: Origin[mut=True] = MutAnyOrigin,
    con_o: Origin[mut=True] = MutAnyOrigin,
    vlogits_o: Origin[mut=True] = MutAnyOrigin,
    svlogits_o: Origin[mut=True] = MutAnyOrigin,
    pmean_o: Origin[mut=True] = MutAnyOrigin,
    pstd_raw_o: Origin[mut=True] = MutAnyOrigin,
    bins_o: Origin[mut=True] = MutAnyOrigin,
    d_policy_o: Origin[mut=True] = MutAnyOrigin,
    d_value_o: Origin[mut=True] = MutAnyOrigin,
    grad_vlogits_o: Origin[mut=True] = MutAnyOrigin,
    grad_pmean_o: Origin[mut=True] = MutAnyOrigin,
    grad_pstd_raw_o: Origin[mut=True] = MutAnyOrigin,
](
    act: UnsafePointer[Scalar[DT], act_o],  # [BK,T,ACT]
    rew: UnsafePointer[Scalar[DT], rew_o],  # [BK,T]
    con: UnsafePointer[Scalar[DT], con_o],  # [BK,T]
    vlogits: UnsafePointer[Scalar[DT], vlogits_o],  # [BK,T,BINS]
    svlogits: UnsafePointer[Scalar[DT], svlogits_o],  # [BK,T,BINS]
    pmean: UnsafePointer[Scalar[DT], pmean_o],  # [BK,T,ACT] raw
    pstd_raw: UnsafePointer[Scalar[DT], pstd_raw_o],  # [BK,T,ACT]
    bins: UnsafePointer[Scalar[DT], bins_o],  # [BINS]
    minstd: Scalar[DT],
    maxstd: Scalar[DT],
    lam: Scalar[DT],
    actent: Scalar[DT],
    slowreg: Scalar[DT],
    rscale: Scalar[DT],  # from forward (sg'd)
    d_policy: UnsafePointer[Scalar[DT], d_policy_o],  # [BK,T-1] cotangent
    d_value: UnsafePointer[Scalar[DT], d_value_o],  # [BK,T-1] cotangent
    grad_vlogits: UnsafePointer[Scalar[DT], grad_vlogits_o],  # [BK,T,BINS]
    grad_pmean: UnsafePointer[Scalar[DT], grad_pmean_o],  # [BK,T,ACT]
    grad_pstd_raw: UnsafePointer[Scalar[DT], grad_pstd_raw_o],  # [BK,T,ACT]
    slowtar: Bool = False,  # bootstrap λ-return from slowvalue (EMA target)
) raises:
    """Backward of `imag_loss_cpu`. retnorm (`rscale`), adv, weight are
    treated as constants (the reference stop-grads them). Grads flow to the
    value logits (twohot CE) and the policy mean/std raw (logp + entropy).
    Outputs are ZEROED then written/accumulated."""
    comptime assert T >= 2, "imag_loss needs T >= 2"
    comptime TM1 = T - 1
    for i in range(BK * T * BINS):
        grad_vlogits[i] = 0.0
    for i in range(BK * T * ACT):
        grad_pmean[i] = 0.0
        grad_pstd_raw[i] = 0.0

    # recompute val (= tarval), slowval, weight, ret
    var val = alloc[Scalar[DT]](BK * T)
    var slowval = alloc[Scalar[DT]](BK * T)
    var weight = alloc[Scalar[DT]](BK * T)
    var ret = alloc[Scalar[DT]](BK * TM1)
    for b in range(BK):
        var acc = Scalar[DT](1.0)
        for t in range(T):
            val[b * T + t] = twohot_pred[BINS](
                vlogits, (b * T + t) * BINS, bins
            )
            slowval[b * T + t] = twohot_pred[BINS](
                svlogits, (b * T + t) * BINS, bins
            )
            acc *= con[b * T + t]
            weight[b * T + t] = acc
    for b in range(BK):
        var ret_next = slowval[b * T + (T - 1)] if slowtar else val[
            b * T + (T - 1)
        ]
        var t = T - 2
        while t >= 0:
            var live = con[b * T + t + 1]
            var vboot = slowval[b * T + t + 1] if slowtar else val[
                b * T + t + 1
            ]
            var interm = (
                rew[b * T + t + 1] + (Scalar[DT](1.0) - lam) * live * vboot
            )
            var cur = interm + live * lam * ret_next
            ret[b * TM1 + t] = cur
            ret_next = cur
            t -= 1

    for b in range(BK):
        for t in range(TM1):
            var w = weight[b * T + t]
            var adv = (ret[b * TM1 + t] - val[b * T + t]) / rscale
            # ── policy grads ─────────────────────────────────────────
            # ∂loss/∂logpi = w·(−adv)·d_policy ; ∂loss/∂ent = w·(−actent)·d_policy
            var dpl_dlogp = d_policy[b * TM1 + t] * w * (-adv)
            var dpl_dent = d_policy[b * TM1 + t] * w * (-actent)
            comptime if DISCRETE:
                # unimix categorical: grads flow to `grad_pmean` (= logits);
                # `grad_pstd_raw` stays 0 (zeroed above). k = argmax(act).
                var base = (b * T + t) * ACT
                var k = _argmax[ACT](act, base)
                var sm = alloc[Scalar[DT]](ACT)
                var pp = alloc[Scalar[DT]](ACT)
                cat_softmax_mix[ACT](pmean, base, UNIMIX, sm, pp)
                cat_bwd[ACT](
                    sm, pp, UNIMIX, k, dpl_dlogp, dpl_dent, grad_pmean, base
                )
                sm.free()
                pp.free()
            else:
                for a in range(ACT):
                    var idx = (b * T + t) * ACT + a
                    var mean = tanh(pmean[idx])
                    var s = Scalar[DT](1.0) / (
                        Scalar[DT](1.0)
                        + exp(-(pstd_raw[idx] + Scalar[DT](2.0)))
                    )
                    var std = (maxstd - minstd) * s + minstd
                    var z = (act[idx] - mean) / std
                    var dlogp_dmean = z / std
                    var dlogp_dstd = (z * z - Scalar[DT](1.0)) / std
                    var dent_dstd = Scalar[DT](1.0) / std
                    var dmean_draw = Scalar[DT](1.0) - mean * mean
                    var dstd_draw = (
                        (maxstd - minstd) * s * (Scalar[DT](1.0) - s)
                    )
                    grad_pmean[idx] = dpl_dlogp * dlogp_dmean * dmean_draw
                    grad_pstd_raw[idx] = (
                        dpl_dlogp * dlogp_dstd + dpl_dent * dent_dstd
                    ) * dstd_draw
            # ── value grads (twohot CE vs ret and vs slowval) ────────
            var up = d_value[b * TM1 + t] * w
            twohot_loss_backward[BINS](
                vlogits,
                (b * T + t) * BINS,
                bins,
                ret[b * TM1 + t],
                up,
                grad_vlogits,
            )
            twohot_loss_backward[BINS](
                vlogits,
                (b * T + t) * BINS,
                bins,
                slowval[b * T + t],
                up * slowreg,
                grad_vlogits,
            )

    val.free()
    slowval.free()
    weight.free()
    ret.free()
