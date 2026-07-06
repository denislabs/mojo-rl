"""Dreamer 4 imagination-RL losses (Phase 4 / paper §3.3, eq. 10 + 11).

After behavior cloning, the policy is finetuned by reinforcement learning on
*imagined* rollouts (no environment interaction): the world-model transformer
is FROZEN and only the policy + value heads update. Two losses train on the
imagined trajectory of agent states s_t = h_t:

  • Value head (eq. 10) — TD-λ. Compute the λ-return from the imagined rewards
    and values, then a symexp-twohot cross-entropy of the value logits against
    sg(R_t^λ). Dreamer 4's reward/continue heads are trained (`bc_loss`) against
    the raw buffer arrays, which use the gym "reward-with-action" (LEAVING)
    convention — rew[t]/done[t] belong to the transition a_t : s_t → s_{t+1} —
    so the immediate reward/continue of state t sit at index t (see
    `lambda_returns` for the full note; this is where Dreamer 4 legitimately
    differs from DreamerV3's t+1 indexing, which shifts its reward head onto the
    arriving obs instead):
        R_t^λ = r_t + γ c_t [ (1−λ) v_{t+1} + λ R_{t+1}^λ ],   R_{H−1}^λ = v_{H−1}
        L_value = Σ_t  twohot_ce( v_logits_t , sg(R_t^λ) )

  • Policy head (eq. 11) — PMPO. A *robust* objective that uses only the SIGN
    of the advantage A_t = R_t^λ − v_t (no return/advantage normalization).
    States split into D⁺ = {A_t ≥ 0} and D⁻ = {A_t < 0}; a maximum-likelihood
    loss is averaged separately over each set, balanced by α, plus a behavioral
    prior term (a frozen copy of the BC policy) via a REVERSE KL:
        L_policy = −(1−α)/|D⁺| Σ_{D⁺} ln π_θ(a_i|s_i)
                   +   α  /|D⁻| Σ_{D⁻} ln π_θ(a_i|s_i)
                   +   β  / N   Σ_i KL[ π_θ(·|s_i) ‖ π_prior(·|s_i) ]
    with α = 0.5 (equal focus on positive/negative feedback) and β = 0.3 (a
    weaker prior scale). The "reverse" direction KL[π_θ ‖ π_prior] (vs the
    original PMPO's KL[π_prior ‖ π_θ]) better constrains π_θ to reasonable
    behaviors.

λ-returns follow the validated DreamerV3 recurrence (`dreamerv3/imag_loss.mojo`
/`repl_loss.mojo`): reward + bootstrap indexed at t+1, returns over t∈[0,H−2),
advantage adv[t] = ret[t] − val[t]. The categorical policy reuses
`dreamerv3/dists_discrete` (unimix softmax, logp/entropy + their gradients);
the reverse-KL gradient is derived here (net-new). All CPU flat-pointer scalar
helpers, matching the BC/shortcut loss style; GPU deferred.
"""

from std.memory import alloc
from std.math import log, exp

from mojo_rl.nn.constants import DT
from .heads import Dreamer4ValueHead
from ..dreamerv3.twohot import twohot_pred, twohot_loss, twohot_loss_backward


# ─────────────────────────────────────────────────────────────────────────
# Continue/termination head (DreamerV3-style `cont`): a single binary logit per
# state; ĉ = sigmoid(logit) = P(non-terminal). Trained by binary cross-entropy
# vs the real continue flag (1−done); used to discount the λ-return.
# ─────────────────────────────────────────────────────────────────────────
@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    # numerically-stable logistic
    if x >= Scalar[DT](0.0):
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


def continue_pred[
    N: Int,
    logits_o: Origin[mut=True],
    out_c_o: Origin[mut=True],
](
    logits: UnsafePointer[Scalar[DT], logits_o],   # [N]
    out_c: UnsafePointer[Scalar[DT], out_c_o],    # OUT [N] = sigmoid
):
    for i in range(N):
        out_c[i] = _sigmoid(logits[i])


def continue_bce_loss[
    N: Int,
    logits_o: Origin[mut=True],
    target_o: Origin[mut=True],
](
    logits: UnsafePointer[Scalar[DT], logits_o],   # [N] continue logits
    target: UnsafePointer[Scalar[DT], target_o],   # [N] continue flag 0/1
) raises -> Float64:
    """Σ binary cross-entropy of the continue flag (mean is the caller's job)."""
    var loss = Float64(0.0)
    for i in range(N):
        var z = Float64(logits[i])
        var y = Float64(target[i])
        # stable BCE: max(z,0) − z·y + log(1+exp(−|z|))
        var az = z if z >= 0.0 else -z
        var mz = z if z >= 0.0 else 0.0
        loss += mz - z * y + log(1.0 + exp(-az))
    return loss


def continue_bce_backward[
    N: Int,
    logits_o: Origin[mut=True],
    target_o: Origin[mut=True],
    grad_logits_o: Origin[mut=True],
](
    logits: UnsafePointer[Scalar[DT], logits_o],   # [N]
    target: UnsafePointer[Scalar[DT], target_o],   # [N]
    upstream: Scalar[DT],
    grad_logits: UnsafePointer[Scalar[DT], grad_logits_o],  # [N] (zeroed+filled)
):
    """∂BCE/∂logit = upstream·(σ(logit) − target)."""
    for i in range(N):
        grad_logits[i] = upstream * (_sigmoid(logits[i]) - target[i])
from ..dreamerv3.dists_discrete import (
    cat_softmax_mix,
    cat_fwd,
    cat_bwd,
    UNIMIX,
)


# ─────────────────────────────────────────────────────────────────────────
# λ-returns (eq. 10) — "leaving"-convention recurrence.
#
# CONVENTION (differs from DreamerV3's index-t+1 recurrence — see below):
# Dreamer 4's reward/continue heads are trained by `bc_loss` against the raw
# buffer arrays: reward_head(h_t) → rewards[t], continue_head(h_t) → 1−done[t].
# The online buffer records `add_step(obs_t, a_t, done_t, r_t)`, so rewards[t]
# and done[t] are the reward and termination of the transition LEAVING state t
# (a_t : s_t → s_{t+1}) — the gym "reward-with-action" convention. Hence the
# immediate reward/continue of state t are at index t (NOT t+1):
#     R_t = rew[t] + con[t]·[ (1−λ)·val[t+1] + λ·R_{t+1} ],   R_{H−1} = v_{H−1}
# with con[t] = γ·(1−done[t]) — the discount on the FUTURE (val[t+1], R_{t+1}),
# never on the immediate reward.
#
# DreamerV3 uses `rew[t+1]`/`con[t+1]` because ITS training batch SHIFTS the
# reward head onto the ARRIVING observation (`blocks.WMStep`: state←obs[t+1],
# reward target←rew[t]), so its reward head learns the "arriving" convention.
# Dreamer 4 applies no such shift, so mirroring DreamerV3's t+1 indexing here
# was an off-by-one: it dropped each state's own reward and misattributed
# sparse reward credit one step (breaking imagination RL on reward-bearing
# envs). rew/val/con are [B,H]; returns are produced for t ∈ [0, H−1).
# ─────────────────────────────────────────────────────────────────────────
@always_inline
def lambda_returns[
    B: Int, H: Int,
    rew_o: Origin[mut=True],
    val_o: Origin[mut=True],
    con_o: Origin[mut=True],
    out_ret_o: Origin[mut=True],
](
    rew: UnsafePointer[Scalar[DT], rew_o],   # [B,H]  reward LEAVING state t
    val: UnsafePointer[Scalar[DT], val_o],   # [B,H]
    con: UnsafePointer[Scalar[DT], con_o],   # [B,H]  = γ·(1−done[t]) LEAVING t
    lam: Scalar[DT],
    out_ret: UnsafePointer[Scalar[DT], out_ret_o],  # OUT [B,H-1]
):
    comptime assert H >= 2, "lambda_returns needs H >= 2"
    comptime HM1 = H - 1
    for b in range(B):
        var ret_next = val[b * H + (H - 1)]            # R_{H-1}^λ = v_{H-1}
        var t = H - 2
        while t >= 0:
            var live = con[b * H + t]                  # γ·(1−done)_t (LEAVING t)
            var interm = (
                rew[b * H + t]                         # immediate reward LEAVING t
                + (Scalar[DT](1.0) - lam) * live * val[b * H + t + 1]
            )
            var cur = interm + live * lam * ret_next
            out_ret[b * HM1 + t] = cur
            ret_next = cur
            t -= 1


# ─────────────────────────────────────────────────────────────────────────
# Value TD loss (eq. 10): twohot CE of value logits vs sg(λ-return).
# vlogits is [B,H,BINS]; trained over the states t ∈ [0,H−1) that have a return.
# ─────────────────────────────────────────────────────────────────────────
def value_td_loss_cpu[
    B: Int, H: Int, BINS: Int,
    vlogits_o: Origin[mut=True],
    bins_o: Origin[mut=True],
    ret_o: Origin[mut=True],
    out_loss_o: Origin[mut=True],
](
    vlogits: UnsafePointer[Scalar[DT], vlogits_o],  # [B,H,BINS]
    bins: UnsafePointer[Scalar[DT], bins_o],     # [BINS]
    ret: UnsafePointer[Scalar[DT], ret_o],      # [B,H-1] sg target
    out_loss: UnsafePointer[Scalar[DT], out_loss_o],  # OUT [B,H-1]
) raises:
    comptime HM1 = H - 1
    for b in range(B):
        for t in range(HM1):
            out_loss[b * HM1 + t] = twohot_loss[BINS](
                vlogits, (b * H + t) * BINS, bins, ret[b * HM1 + t]
            )


def value_td_loss_backward[
    B: Int, H: Int, BINS: Int,
    vlogits_o: Origin[mut=True],
    bins_o: Origin[mut=True],
    ret_o: Origin[mut=True],
    d_loss_o: Origin[mut=True],
    grad_vlogits_o: Origin[mut=True],
](
    vlogits: UnsafePointer[Scalar[DT], vlogits_o],  # [B,H,BINS]
    bins: UnsafePointer[Scalar[DT], bins_o],     # [BINS]
    ret: UnsafePointer[Scalar[DT], ret_o],      # [B,H-1]
    d_loss: UnsafePointer[Scalar[DT], d_loss_o],   # [B,H-1] cotangent
    grad_vlogits: UnsafePointer[Scalar[DT], grad_vlogits_o],  # [B,H,BINS]
) raises:
    """Backward of `value_td_loss_cpu` (target sg'd). grad_vlogits ZEROED then
    accumulated."""
    comptime HM1 = H - 1
    for i in range(B * H * BINS):
        grad_vlogits[i] = 0.0
    for b in range(B):
        for t in range(HM1):
            twohot_loss_backward[BINS](
                vlogits,
                (b * H + t) * BINS,
                bins,
                ret[b * HM1 + t],
                d_loss[b * HM1 + t],
                grad_vlogits,
            )


# ─────────────────────────────────────────────────────────────────────────
# Reverse-KL of two unimix categoricals (net-new): KL[π_θ ‖ π_prior].
#   p = unimix(softmax(logits)),  q = unimix(softmax(prior_logits)) [frozen]
#   KL  = Σ_a p_a (ln p_a − ln q_a)
#   dKL/dlogits_j = (1−u)·sm_j·( g_j − Σ_a sm_a·g_a ),  g_a = ln p_a + 1 − ln q_a
# `sm` is the PRE-mix softmax of the policy logits (mixed probs are `p`).
# ─────────────────────────────────────────────────────────────────────────
@always_inline
def _reverse_kl_fwd[
    C: Int,
    p_o: Origin[mut=True],
    q_o: Origin[mut=True],
](
    p: UnsafePointer[Scalar[DT], p_o],   # [C] policy mixed probs
    q: UnsafePointer[Scalar[DT], q_o],   # [C] prior  mixed probs
) -> Scalar[DT]:
    var kl = Scalar[DT](0.0)
    for a in range(C):
        kl += p[a] * (log(p[a]) - log(q[a]))
    return kl


@always_inline
def _reverse_kl_bwd[
    C: Int,
    sm_o: Origin[mut=True],
    p_o: Origin[mut=True],
    q_o: Origin[mut=True],
    grad_logits_o: Origin[mut=True],
](
    sm: UnsafePointer[Scalar[DT], sm_o],  # [C] policy PRE-mix softmax
    p: UnsafePointer[Scalar[DT], p_o],   # [C] policy mixed probs
    q: UnsafePointer[Scalar[DT], q_o],   # [C] prior  mixed probs
    u: Scalar[DT],
    upstream: Scalar[DT],
    grad_logits: UnsafePointer[Scalar[DT], grad_logits_o],  # accumulate [.,C]
    base: Int,
):
    var one_m_u = Scalar[DT](1.0) - u
    # g_a = ln p_a + 1 − ln q_a ; shared dot = Σ_a sm_a·g_a
    var dot = Scalar[DT](0.0)
    for a in range(C):
        var g = log(p[a]) + Scalar[DT](1.0) - log(q[a])
        dot += sm[a] * g
    for j in range(C):
        var gj = log(p[j]) + Scalar[DT](1.0) - log(q[j])
        grad_logits[base + j] += upstream * one_m_u * sm[j] * (gj - dot)


# ─────────────────────────────────────────────────────────────────────────
# PMPO policy loss (eq. 11). States indexed flat i = b·(H−1) + t over the
# H−1 imagined steps that have an advantage. `actions` are the sampled class
# ids (stop-grad); `adv` = ret − val (sign only is used). Returns the scalar
# loss and fills the policy-logit gradient.
# ─────────────────────────────────────────────────────────────────────────
def pmpo_policy_loss_cpu[
    B: Int, H: Int, NACT: Int,
    plogits_o: Origin[mut=True],
    prior_logits_o: Origin[mut=True],
    actions_o: Origin[mut=True],
    adv_o: Origin[mut=True],
](
    plogits: UnsafePointer[Scalar[DT], plogits_o],       # [B,H,NACT]
    prior_logits: UnsafePointer[Scalar[DT], prior_logits_o],  # [B,H,NACT] frozen
    actions: UnsafePointer[Scalar[DT], actions_o],       # [B,H] class ids
    adv: UnsafePointer[Scalar[DT], adv_o],           # [B,H-1]
    alpha: Scalar[DT],
    beta: Scalar[DT],
) raises -> Float64:
    """Forward-only PMPO loss (eq. 11). Pure (no grad); use the `_backward`
    twin to fill gradients. Returns the total scalar loss."""
    comptime HM1 = H - 1
    var sm = alloc[Scalar[DT]](NACT)
    var pp = alloc[Scalar[DT]](NACT)
    var qsm = alloc[Scalar[DT]](NACT)
    var qp = alloc[Scalar[DT]](NACT)

    # |D⁺|, |D⁻|
    var n_pos = 0
    var n_neg = 0
    for b in range(B):
        for t in range(HM1):
            if adv[b * HM1 + t] >= Scalar[DT](0.0):
                n_pos += 1
            else:
                n_neg += 1
    var inv_pos = Scalar[DT](1.0) / Scalar[DT](n_pos) if n_pos > 0 else Scalar[DT](0.0)
    var inv_neg = Scalar[DT](1.0) / Scalar[DT](n_neg) if n_neg > 0 else Scalar[DT](0.0)
    var N = B * HM1
    var inv_N = Scalar[DT](1.0) / Scalar[DT](N) if N > 0 else Scalar[DT](0.0)

    var loss = Scalar[DT](0.0)
    for b in range(B):
        for t in range(HM1):
            var base = (b * H + t) * NACT
            var k = Int(Float64(actions[b * H + t]) + 0.5)
            # current-policy logp(a) (also fills sm/pp)
            var r = cat_fwd[NACT](plogits, base, UNIMIX, k, sm, pp)
            var logp = r[0]
            # prior probs at this state
            cat_softmax_mix[NACT](prior_logits, base, UNIMIX, qsm, qp)
            # max-likelihood term (sign of advantage)
            if adv[b * HM1 + t] >= Scalar[DT](0.0):
                loss += -(Scalar[DT](1.0) - alpha) * inv_pos * logp
            else:
                loss += alpha * inv_neg * logp
            # reverse-KL prior term
            loss += beta * inv_N * _reverse_kl_fwd[NACT](pp, qp)

    sm.free()
    pp.free()
    qsm.free()
    qp.free()
    return Float64(loss)


def pmpo_policy_loss_backward[
    B: Int, H: Int, NACT: Int,
    plogits_o: Origin[mut=True],
    prior_logits_o: Origin[mut=True],
    actions_o: Origin[mut=True],
    adv_o: Origin[mut=True],
    grad_plogits_o: Origin[mut=True],
](
    plogits: UnsafePointer[Scalar[DT], plogits_o],       # [B,H,NACT]
    prior_logits: UnsafePointer[Scalar[DT], prior_logits_o],  # [B,H,NACT]
    actions: UnsafePointer[Scalar[DT], actions_o],       # [B,H]
    adv: UnsafePointer[Scalar[DT], adv_o],           # [B,H-1]
    alpha: Scalar[DT],
    beta: Scalar[DT],
    upstream: Scalar[DT],
    grad_plogits: UnsafePointer[Scalar[DT], grad_plogits_o],  # [B,H,NACT]
) raises:
    """Backward of `pmpo_policy_loss_cpu` w.r.t. the policy logits (advantages,
    actions, and prior are sg'd). grad_plogits ZEROED then accumulated.
    `upstream` scales the whole loss (typically 1.0)."""
    comptime HM1 = H - 1
    for i in range(B * H * NACT):
        grad_plogits[i] = 0.0

    var sm = alloc[Scalar[DT]](NACT)
    var pp = alloc[Scalar[DT]](NACT)
    var qsm = alloc[Scalar[DT]](NACT)
    var qp = alloc[Scalar[DT]](NACT)

    var n_pos = 0
    var n_neg = 0
    for b in range(B):
        for t in range(HM1):
            if adv[b * HM1 + t] >= Scalar[DT](0.0):
                n_pos += 1
            else:
                n_neg += 1
    var inv_pos = Scalar[DT](1.0) / Scalar[DT](n_pos) if n_pos > 0 else Scalar[DT](0.0)
    var inv_neg = Scalar[DT](1.0) / Scalar[DT](n_neg) if n_neg > 0 else Scalar[DT](0.0)
    var N = B * HM1
    var inv_N = Scalar[DT](1.0) / Scalar[DT](N) if N > 0 else Scalar[DT](0.0)

    for b in range(B):
        for t in range(HM1):
            var base = (b * H + t) * NACT
            var k = Int(Float64(actions[b * H + t]) + 0.5)
            cat_softmax_mix[NACT](plogits, base, UNIMIX, sm, pp)
            cat_softmax_mix[NACT](prior_logits, base, UNIMIX, qsm, qp)
            # max-likelihood: d_logp coefficient by advantage sign
            var d_logp: Scalar[DT]
            if adv[b * HM1 + t] >= Scalar[DT](0.0):
                d_logp = -(Scalar[DT](1.0) - alpha) * inv_pos
            else:
                d_logp = alpha * inv_neg
            cat_bwd[NACT](
                sm, pp, UNIMIX, k, upstream * d_logp, Scalar[DT](0.0),
                grad_plogits, base,
            )
            # reverse-KL prior
            _reverse_kl_bwd[NACT](
                sm, pp, qp, UNIMIX, upstream * beta * inv_N, grad_plogits, base
            )

    sm.free()
    pp.free()
    qsm.free()
    qp.free()
