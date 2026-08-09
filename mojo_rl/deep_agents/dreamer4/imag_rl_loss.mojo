"""Dreamer 4 imagination-RL losses (Phase 4 / paper §3.3, eq. 10 + 11).

After behavior cloning, the policy is finetuned by reinforcement learning on
*imagined* rollouts (no environment interaction): the world-model transformer
is FROZEN and only the policy + value heads update. Two losses train on the
imagined trajectory of agent states s_t = h_t:

  • Value head (eq. 10) — TD-λ. Compute the λ-return from the imagined rewards
    and values, then a symexp-twohot cross-entropy of the value logits against
    sg(R_t^λ). Dreamer 4 trains the reward/continue heads on the ARRIVING reward
    via the shift in `agent.mojo` bc_train_step (rew_shift[f]=rewards[f−1], action
    token at f = a_{f−1}), so reward_head(h_t) = reward arriving at state t and
    the λ-return indexes rew/con at t+1 (see `lambda_returns` for the decisive
    imagination-rollout argument; matches jax train_policy.py + DreamerV3):
        R_t^λ = r_{t+1} + γ c_{t+1} [ (1−λ) v_{t+1} + λ R_{t+1}^λ ],  R_{H−1}^λ = v_{H−1}
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
    logits: Pointer[Scalar[DT], logits_o],   # [N]
    out_c: Pointer[Scalar[DT], out_c_o],    # OUT [N] = sigmoid
):
    for i in range(N):
        out_c[unsafe_offset=i] = _sigmoid(logits[unsafe_offset=i])


def continue_bce_loss[
    N: Int,
    logits_o: Origin[mut=True],
    target_o: Origin[mut=True],
](
    logits: Pointer[Scalar[DT], logits_o],   # [N] continue logits
    target: Pointer[Scalar[DT], target_o],   # [N] continue flag 0/1
) raises -> Float64:
    """Σ binary cross-entropy of the continue flag (mean is the caller's job)."""
    var loss = Float64(0.0)
    for i in range(N):
        var z = Float64(logits[unsafe_offset=i])
        var y = Float64(target[unsafe_offset=i])
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
    logits: Pointer[Scalar[DT], logits_o],   # [N]
    target: Pointer[Scalar[DT], target_o],   # [N]
    upstream: Scalar[DT],
    grad_logits: Pointer[Scalar[DT], grad_logits_o],  # [N] (zeroed+filled)
):
    """∂BCE/∂logit = upstream·(σ(logit) − target)."""
    for i in range(N):
        grad_logits[unsafe_offset=i] = upstream * (_sigmoid(logits[unsafe_offset=i]) - target[unsafe_offset=i])
from ..dreamerv3.dists_discrete import (
    cat_softmax_mix,
    cat_fwd,
    cat_bwd,
    UNIMIX,
)


# ─────────────────────────────────────────────────────────────────────────
# λ-returns (eq. 10) — ARRIVING-convention recurrence (rew/con at t+1).
#
#     R_t = rew[t+1] + con[t+1]·[ (1−λ)·val[t+1] + λ·R_{t+1} ],  R_{H−1} = v_{H−1}
#
# CONVENTION — why t+1 (do NOT "fix" this to t): Dreamer 4 trains the heads on
# the ARRIVING reward via an explicit SHIFT in `bc_train_step` (`agent.mojo`
# ~L765-776): rew_shift[f] = rewards[f−1] and the action token at frame f =
# one_hot(actions[f−1]) (the action leading INTO frame f); frame 0 is a dummy
# (0 reward, no in-window action). So reward_head(h_t) predicts the reward that
# ARRIVED at state t (from a_{t−1}), and h_t "contains" that action — matching
# the jax reference (train_bc_rew_heads.py: "predict r_t from h_t which contains
# a_t") and DreamerV3's obs[t+1]/rew[t] WMStep shift.
#
# Decisive, convention-independent check in imagination: `imag_rollout` samples
# a'_i from h_i and writes it as the action token of frame i+1, and sets
# out_rew[i]=reward_head(h_i). Hence out_rew[t+1] is the reward for the action
# out_act[t] sampled at state t — so the λ-return that credits out_act[t]
# (advantage A_t = R_t − val[t]) MUST use rew[t+1]/con[t+1]. Using rew[t] credits
# the PREVIOUS action's reward to the current action (a one-step misattribution,
# catastrophic on sparse-reward envs). Validated vs jax train_policy.py:1409-1424.
# rew/val/con are [B,H]; returns are produced for t ∈ [0, H−1).
# ─────────────────────────────────────────────────────────────────────────
@always_inline
def lambda_returns[
    B: Int, H: Int,
    rew_o: Origin[mut=True],
    val_o: Origin[mut=True],
    con_o: Origin[mut=True],
    out_ret_o: Origin[mut=True],
](
    rew: Pointer[Scalar[DT], rew_o],   # [B,H]  reward ARRIVING at state t
    val: Pointer[Scalar[DT], val_o],   # [B,H]
    con: Pointer[Scalar[DT], con_o],   # [B,H]  = γ·(1−done) ARRIVING at t
    lam: Scalar[DT],
    out_ret: Pointer[Scalar[DT], out_ret_o],  # OUT [B,H-1]
):
    comptime assert H >= 2, "lambda_returns needs H >= 2"
    comptime HM1 = H - 1
    for b in range(B):
        var ret_next = val[unsafe_offset=b * H + (H - 1)]            # R_{H-1}^λ = v_{H-1}
        var t = H - 2
        while t >= 0:
            var live = con[unsafe_offset=b * H + t + 1]              # γ·(1−done)_{t+1}
            var interm = (
                rew[unsafe_offset=b * H + t + 1]                     # reward for a'_t (arrives t+1)
                + (Scalar[DT](1.0) - lam) * live * val[unsafe_offset=b * H + t + 1]
            )
            var cur = interm + live * lam * ret_next
            out_ret[unsafe_offset=b * HM1 + t] = cur
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
    vlogits: Pointer[Scalar[DT], vlogits_o],  # [B,H,BINS]
    bins: Pointer[Scalar[DT], bins_o],     # [BINS]
    ret: Pointer[Scalar[DT], ret_o],      # [B,H-1] sg target
    out_loss: Pointer[Scalar[DT], out_loss_o],  # OUT [B,H-1]
) raises:
    comptime HM1 = H - 1
    for b in range(B):
        for t in range(HM1):
            out_loss[unsafe_offset=b * HM1 + t] = twohot_loss[BINS](
                vlogits, (b * H + t) * BINS, bins, ret[unsafe_offset=b * HM1 + t]
            )


def value_td_loss_backward[
    B: Int, H: Int, BINS: Int,
    vlogits_o: Origin[mut=True],
    bins_o: Origin[mut=True],
    ret_o: Origin[mut=True],
    d_loss_o: Origin[mut=True],
    grad_vlogits_o: Origin[mut=True],
](
    vlogits: Pointer[Scalar[DT], vlogits_o],  # [B,H,BINS]
    bins: Pointer[Scalar[DT], bins_o],     # [BINS]
    ret: Pointer[Scalar[DT], ret_o],      # [B,H-1]
    d_loss: Pointer[Scalar[DT], d_loss_o],   # [B,H-1] cotangent
    grad_vlogits: Pointer[Scalar[DT], grad_vlogits_o],  # [B,H,BINS]
) raises:
    """Backward of `value_td_loss_cpu` (target sg'd). grad_vlogits ZEROED then
    accumulated."""
    comptime HM1 = H - 1
    for i in range(B * H * BINS):
        grad_vlogits[unsafe_offset=i] = 0.0
    for b in range(B):
        for t in range(HM1):
            twohot_loss_backward[BINS](
                vlogits,
                (b * H + t) * BINS,
                bins,
                ret[unsafe_offset=b * HM1 + t],
                d_loss[unsafe_offset=b * HM1 + t],
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
    p: Pointer[Scalar[DT], p_o],   # [C] policy mixed probs
    q: Pointer[Scalar[DT], q_o],   # [C] prior  mixed probs
) -> Scalar[DT]:
    var kl = Scalar[DT](0.0)
    for a in range(C):
        kl += p[unsafe_offset=a] * (log(p[unsafe_offset=a]) - log(q[unsafe_offset=a]))
    return kl


@always_inline
def _reverse_kl_bwd[
    C: Int,
    sm_o: Origin[mut=True],
    p_o: Origin[mut=True],
    q_o: Origin[mut=True],
    grad_logits_o: Origin[mut=True],
](
    sm: Pointer[Scalar[DT], sm_o],  # [C] policy PRE-mix softmax
    p: Pointer[Scalar[DT], p_o],   # [C] policy mixed probs
    q: Pointer[Scalar[DT], q_o],   # [C] prior  mixed probs
    u: Scalar[DT],
    upstream: Scalar[DT],
    grad_logits: Pointer[Scalar[DT], grad_logits_o],  # accumulate [.,C]
    base: Int,
):
    var one_m_u = Scalar[DT](1.0) - u
    # g_a = ln p_a + 1 − ln q_a ; shared dot = Σ_a sm_a·g_a
    var dot = Scalar[DT](0.0)
    for a in range(C):
        var g = log(p[unsafe_offset=a]) + Scalar[DT](1.0) - log(q[unsafe_offset=a])
        dot += sm[unsafe_offset=a] * g
    for j in range(C):
        var gj = log(p[unsafe_offset=j]) + Scalar[DT](1.0) - log(q[unsafe_offset=j])
        grad_logits[unsafe_offset=base + j] += upstream * one_m_u * sm[unsafe_offset=j] * (gj - dot)


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
    plogits: Pointer[Scalar[DT], plogits_o],       # [B,H,NACT]
    prior_logits: Pointer[Scalar[DT], prior_logits_o],  # [B,H,NACT] frozen
    actions: Pointer[Scalar[DT], actions_o],       # [B,H] class ids
    adv: Pointer[Scalar[DT], adv_o],           # [B,H-1]
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
            if adv[unsafe_offset=b * HM1 + t] >= Scalar[DT](0.0):
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
            var k = Int(Float64(actions[unsafe_offset=b * H + t]) + 0.5)
            # current-policy logp(a) (also fills sm/pp)
            var r = cat_fwd[NACT](plogits, base, UNIMIX, k, sm, pp)
            var logp = r[0]
            # prior probs at this state
            cat_softmax_mix[NACT](prior_logits, base, UNIMIX, qsm, qp)
            # max-likelihood term (sign of advantage)
            if adv[unsafe_offset=b * HM1 + t] >= Scalar[DT](0.0):
                loss += -(Scalar[DT](1.0) - alpha) * inv_pos * logp
            else:
                loss += alpha * inv_neg * logp
            # reverse-KL prior term
            loss += beta * inv_N * _reverse_kl_fwd[NACT](pp, qp)

    sm.unsafe_free()
    pp.unsafe_free()
    qsm.unsafe_free()
    qp.unsafe_free()
    return Float64(loss)


def pmpo_policy_loss_backward[
    B: Int, H: Int, NACT: Int,
    plogits_o: Origin[mut=True],
    prior_logits_o: Origin[mut=True],
    actions_o: Origin[mut=True],
    adv_o: Origin[mut=True],
    grad_plogits_o: Origin[mut=True],
](
    plogits: Pointer[Scalar[DT], plogits_o],       # [B,H,NACT]
    prior_logits: Pointer[Scalar[DT], prior_logits_o],  # [B,H,NACT]
    actions: Pointer[Scalar[DT], actions_o],       # [B,H]
    adv: Pointer[Scalar[DT], adv_o],           # [B,H-1]
    alpha: Scalar[DT],
    beta: Scalar[DT],
    upstream: Scalar[DT],
    grad_plogits: Pointer[Scalar[DT], grad_plogits_o],  # [B,H,NACT]
) raises:
    """Backward of `pmpo_policy_loss_cpu` w.r.t. the policy logits (advantages,
    actions, and prior are sg'd). grad_plogits ZEROED then accumulated.
    `upstream` scales the whole loss (typically 1.0)."""
    comptime HM1 = H - 1
    for i in range(B * H * NACT):
        grad_plogits[unsafe_offset=i] = 0.0

    var sm = alloc[Scalar[DT]](NACT)
    var pp = alloc[Scalar[DT]](NACT)
    var qsm = alloc[Scalar[DT]](NACT)
    var qp = alloc[Scalar[DT]](NACT)

    var n_pos = 0
    var n_neg = 0
    for b in range(B):
        for t in range(HM1):
            if adv[unsafe_offset=b * HM1 + t] >= Scalar[DT](0.0):
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
            var k = Int(Float64(actions[unsafe_offset=b * H + t]) + 0.5)
            cat_softmax_mix[NACT](plogits, base, UNIMIX, sm, pp)
            cat_softmax_mix[NACT](prior_logits, base, UNIMIX, qsm, qp)
            # max-likelihood: d_logp coefficient by advantage sign
            var d_logp: Scalar[DT]
            if adv[unsafe_offset=b * HM1 + t] >= Scalar[DT](0.0):
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

    sm.unsafe_free()
    pp.unsafe_free()
    qsm.unsafe_free()
    qp.unsafe_free()
