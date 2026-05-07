"""Stage 3 diagnostic: compare three entropy-gradient formulations on the
exact same (mean, log_std_smooth, eps) inputs.

For each per-action element we compute three values for ∂scaled_entropy/∂θ:

    (A) Squashed analytical  — current Mojo `tdmpc2_action_tanh_chain_kernel`:
            ∂SE/∂mean    ≈ ACT · 2·tanh(u)
            ∂SE/∂log_std ≈ ACT · (1 − 2·tanh(u)·std·eps)

    (B) Unsquashed analytical — what reference's `scaled_entropy` algebraically
        simplifies to (== item 11 attempt that saturated):
            ∂SE/∂mean    = 0
            ∂SE/∂log_std = ACT

    (C) Autograd-faithful explicit chain rule — what PyTorch *actually*
        computes for `scaled_entropy = -log_p_post · scaled_lp / (log_p_post + ε)`:
            full chain through every step including the ratio.

If C ≈ B (within FP32 noise), the saturation in item 11 is not caused by
FP32 imprecision in the analytical simplification — it's caused by the
absent squash-correction *noise* that the squashed (A) form provided.

If C differs materially from B, then full graph integration may behave
differently than the analytical simplification.
"""

from std.math import abs as _abs, exp, log, tanh
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype


def _abs_f64(x: Float64) -> Float64:
    return -x if x < 0.0 else x


def _safe_log1m_tanh_sq(u: Float64) -> Float64:
    """Numerically stable log(1 − tanh²(u)) = 2(log(2) − |u| − log(1+exp(−2|u|)))."""
    var au = u if u >= 0.0 else -u
    return 2.0 * (
        Float64(0.6931471805599453) - au - log(1.0 + exp(-2.0 * au))
    )


def run_one_element(
    mean: Float64,
    ls: Float64,
    eps: Float64,
    ACT: Int,
    mut dA_mean: Float64,
    mut dA_ls: Float64,
    mut dB_mean: Float64,
    mut dB_ls: Float64,
    mut dC_mean: Float64,
    mut dC_ls: Float64,
):
    """Compute per-element ∂scaled_entropy/∂(mean, ls) under three formulations:
    A=squashed analytical, B=unsquashed analytical, C=autograd-faithful.

    For C we compute the full per-dim contribution by treating
    log_prob_pre, log_prob_post as if they were equal to ACT × per-dim parts.
    """
    var std = exp(ls)
    var u = mean + std * eps
    var t = tanh(u)

    # Per-dim quantities (use sums conceptually; here ACT is set so that
    # log_prob_pre, log_prob_post are scalar sums over actions but for
    # the single-dim test we use the PER-dim contribution).
    var lp_pre_per_dim = -0.5 * eps * eps - ls - 0.9189385175704956
    var sc_per_dim = _safe_log1m_tanh_sq(u)
    var lp_post_per_dim = lp_pre_per_dim - sc_per_dim

    # Approximate full LP_pre, LP_post as ACT × per_dim (matches
    # symmetric all-dims-equal regime; suitable for diagnostic).
    var LP_pre = Float64(ACT) * lp_pre_per_dim
    var LP_post = Float64(ACT) * lp_post_per_dim

    # ── (A) Squashed analytical (current Mojo) ───────────────────────
    # The kernel implements gradient of -log_p_squashed · ACT, so per dim:
    #   ∂(-LP_post·ACT)/∂mean    = ACT · 2·tanh(u)         (squash term)
    #   ∂(-LP_post·ACT)/∂log_std = ACT · (1 − 2·tanh(u)·std·eps)
    dA_mean = Float64(ACT) * 2.0 * t
    dA_ls = Float64(ACT) * (1.0 - 2.0 * t * std * eps)

    # ── (B) Unsquashed analytical (item 11) ──────────────────────────
    # ∂(-LP_pre·ACT)/∂mean = 0
    # ∂(-LP_pre·ACT)/∂log_std = ACT
    dB_mean = 0.0
    dB_ls = Float64(ACT)

    # ── (C) Autograd-faithful explicit chain rule ────────────────────
    # SE = -LP_post · LP_pre · ACT / (LP_post + ε_div)
    # where ε_div = 1e-8.
    comptime EPS_DIV: Float64 = 1e-8
    var lp_post_eps = LP_post + EPS_DIV
    var scaled_lp = LP_pre * Float64(ACT)
    var e_scale = scaled_lp / lp_post_eps
    var SE = -LP_post * e_scale

    # Local derivatives:
    # ∂SE/∂LP_post = -e_scale + (-LP_post) · (-scaled_lp / lp_post_eps²)
    var dSE_dLPpost = -e_scale + LP_post * scaled_lp / (lp_post_eps * lp_post_eps)
    # ∂SE/∂scaled_lp = -LP_post / lp_post_eps
    var dSE_dscaled_lp = -LP_post / lp_post_eps
    # scaled_lp = LP_pre · ACT  =>  ∂scaled_lp/∂LP_pre = ACT
    var dSE_dLPpre_via_scaled_lp = dSE_dscaled_lp * Float64(ACT)
    # LP_post = LP_pre − sum(squash_corr)  =>  ∂LP_post/∂LP_pre = 1
    var dSE_dLPpre_via_LPpost = dSE_dLPpost * 1.0
    var dSE_dLPpre = dSE_dLPpre_via_scaled_lp + dSE_dLPpre_via_LPpost

    # LP_pre, LP_post are both ACT · per_dim (under all-dims-equal).
    # For a SINGLE dim's contribution (treating other dims as constant
    # in the chain), ∂LP_pre/∂lp_pre_i = 1 and similarly ∂LP_post/∂lp_post_i = 1.
    var dSE_dlp_pre_i = dSE_dLPpre * 1.0
    var dSE_dlp_post_i = dSE_dLPpost * 1.0

    # lp_post_i = lp_pre_i - sc_i  =>  ∂lp_post_i/∂lp_pre_i = 1
    # The term `-sc_i` brings derivatives via u = mean + std·eps
    var dSE_dlp_pre_i_total = dSE_dlp_pre_i + dSE_dlp_post_i * 1.0

    # lp_pre_i = -0.5 eps² - ls - const  =>  ∂lp_pre_i/∂ls = -1, ∂lp_pre_i/∂mean = 0
    var dC_ls_via_lp_pre = dSE_dlp_pre_i_total * (-1.0)
    var dC_mean_via_lp_pre = 0.0

    # lp_post_i = lp_pre_i - sc_i, where sc_i = log(1 - tanh²(u_i))
    # ∂lp_post_i/∂u_i (via -sc_i) = -∂sc_i/∂u_i
    # ∂sc_i/∂u_i = -2·tanh(u_i)·tanh'(u_i) / (1 - tanh²(u_i)) = -2·tanh(u_i)
    # so ∂(-sc_i)/∂u_i = 2·tanh(u_i)
    # u_i = mean + exp(ls)·eps  =>  ∂u_i/∂mean = 1, ∂u_i/∂ls = std·eps
    var dlp_post_i_du_i = 2.0 * t
    var dC_mean_via_squash = dSE_dlp_post_i * dlp_post_i_du_i * 1.0
    var dC_ls_via_squash = dSE_dlp_post_i * dlp_post_i_du_i * (std * eps)

    dC_mean = dC_mean_via_lp_pre + dC_mean_via_squash
    dC_ls = dC_ls_via_lp_pre + dC_ls_via_squash


def main() raises:
    comptime ACT = 6  # HalfCheetah action dim

    print("=" * 78)
    print("Per-element entropy-gradient comparison (HalfCheetah ACT=6)")
    print("=" * 78)
    print()
    print("Cases probe: (mean, log_std, eps) at points exercising different")
    print("regimes: small log_std, large log_std, saturated tanh, fresh init.")
    print()

    # Test points
    var means = InlineArray[Float64, 6](uninitialized=True)
    var lss = InlineArray[Float64, 6](uninitialized=True)
    var epss = InlineArray[Float64, 6](uninitialized=True)
    var labels = InlineArray[String, 6](uninitialized=True)
    means[0] = 0.0
    lss[0] = 0.0
    epss[0] = 1.0
    labels[0] = "fresh init: mean=0 ls=0 eps=1"
    means[1] = 0.5
    lss[1] = -1.0
    epss[1] = 0.5
    labels[1] = "early: mean=0.5 ls=-1 eps=0.5"
    means[2] = 0.0
    lss[2] = 1.5
    epss[2] = 1.5
    labels[2] = "saturated: mean=0 ls=1.5 eps=1.5 → |u|=large"
    means[3] = 0.0
    lss[3] = -3.0
    epss[3] = 0.5
    labels[3] = "converged-ish: mean=0 ls=-3 eps=0.5"
    means[4] = -0.5
    lss[4] = 1.0
    epss[4] = 1.0
    labels[4] = "intermediate: mean=-0.5 ls=1 eps=1"
    means[5] = 0.0
    lss[5] = 2.0
    epss[5] = 2.0
    labels[5] = "ls at upper bound, large eps (kernel saturates)"

    print("--------------------------------------------------------------------------------")
    print("                  ∂SE/∂mean                ∂SE/∂log_std")
    print("                  (A)squash (B)unsq (C)auto   (A)squash (B)unsq (C)auto")
    print("--------------------------------------------------------------------------------")
    var max_BC_mean: Float64 = 0.0
    var max_BC_ls: Float64 = 0.0
    var max_AC_mean: Float64 = 0.0
    var max_AC_ls: Float64 = 0.0

    for k in range(6):
        var a_m: Float64 = 0.0
        var a_l: Float64 = 0.0
        var b_m: Float64 = 0.0
        var b_l: Float64 = 0.0
        var c_m: Float64 = 0.0
        var c_l: Float64 = 0.0
        run_one_element(
            means[k], lss[k], epss[k], ACT,
            a_m, a_l, b_m, b_l, c_m, c_l,
        )
        var bc_m = _abs_f64(b_m - c_m)
        var bc_l = _abs_f64(b_l - c_l)
        var ac_m = _abs_f64(a_m - c_m)
        var ac_l = _abs_f64(a_l - c_l)
        if bc_m > max_BC_mean: max_BC_mean = bc_m
        if bc_l > max_BC_ls: max_BC_ls = bc_l
        if ac_m > max_AC_mean: max_AC_mean = ac_m
        if ac_l > max_AC_ls: max_AC_ls = ac_l

        print(labels[k])
        print(
            "    mean: A=", a_m, "  B=", b_m, "  C=", c_m,
            "  | B-C=", bc_m, "  A-C=", ac_m
        )
        print(
            "    ls:   A=", a_l, "  B=", b_l, "  C=", c_l,
            "  | B-C=", bc_l, "  A-C=", ac_l
        )
        print()

    print("=" * 78)
    print("Summary across all test cases:")
    print("  max |B − C| on mean:", max_BC_mean)
    print("  max |B − C| on log_std:", max_BC_ls)
    print("  max |A − C| on mean:", max_AC_mean)
    print("  max |A − C| on log_std:", max_AC_ls)
    print()
    print("Interpretation:")
    print("  - If |B−C| ≈ 0: autograd-faithful (C) matches unsquashed analytical (B,")
    print("    item 11). Stage 4 graph integration would reproduce item 11's saturation.")
    print("    Saturation lives elsewhere (Q-side, encoder, running_scale).")
    print("  - If |A−C| ≈ 0: autograd-faithful (C) matches squashed (A, current Mojo).")
    print("    Reference and current Mojo would compute the same gradient — surprising.")
    print("  - If neither: Stage 4 would produce a NEW gradient distinct from both.")
    print("    Worth running A/B training to see effect.")
