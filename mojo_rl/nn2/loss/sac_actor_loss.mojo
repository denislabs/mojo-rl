"""SAC actor backward helpers — squashed-Gaussian reparameterized loss.

Phase 7.3. Bespoke (NOT `Loss`-conforming) — SAC actor loss takes 5+
input tensors (actor_output, z noise, grad_action from twin critics,
alpha, action_scale) so it would distort the `Loss(logits, targets)`
trait. Free functions instead of a struct because there is no per-call
state to amortize (no GPU scratch, no parameters).

Loss (mean over batch):
    L = E_b[α · Σ_j log_prob_j(a_b) − min(Q1(s_b, a_b), Q2(s_b, a_b))]

Squashed-Gaussian reparameterization (CleanRL-style):
    log_std_j = clamp(actor_output[b, ACT+j], -5, 2)
    std_j     = exp(log_std_j)
    pre_j     = actor_output[b, j] + std_j · z[b, j]
    y_j       = tanh(pre_j)
    a_j       = action_scale · y_j
    log_prob_j = log_N(z; 0, 1) − log(std) − log(action_scale·(1 − y²) + ε)
              = -0.5·z² − log_std − 0.5·log(2π)
                − log(action_scale·(1 − y²) + ε)

Backward emits grad_actor_output[BATCH, 2*ACT] = [d_L/d_mu | d_L/d_log_std].

The caller pre-supplies grad_action[BATCH, ACT] = d_L/d_a, already
including the -1/BATCH factor and the min-mask from the twin critic
backwards. We then add the entropy term (α · d_log_prob/d_·) and chain
through the reparam squashed-Gaussian to mu and log_std.

Phase 7 ships CPU only (SAC Pendulum is CPU-only). GPU path follows the
same shape (one kernel) when the first GPU SAC env lands.
"""

from std.math import exp, log, tanh
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] = 2.0
comptime EPS_TANH_CORR: Scalar[DT] = 1e-6  # log(c·(1-y²) + ε)
comptime LOG_2PI: Scalar[DT] = 1.8378770664093453


def _clamp_log_std(ls: Scalar[DT]) -> Scalar[DT]:
    if ls < LOG_STD_MIN:
        return LOG_STD_MIN
    elif ls > LOG_STD_MAX:
        return LOG_STD_MAX
    return ls


def squashed_gaussian_sample[
    ACT: Int, BATCH: Int,
](
    actor_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    z: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    action_scale: Scalar[DT],
    mut action: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
    mut log_prob: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    """Compute action[b, j] and log_prob[b] from (actor_output, z).

    Inputs (all rank-2 except log_prob):
        actor_output [BATCH, 2*ACT]   [mu | log_std]
        z           [BATCH, ACT]      pre-sampled noise (caller owns RNG)

    Outputs:
        action      [BATCH, ACT]      action_scale · tanh(mu + exp(clamp(log_std))·z)
        log_prob    [BATCH]           per-sample squashed-Gaussian log-prob,
                                      summed over action dims.

    No clamping on action — already in [-action_scale, action_scale] by tanh.
    """
    comptime assert actor_output.flat_rank == 2, "actor_output rank-2"
    comptime assert z.flat_rank == 2, "z rank-2"
    comptime assert action.flat_rank == 2, "action rank-2"
    comptime assert log_prob.flat_rank == 1, "log_prob rank-1"
    comptime assert ACT >= 1, "ACT >= 1"

    for b in range(BATCH):
        var lp_total: Scalar[DT] = 0.0
        for j in range(ACT):
            var mu = actor_output[b, j]
            var ls = _clamp_log_std(actor_output[b, ACT + j])
            var std = exp(ls)
            var zj = z[b, j]
            var pre = mu + std * zj
            var y = tanh(pre)
            action[b, j] = action_scale * y
            # Log-prob: log_N(x_t | mu, std) - log(c·(1-y²) + ε)
            #         = -0.5·z² - log_std - 0.5·log(2π) - log(c·(1-y²) + ε)
            var one_minus_y2 = Scalar[DT](1.0) - y * y
            var corr = action_scale * one_minus_y2 + EPS_TANH_CORR
            lp_total += (
                Scalar[DT](-0.5) * zj * zj
                - ls
                - Scalar[DT](0.5) * LOG_2PI
                - log(corr)
            )
        log_prob[b] = lp_total


def sac_actor_backward[
    ACT: Int, BATCH: Int,
](
    actor_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    z: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    grad_action: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    alpha: Scalar[DT],
    action_scale: Scalar[DT],
    mut grad_actor_output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    """Compute grad_actor_output = [d_L/d_mu | d_L/d_log_std].

    Inputs (per batch element b, action dim j):
        actor_output[b, j]         mu_j   (state-dependent mean)
        actor_output[b, ACT+j]     log_std_j  (state-dependent log-std)
        z[b, j]                    reparam noise used during sampling
        grad_action[b, j]          d_L/d_a_j — from critic backwards;
                                   ALREADY includes -1/BATCH and min-mask.
        alpha                      entropy temperature
        action_scale               env action scale

    The entropy term contributes (α / BATCH) per sample. The squashed-
    Gaussian Jacobian carries the chain rule through reparameterization.

    Derivation:
        y       = tanh(mu + std·z)
        a       = c · y
        L_b     = α · log_prob_b − min_q_b
        log_prob_b_j = -0.5·z² − log_std − 0.5·log(2π) − log(c·(1-y²)+ε)
        d a_j / d mu_j      = c · (1 - y²)
        d a_j / d log_std_j = c · (1 - y²) · z · std
        d log_prob_j / d mu_j      = 2·y·c·(1-y²) / (c·(1-y²)+ε)
        d log_prob_j / d log_std_j = -1 + 2·y·c·(1-y²)·z·std / (c·(1-y²)+ε)

    Total gradient (per b, j):
        grad_mu       = grad_action · c·(1-y²)
                      + (α / BATCH) · 2·y·c·(1-y²) / (c·(1-y²)+ε)
        grad_log_std  = grad_action · c·(1-y²)·z·std
                      + (α / BATCH) · (-1 + 2·y·c·(1-y²)·z·std/(c·(1-y²)+ε))

    log_std clamp masking: if the un-clamped log_std was outside [-5, 2],
    the gradient through it should be zero (saturated clamp boundary).
    """
    comptime assert actor_output.flat_rank == 2, "actor_output rank-2"
    comptime assert z.flat_rank == 2, "z rank-2"
    comptime assert grad_action.flat_rank == 2, "grad_action rank-2"
    comptime assert grad_actor_output.flat_rank == 2, "grad_actor_output rank-2"
    comptime assert ACT >= 1, "ACT >= 1"

    var inv_batch: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BATCH)

    for b in range(BATCH):
        for j in range(ACT):
            var mu = actor_output[b, j]
            var ls_raw = actor_output[b, ACT + j]
            var ls = _clamp_log_std(ls_raw)
            var ls_clamped = (ls_raw < LOG_STD_MIN) or (ls_raw > LOG_STD_MAX)

            var std = exp(ls)
            var zj = z[b, j]
            var pre = mu + std * zj
            var y = tanh(pre)
            var one_minus_y2 = Scalar[DT](1.0) - y * y
            var c_corr = action_scale * one_minus_y2
            var corr = c_corr + EPS_TANH_CORR

            # Chain factors.
            var da_dmu = action_scale * one_minus_y2
            var da_dls = action_scale * one_minus_y2 * zj * std
            var dlp_dmu = (Scalar[DT](2.0) * y * c_corr) / corr
            var dlp_dls = (
                Scalar[DT](-1.0)
                + (Scalar[DT](2.0) * y * c_corr * zj * std) / corr
            )

            var ga = grad_action[b, j]
            var entropy_scalar = alpha * inv_batch

            var gmu = ga * da_dmu + entropy_scalar * dlp_dmu
            var gls = ga * da_dls + entropy_scalar * dlp_dls

            grad_actor_output[b, j] = gmu
            if ls_clamped:
                grad_actor_output[b, ACT + j] = 0.0
            else:
                grad_actor_output[b, ACT + j] = gls


def sac_actor_loss_value[
    BATCH: Int,
](
    log_prob: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    min_q: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    alpha: Scalar[DT],
) raises -> Scalar[DT]:
    """Loss scalar for logging: mean_b(α · log_prob_b − min_q_b)."""
    comptime assert log_prob.flat_rank == 1, "log_prob rank-1"
    comptime assert min_q.flat_rank == 1, "min_q rank-1"
    var total: Scalar[DT] = 0.0
    for b in range(BATCH):
        total += alpha * log_prob[b] - min_q[b]
    return total / Scalar[DT](BATCH)
