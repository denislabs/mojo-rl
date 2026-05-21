"""Canonical squashed-Gaussian forward/backward (Phase E).

Single source of truth for the SAC reparameterized squashed-Gaussian
sampling + log-prob math. Today this math is duplicated in three places
(audit P1 finding):

  1. `nn2/primitives/rsample.mojo` — as `RSample[ACT]` Module.
  2. `nn2/loss/sac_actor_loss.mojo` — as free functions (basically the
     same as here).
  3. `nn2/loss/sac_actor_loss_cg.mojo` — composes the Module form.

Phase E adds this file as the canonical implementation. `rsample_v2.mojo`,
`sac_actor_loss_v2.mojo`, and `sac_actor_loss_cg_v2.mojo` all delegate
here. The split between `grad_action` and `grad_log_prob` in the
backward (vs v1 SAC's hard-coded fold-in of `α/BATCH`) is the
ergonomic win: any caller (PPO continuous, max-ent RL, offline) can
plug in their own scaling.

# Forward

    action[b, j]   = action_scale · tanh(mu_j + exp(clamp(log_std_j)) · z_j)
    log_prob[b]    = Σ_j ( -0.5·z_j²  - log_std_j  - 0.5·log(2π)
                            - log(action_scale·(1-y²) + ε) )

# Backward (per-element chain factors)

    y               = tanh(mu + std·z)
    da/dmu          = c · (1 - y²)             c = action_scale
    da/dlog_std     = c · (1 - y²) · z · std
    dlp/dmu         = 2·y·c·(1-y²) / corr      corr = c·(1-y²) + ε
    dlp/dlog_std    = -1 + 2·y·c·(1-y²)·z·std / corr

    grad_mu[b, j]      = grad_action[b, j]  · da/dmu
                        + grad_log_prob[b]  · dlp/dmu
    grad_log_std[b, j] = grad_action[b, j]  · da/dlog_std
                        + grad_log_prob[b]  · dlp/dlog_std

Log-std clamp mask: when raw log_std is outside [LOG_STD_MIN,
LOG_STD_MAX], the un-clamped value didn't affect downstream — its
grad is zeroed.

CPU only for now. GPU port lands with the first GPU SAC env.
"""

from std.math import exp, log, tanh as ftanh
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] =  2.0
comptime EPS_TANH_CORR: Scalar[DT] = 1e-6
comptime LOG_2PI: Scalar[DT] = 1.8378770664093453


def _clamp_log_std(ls: Scalar[DT]) -> Scalar[DT]:
    if ls < LOG_STD_MIN:
        return LOG_STD_MIN
    elif ls > LOG_STD_MAX:
        return LOG_STD_MAX
    return ls


def squashed_gaussian_forward[ACT: Int, BATCH: Int](
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
    """Compute (action, log_prob) from (mu, log_std, z).

    actor_output [BATCH, 2*ACT]   packed [mu | log_std]
    z            [BATCH, ACT]     reparam noise (caller draws)
    action       [BATCH, ACT]     squashed sample
    log_prob     [BATCH]          per-sample log-prob (sum over j)
    """
    comptime assert actor_output.flat_rank == 2, "actor_output rank-2 [BATCH, 2*ACT]"
    comptime assert z.flat_rank == 2,            "z rank-2 [BATCH, ACT]"
    comptime assert action.flat_rank == 2,       "action rank-2 [BATCH, ACT]"
    comptime assert log_prob.flat_rank == 1,     "log_prob rank-1 [BATCH]"
    comptime assert ACT >= 1, "ACT >= 1"

    for b in range(BATCH):
        var lp: Scalar[DT] = 0.0
        for j in range(ACT):
            var mu = actor_output[b, j]
            var ls = _clamp_log_std(actor_output[b, ACT + j])
            var std = exp(ls)
            var zj = z[b, j]
            var pre = mu + std * zj
            var y = ftanh(pre)
            action[b, j] = action_scale * y
            var corr = action_scale * (Scalar[DT](1.0) - y * y) + EPS_TANH_CORR
            lp += (
                Scalar[DT](-0.5) * zj * zj
                - ls
                - Scalar[DT](0.5) * LOG_2PI
                - log(corr)
            )
        log_prob[b] = lp


def squashed_gaussian_backward[ACT: Int, BATCH: Int](
    actor_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    z: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    grad_action: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    grad_log_prob: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    action_scale: Scalar[DT],
    mut grad_actor_output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    """Chain (grad_action, grad_log_prob) back to grad_[mu | log_std].

    grad_action      [BATCH, ACT]
    grad_log_prob    [BATCH]
    grad_actor_output [BATCH, 2*ACT]

    The caller pre-bakes any BATCH-normalization / α-scaling into
    `grad_action` / `grad_log_prob`. This function is dimensionally
    pure — it only knows the squashed-Gaussian Jacobian.
    """
    comptime assert actor_output.flat_rank == 2,      "actor_output rank-2"
    comptime assert z.flat_rank == 2,                 "z rank-2"
    comptime assert grad_action.flat_rank == 2,       "grad_action rank-2"
    comptime assert grad_log_prob.flat_rank == 1,     "grad_log_prob rank-1"
    comptime assert grad_actor_output.flat_rank == 2, "grad_actor_output rank-2"
    comptime assert ACT >= 1, "ACT >= 1"

    for b in range(BATCH):
        var glp = grad_log_prob[b]
        for j in range(ACT):
            var mu = actor_output[b, j]
            var ls_raw = actor_output[b, ACT + j]
            var ls = _clamp_log_std(ls_raw)
            var ls_clamped = (ls_raw < LOG_STD_MIN) or (ls_raw > LOG_STD_MAX)

            var std = exp(ls)
            var zj = z[b, j]
            var pre = mu + std * zj
            var y = ftanh(pre)
            var one_minus_y2 = Scalar[DT](1.0) - y * y
            var c_om = action_scale * one_minus_y2
            var corr = c_om + EPS_TANH_CORR

            var da_dmu = c_om
            var da_dls = c_om * zj * std
            var dlp_dmu = (Scalar[DT](2.0) * y * c_om) / corr
            var dlp_dls = (
                Scalar[DT](-1.0) + (Scalar[DT](2.0) * y * c_om * zj * std) / corr
            )

            var ga = grad_action[b, j]
            var gmu = ga * da_dmu + glp * dlp_dmu
            var gls = ga * da_dls + glp * dlp_dls

            grad_actor_output[b, j] = gmu
            if ls_clamped:
                grad_actor_output[b, ACT + j] = Scalar[DT](0.0)
            else:
                grad_actor_output[b, ACT + j] = gls
