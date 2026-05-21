"""SAC actor backward helpers — Phase E retrofit.

Same surface as v1 `sac_actor_loss.mojo` (free functions; not
`Loss`-conforming because SAC actor loss takes 5+ tensors which would
distort the `Loss(logits, targets)` trait). The math body is now thin
wrappers over the canonical pair in `loss/squashed_gaussian.mojo`.

Loss (mean over batch):
    L = E_b[α · log_prob_b − min(Q1(s_b, a_b), Q2(s_b, a_b))]

Caller supplies `grad_action[BATCH, ACT] = d_L/d_a` (already
including the `-1/BATCH` factor and the min-mask from twin-critic
backwards). This wrapper builds `grad_log_prob[b] = α/BATCH` and
chains through the canonical squashed-Gaussian backward.
"""

from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from .squashed_gaussian import (
    squashed_gaussian_forward,
    squashed_gaussian_backward,
)


# Re-export the forward under v1's name for backward compatibility.
def squashed_gaussian_sample[ACT: Int, BATCH: Int](
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
    """v1-compatible alias for `squashed_gaussian_forward`."""
    squashed_gaussian_forward[ACT, BATCH](
        actor_output, z, action_scale, action, log_prob,
    )


def sac_actor_backward[ACT: Int, BATCH: Int](
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
    """Same surface as v1's `sac_actor_backward`. Internally builds
    `grad_log_prob = α/BATCH` and delegates to the canonical
    `squashed_gaussian_backward`.

    The α-fold-in stays here (not in the canonical pair) precisely
    because SAC chooses to bake α/BATCH into grad_log_prob; PPO or
    offline algorithms use different coefficients. By keeping the
    canonical pair pure ("only knows the squashed-Gaussian Jacobian")
    we let each algorithm decide its own entropy/scaling story.
    """
    comptime assert actor_output.flat_rank == 2, "actor_output rank-2"
    comptime assert z.flat_rank == 2, "z rank-2"
    comptime assert grad_action.flat_rank == 2, "grad_action rank-2"
    comptime assert grad_actor_output.flat_rank == 2, "grad_actor_output rank-2"
    comptime assert ACT >= 1, "ACT >= 1"

    var inv_batch: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    var entropy_scalar = alpha * inv_batch

    # Build grad_log_prob[b] = α/BATCH (constant across batch).
    var glp_buf = List[Scalar[DT]](length=BATCH, fill=entropy_scalar)
    var glp_t = TileTensor(glp_buf, row_major[BATCH]())

    squashed_gaussian_backward[ACT, BATCH](
        actor_output, z, grad_action, glp_t, action_scale, grad_actor_output,
    )


def sac_actor_loss_value[BATCH: Int](
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
