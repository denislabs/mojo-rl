"""Planner-side GPU kernels for MPPI.

These are the planner-generic pieces of the MPPI pipeline — sampling
actions, accumulating returns, softmax weighting, mean/std refit,
action selection. They contain **no agent-specific logic**
(no Q-decoding, no policy heads, no categorical bins): the
``RolloutCallbackGPU`` adapter is responsible for everything between
"take z and an action" and "produce z_next + scalar reward".

For Phase 2 the canonical kernel definitions still live in
``mojo_rl/deep_agents/tdmpc2/kernels.mojo`` — this module re-exports
them so the new planner doesn't reach across the tdmpc2 namespace
directly. The only new kernel added here is
``mppi_accum_reward_scalar_kernel``, the trait-friendly counterpart
to the existing decoding accumulator (the callback decodes; this
kernel just accumulates the scalar).

Phase-2 cleanup TODO: once ``tdmpc2/mppi.mojo`` itself migrates to
the new planner (task 38), relocate the canonical defs from
``tdmpc2/kernels.mojo`` here and turn the tdmpc2 file into a thin
re-export shim — preserving source compatibility for any external
callers.
"""

from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor

# Re-exports: agent-agnostic kernels currently defined in tdmpc2.
# These have no TDMPC2 dependencies in their body; just re-homed here.
from mojo_rl.deep_agents.tdmpc2.kernels import (
    mppi_broadcast_z0_zero_returns_batched_kernel,
    mppi_sample_actions_batched_kernel,
    mppi_copy_z_kernel,
    mppi_add_terminal_value_kernel,
    mppi_softmax_weights_kernel,
    mppi_weighted_mean_std_kernel,
    mppi_select_action_kernel,
)


@always_inline
def mppi_accum_reward_scalar_kernel[
    dtype: DType,
    TOTAL_SAMPLES: Int,
](
    reward_step: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES), MutAnyOrigin
    ],
    returns: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES), MutAnyOrigin
    ],
    discount: Scalar[dtype],
) where dtype.is_floating_point():
    """Accumulate pre-decoded per-step rewards into discounted returns.

    ``returns[i] += discount * reward_step[i]``. Unlike the legacy
    ``mppi_accumulate_reward_kernel``, this kernel does **not** decode
    categorical logits — the callback's ``rollout_step_gpu`` is
    expected to have already produced a scalar reward per batch row.
    This is the planner-side half of the trait split.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= TOTAL_SAMPLES:
        return
    returns[i] = returns[i] + discount * Scalar[dtype](reward_step[i][0])
