"""Policy gradient strategies for on-policy agents.

Stateless strategy types with @staticmethod methods and compile-time flags.
Each implementation computes d_logits for the actor backward pass (CPU + GPU).

Implementations:
  - VanillaPG: vanilla policy gradient (A2C) — d_logits = -advantage * d_log_prob - entropy_coef * d_entropy
  - ClippedSurrogate: PPO clipped surrogate — ratio clipping with KL divergence output
"""

from layout import Layout, LayoutTensor
from std.math import exp, log
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import dtype, TPB


# =============================================================================
# GPU Kernels
# =============================================================================


@always_inline
fn _vanilla_pg_actor_grad_kernel[
    d: DType where d.is_floating_point(),
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs (same signature as ppo_actor_grad_with_kl_kernel for interop)
    grad_logits: LayoutTensor[
        d, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    kl_divergences: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    entropies: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_flags: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs
    logits: LayoutTensor[
        d, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    old_log_probs: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    advantages: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    actions: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_epsilon: Scalar[d],
    entropy_coef: Scalar[d],
    batch_size: Int,
):
    """Vanilla policy gradient kernel for A2C (GPU).

    Same signature as ppo_actor_grad_with_kl_kernel so the agent can dispatch
    uniformly. old_log_probs and clip_epsilon are unused. KL and clip_flags
    are zeroed out.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    var action = Int(actions[b])
    var advantage = advantages[b]

    # Softmax probabilities
    var max_logit = logits[b, 0]
    for a in range(1, NUM_ACTIONS):
        if logits[b, a] > max_logit:
            max_logit = logits[b, a]

    var sum_exp = max_logit - max_logit  # Zero with correct type
    for a in range(NUM_ACTIONS):
        var logit_val = logits[b, a] - max_logit
        sum_exp = sum_exp + exp(logit_val)

    var probs = InlineArray[Scalar[d], NUM_ACTIONS](fill=Scalar[d](0.0))
    for a in range(NUM_ACTIONS):
        var logit_val = logits[b, a] - max_logit
        var prob_val = exp(logit_val) / sum_exp
        probs[a] = Scalar[d](prob_val[0])

    # Entropy: H = -sum(p * log(p))
    var ent: Scalar[d] = 0.0
    for a in range(NUM_ACTIONS):
        if probs[a] > Scalar[d](1e-10):
            var p_log = Float32(probs[a]) + Float32(1e-8)
            ent = ent - probs[a] * Scalar[d](log(p_log))
    entropies[b] = ent

    # No clipping or KL for vanilla PG
    kl_divergences[b] = Scalar[d](0.0)
    clip_flags[b] = Scalar[d](0.0)

    # Vanilla policy gradient: -advantage * d_log_prob - entropy_coef * d_entropy
    for a in range(NUM_ACTIONS):
        var d_log_prob: Scalar[d]
        if a == action:
            d_log_prob = Scalar[d](1.0) - probs[a]
        else:
            d_log_prob = -probs[a]

        var prob_for_log_ent = Float32(probs[a]) + Float32(1e-8)
        var log_prob_ent = Scalar[d](log(prob_for_log_ent))
        var d_entropy = -probs[a] * (Scalar[d](1.0) + log_prob_ent)

        grad_logits[b, a] = (
            -advantage * d_log_prob - entropy_coef * d_entropy
        ) / Scalar[d](BATCH_SIZE)


# =============================================================================
# PolicyGradient trait
# =============================================================================


trait PolicyGradient:
    """Trait for policy gradient strategies (CPU + GPU)."""

    comptime NEEDS_OLD_LOG_PROB: Bool
    comptime NEEDS_CLIP_EPSILON: Bool

    @staticmethod
    fn compute_d_logits[
        ACTIONS: Int,
    ](
        probs: InlineArray[Scalar[dtype], ACTIONS],
        action: Int,
        new_log_prob: Scalar[dtype],
        old_log_prob: Scalar[dtype],
        advantage: Scalar[dtype],
        clip_eps: Float64,
        entropy_coef: Float64,
        mut d_logits: InlineArray[Scalar[dtype], ACTIONS],
    ) -> None:
        ...

    @staticmethod
    fn compute_d_logits_gpu[
        BATCH_SIZE: Int,
        NUM_ACTIONS: Int,
    ](
        ctx: DeviceContext,
        grad_logits: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
        ],
        kl_divergences: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        entropies: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        clip_flags: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        logits: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
        ],
        old_log_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        advantages: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        clip_epsilon: Float64,
        entropy_coef: Float64,
    ) raises -> None:
        ...


# =============================================================================
# VanillaPG — vanilla policy gradient (A2C)
# =============================================================================


struct VanillaPG(PolicyGradient):
    """Vanilla policy gradient: d_logits = -advantage * d_log_prob - entropy_coef * d_entropy.

    Used by A2C. No ratio clipping, no old log probs needed.
    """

    comptime NEEDS_OLD_LOG_PROB: Bool = False
    comptime NEEDS_CLIP_EPSILON: Bool = False

    @staticmethod
    fn compute_d_logits[
        ACTIONS: Int,
    ](
        probs: InlineArray[Scalar[dtype], ACTIONS],
        action: Int,
        new_log_prob: Scalar[dtype],
        old_log_prob: Scalar[dtype],
        advantage: Scalar[dtype],
        clip_eps: Float64,
        entropy_coef: Float64,
        mut d_logits: InlineArray[Scalar[dtype], ACTIONS],
    ) -> None:
        """Compute vanilla policy gradient d_logits."""
        for a in range(ACTIONS):
            var d_lp: Scalar[dtype]
            if a == action:
                d_lp = Scalar[dtype](1.0) - probs[a]
            else:
                d_lp = -probs[a]
            var d_ent = -probs[a] * (
                Scalar[dtype](1.0) + log(probs[a] + Scalar[dtype](1e-8))
            )
            d_logits[a] = (
                -advantage * d_lp - Scalar[dtype](entropy_coef) * d_ent
            )

    @staticmethod
    fn compute_d_logits_gpu[
        BATCH_SIZE: Int,
        NUM_ACTIONS: Int,
    ](
        ctx: DeviceContext,
        grad_logits: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
        ],
        kl_divergences: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        entropies: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        clip_flags: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        logits: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
        ],
        old_log_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        advantages: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        clip_epsilon: Float64,
        entropy_coef: Float64,
    ) raises -> None:
        """Launch vanilla PG GPU kernel."""
        comptime kernel = _vanilla_pg_actor_grad_kernel[
            dtype, BATCH_SIZE, NUM_ACTIONS
        ]
        comptime blocks = (BATCH_SIZE + TPB - 1) // TPB
        ctx.enqueue_function[kernel, kernel](
            grad_logits,
            kl_divergences,
            entropies,
            clip_flags,
            logits,
            old_log_probs,
            advantages,
            actions,
            Scalar[dtype](clip_epsilon),
            Scalar[dtype](entropy_coef),
            BATCH_SIZE,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )


# =============================================================================
# ClippedSurrogate — PPO clipped surrogate objective
# =============================================================================


struct ClippedSurrogate(PolicyGradient):
    """PPO clipped surrogate: ratio clipping with entropy bonus.

    Used by PPO. Requires old log probs and clip epsilon.
    """

    comptime NEEDS_OLD_LOG_PROB: Bool = True
    comptime NEEDS_CLIP_EPSILON: Bool = True

    @staticmethod
    fn compute_d_logits[
        ACTIONS: Int,
    ](
        probs: InlineArray[Scalar[dtype], ACTIONS],
        action: Int,
        new_log_prob: Scalar[dtype],
        old_log_prob: Scalar[dtype],
        advantage: Scalar[dtype],
        clip_eps: Float64,
        entropy_coef: Float64,
        mut d_logits: InlineArray[Scalar[dtype], ACTIONS],
    ) -> None:
        """Compute PPO clipped surrogate d_logits."""
        var ratio = exp(new_log_prob - old_log_prob)

        var is_clipped = (ratio < Scalar[dtype](1.0 - clip_eps)) or (
            ratio > Scalar[dtype](1.0 + clip_eps)
        )

        for a in range(ACTIONS):
            var d_lp: Scalar[dtype]
            if a == action:
                d_lp = Scalar[dtype](1.0) - probs[a]
            else:
                d_lp = -probs[a]
            var d_ent = -probs[a] * (
                Scalar[dtype](1.0) + log(probs[a] + Scalar[dtype](1e-8))
            )
            if is_clipped:
                # Clipped: only entropy gradient flows
                d_logits[a] = -Scalar[dtype](entropy_coef) * d_ent
            else:
                # Unclipped: full policy gradient + entropy
                d_logits[a] = (
                    -advantage * ratio * d_lp
                    - Scalar[dtype](entropy_coef) * d_ent
                )

    @staticmethod
    fn compute_d_logits_gpu[
        BATCH_SIZE: Int,
        NUM_ACTIONS: Int,
    ](
        ctx: DeviceContext,
        grad_logits: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
        ],
        kl_divergences: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        entropies: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        clip_flags: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        logits: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
        ],
        old_log_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        advantages: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        clip_epsilon: Float64,
        entropy_coef: Float64,
    ) raises -> None:
        """Launch PPO clipped surrogate GPU kernel."""
        from mojo_rl.deep_agents.ppo.kernels import (
            ppo_actor_grad_with_kl_kernel,
        )

        comptime kernel = ppo_actor_grad_with_kl_kernel[
            dtype, BATCH_SIZE, NUM_ACTIONS
        ]
        comptime blocks = (BATCH_SIZE + TPB - 1) // TPB
        ctx.enqueue_function[kernel, kernel](
            grad_logits,
            kl_divergences,
            entropies,
            clip_flags,
            logits,
            old_log_probs,
            advantages,
            actions,
            Scalar[dtype](clip_epsilon),
            Scalar[dtype](entropy_coef),
            BATCH_SIZE,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )
