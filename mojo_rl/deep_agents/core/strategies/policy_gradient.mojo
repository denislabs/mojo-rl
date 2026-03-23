"""Policy gradient strategies for on-policy agents.

Stateless strategy types with @staticmethod methods and compile-time flags.
Each implementation computes d_logits for the actor backward pass (CPU + GPU).

Implementations:
  - VanillaPG: vanilla policy gradient (A2C) — d_logits = -advantage * d_log_prob - entropy_coef * d_entropy
  - ClippedSurrogate: PPO clipped surrogate — ratio clipping with KL divergence output
  - AutodiffVanillaPG: vanilla PG using DiffOp backward math (CategoricalLogProbOp.vjp)
  - AutodiffClippedSurrogate: PPO clipped surrogate using DiffOp backward math
    (CategoricalLogProbOp.vjp, RatioOp.vjp, ClipSurrogateOp.vjp)
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
def _vanilla_pg_actor_grad_kernel[
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
    def compute_d_logits[
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
    def compute_d_logits_gpu[
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
    def compute_d_logits[
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
    def compute_d_logits_gpu[
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
    def compute_d_logits[
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
    def compute_d_logits_gpu[
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
        from mojo_rl.deep_agents.core.kernels import (
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


# =============================================================================
# GPU Kernel — autodiff-style vanilla PG
# =============================================================================


@always_inline
def _autodiff_vanilla_pg_actor_grad_kernel[
    d: DType where d.is_floating_point(),
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    grad_logits: LayoutTensor[
        d, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    kl_divergences: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    entropies: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_flags: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
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
    """Autodiff-style vanilla PG kernel (GPU).

    Uses CategoricalLogProbOp backward math: d(log_softmax[i])/d(logit[j]) = delta_ij - softmax_j.
    Same output signature as PPO kernel for uniform dispatch.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    var action = Int(rebind[Scalar[d]](actions[b]))
    var adv = rebind[Scalar[d]](advantages[b])

    # --- CategoricalLogProbOp forward: softmax probabilities ---
    var max_logit = rebind[Scalar[d]](logits[b, 0])
    for a in range(1, NUM_ACTIONS):
        var la = rebind[Scalar[d]](logits[b, a])
        if la > max_logit:
            max_logit = la

    var sum_exp = Scalar[d](0.0)
    for a in range(NUM_ACTIONS):
        sum_exp = sum_exp + exp(rebind[Scalar[d]](logits[b, a]) - max_logit)

    var probs = InlineArray[Scalar[d], NUM_ACTIONS](fill=Scalar[d](0.0))
    for a in range(NUM_ACTIONS):
        var e = exp(rebind[Scalar[d]](logits[b, a]) - max_logit)
        probs[a] = e / sum_exp

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

    # --- Backprop: grad_log_prob = -advantage ---
    var g = -adv
    for a in range(NUM_ACTIONS):
        var indicator = Scalar[d](1.0) if a == action else Scalar[d](0.0)
        var d_lp = g * (indicator - probs[a])

        var prob_for_log = Float32(probs[a]) + Float32(1e-8)
        var d_ent = -probs[a] * (Scalar[d](1.0) + Scalar[d](log(prob_for_log)))

        grad_logits[b, a] = (d_lp - entropy_coef * d_ent) / Scalar[d](
            BATCH_SIZE
        )


# =============================================================================
# GPU Kernel — autodiff-style clipped surrogate
# =============================================================================


@always_inline
def _autodiff_clipped_surrogate_actor_grad_kernel[
    d: DType where d.is_floating_point(),
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
    eps: Float64 = 0.2,
](
    grad_logits: LayoutTensor[
        d, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    kl_divergences: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    entropies: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_flags: LayoutTensor[d, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
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
    """Autodiff-style PPO clipped surrogate kernel (GPU).

    Chains DiffOp backward math:
      1. ClipSurrogateOp.vjp  -> grad_ratio
      2. RatioOp.vjp          -> grad_log_prob = grad_ratio * ratio
      3. CategoricalLogProbOp.vjp -> grad_logits[j] = grad_log_prob * (delta_ij - prob[j])
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    var action = Int(rebind[Scalar[d]](actions[b]))
    var adv = rebind[Scalar[d]](advantages[b])
    var old_lp = rebind[Scalar[d]](old_log_probs[b])

    # --- CategoricalLogProbOp forward: softmax + log_prob ---
    var max_logit = rebind[Scalar[d]](logits[b, 0])
    for a in range(1, NUM_ACTIONS):
        var la = rebind[Scalar[d]](logits[b, a])
        if la > max_logit:
            max_logit = la

    var sum_exp = Scalar[d](0.0)
    for a in range(NUM_ACTIONS):
        sum_exp = sum_exp + exp(rebind[Scalar[d]](logits[b, a]) - max_logit)

    var probs = InlineArray[Scalar[d], NUM_ACTIONS](fill=Scalar[d](0.0))
    for a in range(NUM_ACTIONS):
        var e = exp(rebind[Scalar[d]](logits[b, a]) - max_logit)
        probs[a] = e / sum_exp

    var new_log_prob = Scalar[d](log(Float32(probs[action]) + Float32(1e-8)))

    # --- RatioOp forward: ratio = exp(log_prob - old_log_prob) ---
    var ratio = exp(new_log_prob - old_lp)

    # --- ClipSurrogateOp forward + cache grad multiplier ---
    var lo = Scalar[d](1.0 - eps)
    var hi = Scalar[d](1.0 + eps)

    var surr1 = ratio * adv
    var clipped_ratio = ratio
    if clipped_ratio < lo:
        clipped_ratio = lo
    if clipped_ratio > hi:
        clipped_ratio = hi
    var surr2 = clipped_ratio * adv

    # ClipSurrogateOp.vjp: cached gradient multiplier for ratio
    var grad_mult: Scalar[d]
    if surr1 <= surr2:
        grad_mult = -adv
        clip_flags[b] = Scalar[d](0.0)
    else:
        clip_flags[b] = Scalar[d](1.0)
        if ratio >= lo and ratio <= hi:
            grad_mult = -adv
        else:
            grad_mult = Scalar[d](0.0)

    # ClipSurrogateOp.vjp: upstream grad = 1.0 (scalar loss), so grad_ratio = grad_mult
    var grad_ratio = grad_mult

    # --- RatioOp.vjp: grad_log_prob = grad_ratio * ratio ---
    var grad_log_prob = grad_ratio * ratio

    # KL divergence approximation: (ratio - 1) - log(ratio)
    var log_ratio = new_log_prob - old_lp
    var kl = (ratio - Scalar[d](1.0)) - log_ratio
    if kl < Scalar[d](0.0):
        kl = Scalar[d](0.0)
    kl_divergences[b] = kl

    # Entropy: H = -sum(p * log(p))
    var ent: Scalar[d] = 0.0
    for a in range(NUM_ACTIONS):
        if probs[a] > Scalar[d](1e-10):
            var p_log = Float32(probs[a]) + Float32(1e-8)
            ent = ent - probs[a] * Scalar[d](log(p_log))
    entropies[b] = ent

    # --- CategoricalLogProbOp.vjp: d_logit[j] = grad_log_prob * (delta_ij - prob[j]) ---
    for a in range(NUM_ACTIONS):
        var indicator = Scalar[d](1.0) if a == action else Scalar[d](0.0)
        var d_lp = grad_log_prob * (indicator - probs[a])

        # Entropy gradient
        var prob_for_log = Float32(probs[a]) + Float32(1e-8)
        var d_ent = -probs[a] * (Scalar[d](1.0) + Scalar[d](log(prob_for_log)))

        grad_logits[b, a] = (d_lp - entropy_coef * d_ent) / Scalar[d](
            BATCH_SIZE
        )


# =============================================================================
# AutodiffVanillaPG — vanilla policy gradient using DiffOp backward math
# =============================================================================


struct AutodiffVanillaPG(PolicyGradient):
    """Vanilla policy gradient using CategoricalLogProbOp backward math.

    Equivalent to VanillaPG but derives gradients from the DiffOp vjp formulas:
      - CategoricalLogProbOp.vjp: d(log_softmax[i])/d(logit[j]) = delta_ij - softmax_j

    No ratio clipping, no old log probs needed (same as VanillaPG).
    """

    comptime NEEDS_OLD_LOG_PROB: Bool = False
    comptime NEEDS_CLIP_EPSILON: Bool = False

    @staticmethod
    def compute_d_logits[
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
        """Compute vanilla PG d_logits using CategoricalLogProbOp.vjp math.

        Chain: loss = -advantage * log_prob
        Upstream grad into CategoricalLogProbOp: g = -advantage
        CategoricalLogProbOp.vjp: d_logit[j] = g * (delta(j==action) - prob[j])
        """
        # Upstream gradient from loss = -advantage * log_prob
        var g = -advantage

        for a in range(ACTIONS):
            # CategoricalLogProbOp.vjp: indicator - prob
            var indicator = Scalar[dtype](1.0) if a == action else Scalar[
                dtype
            ](0.0)
            var d_lp = g * (indicator - probs[a])

            # Entropy gradient: d(-sum(p*log(p)))/d(logit[j]) = -p[j]*(1 + log(p[j]))
            var d_ent = -probs[a] * (
                Scalar[dtype](1.0) + log(probs[a] + Scalar[dtype](1e-8))
            )

            d_logits[a] = d_lp - Scalar[dtype](entropy_coef) * d_ent

    @staticmethod
    def compute_d_logits_gpu[
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
        """Launch autodiff-style vanilla PG GPU kernel."""
        comptime kernel = _autodiff_vanilla_pg_actor_grad_kernel[
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
# AutodiffClippedSurrogate — PPO clipped surrogate using DiffOp backward math
# =============================================================================


struct AutodiffClippedSurrogate[clip_eps: Float64 = 0.2](PolicyGradient):
    """PPO clipped surrogate using chained DiffOp backward math.

    Derives gradients by chaining vjp formulas from the PPO DiffOps:
      1. ClipSurrogateOp.vjp:     grad_ratio = grad_mult (cached multiplier)
      2. RatioOp.vjp:             grad_log_prob = grad_ratio * ratio
      3. CategoricalLogProbOp.vjp: d_logit[j] = grad_log_prob * (delta_ij - prob[j])

    This is mathematically equivalent to ClippedSurrogate but the backward pass
    is structured as a chain of op-level vjps rather than hand-derived gradients.
    """

    comptime NEEDS_OLD_LOG_PROB: Bool = True
    comptime NEEDS_CLIP_EPSILON: Bool = True

    @staticmethod
    def compute_d_logits[
        ACTIONS: Int,
    ](
        probs: InlineArray[Scalar[dtype], ACTIONS],
        action: Int,
        new_log_prob: Scalar[dtype],
        old_log_prob: Scalar[dtype],
        advantage: Scalar[dtype],
        clip_epsilon: Float64,
        entropy_coef: Float64,
        mut d_logits: InlineArray[Scalar[dtype], ACTIONS],
    ) -> None:
        """Compute PPO clipped surrogate d_logits using chained DiffOp vjps.

        Forward chain:
          log_prob = CategoricalLogProb(logits, action)
          ratio = RatioOp(log_prob, old_log_prob) = exp(log_prob - old_log_prob)
          loss = ClipSurrogateOp(ratio, advantage) = -min(r*A, clip(r)*A)

        Backward chain (upstream grad = 1.0 from scalar loss):
          ClipSurrogateOp.vjp -> grad_ratio
          RatioOp.vjp         -> grad_log_prob = grad_ratio * ratio
          CategoricalLogProbOp.vjp -> d_logit[j] = grad_log_prob * (delta_ij - prob[j])
        """
        # --- RatioOp forward (cache ratio for backward) ---
        var ratio = exp(new_log_prob - old_log_prob)

        # --- ClipSurrogateOp forward + vjp ---
        var lo = Scalar[dtype](1.0 - clip_epsilon)
        var hi = Scalar[dtype](1.0 + clip_epsilon)

        var surr1 = ratio * advantage
        var clipped_ratio = ratio
        if clipped_ratio < lo:
            clipped_ratio = lo
        if clipped_ratio > hi:
            clipped_ratio = hi
        var surr2 = clipped_ratio * advantage

        # ClipSurrogateOp.vjp: cached gradient multiplier
        var grad_mult: Scalar[dtype]
        if surr1 <= surr2:
            # Unclipped branch: grad_ratio = -advantage
            grad_mult = -advantage
        else:
            # Clipped branch: gradient is -advantage if ratio in range, else 0
            if ratio >= lo and ratio <= hi:
                grad_mult = -advantage
            else:
                grad_mult = Scalar[dtype](0.0)

        # ClipSurrogateOp.vjp output (upstream = 1.0)
        var grad_ratio = grad_mult

        # --- RatioOp.vjp: grad_log_prob = grad_ratio * ratio ---
        var grad_log_prob = grad_ratio * ratio

        # --- CategoricalLogProbOp.vjp: d_logit[j] = g * (delta_ij - prob[j]) ---
        for a in range(ACTIONS):
            var indicator = Scalar[dtype](1.0) if a == action else Scalar[
                dtype
            ](0.0)
            var d_lp = grad_log_prob * (indicator - probs[a])

            # Entropy gradient: d(-sum(p*log(p)))/d(logit[j]) = -p[j]*(1+log(p[j]))
            var d_ent = -probs[a] * (
                Scalar[dtype](1.0) + log(probs[a] + Scalar[dtype](1e-8))
            )

            d_logits[a] = d_lp - Scalar[dtype](entropy_coef) * d_ent

    @staticmethod
    def compute_d_logits_gpu[
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
        """Launch autodiff-style PPO clipped surrogate GPU kernel."""
        comptime kernel = _autodiff_clipped_surrogate_actor_grad_kernel[
            dtype, BATCH_SIZE, NUM_ACTIONS, Self.clip_eps
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
