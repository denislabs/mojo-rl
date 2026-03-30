"""Target value computation strategies for off-policy agents.

Pure computation strategies — compute TD target values from critic outputs.
No network calls, no workspace needed.

Uniform signature: unused args (q2 for single critic, log_probs/alpha for
non-SAC) are passed but ignored — following the nn/ pattern where cache is
passed even to layers that don't use it.

Implementations:
  - SingleQTarget: r + γ * Q (DDPG)
  - TwinQTarget: r + γ * min(Q1, Q2) (TD3)
  - EntropicTwinQTarget: r + γ * (min(Q1, Q2) - α * log_π) (SAC)
"""

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.deep_agents.core.kernels import (
    td_target_continuous_kernel,
    td_target_min_twin_kernel,
)


trait TargetValue:
    """Trait for target value strategies."""

    comptime NEEDS_TWIN_Q: Bool
    comptime NEEDS_LOG_PROBS: Bool

    @staticmethod
    def compute_cpu[
        BATCH: Int
    ](
        q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        mut targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
        alpha: Float64,
    ):
        ...

    @staticmethod
    def compute_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        mut targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
        alpha_buf: DeviceBuffer[dtype],
    ) raises:
        ...


def _safe_q(val: Float64) -> Float64:
    """Guard NaN in Q values."""
    if val != val:
        return 0.0
    return val


def _clamp_target(mut tgt: Float64):
    """Guard NaN and clamp to [-1000, 1000]."""
    if tgt != tgt:
        tgt = 0.0
    elif tgt > 1000.0:
        tgt = 1000.0
    elif tgt < -1000.0:
        tgt = -1000.0


# =============================================================================
# SingleQTarget — DDPG: r + γ * Q
# =============================================================================


struct SingleQTarget(TargetValue):
    """Single critic TD target: r + γ * Q_target * (1 - done).

    Used by DDPG. q2, log_probs, and alpha args are ignored.
    """

    comptime NEEDS_TWIN_Q: Bool = False
    comptime NEEDS_LOG_PROBS: Bool = False

    @staticmethod
    def compute_cpu[
        BATCH: Int
    ](
        q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        mut targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
        alpha: Float64,
    ):
        """Compute r + γ * Q * (1 - done) for each sample."""
        for b in range(BATCH):
            var q = _safe_q(Float64(q1.ptr[b]))
            var dm = 1.0 - Float64(dones.ptr[b])
            var tgt = Float64(rewards.ptr[b]) + gamma * q * dm
            _clamp_target(tgt)
            targets.ptr[b] = Scalar[dtype](tgt)

    @staticmethod
    def compute_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        mut targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
        alpha_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU dispatch: r + γ * Q * (1 - done)."""
        comptime BLOCKS = (BATCH + TPB - 1) // TPB
        var gamma_s = Scalar[dtype](gamma)

        @always_inline
        def kernel_wrapper(
            td_targets: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            next_q: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            dn: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            g: Scalar[dtype],
        ):
            td_target_continuous_kernel[dtype, BATCH](
                td_targets, rew, next_q, dn, g
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            targets,
            rewards,
            q1,
            dones,
            gamma_s,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


# =============================================================================
# TwinQTarget — TD3: r + γ * min(Q1, Q2)
# =============================================================================


struct TwinQTarget(TargetValue):
    """Twin critic TD target: r + γ * min(Q1, Q2) * (1 - done).

    Used by TD3. log_probs and alpha args are ignored.
    """

    comptime NEEDS_TWIN_Q: Bool = True
    comptime NEEDS_LOG_PROBS: Bool = False

    @staticmethod
    def compute_cpu[
        BATCH: Int
    ](
        q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        mut targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
        alpha: Float64,
    ):
        """Compute r + γ * min(Q1, Q2) * (1 - done) for each sample."""
        for b in range(BATCH):
            var q_1 = _safe_q(Float64(q1.ptr[b]))
            var q_2 = _safe_q(Float64(q2.ptr[b]))
            var min_q = q_1 if q_1 < q_2 else q_2
            var dm = 1.0 - Float64(dones.ptr[b])
            var tgt = Float64(rewards.ptr[b]) + gamma * min_q * dm
            _clamp_target(tgt)
            targets.ptr[b] = Scalar[dtype](tgt)

    @staticmethod
    def compute_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        mut targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
        alpha_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU dispatch: r + γ * min(Q1, Q2) * (1 - done)."""
        comptime BLOCKS = (BATCH + TPB - 1) // TPB
        var gamma_s = Scalar[dtype](gamma)
        var alpha_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](alpha_buf.unsafe_ptr())

        @always_inline
        def kernel_wrapper(
            td_targets: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q1_v: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q2_v: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            dn: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            lp: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            g: Scalar[dtype],
            a: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
        ):
            td_target_min_twin_kernel[dtype, BATCH, False](
                td_targets, rew, q1_v, q2_v, dn, lp, g, a
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            targets,
            rewards,
            q1,
            q2,
            dones,
            log_probs,
            gamma_s,
            alpha_t,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


# =============================================================================
# EntropicTwinQTarget — SAC: r + γ * (min(Q1, Q2) - α * log_π)
# =============================================================================


struct EntropicTwinQTarget(TargetValue):
    """Entropic twin critic target: r + γ*(min(Q1,Q2) - α*log_π)*(1 - done).

    Used by SAC. All args are used.
    """

    comptime NEEDS_TWIN_Q: Bool = True
    comptime NEEDS_LOG_PROBS: Bool = True

    @staticmethod
    def compute_cpu[
        BATCH: Int
    ](
        q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        mut targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
        alpha: Float64,
    ):
        """Compute r + γ * (min(Q1,Q2) - α*log_π) * (1 - done)."""
        for b in range(BATCH):
            var q_1 = _safe_q(Float64(q1.ptr[b]))
            var q_2 = _safe_q(Float64(q2.ptr[b]))
            var min_q = q_1 if q_1 < q_2 else q_2
            var lp = Float64(log_probs.ptr[b])
            var dm = 1.0 - Float64(dones.ptr[b])
            var tgt = (
                Float64(rewards.ptr[b]) + gamma * (min_q - alpha * lp) * dm
            )
            _clamp_target(tgt)
            targets.ptr[b] = Scalar[dtype](tgt)

    @staticmethod
    def compute_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        mut targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
        alpha_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU dispatch: r + γ * (min(Q1,Q2) - α*log_π) * (1 - done)."""
        comptime BLOCKS = (BATCH + TPB - 1) // TPB
        var gamma_s = Scalar[dtype](gamma)
        var alpha_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](alpha_buf.unsafe_ptr())

        @always_inline
        def kernel_wrapper(
            td_targets: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q1_v: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q2_v: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            dn: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            lp: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            g: Scalar[dtype],
            a: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
        ):
            td_target_min_twin_kernel[dtype, BATCH, True](
                td_targets, rew, q1_v, q2_v, dn, lp, g, a
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            targets,
            rewards,
            q1,
            q2,
            dones,
            log_probs,
            gamma_s,
            alpha_t,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
