"""Q-value target computation strategies for DQN family agents.

Stateless strategy types for computing TD targets.

Implementations:
  - StandardQTarget: target = r + gamma * max_a Q_target(s', a) * (1-done)
  - DoubleQTarget: target = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1-done)
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB


trait QTarget:
    """Trait for Q-value target computation strategies."""

    comptime IS_DOUBLE: Bool

    @staticmethod
    def compute_targets_cpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        online_next_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        target_next_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        rewards: InlineArray[Scalar[dtype], BATCH],
        dones: InlineArray[Scalar[dtype], BATCH],
        mut targets: InlineArray[Scalar[dtype], BATCH],
        gamma: Float64,
    ) -> None:
        ...

    @staticmethod
    def compute_targets_gpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        ctx: DeviceContext,
        targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        online_next_q: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        target_next_q: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
    ) raises -> None:
        ...


# =============================================================================
# StandardQTarget — standard DQN target
# =============================================================================


struct StandardQTarget(QTarget):
    """Standard DQN: target = r + gamma * max_a Q_target(s', a) * (1-done).

    Does not use online_next_q — only uses target network Q-values.
    """

    comptime IS_DOUBLE: Bool = False

    @staticmethod
    def compute_targets_cpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        online_next_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        target_next_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        rewards: InlineArray[Scalar[dtype], BATCH],
        dones: InlineArray[Scalar[dtype], BATCH],
        mut targets: InlineArray[Scalar[dtype], BATCH],
        gamma: Float64,
    ) -> None:
        """Compute standard DQN TD targets."""
        for b in range(BATCH):
            var max_nq = target_next_q[b * ACTIONS]
            for a in range(1, ACTIONS):
                var q = target_next_q[b * ACTIONS + a]
                if q > max_nq:
                    max_nq = q
            var dm = Scalar[dtype](1.0) - dones[b]
            targets[b] = rewards[b] + Scalar[dtype](gamma) * max_nq * dm

    @staticmethod
    def compute_targets_gpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        ctx: DeviceContext,
        targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        online_next_q: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        target_next_q: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
    ) raises -> None:
        """Compute standard DQN TD targets on GPU."""
        from mojo_rl.deep_agents.core.kernels import dqn_td_target_kernel

        var gamma_s = Scalar[dtype](gamma)

        @always_inline
        def td_wrapper(
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            nq: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            don: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            g: Scalar[dtype],
        ):
            dqn_td_target_kernel[dtype, BATCH, ACTIONS](tgt, nq, rew, don, g)

        ctx.enqueue_function[td_wrapper, td_wrapper](
            targets,
            target_next_q,
            rewards,
            dones,
            gamma_s,
            grid_dim=((BATCH + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )


# =============================================================================
# DoubleQTarget — Double DQN target
# =============================================================================


struct DoubleQTarget(QTarget):
    """Double DQN: target = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1-done).

    Uses pre-combined online Q-values for action selection and target Q-values
    for evaluation.
    """

    comptime IS_DOUBLE: Bool = True

    @staticmethod
    def compute_targets_cpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        online_next_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        target_next_q: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        rewards: InlineArray[Scalar[dtype], BATCH],
        dones: InlineArray[Scalar[dtype], BATCH],
        mut targets: InlineArray[Scalar[dtype], BATCH],
        gamma: Float64,
    ) -> None:
        """Compute Double DQN TD targets using online Q-values for action selection.
        """
        for b in range(BATCH):
            var best_a = 0
            var best_q = online_next_q[b * ACTIONS]
            for a in range(1, ACTIONS):
                var q = online_next_q[b * ACTIONS + a]
                if q > best_q:
                    best_q = q
                    best_a = a
            var nq = target_next_q[b * ACTIONS + best_a]
            var dm = Scalar[dtype](1.0) - dones[b]
            targets[b] = rewards[b] + Scalar[dtype](gamma) * nq * dm

    @staticmethod
    def compute_targets_gpu[
        BATCH: Int,
        ACTIONS: Int,
    ](
        ctx: DeviceContext,
        targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        online_next_q: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        target_next_q: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        gamma: Float64,
    ) raises -> None:
        """Compute Double DQN TD targets on GPU."""
        from mojo_rl.deep_agents.core.kernels import dqn_double_td_target_kernel

        var gamma_s = Scalar[dtype](gamma)

        @always_inline
        def double_td_wrapper(
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            onq: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            tnq: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            don: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            g: Scalar[dtype],
        ):
            dqn_double_td_target_kernel[dtype, BATCH, ACTIONS](
                tgt, onq, tnq, rew, don, g
            )

        ctx.enqueue_function[double_td_wrapper, double_td_wrapper](
            targets,
            online_next_q,
            target_next_q,
            rewards,
            dones,
            gamma_s,
            grid_dim=((BATCH + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )
