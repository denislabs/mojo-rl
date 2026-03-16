"""Q-value output strategies for DQN family agents.

Controls how raw network output is converted to Q-values and how
gradients are transformed back through the output layer.

Implementations:
  - DirectQ: Identity — raw output IS Q-values (standard DQN, Double DQN)
  - DuelingQ: Q = V + (A - mean(A)), with corresponding gradient transform
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB


trait QOutput:
    """Trait for Q-value output strategies."""

    comptime IS_DUELING: Bool

    @staticmethod
    fn combine_cpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        raw_out: InlineArray[Scalar[dtype], BATCH * RAW_OUT],
        mut q_values: InlineArray[Scalar[dtype], BATCH * ACTIONS],
    ) -> None:
        ...

    @staticmethod
    fn grad_transform_cpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        dq: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        mut d_raw: InlineArray[Scalar[dtype], BATCH * RAW_OUT],
    ) -> None:
        ...

    @staticmethod
    fn combine_gpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        ctx: DeviceContext,
        raw_out: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
        q_values: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
    ) raises -> None:
        ...

    @staticmethod
    fn grad_transform_gpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        ctx: DeviceContext,
        dq: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
        d_raw: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
    ) raises -> None:
        ...


# =============================================================================
# DirectQ — Identity output (standard DQN, Double DQN)
# =============================================================================


struct DirectQ(QOutput):
    """Identity: raw network output = Q-values, grad passthrough.

    Used by standard DQN and Double DQN where the network directly
    outputs Q-values for each action.
    """

    comptime IS_DUELING: Bool = False

    @staticmethod
    fn combine_cpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        raw_out: InlineArray[Scalar[dtype], BATCH * RAW_OUT],
        mut q_values: InlineArray[Scalar[dtype], BATCH * ACTIONS],
    ) -> None:
        """Identity: copy raw output to q_values."""
        for i in range(BATCH * ACTIONS):
            q_values[i] = raw_out[i]

    @staticmethod
    fn grad_transform_cpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        dq: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        mut d_raw: InlineArray[Scalar[dtype], BATCH * RAW_OUT],
    ) -> None:
        """Identity: copy dq to d_raw."""
        for i in range(BATCH * ACTIONS):
            d_raw[i] = dq[i]

    @staticmethod
    fn combine_gpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        ctx: DeviceContext,
        raw_out: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
        q_values: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
    ) raises -> None:
        """Identity: copy raw output to q_values on GPU."""

        @always_inline
        fn copy_kernel(
            src: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
            dst: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * ACTIONS:
                return
            var b = idx // ACTIONS
            var a = idx % ACTIONS
            dst[b, a] = src[b, a]

        comptime TOTAL = BATCH * ACTIONS
        ctx.enqueue_function[copy_kernel, copy_kernel](
            raw_out,
            q_values,
            grid_dim=((TOTAL + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn grad_transform_gpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        ctx: DeviceContext,
        dq: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
        d_raw: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
    ) raises -> None:
        """Identity: copy dq to d_raw on GPU."""

        @always_inline
        fn copy_kernel(
            src: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
            dst: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * ACTIONS:
                return
            var b = idx // ACTIONS
            var a = idx % ACTIONS
            dst[b, a] = src[b, a]

        comptime TOTAL = BATCH * ACTIONS
        ctx.enqueue_function[copy_kernel, copy_kernel](
            dq,
            d_raw,
            grid_dim=((TOTAL + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )


# =============================================================================
# DuelingQ — Dueling architecture output
# =============================================================================


struct DuelingQ(QOutput):
    """Dueling: Q = V + (A - mean(A)), with corresponding gradient transform.

    RAW_OUT = 1 + ACTIONS (V stream + A stream from Parallel model).
    V(s) is at index 0, A(s, a_i) at indices 1..ACTIONS.
    """

    comptime IS_DUELING: Bool = True

    @staticmethod
    fn combine_cpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        raw_out: InlineArray[Scalar[dtype], BATCH * RAW_OUT],
        mut q_values: InlineArray[Scalar[dtype], BATCH * ACTIONS],
    ) -> None:
        """Compute Q = V + (A - mean(A))."""
        for b in range(BATCH):
            var v_s = raw_out[b * RAW_OUT]
            var mean_adv = Scalar[dtype](0.0)
            for a in range(ACTIONS):
                mean_adv += raw_out[b * RAW_OUT + 1 + a]
            mean_adv /= Scalar[dtype](ACTIONS)
            for a in range(ACTIONS):
                var adv = raw_out[b * RAW_OUT + 1 + a]
                q_values[b * ACTIONS + a] = v_s + (adv - mean_adv)

    @staticmethod
    fn grad_transform_cpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        dq: InlineArray[Scalar[dtype], BATCH * ACTIONS],
        mut d_raw: InlineArray[Scalar[dtype], BATCH * RAW_OUT],
    ) -> None:
        """Transform dQ to dueling gradients: dV = sum(dQ), dA_i = dQ_i - mean(dQ)."""
        for b in range(BATCH):
            var sum_dq = Scalar[dtype](0.0)
            for a in range(ACTIONS):
                sum_dq += dq[b * ACTIONS + a]
            # dV = sum(dQ)
            d_raw[b * RAW_OUT] = sum_dq
            # dA_i = dQ_i - (1/n) * sum(dQ)
            var one_over_n = Scalar[dtype](1.0) / Scalar[dtype](ACTIONS)
            for a in range(ACTIONS):
                d_raw[b * RAW_OUT + 1 + a] = dq[b * ACTIONS + a] - one_over_n * sum_dq

    @staticmethod
    fn combine_gpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        ctx: DeviceContext,
        raw_out: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
        q_values: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
    ) raises -> None:
        """Compute Q = V + (A - mean(A)) on GPU. One thread per batch sample."""
        from mojo_rl.deep_agents.dueling_dqn.kernels import dueling_combine_kernel

        @always_inline
        fn combine_wrapper(
            q_out: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
            d_out: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
        ):
            dueling_combine_kernel[dtype, BATCH, ACTIONS, RAW_OUT](q_out, d_out)

        ctx.enqueue_function[combine_wrapper, combine_wrapper](
            q_values,
            raw_out,
            grid_dim=((BATCH + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn grad_transform_gpu[
        BATCH: Int,
        ACTIONS: Int,
        RAW_OUT: Int,
    ](
        ctx: DeviceContext,
        dq: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
        d_raw: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
    ) raises -> None:
        """Transform dQ to dueling gradients on GPU. One thread per batch sample."""
        from mojo_rl.deep_agents.dueling_dqn.kernels import dueling_grad_kernel

        @always_inline
        fn grad_wrapper(
            d_out: LayoutTensor[dtype, Layout.row_major(BATCH, RAW_OUT), MutAnyOrigin],
            dq_in: LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin],
        ):
            dueling_grad_kernel[dtype, BATCH, ACTIONS, RAW_OUT](d_out, dq_in)

        ctx.enqueue_function[grad_wrapper, grad_wrapper](
            d_raw,
            dq,
            grid_dim=((BATCH + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )
