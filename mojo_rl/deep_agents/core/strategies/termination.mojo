"""Termination functions for model-based rollouts.

Used by MBPO to determine if a model-predicted state is terminal,
since the dynamics model does not predict termination signals.
Each environment defines its own termination criteria.

GPU support: is_terminal_gpu kernel checks termination for a batch of
observations in parallel. One thread per batch element.
"""

from layout import Layout, LayoutTensor
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.nn.constants import dtype


trait TerminationFn:
    """Environment-specific termination check for model rollouts."""

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        ...

    @staticmethod
    def is_terminal_gpu[
        BATCH: Int, OBS_DIM: Int
    ](
        ctx: DeviceContext,
        next_obs: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
    ) raises:
        """GPU batch termination check. Sets dones[b] = 1.0 if terminal.

        Default: no termination (memset to 0). Override for envs with
        termination conditions.

        Args:
            ctx: GPU device context.
            next_obs: Predicted next observations [BATCH * OBS_DIM].
            dones: Output done flags [BATCH]. 1.0 = terminal.
        """
        ...


struct NeverTerminate(TerminationFn):
    """No early termination. Use for environments without termination
    conditions (e.g., HalfCheetah, Swimmer, Walker2d)."""

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        return False

    @staticmethod
    def is_terminal_gpu[
        BATCH: Int, OBS_DIM: Int
    ](
        ctx: DeviceContext,
        next_obs: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
    ) raises:
        ctx.enqueue_memset(dones, 0)


struct HopperTerminate(TerminationFn):
    """Hopper-v2/v4: terminate if height < 0.7 or |angle| > 0.2.

    Observation layout: [z_pos, angle, ...].
    """

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        var height = Float64(obs[0])
        var angle = Float64(obs[1])
        return height < 0.7 or angle > 0.2 or angle < -0.2

    @staticmethod
    def is_terminal_gpu[
        BATCH: Int, OBS_DIM: Int
    ](
        ctx: DeviceContext,
        next_obs: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
    ) raises:
        comptime TPB = 256
        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @parameter
        @always_inline
        def hopper_term_kernel(
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
            ],
            d: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var height = rebind[Scalar[dtype]](obs[b, 0])
            var angle = rebind[Scalar[dtype]](obs[b, 1])
            if (
                height < Scalar[dtype](0.7)
                or angle > Scalar[dtype](0.2)
                or angle < Scalar[dtype](-0.2)
            ):
                d[b] = Scalar[dtype](1.0)
            else:
                d[b] = Scalar[dtype](0.0)

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
        ](next_obs.unsafe_ptr())
        var d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](dones.unsafe_ptr())
        ctx.enqueue_function[hopper_term_kernel, hopper_term_kernel](
            obs_t, d_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )


struct AntTerminate(TerminationFn):
    """Ant-v2/v4: terminate if height not in [0.2, 1.0].

    Observation layout: [x_pos, y_pos, z_pos, ...] (z_pos at index 0
    after removing x,y from obs in standard Gym wrapper).
    """

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        var height = Float64(obs[0])
        return height < 0.2 or height > 1.0

    @staticmethod
    def is_terminal_gpu[
        BATCH: Int, OBS_DIM: Int
    ](
        ctx: DeviceContext,
        next_obs: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
    ) raises:
        comptime TPB = 256
        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @parameter
        @always_inline
        def ant_term_kernel(
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
            ],
            d: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var height = rebind[Scalar[dtype]](obs[b, 0])
            if (
                height < Scalar[dtype](0.2)
                or height > Scalar[dtype](1.0)
            ):
                d[b] = Scalar[dtype](1.0)
            else:
                d[b] = Scalar[dtype](0.0)

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
        ](next_obs.unsafe_ptr())
        var d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](dones.unsafe_ptr())
        ctx.enqueue_function[ant_term_kernel, ant_term_kernel](
            obs_t, d_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )


struct InvertedPendulumTerminate(TerminationFn):
    """InvertedPendulum: terminate if |angle| > 0.2.

    Observation layout: [x, x_dot, theta, theta_dot].
    """

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        var angle = Float64(obs[2])
        return angle > 0.2 or angle < -0.2

    @staticmethod
    def is_terminal_gpu[
        BATCH: Int, OBS_DIM: Int
    ](
        ctx: DeviceContext,
        next_obs: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
    ) raises:
        comptime TPB = 256
        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @parameter
        @always_inline
        def pendulum_term_kernel(
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
            ],
            d: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var angle = rebind[Scalar[dtype]](obs[b, 2])
            if angle > Scalar[dtype](0.2) or angle < Scalar[dtype](-0.2):
                d[b] = Scalar[dtype](1.0)
            else:
                d[b] = Scalar[dtype](0.0)

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
        ](next_obs.unsafe_ptr())
        var d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](dones.unsafe_ptr())
        ctx.enqueue_function[pendulum_term_kernel, pendulum_term_kernel](
            obs_t, d_t, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
