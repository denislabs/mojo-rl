"""DQN CNN Agent — Deep Q-Network with convolutional image processing.

Uses the Nature DQN architecture (Mnih et al., 2015) for pixel observations:
  Conv2D[4, 32, 8, 4] → ReLU → Conv2D[32, 64, 4, 2] → ReLU →
  Conv2D[64, 64, 3, 1] → ReLU → Flatten → Dense[3136, 512] → ReLU →
  Dense[512, num_actions]

Input: 4 × 84 × 84 stacked grayscale frames (28224 floats)
Output: Q-values for each discrete action

Usage:
    from deep_agents.dqn_cnn import DQNCNNAgent
    from envs.arcade_games.pong import PongPixelEnv

    var agent = DQNCNNAgent[num_actions=3]()
    var ctx = DeviceContext()
    var metrics = agent.train_gpu[PongPixelEnv[DType.float32]](ctx, num_steps=1_000_000)
"""

from std.math import exp
from std.random import random_float64, seed

from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE, TPB
from nn.model import Sequential, Model
from nn.optimizer import Adam, Optimizer
from nn.initializer import Kaiming
from nn.training import Network, NetworkState, GPUNetworkState
from nn.autodiff import (
    AutoDiffChain,
    MatMul,
    BiasAdd,
    ReLUOp,
    Flatten,
    Conv2D,
)
from deep_agents.core import (
    fill_inline,
    obs_to_inline,
    OffPolicyDiscreteState,
    OffPolicyDiscreteAgent,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
    run_offpolicy_discrete_train_gpu,
    Checkpointable,
)
from deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
)
from nn.gpu import (
    xorshift32,
    random_uniform,
)
from core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from deep_agents.dqn.state import DQNGPUState, DQNCPUState
from deep_agents.dqn.kernels import (
    dqn_td_target_kernel,
    dqn_double_td_target_kernel,
)


# =============================================================================
# Nature DQN CNN Architecture
# =============================================================================

# Input: 4 channels (frame stack) × 84 height × 84 width
# Conv1: 32 filters, 8×8 kernel, stride 4 → 32 × 20 × 20 = 12800
# Conv2: 64 filters, 4×4 kernel, stride 2 → 64 × 9 × 9 = 5184
# Conv3: 64 filters, 3×3 kernel, stride 1 → 64 × 7 × 7 = 3136
# Flatten → 3136
# Dense: 3136 → 512 (ReLU)
# Dense: 512 → num_actions

comptime CONV1_OUT_H: Int = (84 + 2 * 0 - 8) // 4 + 1  # = 20
comptime CONV1_OUT_W: Int = (84 + 2 * 0 - 8) // 4 + 1  # = 20
comptime CONV1_FLAT: Int = 32 * CONV1_OUT_H * CONV1_OUT_W  # = 12800

comptime CONV2_OUT_H: Int = (CONV1_OUT_H + 2 * 0 - 4) // 2 + 1  # = 9
comptime CONV2_OUT_W: Int = (CONV1_OUT_W + 2 * 0 - 4) // 2 + 1  # = 9
comptime CONV2_FLAT: Int = 64 * CONV2_OUT_H * CONV2_OUT_W  # = 5184

comptime CONV3_OUT_H: Int = (CONV2_OUT_H + 2 * 0 - 3) // 1 + 1  # = 7
comptime CONV3_OUT_W: Int = (CONV2_OUT_W + 2 * 0 - 3) // 1 + 1  # = 7
comptime CONV3_FLAT: Int = 64 * CONV3_OUT_H * CONV3_OUT_W  # = 3136

comptime FC_HIDDEN: Int = 512


# =============================================================================
# DQN CNN Agent
# =============================================================================


struct DQNCNNAgent[
    num_actions: Int,
    buffer_capacity: Int = 10000,
    batch_size: Int = 32,
    n_envs: Int = 64,
    double_dqn: Bool = True,
    lr: Float64 = 0.00025,
](OffPolicyDiscreteAgent & GPUOffPolicyAgent & Checkpointable):
    """Deep Q-Network agent with CNN for pixel observations.

    Uses the Nature DQN architecture for processing 4×84×84 image observations.

    Parameters:
        num_actions: Number of discrete actions.
        buffer_capacity: Replay buffer capacity (default: 10000, smaller for pixel obs).
        batch_size: Training batch size (default: 32, smaller for pixel obs).
        n_envs: Parallel GPU environments (default: 64, smaller for pixel obs).
        double_dqn: Use Double DQN (default: True).
        lr: Adam learning rate (default: 0.00025, Nature DQN value).
    """

    comptime OBS = 4 * 84 * 84  # 28224 = PIXEL_OBS_DIM
    comptime ACTIONS = Self.num_actions
    comptime BATCH = Self.batch_size

    # Nature DQN CNN architecture
    comptime Q_Model = Sequential[
        # Conv block 1: 4×84×84 → 32×20×20
        AutoDiffChain[
            Conv2D[4, 32, 8, 4, 0, 84, 84],
            ReLUOp[CONV1_FLAT],
        ],
        # Conv block 2: 32×20×20 → 64×9×9
        AutoDiffChain[
            Conv2D[32, 64, 4, 2, 0, CONV1_OUT_H, CONV1_OUT_W],
            ReLUOp[CONV2_FLAT],
        ],
        # Conv block 3: 64×9×9 → 64×7×7
        AutoDiffChain[
            Conv2D[64, 64, 3, 1, 0, CONV2_OUT_H, CONV2_OUT_W],
            ReLUOp[CONV3_FLAT],
        ],
        # Flatten + Dense head: 3136 → 512 → num_actions
        AutoDiffChain[
            Flatten[CONV3_FLAT],
            MatMul[CONV3_FLAT, FC_HIDDEN],
            BiasAdd[FC_HIDDEN],
            ReLUOp[FC_HIDDEN],
        ],
        AutoDiffChain[
            MatMul[FC_HIDDEN, Self.ACTIONS],
            BiasAdd[Self.ACTIONS],
        ],
    ]
    comptime Q_Network = Network[Self.Q_Model, Adam[Self.lr]]

    # CPU state type
    comptime CPUStateType = DQNCPUState[
        Self.Q_Model,
        Adam[Self.lr],
        Self.buffer_capacity,
        Self.OBS,
        Self.batch_size,
    ]

    # GPUOffPolicyAgent required compile-time constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = 1  # discrete action as scalar
    comptime BUFFER_CAPACITY: Int = Self.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GPUStateType = DQNGPUState[
        Self.Q_Model,
        Adam[Self.lr],
        Self.buffer_capacity,
        Self.OBS,
        Self.num_actions,
        Self.batch_size,
        Self.n_envs,
    ]

    # CPU state
    var state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var tau: Float64

    # Exploration
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64

    # Training state
    var train_step_count: Int

    # Auto-checkpoint
    var checkpoint_every: Int
    var checkpoint_path: String

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.01,
        epsilon_decay: Float64 = 0.9995,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        self.state = Self.CPUStateType()
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    # =========================================================================
    # Action Selection
    # =========================================================================

    fn _select_action_inline(
        self,
        cpu_state: Self.CPUStateType,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        greedy: Bool,
    ) -> Int:
        if not greedy and random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.num_actions))

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](obs.unsafe_ptr()))
        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        var q_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](q_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.Q_Network.forward[1](obs_t, q_t, p)

        var best_action = 0
        var best_q = q_arr[0]
        for a in range(1, Self.num_actions):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best_action = a
        return best_action

    # =========================================================================
    # CPU Training Step
    # =========================================================================

    fn do_cpu_train_step(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        if not cpu_state.buffer.is_ready[Self.batch_size]():
            return 0.0

        var batch_obs = InlineArray[Scalar[dtype], Self.batch_size * Self.OBS](
            uninitialized=True
        )
        var batch_actions_tmp = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_rewards = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[
            Scalar[dtype], Self.batch_size * Self.OBS
        ](uninitialized=True)
        var batch_dones = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        var batch_act1 = InlineArray[Scalar[dtype], Self.batch_size * 1](
            uninitialized=True
        )
        cpu_state.buffer.sample[Self.batch_size](
            batch_obs, batch_act1, batch_rewards, batch_next_obs, batch_dones
        )
        for i in range(Self.batch_size):
            batch_actions_tmp[i] = batch_act1[i]

        var obs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.OBS),
            MutAnyOrigin,
        ](batch_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.OBS),
            MutAnyOrigin,
        ](batch_next_obs.unsafe_ptr())

        # Forward: online with cache
        var q_arr = InlineArray[Scalar[dtype], Self.batch_size * Self.ACTIONS](
            uninitialized=True
        )
        var q_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.ACTIONS),
            MutAnyOrigin,
        ](q_arr.unsafe_ptr())
        var cache_arr = InlineArray[
            Scalar[dtype], Self.batch_size * Self.Q_Model.CACHE_SIZE
        ](uninitialized=True)
        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.Q_Model.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_arr.unsafe_ptr())

        var p_online = cpu_state.online.params_view()
        Self.Q_Network.forward_with_cache[Self.batch_size](
            obs_t, q_t, p_online, cache_t
        )

        # Forward: target
        var next_q_arr = InlineArray[
            Scalar[dtype], Self.batch_size * Self.ACTIONS
        ](uninitialized=True)
        var next_q_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.ACTIONS),
            MutAnyOrigin,
        ](next_q_arr.unsafe_ptr())
        var p_target = cpu_state.target.params_view()
        Self.Q_Network.forward[Self.batch_size](next_obs_t, next_q_t, p_target)

        # TD targets
        var targets = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        comptime if Self.double_dqn:
            var online_next_arr = InlineArray[
                Scalar[dtype], Self.batch_size * Self.ACTIONS
            ](uninitialized=True)
            var online_next_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.ACTIONS),
                MutAnyOrigin,
            ](online_next_arr.unsafe_ptr())
            Self.Q_Network.forward[Self.batch_size](
                next_obs_t, online_next_t, p_online
            )

            for b in range(Self.batch_size):
                var best_action = 0
                var best_q = online_next_arr[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = online_next_arr[b * Self.ACTIONS + a]
                    if q > best_q:
                        best_q = q
                        best_action = a
                var next_q = next_q_arr[b * Self.ACTIONS + best_action]
                var done_mask = Scalar[dtype](1.0) - batch_dones[b]
                targets[b] = (
                    batch_rewards[b]
                    + Scalar[dtype](self.gamma) * next_q * done_mask
                )
        else:
            for b in range(Self.batch_size):
                var max_next_q = next_q_arr[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = next_q_arr[b * Self.ACTIONS + a]
                    if q > max_next_q:
                        max_next_q = q
                var done_mask = Scalar[dtype](1.0) - batch_dones[b]
                targets[b] = (
                    batch_rewards[b]
                    + Scalar[dtype](self.gamma) * max_next_q * done_mask
                )

        # Gradient (masked MSE)
        var grad_arr = InlineArray[
            Scalar[dtype], Self.batch_size * Self.ACTIONS
        ](uninitialized=True)
        var total_loss: Float64 = 0.0

        for b in range(Self.batch_size):
            var action = Int(batch_actions_tmp[b])
            var q_pred = q_arr[b * Self.ACTIONS + action]
            var td_error = q_pred - targets[b]
            total_loss += Float64(td_error * td_error)

            for a in range(Self.ACTIONS):
                if a == action:
                    grad_arr[b * Self.ACTIONS + a] = (
                        Scalar[dtype](2.0)
                        * td_error
                        / Scalar[dtype](Self.batch_size)
                    )
                else:
                    grad_arr[b * Self.ACTIONS + a] = Scalar[dtype](0.0)

        total_loss /= Float64(Self.batch_size)

        # Backward
        var grad_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.ACTIONS),
            MutAnyOrigin,
        ](grad_arr.unsafe_ptr())
        var grad_in_arr = InlineArray[
            Scalar[dtype], Self.batch_size * Self.OBS
        ](uninitialized=True)
        var grad_in_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.OBS),
            MutAnyOrigin,
        ](grad_in_arr.unsafe_ptr())
        var g = cpu_state.online.grads_view()

        cpu_state.online.zero_grads()
        Self.Q_Network.backward[Self.batch_size](
            grad_t, grad_in_t, p_online, cache_t, g
        )
        cpu_state.online.optimizer_step()

        cpu_state.target.soft_update_from(cpu_state.online, self.tau)
        self.train_step_count += 1

        return total_loss

    fn decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        return self.epsilon

    fn get_train_steps(self) -> Int:
        return self.train_step_count

    # =========================================================================
    # OffPolicyDiscreteAgent trait
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    fn select_action[
        dt: DType
    ](mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[dt]]) -> Int:
        var obs_inline = obs_to_inline[Self.OBS, dt](obs)
        return self._select_action_inline(cpu_state, obs_inline, False)

    fn store_transition[
        dt: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[dt]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[dt]],
        done: Bool,
    ) -> None:
        cpu_state.store[dt](obs, action, reward, next_obs, done)

    fn decay_explore(mut self) -> None:
        self.decay_epsilon()

    fn get_explore_rate(self) -> Float64:
        return self.epsilon

    fn random_action(self) -> Int:
        return Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS

    fn select_greedy_action(
        self,
        cpu_state: Self.CPUStateType,
        obs: List[Float64],
    ) -> Int:
        var obs_inline = obs_to_inline[Self.OBS, DType.float64](obs)
        return self._select_action_inline(cpu_state, obs_inline, True)

    # =========================================================================
    # GPUOffPolicyAgent trait
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.download_to(self.state.online, ctx)
        gpu_state.target.download_to(self.state.target, ctx)

    fn select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var q_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.env_q_buf.unsafe_ptr())
        var p = gpu_state.online.params_view()
        Self.Q_Network.forward_gpu[N_ENVS](ctx, obs_t, q_t, p, gpu_state.inf_ws)

        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var epsilon_s = Scalar[dtype](self.epsilon)
        var seed_s = Scalar[DType.uint32](
            UInt32(self.train_step_count * 2654435761)
        )

        @always_inline
        fn argmax_wrapper(
            eps: Scalar[dtype],
            q_vals: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            acts: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            base_seed: Scalar[DType.uint32],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= N_ENVS:
                return

            var rng = xorshift32(
                Scalar[DType.uint32](b * 2654435761) + base_seed
            )
            var rand_result = random_uniform[dtype](rng)
            var rand_val = rand_result[0]
            rng = rand_result[1]

            if rand_val < eps:
                var action_result = random_uniform[dtype](rng)
                acts[b] = Scalar[dtype](
                    Int(action_result[0] * Scalar[dtype](Self.ACTIONS))
                )
                return

            var best_q = q_vals[b, 0]
            var best_action = 0
            for a in range(1, Self.ACTIONS):
                var qv = q_vals[b, a]
                if qv > best_q:
                    best_q = qv
                    best_action = a
            acts[b] = Scalar[dtype](best_action)

        ctx.enqueue_function[argmax_wrapper, argmax_wrapper](
            epsilon_s,
            q_t,
            actions_t,
            seed_s,
            grid_dim=((N_ENVS + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        comptime BATCH = Self.batch_size
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        # Sample batch
        gpu_state.buffer.sample[BATCH](
            ctx,
            UInt32(self.train_step_count),
            gpu_state.s_obs,
            gpu_state.s_act,
            gpu_state.s_rew,
            gpu_state.s_nobs,
            gpu_state.s_done,
            gpu_state.s_idx,
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_nobs.unsafe_ptr())
        var q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.q_values.unsafe_ptr())
        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.next_q_values.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Q_Model.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu_state.cache.unsafe_ptr())
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_rew.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_done.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_act.unsafe_ptr())
        var grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.grad_output.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.grad_input.unsafe_ptr())

        # Online forward with cache
        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()
        Self.Q_Network.forward_gpu_with_cache[BATCH](
            ctx, obs_t, q_t, p_online, cache_t, gpu_state.train_ws
        )

        # Target forward
        Self.Q_Network.forward_gpu[BATCH](
            ctx, next_obs_t, next_q_t, p_target, gpu_state.train_ws
        )

        # TD targets
        var gamma_s = Scalar[dtype](self.gamma)

        comptime if Self.double_dqn:
            var online_next_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ](gpu_state.online_next_q.unsafe_ptr())
            Self.Q_Network.forward_gpu[BATCH](
                ctx, next_obs_t, online_next_t, p_online, gpu_state.train_ws
            )

            @always_inline
            fn double_td_wrapper(
                tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                onq: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                tnq: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                don: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                g: Scalar[dtype],
            ):
                dqn_double_td_target_kernel[dtype, BATCH, Self.ACTIONS](
                    tgt, onq, tnq, rew, don, g
                )

            ctx.enqueue_function[double_td_wrapper, double_td_wrapper](
                targets_t,
                online_next_t,
                next_q_t,
                rewards_t,
                dones_t,
                gamma_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
        else:

            @always_inline
            fn td_wrapper(
                tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                nq: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                don: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                g: Scalar[dtype],
            ):
                dqn_td_target_kernel[dtype, BATCH, Self.ACTIONS](
                    tgt, nq, rew, don, g
                )

            ctx.enqueue_function[td_wrapper, td_wrapper](
                targets_t,
                next_q_t,
                rewards_t,
                dones_t,
                gamma_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        # Gradient kernel (masked MSE)
        @always_inline
        fn grad_wrapper(
            grd: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            qv: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            act: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var action = Int(act[b])
            var q_pred = qv[b, action]
            var td_error = q_pred - tgt[b]
            for a in range(Self.ACTIONS):
                if a == action:
                    grd[b, a] = (
                        Scalar[dtype](2.0) * td_error / Scalar[dtype](BATCH)
                    )
                else:
                    grd[b, a] = Scalar[dtype](0.0)

        ctx.enqueue_function[grad_wrapper, grad_wrapper](
            grad_t,
            q_t,
            targets_t,
            actions_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # Backward + optimizer step
        var g = gpu_state.online.grads_view()
        gpu_state.online.zero_grads(ctx)
        Self.Q_Network.backward_gpu[BATCH](
            ctx, grad_t, grad_in_t, p_online, cache_t, g, gpu_state.train_ws
        )
        gpu_state.online.optimizer_step(ctx)

        self.train_step_count += 1

    fn get_action_scale(self) -> Float64:
        return 1.0

    fn get_total_steps(self) -> Int:
        return self.train_step_count

    fn set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        gpu_state.target.soft_update_from_gpu(gpu_state.online, self.tau, ctx)

    # =========================================================================
    # High-level training
    # =========================================================================

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 5000,
        warmup_steps: Int = 1000,
        train_every: Int = 4,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        var algo_name = String("DQN CNN")
        if Self.double_dqn:
            algo_name = String("Double DQN CNN")
        var cpu_state = Self.CPUStateType()
        var checkpoint_every = self.checkpoint_every
        var checkpoint_path = self.checkpoint_path
        var metrics = run_offpolicy_discrete_train(
            self,
            cpu_state,
            env,
            num_episodes,
            max_steps_per_episode,
            warmup_steps,
            train_every,
            checkpoint_every,
            checkpoint_path,
            verbose,
            print_every,
            environment_name,
            algo_name,
        )
        self.state = cpu_state^
        return metrics

    fn evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 5000,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
        greedy: Bool = True,
    ) raises -> Float64:
        var metrics = run_offpolicy_discrete_eval(
            self,
            self.state,
            env,
            num_episodes=num_episodes,
            max_steps=max_steps_per_episode,
            verbose=verbose,
            render=render,
            frame_delay_ms=frame_delay_ms,
        )
        return metrics.mean_reward()

    fn train_gpu[
        E: GPUDiscreteEnv,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 5000,
        gradient_steps: Int = 0,
        sync_every: Int = 5000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        var algo_name = String(
            "DQN CNN (GPU)" if not Self.double_dqn else "Double DQN CNN (GPU)"
        )
        return run_offpolicy_discrete_train_gpu[E, Self](
            self,
            ctx,
            num_steps,
            warmup_steps=warmup_steps,
            gradient_steps=gradient_steps,
            sync_every=sync_every,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
        )

    # =========================================================================
    # Checkpoint
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        comptime PARAM_SIZE = Self.Q_Network.PARAM_SIZE
        comptime STATE_SIZE = PARAM_SIZE * Adam[Self.lr].STATE_PER_PARAM

        var content = write_checkpoint_header(
            "dqn_cnn_agent", PARAM_SIZE, STATE_SIZE
        )
        content += self.state.online.write_sections("online_")
        content += self.state.target.write_sections("target_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("lr=" + String(Self.lr))
        metadata.append("epsilon=" + String(self.epsilon))
        metadata.append("epsilon_min=" + String(self.epsilon_min))
        metadata.append("epsilon_decay=" + String(self.epsilon_decay))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        var content = read_checkpoint_file(filepath)
        var header = parse_checkpoint_header(content)

        comptime PARAM_SIZE = Self.Q_Network.PARAM_SIZE
        if header.param_size != PARAM_SIZE:
            print(
                "Warning: checkpoint param_size ("
                + String(header.param_size)
                + ") != PARAM_SIZE ("
                + String(PARAM_SIZE)
                + ")"
            )

        self.state.online.read_sections(content, "online_")
        self.state.target.read_sections(content, "target_")

        var metadata = read_metadata_section(content)

        var gamma_str = get_metadata_value(metadata, "gamma")
        if len(gamma_str) > 0:
            self.gamma = atof(gamma_str)

        var tau_str = get_metadata_value(metadata, "tau")
        if len(tau_str) > 0:
            self.tau = atof(tau_str)

        var epsilon_str = get_metadata_value(metadata, "epsilon")
        if len(epsilon_str) > 0:
            self.epsilon = atof(epsilon_str)

        var epsilon_min_str = get_metadata_value(metadata, "epsilon_min")
        if len(epsilon_min_str) > 0:
            self.epsilon_min = atof(epsilon_min_str)

        var epsilon_decay_str = get_metadata_value(metadata, "epsilon_decay")
        if len(epsilon_decay_str) > 0:
            self.epsilon_decay = atof(epsilon_decay_str)

        var train_step_str = get_metadata_value(metadata, "train_step_count")
        if len(train_step_str) > 0:
            self.train_step_count = Int(atol(train_step_str))
