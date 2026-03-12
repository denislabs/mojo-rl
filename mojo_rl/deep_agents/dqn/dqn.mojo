"""DQN Agent using the new trait-based deep learning architecture.

This DQN implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- GPUNetworkState for device-side training (local to train_gpu)
- GPUReplayBuffer for GPU-side experience replay (local to train_gpu)
- Sequential composition for Q-networks

Features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Epsilon-greedy exploration with decay
- Target network with soft updates
- Double DQN to reduce overestimation bias (optional)
- GPU support via GPUOffPolicyAgent trait + run_offpolicy_discrete_train_gpu
- lr is a compile-time parameter (Adam LR baked in at compile time)

Usage:
    from mojo_rl.deep_agents.dqn import DQNAgent
    from mojo_rl.envs import CartPoleNative

    var env = CartPoleNative()
    var agent = DQNAgent[4, 2, 64, 10000, 32]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=200)

    # GPU Training (step-based, using shared off-policy GPU loop)
    var ctx = DeviceContext()
    var metrics_gpu = agent.train_gpu[CartPoleGPUEnv](ctx, num_steps=100000)

    # Evaluate
    var avg_reward = agent.evaluate(env, num_episodes=10, greedy=True)
"""

from std.math import exp
from std.random import random_float64, seed

from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TILE, TPB
from mojo_rl.nn.model import Linear, Sequential, LinearReLU
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.deep_agents.core import (
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
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
)
from mojo_rl.nn.gpu import (
    xorshift32,
    random_uniform,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from .state import DQNGPUState, DQNCPUState
from .kernels import dqn_td_target_kernel, dqn_double_td_target_kernel
from mojo_rl.deep_agents.core.perf_timer import PerfTimer
from mojo_rl.nn.model.model import PerfTimerPtr


# =============================================================================
# DQN Agent
# =============================================================================


struct DQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 64,
    buffer_capacity: Int = 10000,
    batch_size: Int = 256,
    n_envs: Int = 1024,
    double_dqn: Bool = True,
    lr: Float64 = 0.001,
    profile: Int = 0,
](OffPolicyDiscreteAgent & GPUOffPolicyAgent & Checkpointable):
    """Deep Q-Network agent — unified CPU + GPU.

    DQN is an off-policy value-based algorithm for discrete action spaces.

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size (default: 64).
        buffer_capacity: Replay buffer capacity (default: 10000).
        batch_size: Training batch size for gradient updates (default: 256).
        n_envs: Number of parallel environments for GPU training (default: 1024).
        double_dqn: Use Double DQN (default: True).
        lr: Adam learning rate — compile-time (default: 0.001).

    Note on batch_size vs n_envs (GPU training):
        n_envs controls parallel data collection; batch_size controls update size.
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime BATCH = Self.batch_size

    # Q-network: obs → hidden (ReLU) → hidden (ReLU) → num_actions
    comptime Q_Model = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.ACTIONS],
    ]
    comptime Q_Network = Network[Self.Q_Model, Adam[Self.lr]]

    # CPU state type (online + target networks + replay buffer)
    comptime CPUStateType = DQNCPUState[
        Self.Q_Model,
        Adam[Self.lr],
        Self.buffer_capacity,
        Self.obs_dim,
        Self.batch_size,
    ]

    # GPUOffPolicyAgent required compile-time constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = 1  # discrete action stored as float scalar index
    comptime BUFFER_CAPACITY: Int = Self.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GPUStateType = DQNGPUState[
        Self.Q_Model,
        Adam[Self.lr],
        Self.buffer_capacity,
        Self.obs_dim,
        Self.num_actions,
        Self.batch_size,
        Self.n_envs,
    ]

    # CPU state: persistent for evaluate() and checkpointing
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

    # Level-2 profiler: sub-phases of do_gpu_train_step
    var train_timer: PerfTimer[Self.profile >= 1]

    # Level-3 profiler: per-layer timing (base slot indices into train_timer)
    var online_fwd_base: Int
    var target_fwd_base: Int
    var online_bwd_base: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.01,
        epsilon_decay: Float64 = 0.995,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize DQN agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update rate for target network (default: 0.005).
            epsilon: Initial exploration rate (default: 1.0).
            epsilon_min: Minimum exploration rate (default: 0.01).
            epsilon_decay: Epsilon decay per episode (default: 0.995).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path for auto-checkpointing.
        """
        self.state = Self.CPUStateType()

        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.train_timer = PerfTimer[Self.profile >= 1]()
        self.online_fwd_base = 0
        self.target_fwd_base = 0
        self.online_bwd_base = 0
        comptime if Self.profile >= 2:
            _ = self.train_timer.add_slot("sample_batch")       # 0
            _ = self.train_timer.add_slot("online_forward")     # 1
            _ = self.train_timer.add_slot("target_forward")     # 2
            _ = self.train_timer.add_slot("td_targets")         # 3
            _ = self.train_timer.add_slot("grad_kernel")        # 4
            _ = self.train_timer.add_slot("backward_update")    # 5
        comptime if Self.profile >= 3:
            # L3 slots as children of L2 sub-phases
            # slot 1 = online_forward, 2 = target_forward, 5 = backward_update
            self.online_fwd_base = Self.Q_Model.register_forward_slots(
                self.train_timer, parent=1
            )
            self.target_fwd_base = Self.Q_Model.register_forward_slots(
                self.train_timer, parent=2
            )
            self.online_bwd_base = Self.Q_Model.register_backward_slots(
                self.train_timer, parent=5
            )

    fn _perf_ptr(mut self) -> PerfTimerPtr:
        """Return opaque timer pointer for L3 profiling (null when profile < 3)."""
        comptime if Self.profile >= 3:
            return UnsafePointer(to=self.train_timer).bitcast[NoneType]()
        else:
            return PerfTimerPtr(unsafe_from_address=0)

    # =========================================================================
    # Action Selection
    # =========================================================================

    fn _select_action_inline(
        self,
        cpu_state: Self.CPUStateType,
        obs: InlineArray[Scalar[dtype], Self.obs_dim],
        greedy: Bool,
    ) -> Int:
        """Internal greedy/epsilon-greedy action selection using InlineArray obs.
        """
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
        """Perform one CPU training step.

        Args:
            cpu_state: CPU state with replay buffer and network weights.

        Returns:
            Loss value (0 if buffer not ready).
        """
        if not cpu_state.buffer.is_ready[Self.batch_size]():
            return 0.0

        var batch_obs = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)
        var batch_actions_tmp = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_rewards = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)
        var batch_dones = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        # action_dim=1 — use a length-1 row per sample then flatten
        var batch_act1 = InlineArray[Scalar[dtype], Self.batch_size * 1](
            uninitialized=True
        )
        cpu_state.buffer.sample[Self.batch_size](
            batch_obs, batch_act1, batch_rewards, batch_next_obs, batch_dones
        )
        for i in range(Self.batch_size):
            batch_actions_tmp[i] = batch_act1[i]

        # LayoutTensor views over sampled data
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

        # Forward: target (no cache)
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

        # Compute TD targets
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

        # Compute gradient (MSE, masked to taken action)
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
        """Decay exploration rate (call at end of each episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Get current exploration rate."""
        return self.epsilon

    fn get_train_steps(self) -> Int:
        """Get total training steps performed."""
        return self.train_step_count

    # =========================================================================
    # OffPolicyDiscreteAgent trait conformance
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        """Allocate a fresh CPUStateType (networks + replay buffer)."""
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
    # GPUOffPolicyAgent trait conformance
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for DQN training.

        Does NOT upload CPU weights — call upload_to_gpu after this.
        """
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network weights to GPU online and target networks."""
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
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
        """Forward Q-network on GPU for N_ENVS environments + epsilon-greedy selection.

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with Q-network and inference scratch buffers.
            obs_buf: Observations [N_ENVS * obs_dim].
            actions_buf: Output actions [N_ENVS] (float scalar indices).
        """
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
            UInt32(self.get_total_steps() * 2654435761)
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
        """One DQN training step on GPU: sample → TD targets → backward → update.

        Soft-update of target network is handled separately by soft_update_targets_gpu.

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with replay buffer, networks, and scratch buffers.
        """
        comptime BATCH = Self.batch_size
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        # ---- Phase 1: Sample batch ----
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_mark(ctx)
        gpu_state.buffer.sample[BATCH](
            ctx,
            UInt32(self.train_step_count * (BATCH + 1)),
            gpu_state.s_obs,
            gpu_state.s_act,
            gpu_state.s_rew,
            gpu_state.s_nobs,
            gpu_state.s_done,
            gpu_state.s_idx,
        )
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(0, ctx)
            self.train_timer.mark()

        # LayoutTensor views for sampled batch
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

        # ---- Phase 2: Online forward with cache ----
        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()
        Self.Q_Network.forward_gpu_with_cache[BATCH](
            ctx, obs_t, q_t, p_online, cache_t, gpu_state.train_ws,
            perf=self._perf_ptr(), perf_slot=self.online_fwd_base,
        )
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(1, ctx)
            self.train_timer.mark()

        # ---- Phase 3: Target forward (no cache) ----
        Self.Q_Network.forward_gpu[BATCH](
            ctx, next_obs_t, next_q_t, p_target, gpu_state.train_ws,
            perf=self._perf_ptr(), perf_slot=self.target_fwd_base,
        )

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(2, ctx)
            self.train_timer.mark()

        # ---- Phase 4: Compute TD targets ----
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

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(3, ctx)
            self.train_timer.mark()

        # ---- Phase 5: Gradient kernel (masked MSE grad) ----
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

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(4, ctx)
            self.train_timer.mark()

        # ---- Phase 6: Backward + optimizer step ----
        var g = gpu_state.online.grads_view()
        gpu_state.online.zero_grads(ctx)
        Self.Q_Network.backward_gpu[BATCH](
            ctx, grad_t, grad_in_t, p_online, cache_t, g, gpu_state.train_ws,
            perf=self._perf_ptr(), perf_slot=self.online_bwd_base,
        )
        gpu_state.online.optimizer_step(ctx)
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(5, ctx)

        self.train_step_count += 1

    fn get_action_scale(self) -> Float64:
        return 1.0  # Discrete actions don't use action_scale

    fn get_total_steps(self) -> Int:
        return self.train_step_count

    fn set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    fn decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        """Linear epsilon schedule matching CleanRL:
        epsilon = max(end_e, start_e + (end_e - start_e) * t / duration).
        Exploration fraction = 0.5 (decay over first half of training).
        """
        var duration = Float64(num_steps) * 0.5  # exploration_fraction = 0.5
        var slope = (self.epsilon_min - 1.0) / duration
        self.epsilon = max(
            self.epsilon_min,
            slope * Float64(total_steps) + 1.0,
        )

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Soft-update target Q-network on GPU: θ_t ← τ*θ + (1-τ)*θ_t."""
        gpu_state.target.soft_update_from_gpu(gpu_state.online, self.tau, ctx)

    # =========================================================================
    # High-level CPU training and evaluation
    # =========================================================================

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        warmup_steps: Int = 1000,
        train_every: Int = 4,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train the DQN agent on a continuous-state environment.

        Args:
            env: Environment (must implement BoxDiscreteActionEnv).
            num_episodes: Number of training episodes.
            max_steps_per_episode: Maximum steps per episode (default: 500).
            warmup_steps: Random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 4).
            verbose: Print progress (default: False).
            print_every: Print every N episodes if verbose (default: 10).
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        var algo_name = String("DQN")
        if Self.double_dqn:
            algo_name = String("Double DQN")
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
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
        greedy: Bool = True,
    ) raises -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: Environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps_per_episode: Maximum steps per episode (default: 500).
            verbose: Print per-episode results (default: False).
            render: Render the environment (default: False).
            frame_delay_ms: Delay between frames in ms (default: 16).
            greedy: Unused (always greedy via select_greedy_action).

        Returns:
            Average reward across episodes.
        """
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

    # =========================================================================
    # GPU Training — delegates to shared run_offpolicy_discrete_train_gpu
    # =========================================================================

    fn train_gpu[
        E: GPUDiscreteEnv,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
        gradient_steps: Int = 0,
        sync_every: Int = 5000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train on GPU using the shared off-policy discrete GPU loop.

        GPU state (networks, replay buffer, scratch buffers) is created
        locally for the duration of training and freed when the method returns.
        After this call self.state.online / target hold the trained GPU weights,
        so evaluate() works immediately.

        All step-based parameters are in total env transitions (n_envs per
        loop iteration), matching on-policy convention.

        Parameters:
            E: GPU environment type implementing GPUDiscreteEnv.
            PROFILE: Whether to profile the training loop.

        Args:
            ctx: GPU device context.
            num_steps: Total env transitions across all parallel envs.
            warmup_steps: Transitions before training starts (default: 1000).
            gradient_steps: Training steps per env collection iteration.
                0 (default) = n_envs for 1:1 replay ratio.
            sync_every: GPU→CPU sync interval in transitions (default: 5000).
            verbose: Print progress (default: False).
            print_every: Print interval in transitions (default: 50000).
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics with episode-level statistics.
        """
        var algo_name = String(
            "DQN (GPU)" if not Self.double_dqn else "Double DQN (GPU)"
        )

        # Create profiling timer with L1 slots (off-policy phases)
        var timer = PerfTimer[Self.profile >= 1]()
        _ = timer.add_slot("copy_prev_obs")
        _ = timer.add_slot("select_actions")
        _ = timer.add_slot("env_step")
        _ = timer.add_slot("buffer_store")
        _ = timer.add_slot("episode_tracking")
        _ = timer.add_slot("reset")
        _ = timer.add_slot("train_step")
        _ = timer.add_slot("gpu_cpu_sync")

        var metrics = run_offpolicy_discrete_train_gpu[E, Self, Self.profile](
            self,
            ctx,
            num_steps,
            timer,
            warmup_steps=warmup_steps,
            gradient_steps=gradient_steps,
            sync_every=sync_every,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
        )

        # Merge L2 (+ L3 descendants) as children of train_step (slot 6)
        comptime if Self.profile >= 2:
            timer.merge_children(6, self.train_timer)

        comptime if Self.profile >= 1:
            timer.print_report(algo_name + " Profile")

        return metrics^

    # =========================================================================
    # Checkpoint Save / Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save DQN agent state to a checkpoint file.

        Saves online and target network params + optimizer states,
        plus epsilon and training counters.  Replay buffer is NOT saved.

        Args:
            filepath: Destination path (e.g. "dqn_agent.ckpt").
        """
        comptime PARAM_SIZE = Self.Q_Network.PARAM_SIZE
        comptime STATE_SIZE = PARAM_SIZE * Adam[Self.lr].STATE_PER_PARAM

        var content = write_checkpoint_header(
            "dqn_agent", PARAM_SIZE, STATE_SIZE
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
        """Load DQN agent state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
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
