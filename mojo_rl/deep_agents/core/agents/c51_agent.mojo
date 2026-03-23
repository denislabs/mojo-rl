"""C51 (Categorical DQN) agent — distributional reinforcement learning.

C51 models Q-values as discrete distributions over NUM_ATOMS bins in [V_MIN, V_MAX]
instead of scalar values. Uses cross-entropy loss with Bellman-projected target
distributions.

Key differences from standard DQN:
  - Network outputs ACTIONS * NUM_ATOMS logits (distribution per action)
  - Action selection: Q(s,a) = sum_i(softmax(logits_a)_i * z_i)
  - Target: Bellman projection of target distribution onto support
  - Loss: Cross-entropy between predicted and projected target distribution

Reference: Bellemare, Dabney, Munos (2017) "A Distributional Perspective on RL"

GPU support via GPUOffPolicyAgent trait + run_offpolicy_discrete_train_gpu.
"""

from std.math import exp, log, floor, ceil
from std.random import random_float64
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor
from mojo_rl.deep_agents.core import (
    run_offpolicy_discrete_train_gpu,
    PerfTimer,
)
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model, Linear, LinearReLU, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import (
    Network,
    NetworkState,
    NetworkPair,
    GPUNetworkState,
)
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.loss.two_hot import compute_bins
from mojo_rl.nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
)

from mojo_rl.deep_agents.core import (
    OffPolicyDiscreteState,
    OffPolicyDiscreteAgent,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    Checkpointable,
)
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.eval import (
    run_offpolicy_discrete_eval,
)


# =============================================================================
# C51 Config Trait
# =============================================================================


trait CategoricalDQNConfig:
    """Compile-time config for C51 (categorical DQN) agents."""

    comptime NAME: String
    comptime obs_dim: Int
    comptime num_actions: Int
    comptime num_atoms: Int
    comptime v_min: Float64
    comptime v_max: Float64
    comptime batch_size: Int
    comptime buffer_capacity: Int
    comptime QModel: Model
    comptime QOpt: Optimizer


# =============================================================================
# C51 Config
# =============================================================================


struct C51Config[
    OBS: Int,
    ACT: Int,
    NUM_ATOMS: Int = 51,
    V_MIN: Float64 = -10.0,
    V_MAX: Float64 = 10.0,
    HIDDEN: Int = 128,
    HIDDEN2: Int = 128,
    CAP: Int = 10000,
    BS: Int = 32,
    lr: Float64 = 2.5e-4,
](CategoricalDQNConfig):
    """C51 (Categorical DQN) config.

    Network outputs ACT * NUM_ATOMS logits — a categorical distribution per action.

    Parameters:
        OBS: Observation dimension.
        ACT: Number of discrete actions.
        NUM_ATOMS: Number of atoms in the categorical distribution (default: 51).
        V_MIN: Minimum support value (default: -10.0).
        V_MAX: Maximum support value (default: 10.0).
        HIDDEN: First hidden layer size.
        HIDDEN2: Second hidden layer size.
        CAP: Replay buffer capacity.
        BS: Training batch size.
        lr: Learning rate.
    """

    comptime NAME: String = "C51"
    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime num_atoms: Int = Self.NUM_ATOMS
    comptime v_min: Float64 = Self.V_MIN
    comptime v_max: Float64 = Self.V_MAX
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN2],
        Linear[Self.HIDDEN2, Self.ACT * Self.NUM_ATOMS],
    ]
    comptime QOpt = Adam[Self.lr]


# =============================================================================
# C51 CPU State
# =============================================================================


struct C51CPUState[
    QModel: Model,
    QOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    batch_size: Int,
    num_atoms: Int,
    v_min: Float64,
    v_max: Float64,
](Movable, OffPolicyDiscreteState):
    """CPU state for C51: online + target Q-networks + replay buffer + bins."""

    comptime BUFFER_DTYPE = dtype

    var online: NetworkState[Self.QModel, Self.QOpt]
    var target: NetworkState[Self.QModel, Self.QOpt]
    var buffer: HeapReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, dtype]
    var bins: InlineArray[Float32, Self.num_atoms]

    def __init__(out self):
        self.online = NetworkState[Self.QModel, Self.QOpt]()
        self.online.initialize[Xavier[]]()
        self.target = NetworkState[Self.QModel, Self.QOpt]()
        self.target.initialize[Xavier[]]()
        # Copy online -> target
        self.target.copy_params_from(self.online)
        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()
        self.bins = compute_bins[Self.num_atoms](
            Float32(Self.v_min), Float32(Self.v_max)
        )

    def store[
        d: DType
    ](
        mut self,
        obs: List[Scalar[d]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        var obs_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], 1](
            uninitialized=True
        )
        act_arr[0] = Scalar[Self.BUFFER_DTYPE](action)
        self.buffer.add(
            obs_arr,
            act_arr,
            Scalar[Self.BUFFER_DTYPE](reward),
            next_arr,
            done,
        )

    def is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# C51 GPU State
# =============================================================================


struct C51GPUState[
    QModel: Model,
    QOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    num_actions: Int,
    num_atoms: Int,
    batch_size: Int,
    max_n_envs: Int,
](GPUOffPolicyState):
    """GPU-resident state for C51 training.

    Extends the standard DQN GPU state with distributional buffers:
      - target_dist: Projected target distribution [batch_size * num_atoms]
      - bins_buf: Pre-computed bin values on GPU [num_atoms]
      - expected_q: Expected Q-values for action selection [batch_size * num_actions]
    """

    comptime Q_Net = Network[Self.QModel, Self.QOpt]
    comptime CACHE_SIZE = Self.QModel.CACHE_SIZE
    comptime WS_PER_SAMPLE = Self.Q_Net.WORKSPACE_SIZE_PER_SAMPLE
    comptime RAW_OUT = Self.QModel.OUT_DIM  # num_actions * num_atoms

    # GPU network states (online + target)
    var online: GPUNetworkState[Self.QModel, Self.QOpt]
    var target: GPUNetworkState[Self.QModel, Self.QOpt]

    # GPU replay buffer
    var buffer: GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim]

    # Inference buffers (max_n_envs sized)
    var env_raw_buf: DeviceBuffer[dtype]  # [max_n_envs * RAW_OUT]
    var env_q_buf: DeviceBuffer[dtype]  # [max_n_envs * num_actions]
    var inf_ws: DeviceBuffer[dtype]  # [max(1, max_n_envs * WS_PER_SAMPLE)]

    # Training scratch -- replay sample output
    var s_obs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_act: DeviceBuffer[dtype]  # [batch_size]
    var s_rew: DeviceBuffer[dtype]  # [batch_size]
    var s_nobs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_done: DeviceBuffer[dtype]  # [batch_size]
    var s_idx: DeviceBuffer[DType.int32]  # [batch_size]

    # Training scratch -- raw forward output (logits)
    var q_raw: DeviceBuffer[dtype]  # [batch_size * RAW_OUT]
    var next_q_raw: DeviceBuffer[dtype]  # [batch_size * RAW_OUT]
    var online_next_q_raw: DeviceBuffer[dtype]  # [batch_size * RAW_OUT]

    # Training scratch -- expected Q-values (for action selection)
    var expected_q: DeviceBuffer[dtype]  # [batch_size * num_actions]
    var next_expected_q: DeviceBuffer[dtype]  # [batch_size * num_actions]

    # C51-specific: target distribution + bins
    var target_dist: DeviceBuffer[dtype]  # [batch_size * num_atoms]
    var bins_buf: DeviceBuffer[dtype]  # [num_atoms]

    # Training scratch -- cache, gradients
    var cache: DeviceBuffer[dtype]  # [batch_size * CACHE_SIZE]
    var grad_raw: DeviceBuffer[dtype]  # [batch_size * RAW_OUT]
    var grad_input: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var train_ws: DeviceBuffer[dtype]  # [max(1, batch_size * WS_PER_SAMPLE)]

    # Diagnostic host buffers
    var diag_raw_host: HostBuffer[dtype]  # [batch_size * RAW_OUT]
    var diag_act_host: HostBuffer[dtype]  # [batch_size]
    var diag_rew_host: HostBuffer[dtype]  # [batch_size]
    var diag_done_host: HostBuffer[dtype]  # [batch_size]

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers."""
        self.online = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.target = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.buffer = GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim](ctx)

        # Inference buffers
        self.env_raw_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs * Self.RAW_OUT
        )
        self.env_q_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs * Self.num_actions
        )
        var inf_ws_size = max(1, Self.max_n_envs * Self.WS_PER_SAMPLE)
        self.inf_ws = ctx.enqueue_create_buffer[dtype](inf_ws_size)

        # Replay sample output
        self.s_obs = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )
        self.s_act = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.s_rew = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.s_nobs = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )
        self.s_done = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](Self.batch_size)

        # Raw forward output buffers
        var batch_raw_size = Self.batch_size * Self.RAW_OUT
        self.q_raw = ctx.enqueue_create_buffer[dtype](batch_raw_size)
        self.next_q_raw = ctx.enqueue_create_buffer[dtype](batch_raw_size)
        self.online_next_q_raw = ctx.enqueue_create_buffer[dtype](
            batch_raw_size
        )

        # Expected Q buffers
        var batch_q_size = Self.batch_size * Self.num_actions
        self.expected_q = ctx.enqueue_create_buffer[dtype](batch_q_size)
        self.next_expected_q = ctx.enqueue_create_buffer[dtype](batch_q_size)

        # C51-specific buffers
        self.target_dist = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.num_atoms
        )
        self.bins_buf = ctx.enqueue_create_buffer[dtype](Self.num_atoms)

        # Cache, gradients
        self.cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CACHE_SIZE
        )
        self.grad_raw = ctx.enqueue_create_buffer[dtype](batch_raw_size)
        self.grad_input = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )
        var train_ws_size = max(1, Self.batch_size * Self.WS_PER_SAMPLE)
        self.train_ws = ctx.enqueue_create_buffer[dtype](train_ws_size)

        # Diagnostics
        self.diag_raw_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size * Self.RAW_OUT
        )
        self.diag_act_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        self.diag_rew_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        self.diag_done_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )

    # GPUOffPolicyState required methods

    def gpu_store[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        rewards_buf: DeviceBuffer[dtype],
        obs_buf: DeviceBuffer[dtype],
        dones_buf: DeviceBuffer[dtype],
    ) raises -> None:
        self.buffer.store[N_ENVS](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, dones_buf
        )

    def gpu_buffer_is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# Helper: compute expected Q from distributional logits (CPU)
# =============================================================================


def _expected_q_from_logits[
    BATCH: Int,
    ACTIONS: Int,
    NUM_ATOMS: Int,
    RAW_OUT: Int,
](
    raw_out: InlineArray[Scalar[dtype], BATCH * RAW_OUT],
    bins: InlineArray[Float32, NUM_ATOMS],
    mut q_values: InlineArray[Scalar[dtype], BATCH * ACTIONS],
) -> None:
    """Compute expected Q-value for each action from distributional logits.

    Q(s,a) = sum_i(softmax(logits_a)_i * z_i)

    Args:
        raw_out: Network output [BATCH, RAW_OUT] where RAW_OUT = ACTIONS * NUM_ATOMS.
        bins: Bin support values [NUM_ATOMS].
        q_values: Output expected Q-values [BATCH, ACTIONS].
    """
    for b in range(BATCH):
        for a in range(ACTIONS):
            var base = b * RAW_OUT + a * NUM_ATOMS
            # Numerically stable softmax
            var max_val = raw_out[base]
            for i in range(1, NUM_ATOMS):
                if raw_out[base + i] > max_val:
                    max_val = raw_out[base + i]
            var sum_exp = Scalar[dtype](0.0)
            for i in range(NUM_ATOMS):
                sum_exp += exp(raw_out[base + i] - max_val)
            var expected = Scalar[dtype](0.0)
            for i in range(NUM_ATOMS):
                var prob = exp(raw_out[base + i] - max_val) / sum_exp
                expected += prob * Scalar[dtype](bins[i])
            q_values[b * ACTIONS + a] = expected


# =============================================================================
# GenericC51Agent
# =============================================================================


struct GenericC51Agent[
    Config: CategoricalDQNConfig,
    n_envs: Int = 1024,
    L: Logger = NoOpLogger,
](OffPolicyDiscreteAgent & GPUOffPolicyAgent & Checkpointable):
    """C51 (Categorical DQN) agent.

    Models Q-values as categorical distributions over NUM_ATOMS bins.
    Uses Double DQN (online for action selection, target for evaluation).

    Parameters:
        Config: C51Config with network architecture and distributional params.
        n_envs: Number of parallel environments for GPU training.
        L: Logger type for diagnostic logging.
    """

    comptime OBS: Int = Self.Config.QModel.IN_DIM
    comptime RAW_OUT: Int = Self.Config.QModel.OUT_DIM  # ACTIONS * NUM_ATOMS
    comptime ACTIONS: Int = Self.Config.num_actions
    comptime NUM_ATOMS: Int = Self.Config.num_atoms
    comptime BATCH: Int = Self.Config.batch_size
    comptime Q_CS: Int = Self.Config.QModel.CACHE_SIZE
    comptime QNet = Network[Self.Config.QModel, Self.Config.QOpt]

    comptime CPUStateType = C51CPUState[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.batch_size,
        Self.Config.num_atoms,
        Self.Config.v_min,
        Self.Config.v_max,
    ]

    # GPUOffPolicyAgent compile-time constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = 1
    comptime BUFFER_CAPACITY: Int = Self.Config.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GPUStateType = C51GPUState[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.num_actions,
        Self.Config.num_atoms,
        Self.Config.batch_size,
        Self.n_envs,
    ]

    # Persistent CPU state (for evaluate() after train/train_gpu)
    var state: Self.CPUStateType

    var gamma: Float64
    var tau: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64
    var exploration_fraction: Float64
    var target_update_freq: Int
    var train_step_count: Int
    var target_total_steps: Int
    var _target_update_ctr: Int
    var checkpoint_every: Int
    var checkpoint_path: String

    # Logger
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    def __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 1.0,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.05,
        epsilon_decay: Float64 = 0.995,
        exploration_fraction: Float64 = 0.5,
        target_update_freq: Int = 500,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        target_total_steps: Int = 0,
    ):
        self.state = Self.CPUStateType()
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.exploration_fraction = exploration_fraction
        self.target_update_freq = target_update_freq
        self.train_step_count = 0
        self.target_total_steps = target_total_steps
        self._target_update_ctr = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    def make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    # =========================================================================
    # Action selection (CPU)
    # =========================================================================

    def select_action[
        d: DType
    ](mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[d]]) -> Int:
        # Epsilon-greedy
        if random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.ACTIONS))

        var obs_arr = obs_to_inline[Self.OBS, d](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var raw_arr = InlineArray[Scalar[dtype], Self.RAW_OUT](
            uninitialized=True
        )
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.RAW_OUT), MutAnyOrigin
        ](raw_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.QNet.forward[1](obs_t, raw_t, p)

        # Compute expected Q from distributional output
        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        _expected_q_from_logits[1, Self.ACTIONS, Self.NUM_ATOMS, Self.RAW_OUT](
            raw_arr, cpu_state.bins, q_arr
        )

        var best = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best = a
        return best

    def store_transition[
        d: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[d]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        cpu_state.store[d](obs, action, reward, next_obs, done)

    # =========================================================================
    # CPU Training Step (C51-specific)
    # =========================================================================

    def do_cpu_train_step(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> Float64:
        """C51 training step: distributional Bellman projection + cross-entropy loss.
        """
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

        comptime ATOMS = Self.NUM_ATOMS

        # Sample batch
        var b_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_act1 = InlineArray[Scalar[dtype], Self.BATCH * 1](
            uninitialized=True
        )
        var b_rew = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var b_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_done = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        cpu_state.buffer.sample[Self.BATCH](
            b_obs, b_act1, b_rew, b_next, b_done
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_next.unsafe_ptr())

        # Online forward with cache
        var raw_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.RAW_OUT](
            uninitialized=True
        )
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](raw_arr.unsafe_ptr())
        var cache_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.Q_CS](
            uninitialized=True
        )
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.Q_CS), MutAnyOrigin
        ](cache_arr.unsafe_ptr())
        var p_online = cpu_state.online.params_view()
        Self.QNet.forward_with_cache[Self.BATCH](
            obs_t, raw_t, p_online, cache_t
        )

        # Target forward on next_obs
        var target_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        var target_raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](target_raw_arr.unsafe_ptr())
        var p_target = cpu_state.target.params_view()
        Self.QNet.forward[Self.BATCH](next_obs_t, target_raw_t, p_target)

        # Online forward on next_obs (Double DQN: online selects action)
        var online_next_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        var online_next_raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](online_next_raw_arr.unsafe_ptr())
        Self.QNet.forward[Self.BATCH](next_obs_t, online_next_raw_t, p_online)

        # Compute expected Q from online_next for action selection (Double DQN)
        var online_next_q = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        _expected_q_from_logits[Self.BATCH, Self.ACTIONS, ATOMS, Self.RAW_OUT](
            online_next_raw_arr, cpu_state.bins, online_next_q
        )

        # Distributional parameters
        var v_min = Float64(Self.Config.v_min)
        var v_max = Float64(Self.Config.v_max)
        var dz = (v_max - v_min) / Float64(ATOMS - 1)

        # Compute gradient in raw output space
        var grad_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](fill=Scalar[dtype](0.0))

        var total_loss: Float64 = 0.0

        for b in range(Self.BATCH):
            # 1. Select best next action using online network (Double DQN)
            var best_next_a = 0
            var best_next_q = online_next_q[b * Self.ACTIONS]
            for a in range(1, Self.ACTIONS):
                var q = online_next_q[b * Self.ACTIONS + a]
                if q > best_next_q:
                    best_next_q = q
                    best_next_a = a

            # 2. Get target distribution for best action: softmax(target_logits[a*])
            var target_base = b * Self.RAW_OUT + best_next_a * ATOMS
            var target_max = target_raw_arr[target_base]
            for i in range(1, ATOMS):
                if target_raw_arr[target_base + i] > target_max:
                    target_max = target_raw_arr[target_base + i]
            var target_sum_exp = Scalar[dtype](0.0)
            for i in range(ATOMS):
                target_sum_exp += exp(
                    target_raw_arr[target_base + i] - target_max
                )
            var target_probs = InlineArray[Scalar[dtype], ATOMS](
                fill=Scalar[dtype](0.0)
            )
            for i in range(ATOMS):
                target_probs[i] = (
                    exp(target_raw_arr[target_base + i] - target_max)
                    / target_sum_exp
                )

            # 3. Bellman projection: Tz_j = clip(r + gamma * z_j, v_min, v_max)
            var projected = InlineArray[Scalar[dtype], ATOMS](
                fill=Scalar[dtype](0.0)
            )
            var r = Float64(b_rew[b])
            var dm = Float64(Scalar[dtype](1.0) - b_done[b])

            for j in range(ATOMS):
                var tz = r + self.gamma * Float64(cpu_state.bins[j]) * dm
                # Clamp to [v_min, v_max]
                if tz < v_min:
                    tz = v_min
                if tz > v_max:
                    tz = v_max
                # Project onto support
                var bj = (tz - v_min) / dz
                var l_idx = Int(floor(bj))
                var u_idx = Int(ceil(bj))
                if l_idx == u_idx:
                    projected[l_idx] += target_probs[j]
                else:
                    var u_weight = Scalar[dtype](bj - Float64(l_idx))
                    var l_weight = Scalar[dtype](1.0) - u_weight
                    projected[l_idx] += target_probs[j] * l_weight
                    projected[u_idx] += target_probs[j] * u_weight

            # 4. Cross-entropy loss and gradient for taken action
            var action = Int(b_act1[b])
            var pred_base = b * Self.RAW_OUT + action * ATOMS

            # Softmax of predicted logits for taken action
            var pred_max = raw_arr[pred_base]
            for i in range(1, ATOMS):
                if raw_arr[pred_base + i] > pred_max:
                    pred_max = raw_arr[pred_base + i]
            var pred_sum_exp = Scalar[dtype](0.0)
            for i in range(ATOMS):
                pred_sum_exp += exp(raw_arr[pred_base + i] - pred_max)
            var log_sum_exp = pred_max + log(pred_sum_exp)

            # Cross-entropy loss: -sum(m_i * log_softmax(logits_i))
            var sample_loss: Float64 = 0.0
            for i in range(ATOMS):
                var log_sm = Float64(raw_arr[pred_base + i]) - Float64(
                    log_sum_exp
                )
                sample_loss -= Float64(projected[i]) * log_sm
            total_loss += sample_loss

            # Gradient: (softmax(logits) - projected_target) / BATCH
            for i in range(ATOMS):
                var sm = exp(raw_arr[pred_base + i] - pred_max) / pred_sum_exp
                grad_raw_arr[pred_base + i] = (sm - projected[i]) / Scalar[
                    dtype
                ](Self.BATCH)

        var loss = total_loss / Float64(Self.BATCH)

        # Backward + optimizer step
        var grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](grad_raw_arr.unsafe_ptr())
        var d_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](d_obs.unsafe_ptr())
        var g = cpu_state.online.grads_view()
        cpu_state.online.zero_grads()
        Self.QNet.backward[Self.BATCH](grad_t, d_obs_t, p_online, cache_t, g)
        cpu_state.online.optimizer_step()

        self.train_step_count += 1

        # ---- Diagnostic logging ----
        if self.logger and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count

                # Expected Q-value stats (from online_next used for action selection)
                var online_q = InlineArray[
                    Scalar[dtype], Self.BATCH * Self.ACTIONS
                ](uninitialized=True)
                _expected_q_from_logits[
                    Self.BATCH, Self.ACTIONS, ATOMS, Self.RAW_OUT
                ](raw_arr, cpu_state.bins, online_q)

                var q_min = Float64(online_q[0])
                var q_max = Float64(online_q[0])
                var q_sum: Float64 = 0.0
                for i in range(Self.BATCH * Self.ACTIONS):
                    var v = Float64(online_q[i])
                    q_sum += v
                    if v < q_min:
                        q_min = v
                    if v > q_max:
                        q_max = v
                self.logger[].log_scalar(
                    "q_mean",
                    q_sum / Float64(Self.BATCH * Self.ACTIONS),
                    step,
                )
                self.logger[].log_scalar("q_min", q_min, step)
                self.logger[].log_scalar("q_max", q_max, step)

                # CE loss
                self.logger[].log_scalar("loss", loss, step)

                # Distribution entropy (how peaked/spread the predicted dist is)
                var entropy_sum: Float64 = 0.0
                for b in range(Self.BATCH):
                    var action = Int(b_act1[b])
                    var pred_base = b * Self.RAW_OUT + action * ATOMS
                    var pred_max2 = raw_arr[pred_base]
                    for i in range(1, ATOMS):
                        if raw_arr[pred_base + i] > pred_max2:
                            pred_max2 = raw_arr[pred_base + i]
                    var se2 = Scalar[dtype](0.0)
                    for i in range(ATOMS):
                        se2 += exp(raw_arr[pred_base + i] - pred_max2)
                    var h: Float64 = 0.0
                    for i in range(ATOMS):
                        var p = Float64(
                            exp(raw_arr[pred_base + i] - pred_max2) / se2
                        )
                        if p > 1e-8:
                            h -= p * log(p)
                    entropy_sum += h
                self.logger[].log_scalar(
                    "dist_entropy_mean",
                    entropy_sum / Float64(Self.BATCH),
                    step,
                )
            except:
                pass

        # Target update (hard or soft)
        if (
            self.train_step_count - self._target_update_ctr
            >= self.target_update_freq
        ):
            self._target_update_ctr = self.train_step_count
            if self.tau >= 1.0:
                cpu_state.target.copy_params_from(cpu_state.online)
            else:
                cpu_state.target.soft_update_from(cpu_state.online, self.tau)
        return loss

    def decay_explore(mut self) -> None:
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def get_explore_rate(self) -> Float64:
        return self.epsilon

    def random_action(self) -> Int:
        return Int(random_float64() * Float64(Self.ACTIONS))

    def select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> Int:
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var raw_arr = InlineArray[Scalar[dtype], Self.RAW_OUT](
            uninitialized=True
        )
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.RAW_OUT), MutAnyOrigin
        ](raw_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.QNet.forward[1](obs_t, raw_t, p)

        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        _expected_q_from_logits[1, Self.ACTIONS, Self.NUM_ATOMS, Self.RAW_OUT](
            raw_arr, cpu_state.bins, q_arr
        )

        var best = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best = a
        return best

    # =========================================================================
    # Checkpointable
    # =========================================================================

    def save_checkpoint(self, path: String) raises -> None:
        comptime PARAM_SIZE = Self.QNet.PARAM_SIZE
        comptime STATE_SIZE = PARAM_SIZE * Self.Config.QOpt.STATE_PER_PARAM

        var content = write_checkpoint_header(
            "c51_agent", PARAM_SIZE, STATE_SIZE
        )
        content += self.state.online.write_sections("online_")
        content += self.state.target.write_sections("target_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("epsilon=" + String(self.epsilon))
        metadata.append("epsilon_min=" + String(self.epsilon_min))
        metadata.append("epsilon_decay=" + String(self.epsilon_decay))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(path, content)

    def load_checkpoint(mut self, path: String) raises -> None:
        var content = read_checkpoint_file(path)
        _ = parse_checkpoint_header(content)

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

    # =========================================================================
    # CPU Convenience training
    # =========================================================================

    def train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 300,
        max_steps_per_episode: Int = 500,
        warmup_steps: Int = 1000,
        train_every: Int = 4,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train C51 agent on a discrete-action environment."""
        from mojo_rl.deep_agents.core.training.offpolicy_train import (
            run_offpolicy_discrete_train,
        )

        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = Self.CPUStateType()
        var ckpt_path = String(self.checkpoint_path)
        var algo_name = Self.Config.NAME
        var metrics = run_offpolicy_discrete_train[E, Self, Self.L](
            self,
            cpu_state,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            warmup_steps=warmup_steps,
            train_every=train_every,
            checkpoint_every=self.checkpoint_every,
            checkpoint_path=ckpt_path,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
            logger=logger,
        )
        self.state = cpu_state^
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
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

    # =========================================================================
    # GPUOffPolicyAgent trait conformance
    # =========================================================================

    def get_action_scale(self) -> Float64:
        return 1.0

    def get_total_steps(self) -> Int:
        return self.train_step_count

    def set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    def make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(ctx)

    def upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)
        # Upload bins to GPU
        var bins_host = HostBuffer[dtype](ctx, Self.NUM_ATOMS)
        for i in range(Self.NUM_ATOMS):
            bins_host[i] = Scalar[dtype](self.state.bins[i])
        ctx.enqueue_copy(gpu_state.bins_buf, bins_host)

    def download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.download_to(self.state.online, ctx)
        gpu_state.target.download_to(self.state.target, ctx)

    def select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Forward Q-network on GPU + compute expected Q + epsilon-greedy."""
        # Forward pass: obs -> raw logits
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.env_raw_buf.unsafe_ptr())
        var p = gpu_state.online.params_view()
        Self.QNet.forward_gpu[N_ENVS](ctx, obs_t, raw_t, p, gpu_state.inf_ws)

        # Compute expected Q from logits + epsilon-greedy argmax
        var q_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.env_q_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var bins_t = LayoutTensor[
            dtype, Layout.row_major(Self.NUM_ATOMS), MutAnyOrigin
        ](gpu_state.bins_buf.unsafe_ptr())
        var epsilon_s = Scalar[dtype](self.epsilon)
        var seed_val = Scalar[DType.uint64](
            UInt64(self.get_total_steps()) * UInt64(2654435761)
        )

        @always_inline
        def c51_select_kernel(
            eps: Scalar[dtype],
            raw: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
            ],
            q_out: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            bins: LayoutTensor[
                dtype, Layout.row_major(Self.NUM_ATOMS), MutAnyOrigin
            ],
            acts: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            base_seed: Scalar[DType.uint64],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= N_ENVS:
                return

            # Compute expected Q for each action
            for a in range(Self.ACTIONS):
                var base = a * Self.NUM_ATOMS
                # Stable softmax
                var max_val = rebind[Scalar[dtype]](raw[b, base])
                for i in range(1, Self.NUM_ATOMS):
                    var v = rebind[Scalar[dtype]](raw[b, base + i])
                    if v > max_val:
                        max_val = v
                var sum_exp: Scalar[dtype] = 0.0
                for i in range(Self.NUM_ATOMS):
                    sum_exp += exp(
                        rebind[Scalar[dtype]](raw[b, base + i]) - max_val
                    )
                var expected: Scalar[dtype] = 0.0
                for i in range(Self.NUM_ATOMS):
                    var prob = (
                        exp(rebind[Scalar[dtype]](raw[b, base + i]) - max_val)
                        / sum_exp
                    )
                    expected += prob * rebind[Scalar[dtype]](bins[i])
                q_out[b, a] = expected

            # Epsilon-greedy
            var rng = PhiloxRandom(
                seed=UInt64(base_seed) + UInt64(b),
                offset=0,
            )
            var rand_vals = rng.step_uniform()
            var rand_val = Scalar[dtype](rand_vals[0])

            if rand_val < eps:
                acts[b] = Scalar[dtype](
                    Int(
                        Scalar[dtype](rand_vals[1])
                        * Scalar[dtype](Self.ACTIONS)
                    )
                    % Self.ACTIONS
                )
                return

            var best_q = q_out[b, 0]
            var best_action = 0
            for a in range(1, Self.ACTIONS):
                var qv = q_out[b, a]
                if qv > best_q:
                    best_q = qv
                    best_action = a
            acts[b] = Scalar[dtype](best_action)

        ctx.enqueue_function[c51_select_kernel, c51_select_kernel](
            epsilon_s,
            raw_t,
            q_t,
            bins_t,
            actions_t,
            seed_val,
            grid_dim=((N_ENVS + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    def do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """C51 GPU training step: sample -> project distribution -> CE grad -> backward.
        """
        comptime BATCH = Self.BATCH
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime ATOMS = Self.NUM_ATOMS

        # ---- Phase 1: Sample batch ----
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

        # LayoutTensor views
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_nobs.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_act.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_rew.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_done.unsafe_ptr())

        var q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.q_raw.unsafe_ptr())
        var next_q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.next_q_raw.unsafe_ptr())
        var online_next_q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.online_next_q_raw.unsafe_ptr())
        var next_eq_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.next_expected_q.unsafe_ptr())
        var bins_t = LayoutTensor[dtype, Layout.row_major(ATOMS), MutAnyOrigin](
            gpu_state.bins_buf.unsafe_ptr()
        )
        var grad_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.grad_raw.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Q_CS), MutAnyOrigin
        ](gpu_state.cache.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.grad_input.unsafe_ptr())

        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()

        # ---- Phase 2: Online forward with cache ----
        Self.QNet.forward_gpu_with_cache[BATCH](
            ctx,
            obs_t,
            q_raw_t,
            p_online,
            cache_t,
            gpu_state.train_ws,
        )

        # ---- Phase 3: Target forward on next_obs ----
        Self.QNet.forward_gpu[BATCH](
            ctx,
            next_obs_t,
            next_q_raw_t,
            p_target,
            gpu_state.train_ws,
        )

        # ---- Phase 3b: Online forward on next_obs (Double DQN) ----
        Self.QNet.forward_gpu[BATCH](
            ctx,
            next_obs_t,
            online_next_q_raw_t,
            p_online,
            gpu_state.train_ws,
        )

        # ---- Phase 4: Compute expected Q from online_next for action selection ----
        @always_inline
        def expected_q_kernel(
            raw: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
            ],
            bins: LayoutTensor[dtype, Layout.row_major(ATOMS), MutAnyOrigin],
            eq: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            for a in range(Self.ACTIONS):
                var base = a * ATOMS
                var max_val = rebind[Scalar[dtype]](raw[b, base])
                for i in range(1, ATOMS):
                    var v = rebind[Scalar[dtype]](raw[b, base + i])
                    if v > max_val:
                        max_val = v
                var sum_exp: Scalar[dtype] = 0.0
                for i in range(ATOMS):
                    sum_exp += exp(
                        rebind[Scalar[dtype]](raw[b, base + i]) - max_val
                    )
                var expected: Scalar[dtype] = 0.0
                for i in range(ATOMS):
                    var prob = (
                        exp(rebind[Scalar[dtype]](raw[b, base + i]) - max_val)
                        / sum_exp
                    )
                    expected += prob * rebind[Scalar[dtype]](bins[i])
                eq[b, a] = expected

        ctx.enqueue_function[expected_q_kernel, expected_q_kernel](
            online_next_q_raw_t,
            bins_t,
            next_eq_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 5: Bellman projection + CE gradient (combined kernel) ----
        var gamma_s = Scalar[dtype](self.gamma)
        var v_min_s = Scalar[dtype](Self.Config.v_min)
        var v_max_s = Scalar[dtype](Self.Config.v_max)
        var dz_s = Scalar[dtype](
            (Self.Config.v_max - Self.Config.v_min) / Float64(ATOMS - 1)
        )

        @always_inline
        def c51_project_grad_kernel(
            online_raw: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
            ],
            target_raw: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
            ],
            next_eq: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            bins: LayoutTensor[dtype, Layout.row_major(ATOMS), MutAnyOrigin],
            actions: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            grad: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
            ],
            gamma: Scalar[dtype],
            vmin: Scalar[dtype],
            vmax: Scalar[dtype],
            dz: Scalar[dtype],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return

            # Zero gradient for this sample
            for i in range(Self.RAW_OUT):
                grad[b, i] = Scalar[dtype](0.0)

            # 1. Select best next action from expected Q (Double DQN)
            var best_a = 0
            var best_q = rebind[Scalar[dtype]](next_eq[b, 0])
            for a in range(1, Self.ACTIONS):
                var q = rebind[Scalar[dtype]](next_eq[b, a])
                if q > best_q:
                    best_q = q
                    best_a = a

            # 2. Target distribution softmax for best_a
            var t_base = best_a * ATOMS
            var t_max = rebind[Scalar[dtype]](target_raw[b, t_base])
            for i in range(1, ATOMS):
                var v = rebind[Scalar[dtype]](target_raw[b, t_base + i])
                if v > t_max:
                    t_max = v
            var t_sum_exp: Scalar[dtype] = 0.0
            for i in range(ATOMS):
                t_sum_exp += exp(
                    rebind[Scalar[dtype]](target_raw[b, t_base + i]) - t_max
                )

            # 3. Bellman projection
            var projected = InlineArray[Scalar[dtype], ATOMS](
                fill=Scalar[dtype](0.0)
            )
            var r = rebind[Scalar[dtype]](rewards[b])
            var dm = Scalar[dtype](1.0) - rebind[Scalar[dtype]](dones[b])

            for j in range(ATOMS):
                var t_prob = (
                    exp(
                        rebind[Scalar[dtype]](target_raw[b, t_base + j]) - t_max
                    )
                    / t_sum_exp
                )
                var tz = r + gamma * rebind[Scalar[dtype]](bins[j]) * dm
                # Clamp
                if tz < vmin:
                    tz = vmin
                if tz > vmax:
                    tz = vmax
                var bj = (tz - vmin) / dz
                var l_idx = Int(floor(bj))
                var u_idx = Int(ceil(bj))
                if l_idx == u_idx:
                    projected[l_idx] = projected[l_idx] + t_prob
                else:
                    var u_w = bj - Scalar[dtype](l_idx)
                    var l_w = Scalar[dtype](1.0) - u_w
                    projected[l_idx] = projected[l_idx] + t_prob * l_w
                    projected[u_idx] = projected[u_idx] + t_prob * u_w

            # 4. CE gradient for taken action
            var action = Int(rebind[Scalar[dtype]](actions[b]))
            var p_base = action * ATOMS
            var p_max = rebind[Scalar[dtype]](online_raw[b, p_base])
            for i in range(1, ATOMS):
                var v = rebind[Scalar[dtype]](online_raw[b, p_base + i])
                if v > p_max:
                    p_max = v
            var p_sum_exp: Scalar[dtype] = 0.0
            for i in range(ATOMS):
                p_sum_exp += exp(
                    rebind[Scalar[dtype]](online_raw[b, p_base + i]) - p_max
                )
            for i in range(ATOMS):
                var sm = (
                    exp(
                        rebind[Scalar[dtype]](online_raw[b, p_base + i]) - p_max
                    )
                    / p_sum_exp
                )
                grad[b, p_base + i] = (sm - projected[i]) / Scalar[dtype](BATCH)

        ctx.enqueue_function[c51_project_grad_kernel, c51_project_grad_kernel](
            q_raw_t,
            next_q_raw_t,
            next_eq_t,
            bins_t,
            actions_t,
            rewards_t,
            dones_t,
            grad_raw_t,
            gamma_s,
            v_min_s,
            v_max_s,
            dz_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 6: Backward + optimizer step ----
        var g = gpu_state.online.grads_view()
        gpu_state.online.zero_grads(ctx)
        Self.QNet.backward_gpu[BATCH](
            ctx,
            grad_raw_t,
            grad_in_t,
            p_online,
            cache_t,
            g,
            gpu_state.train_ws,
        )
        gpu_state.online.optimizer_step(ctx)

        self.train_step_count += 1

        # ---- GPU Diagnostic logging ----
        if (
            self.logger
            and self.diag_every > 0
            and self.train_step_count % self.diag_every == 0
        ):
            try:
                # Copy raw logits and sample data to host
                ctx.enqueue_copy(gpu_state.diag_raw_host, gpu_state.q_raw)
                ctx.enqueue_copy(gpu_state.diag_act_host, gpu_state.s_act)
                ctx.enqueue_copy(gpu_state.diag_rew_host, gpu_state.s_rew)
                ctx.enqueue_copy(gpu_state.diag_done_host, gpu_state.s_done)
                ctx.synchronize()

                var step = self.train_step_count

                # Compute expected Q from raw logits on host
                var raw_host_arr = InlineArray[
                    Scalar[dtype], BATCH * Self.RAW_OUT
                ](uninitialized=True)
                for i in range(BATCH * Self.RAW_OUT):
                    raw_host_arr[i] = gpu_state.diag_raw_host[i]
                var bins_host = InlineArray[Scalar[dtype], ATOMS](
                    uninitialized=True
                )
                for i in range(ATOMS):
                    bins_host[i] = Scalar[dtype](
                        Self.Config.v_min
                        + Float64(i)
                        * (Self.Config.v_max - Self.Config.v_min)
                        / Float64(ATOMS - 1)
                    )
                var eq_host = InlineArray[Scalar[dtype], BATCH * Self.ACTIONS](
                    uninitialized=True
                )
                _expected_q_from_logits[
                    BATCH, Self.ACTIONS, ATOMS, Self.RAW_OUT
                ](raw_host_arr, bins_host, eq_host)

                # Q-value stats
                var q_min = Float64(eq_host[0])
                var q_max = Float64(eq_host[0])
                var q_sum: Float64 = 0.0
                for i in range(BATCH * Self.ACTIONS):
                    var v = Float64(eq_host[i])
                    q_sum += v
                    if v < q_min:
                        q_min = v
                    if v > q_max:
                        q_max = v
                self.logger[].log_scalar(
                    "q_mean",
                    q_sum / Float64(BATCH * Self.ACTIONS),
                    step,
                )
                self.logger[].log_scalar("q_min", q_min, step)
                self.logger[].log_scalar("q_max", q_max, step)

                # Done fraction and reward stats from sampled batch
                var done_count: Float64 = 0.0
                var rew_sum: Float64 = 0.0
                var rew_min = Float64(gpu_state.diag_rew_host[0])
                var rew_max = Float64(gpu_state.diag_rew_host[0])
                for b in range(BATCH):
                    done_count += Float64(gpu_state.diag_done_host[b])
                    var r = Float64(gpu_state.diag_rew_host[b])
                    rew_sum += r
                    if r < rew_min:
                        rew_min = r
                    if r > rew_max:
                        rew_max = r
                self.logger[].log_scalar(
                    "done_fraction",
                    done_count / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar(
                    "reward_mean",
                    rew_sum / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar("reward_min", rew_min, step)
                self.logger[].log_scalar("reward_max", rew_max, step)

                # Distribution entropy (how peaked/spread the predicted dist is)
                var entropy_sum: Float64 = 0.0
                for b in range(BATCH):
                    var action = Int(Float64(gpu_state.diag_act_host[b]))
                    var pred_base = b * Self.RAW_OUT + action * ATOMS
                    var pred_max2 = Float64(gpu_state.diag_raw_host[pred_base])
                    for i in range(1, ATOMS):
                        var v = Float64(gpu_state.diag_raw_host[pred_base + i])
                        if v > pred_max2:
                            pred_max2 = v
                    var se2: Float64 = 0.0
                    for i in range(ATOMS):
                        se2 += exp(
                            Float64(gpu_state.diag_raw_host[pred_base + i])
                            - pred_max2
                        )
                    var h: Float64 = 0.0
                    for i in range(ATOMS):
                        var p = (
                            exp(
                                Float64(gpu_state.diag_raw_host[pred_base + i])
                                - pred_max2
                            )
                            / se2
                        )
                        if p > 1e-8:
                            h -= p * log(p)
                    entropy_sum += h
                self.logger[].log_scalar(
                    "dist_entropy_mean",
                    entropy_sum / Float64(BATCH),
                    step,
                )
            except:
                pass

    def soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        if (
            self.train_step_count - self._target_update_ctr
            >= self.target_update_freq
        ):
            gpu_state.target.soft_update_from_gpu(
                gpu_state.online, self.tau, ctx
            )
            self._target_update_ctr = self.train_step_count

    def decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        var duration = Float64(num_steps) * self.exploration_fraction
        var slope = (self.epsilon_min - 1.0) / duration
        self.epsilon = max(self.epsilon_min, slope * Float64(total_steps) + 1.0)

    # =========================================================================
    # GPU Training convenience
    # =========================================================================

    def train_gpu[
        E: GPUDiscreteEnv,
        CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
        gradient_steps: Int = 0,
        sync_every: Int = 5000,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        target_total_steps: Int = 0,
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train C51 on GPU."""
        self.logger = logger
        self.diag_every = diag_every
        self.target_total_steps = target_total_steps
        var timer = PerfTimer[False]()
        var algo_name = Self.Config.NAME
        return run_offpolicy_discrete_train_gpu[
            E, Self, 0, Self.L, CurriculumType
        ](
            self,
            ctx,
            num_steps,
            timer,
            logger=logger,
            warmup_steps=warmup_steps,
            gradient_steps=gradient_steps,
            sync_every=sync_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
            target_total_steps=target_total_steps,
        )
