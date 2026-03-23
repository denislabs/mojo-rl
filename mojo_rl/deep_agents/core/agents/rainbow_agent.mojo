"""Rainbow DQN agent — combines all 6 DQN improvements.

Rainbow = C51 (distributional) + Double DQN + PER + Dueling + Noisy Networks + N-step.

Architecture:
  Sequential[
    NoisyLinearReLU[OBS, HIDDEN],
    NoisyLinearReLU[HIDDEN, HIDDEN],
    Parallel[
      Sequential[NoisyLinearReLU[HIDDEN, STREAM_H], NoisyLinear[STREAM_H, NUM_ATOMS]],
      Sequential[NoisyLinearReLU[HIDDEN, STREAM_H], NoisyLinear[STREAM_H, ACT*NUM_ATOMS]],
    ],
  ]
  RAW_OUT = (1 + ACT) * NUM_ATOMS

Dueling distributional combine (per atom):
  Q_logit(s,a,i) = V(i) + A(a,i) - mean_a'(A(a',i))

Training: n-step returns → PER buffer → IS-weighted CE loss → Bellman projection with γ^n.

Reference: Hessel et al., "Rainbow: Combining Improvements in Deep RL" (2018)
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
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Parallel,
    NoisyLinear,
    NoisyLinearReLU,
    Conv2DReLU,
    FlattenLayer,
)
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import (
    Network,
    NetworkState,
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
from mojo_rl.deep_agents.core.replay import (
    PrioritizedReplayBuffer,
    HostPrioritizedReplayBuffer,
    NStepBuffer,
    NStepTransition,
    GPUNStepBuffer,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.eval import run_offpolicy_discrete_eval


# =============================================================================
# Rainbow Config Trait
# =============================================================================


trait RainbowDQNConfig:
    """Compile-time config for Rainbow DQN agents."""

    comptime NAME: String
    comptime obs_dim: Int
    comptime num_actions: Int
    comptime num_atoms: Int
    comptime v_min: Float64
    comptime v_max: Float64
    comptime n_step: Int
    comptime batch_size: Int
    comptime buffer_capacity: Int
    comptime QModel: Model
    comptime QOpt: Optimizer


# =============================================================================
# Rainbow Config
# =============================================================================


struct RainbowConfig[
    OBS: Int,
    ACT: Int,
    NUM_ATOMS: Int = 51,
    V_MIN: Float64 = -10.0,
    V_MAX: Float64 = 10.0,
    HIDDEN: Int = 128,
    STREAM_H: Int = 128,
    N_STEP: Int = 3,
    CAP: Int = 100000,
    BS: Int = 32,
    lr: Float64 = 6.25e-5,
](RainbowDQNConfig):
    """Rainbow DQN config.

    Parameters:
        OBS: Observation dimension.
        ACT: Number of discrete actions.
        NUM_ATOMS: Atoms in categorical distribution (default 51).
        V_MIN: Min support value.
        V_MAX: Max support value.
        HIDDEN: Shared hidden layer size.
        STREAM_H: Dueling stream hidden size.
        N_STEP: N-step return horizon (default 3).
        CAP: Replay buffer capacity.
        BS: Training batch size.
        lr: Learning rate (default 6.25e-5 per Rainbow paper).
    """

    comptime NAME: String = "Rainbow"
    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime num_atoms: Int = Self.NUM_ATOMS
    comptime v_min: Float64 = Self.V_MIN
    comptime v_max: Float64 = Self.V_MAX
    comptime n_step: Int = Self.N_STEP
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    # Dueling noisy network: V(NUM_ATOMS) + A(ACT*NUM_ATOMS)
    comptime QModel = Sequential[
        NoisyLinearReLU[Self.OBS, Self.HIDDEN],
        NoisyLinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Sequential[
                NoisyLinearReLU[Self.HIDDEN, Self.STREAM_H],
                NoisyLinear[Self.STREAM_H, Self.NUM_ATOMS],
            ],
            Sequential[
                NoisyLinearReLU[Self.HIDDEN, Self.STREAM_H],
                NoisyLinear[Self.STREAM_H, Self.ACT * Self.NUM_ATOMS],
            ],
        ],
    ]
    comptime QOpt = Adam[Self.lr]


# =============================================================================
# Rainbow CNN Config (for pixel observations like Atari/Pong)
# =============================================================================


struct RainbowCNNConfig[
    ACT: Int,
    NUM_ATOMS: Int = 51,
    V_MIN: Float64 = -10.0,
    V_MAX: Float64 = 10.0,
    N_STEP: Int = 3,
    CAP: Int = 10000,
    BS: Int = 32,
    lr: Float64 = 6.25e-5,
](RainbowDQNConfig):
    """Rainbow DQN with Nature CNN for 4x84x84 pixel observations.

    Architecture: Conv layers (shared) → NoisyLinear dueling heads.
      Conv2D(4→32, 8×8, stride=4) → ReLU
      Conv2D(32→64, 4×4, stride=2) → ReLU
      Conv2D(64→64, 3×3, stride=1) → ReLU
      Flatten(3136)
      NoisyLinearReLU(3136→512)
      Parallel[
        NoisyLinear(512→NUM_ATOMS),          # V distribution
        NoisyLinear(512→ACT*NUM_ATOMS),      # A distributions
      ]

    Parameters:
        ACT: Number of discrete actions.
        NUM_ATOMS: Atoms in categorical distribution.
        V_MIN: Min support value.
        V_MAX: Max support value.
        N_STEP: N-step return horizon.
        CAP: Replay buffer capacity.
        BS: Training batch size.
        lr: Learning rate.
    """

    comptime NAME: String = "Rainbow CNN"
    comptime obs_dim: Int = 4 * 84 * 84  # 28224
    comptime num_actions: Int = Self.ACT
    comptime num_atoms: Int = Self.NUM_ATOMS
    comptime v_min: Float64 = Self.V_MIN
    comptime v_max: Float64 = Self.V_MAX
    comptime n_step: Int = Self.N_STEP
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        Conv2DReLU[4, 32, 8, 4, 0, 84, 84],
        Conv2DReLU[32, 64, 4, 2, 0, 20, 20],
        Conv2DReLU[64, 64, 3, 1, 0, 9, 9],
        FlattenLayer[64 * 7 * 7],
        NoisyLinearReLU[64 * 7 * 7, 512],
        Parallel[
            NoisyLinear[512, Self.NUM_ATOMS],
            NoisyLinear[512, Self.ACT * Self.NUM_ATOMS],
        ],
    ]
    comptime QOpt = Adam[Self.lr]


# =============================================================================
# Dueling distributional helpers (per-atom combine/reverse)
# =============================================================================


def _dueling_dist_combine[
    BATCH: Int,
    ACTIONS: Int,
    ATOMS: Int,
    RAW_OUT: Int,
](
    raw_out: InlineArray[Scalar[dtype], BATCH * RAW_OUT],
    mut combined: InlineArray[Scalar[dtype], BATCH * ACTIONS * ATOMS],
) -> None:
    """Combine V_dist + A_dist per atom: Q(s,a,i) = V(i) + A(a,i) - mean_a(A(a,i)).

    raw_out layout: [V(ATOMS) | A(ACT*ATOMS)] per sample.
    combined layout: [ACT*ATOMS] per sample (contiguous per action).
    """
    for b in range(BATCH):
        var v_base = b * RAW_OUT
        var a_base = b * RAW_OUT + ATOMS

        for i in range(ATOMS):
            var v_i = raw_out[v_base + i]
            # Compute mean advantage for atom i across actions
            var mean_a = Scalar[dtype](0)
            for a in range(ACTIONS):
                mean_a += raw_out[a_base + a * ATOMS + i]
            mean_a /= Scalar[dtype](ACTIONS)
            # Combine
            for a in range(ACTIONS):
                combined[b * ACTIONS * ATOMS + a * ATOMS + i] = (
                    v_i + raw_out[a_base + a * ATOMS + i] - mean_a
                )


def _dueling_dist_grad_reverse[
    BATCH: Int,
    ACTIONS: Int,
    ATOMS: Int,
    RAW_OUT: Int,
](
    grad_combined: InlineArray[Scalar[dtype], BATCH * ACTIONS * ATOMS],
    mut grad_raw: InlineArray[Scalar[dtype], BATCH * RAW_OUT],
) -> None:
    """Reverse dueling gradient: from combined [ACT*ATOMS] space to raw [RAW_OUT] space.
    """
    for b in range(BATCH):
        var v_base = b * RAW_OUT
        var a_base = b * RAW_OUT + ATOMS

        for i in range(ATOMS):
            var sum_dq = Scalar[dtype](0)
            for a in range(ACTIONS):
                sum_dq += grad_combined[b * ACTIONS * ATOMS + a * ATOMS + i]
            # dV[i] = sum_a(dQ(a,i))
            grad_raw[v_base + i] = sum_dq
            # dA(a,i) = dQ(a,i) - (1/ACT)*sum_a'(dQ(a',i))
            var one_over_n = Scalar[dtype](1.0) / Scalar[dtype](ACTIONS)
            for a in range(ACTIONS):
                grad_raw[a_base + a * ATOMS + i] = (
                    grad_combined[b * ACTIONS * ATOMS + a * ATOMS + i]
                    - one_over_n * sum_dq
                )


# =============================================================================
# Expected Q from combined distributional logits
# =============================================================================


def _rainbow_expected_q[
    BATCH: Int,
    ACTIONS: Int,
    ATOMS: Int,
    COMBINED_SIZE: Int,
](
    combined: InlineArray[Scalar[dtype], BATCH * COMBINED_SIZE],
    bins: InlineArray[Float32, ATOMS],
    mut q_values: InlineArray[Scalar[dtype], BATCH * ACTIONS],
) -> None:
    """Compute expected Q from combined distributional logits."""
    for b in range(BATCH):
        for a in range(ACTIONS):
            var base = b * COMBINED_SIZE + a * ATOMS
            var max_val = combined[base]
            for i in range(1, ATOMS):
                if combined[base + i] > max_val:
                    max_val = combined[base + i]
            var sum_exp = Scalar[dtype](0)
            for i in range(ATOMS):
                sum_exp += exp(combined[base + i] - max_val)
            var expected = Scalar[dtype](0)
            for i in range(ATOMS):
                var prob = exp(combined[base + i] - max_val) / sum_exp
                expected += prob * Scalar[dtype](bins[i])
            q_values[b * ACTIONS + a] = expected


# =============================================================================
# Rainbow CPU State
# =============================================================================


struct RainbowCPUState[
    QModel: Model,
    QOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    batch_size: Int,
    num_atoms: Int,
    n_step: Int,
    v_min: Float64,
    v_max: Float64,
](Movable, OffPolicyDiscreteState):
    """CPU state for Rainbow: networks + PER buffer + n-step buffer + bins."""

    comptime BUFFER_DTYPE = dtype

    var online: NetworkState[Self.QModel, Self.QOpt]
    var target: NetworkState[Self.QModel, Self.QOpt]
    var buffer: PrioritizedReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, dtype
    ]
    var nstep: NStepBuffer[Self.n_step, Self.obs_dim]
    var bins: InlineArray[Float32, Self.num_atoms]

    def __init__(
        out self,
        alpha: Float64 = 0.6,
        beta: Float64 = 0.4,
        gamma: Float64 = 0.99,
    ):
        self.online = NetworkState[Self.QModel, Self.QOpt]()
        self.online.initialize[Xavier[]]()
        self.target = NetworkState[Self.QModel, Self.QOpt]()
        self.target.initialize[Xavier[]]()
        self.target.copy_params_from(self.online)
        self.buffer = PrioritizedReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ](
            alpha=Scalar[dtype](alpha),
            beta=Scalar[dtype](beta),
        )
        self.nstep = NStepBuffer[Self.n_step, Self.obs_dim](gamma=gamma)
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
        """Store transition through n-step buffer → PER."""
        var obs_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))

        var result = self.nstep.add(
            obs_arr,
            Scalar[Self.BUFFER_DTYPE](action),
            Scalar[Self.BUFFER_DTYPE](reward),
            next_arr,
            done,
        )

        if result.valid:
            var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], 1](
                uninitialized=True
            )
            act_arr[0] = result.action
            self.buffer.add(
                result.obs, act_arr, result.reward, result.next_obs, result.done
            )

    def is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# Rainbow GPU State
# =============================================================================


struct RainbowGPUState[
    QModel: Model,
    QOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    num_actions: Int,
    num_atoms: Int,
    n_step: Int,
    batch_size: Int,
    max_n_envs: Int,
](GPUOffPolicyState):
    """GPU state for Rainbow: networks + PER + n-step + distributional buffers.
    """

    comptime Q_Net = Network[Self.QModel, Self.QOpt]
    comptime CACHE_SIZE = Self.QModel.CACHE_SIZE
    comptime WS_PER_SAMPLE = Self.Q_Net.WORKSPACE_SIZE_PER_SAMPLE
    comptime RAW_OUT = Self.QModel.OUT_DIM  # (1 + num_actions) * num_atoms
    comptime COMBINED = Self.num_actions * Self.num_atoms  # after dueling combine

    # Networks
    var online: GPUNetworkState[Self.QModel, Self.QOpt]
    var target: GPUNetworkState[Self.QModel, Self.QOpt]

    # PER buffer (host-memory for large obs compatibility)
    var buffer: HostPrioritizedReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, Self.batch_size, Self.max_n_envs
    ]

    # N-step buffer
    var nstep: GPUNStepBuffer[Self.n_step, Self.obs_dim, Self.max_n_envs]

    # Inference buffers
    var env_raw_buf: DeviceBuffer[dtype]
    var env_q_buf: DeviceBuffer[dtype]
    var inf_ws: DeviceBuffer[dtype]

    # Training -- replay sample
    var s_obs: DeviceBuffer[dtype]
    var s_act: DeviceBuffer[dtype]
    var s_rew: DeviceBuffer[dtype]
    var s_nobs: DeviceBuffer[dtype]
    var s_done: DeviceBuffer[dtype]
    var s_idx: DeviceBuffer[DType.int32]
    var s_weights: DeviceBuffer[dtype]

    # Training -- raw + combined outputs
    var q_raw: DeviceBuffer[dtype]
    var next_q_raw: DeviceBuffer[dtype]
    var online_next_q_raw: DeviceBuffer[dtype]
    var q_combined: DeviceBuffer[dtype]  # after dueling combine
    var next_q_combined: DeviceBuffer[dtype]
    var online_next_q_combined: DeviceBuffer[dtype]
    var expected_q: DeviceBuffer[dtype]  # [batch * num_actions]

    # Training -- targets, cache, gradients
    var cache: DeviceBuffer[dtype]
    var grad_combined: DeviceBuffer[dtype]  # gradient in combined space
    var grad_raw: DeviceBuffer[dtype]  # gradient in raw space (after reverse)
    var grad_input: DeviceBuffer[dtype]
    var train_ws: DeviceBuffer[dtype]
    var td_errors: DeviceBuffer[dtype]  # for PER priority update
    var bins_buf: DeviceBuffer[dtype]

    # Diagnostic host buffers
    var diag_comb_host: HostBuffer[dtype]  # [batch * COMBINED] combined logits
    var diag_act_host: HostBuffer[dtype]  # [batch]
    var diag_rew_host: HostBuffer[dtype]  # [batch]
    var diag_done_host: HostBuffer[dtype]  # [batch]
    var diag_weights_host: HostBuffer[dtype]  # [batch] IS weights
    var diag_td_host: HostBuffer[dtype]  # [batch] CE loss per sample

    def __init__(
        out self,
        ctx: DeviceContext,
        alpha: Float64 = 0.6,
        beta: Float64 = 0.4,
        gamma: Float64 = 0.99,
    ) raises:
        self.online = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.target = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.buffer = HostPrioritizedReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, Self.batch_size,
            Self.max_n_envs,
        ](ctx, alpha=alpha, beta=beta)
        self.nstep = GPUNStepBuffer[Self.n_step, Self.obs_dim, Self.max_n_envs](
            ctx, gamma=gamma
        )

        # Inference
        self.env_raw_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs * Self.RAW_OUT
        )
        self.env_q_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs * Self.num_actions
        )
        self.inf_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.max_n_envs * Self.WS_PER_SAMPLE)
        )

        # Replay sample
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
        self.s_weights = ctx.enqueue_create_buffer[dtype](Self.batch_size)

        # Raw + combined
        var batch_raw = Self.batch_size * Self.RAW_OUT
        var batch_combined = Self.batch_size * Self.COMBINED
        self.q_raw = ctx.enqueue_create_buffer[dtype](batch_raw)
        self.next_q_raw = ctx.enqueue_create_buffer[dtype](batch_raw)
        self.online_next_q_raw = ctx.enqueue_create_buffer[dtype](batch_raw)
        self.q_combined = ctx.enqueue_create_buffer[dtype](batch_combined)
        self.next_q_combined = ctx.enqueue_create_buffer[dtype](batch_combined)
        self.online_next_q_combined = ctx.enqueue_create_buffer[dtype](
            batch_combined
        )
        self.expected_q = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.num_actions
        )

        # Cache, gradients
        self.cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CACHE_SIZE
        )
        self.grad_combined = ctx.enqueue_create_buffer[dtype](batch_combined)
        self.grad_raw = ctx.enqueue_create_buffer[dtype](batch_raw)
        self.grad_input = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )
        self.train_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.batch_size * Self.WS_PER_SAMPLE)
        )
        self.td_errors = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.bins_buf = ctx.enqueue_create_buffer[dtype](Self.num_atoms)

        # Diagnostics
        self.diag_comb_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size * Self.num_actions * Self.num_atoms
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
        self.diag_weights_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        self.diag_td_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )

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
        """Route through n-step buffer then into PER buffer."""
        # N-step accumulation
        self.nstep.process(
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, dones_buf
        )
        # Store compressed transitions into PER buffer
        self.buffer.store[N_ENVS](
            ctx,
            self.nstep.out_obs,
            self.nstep.out_act,
            self.nstep.out_rew,
            self.nstep.out_nobs,
            self.nstep.out_done,
        )

    def gpu_buffer_is_ready(self) -> Bool:
        return self.buffer.gpu_buffer_is_ready()


# =============================================================================
# GenericRainbowAgent
# =============================================================================


struct GenericRainbowAgent[
    Config: RainbowDQNConfig,
    n_envs: Int = 256,
    L: Logger = NoOpLogger,
](OffPolicyDiscreteAgent & GPUOffPolicyAgent & Checkpointable):
    """Rainbow DQN: C51 + Double DQN + PER + Dueling + Noisy + N-step.

    Parameters:
        Config: RainbowConfig with architecture and distributional params.
        n_envs: Parallel environments for GPU training.
        L: Logger type.
    """

    comptime OBS: Int = Self.Config.QModel.IN_DIM
    comptime RAW_OUT: Int = Self.Config.QModel.OUT_DIM
    comptime ACTIONS: Int = Self.Config.num_actions
    comptime NUM_ATOMS: Int = Self.Config.num_atoms
    comptime BATCH: Int = Self.Config.batch_size
    comptime Q_CS: Int = Self.Config.QModel.CACHE_SIZE
    comptime QNet = Network[Self.Config.QModel, Self.Config.QOpt]
    comptime COMBINED: Int = Self.ACTIONS * Self.NUM_ATOMS

    comptime CPUStateType = RainbowCPUState[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.batch_size,
        Self.Config.num_atoms,
        Self.Config.n_step,
        Self.Config.v_min,
        Self.Config.v_max,
    ]

    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = 1
    comptime BUFFER_CAPACITY: Int = Self.Config.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GPUStateType = RainbowGPUState[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.num_actions,
        Self.Config.num_atoms,
        Self.Config.n_step,
        Self.Config.batch_size,
        Self.n_envs,
    ]

    var state: Self.CPUStateType
    var gamma: Float64
    var tau: Float64
    var target_update_freq: Int
    var train_step_count: Int
    var target_total_steps: Int
    var _target_update_ctr: Int
    var checkpoint_every: Int
    var checkpoint_path: String
    var beta_start: Float64
    var beta_frames: Int
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    def __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 1.0,
        target_update_freq: Int = 500,
        alpha: Float64 = 0.5,
        beta: Float64 = 0.4,
        beta_frames: Int = 100000,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        target_total_steps: Int = 0,
    ):
        self.state = Self.CPUStateType(alpha=alpha, beta=beta, gamma=gamma)
        self.gamma = gamma
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.train_step_count = 0
        self.target_total_steps = target_total_steps
        self._target_update_ctr = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.beta_start = beta
        self.beta_frames = beta_frames
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    def make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType(gamma=self.gamma)

    # =========================================================================
    # Action selection (CPU) — noisy forward + dueling combine + argmax
    # =========================================================================

    def select_action[
        d: DType
    ](mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[d]]) -> Int:
        # No epsilon — noise provides exploration
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

        # Dueling combine
        var combined = InlineArray[Scalar[dtype], Self.COMBINED](
            uninitialized=True
        )
        _dueling_dist_combine[1, Self.ACTIONS, Self.NUM_ATOMS, Self.RAW_OUT](
            raw_arr, combined
        )

        # Expected Q
        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        _rainbow_expected_q[1, Self.ACTIONS, Self.NUM_ATOMS, Self.COMBINED](
            combined, cpu_state.bins, q_arr
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
    # CPU Training Step — Rainbow distributional with all 6 components
    # =========================================================================

    def do_cpu_train_step(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> Float64:
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

        comptime ATOMS = Self.NUM_ATOMS
        comptime COMB = Self.COMBINED

        # Beta annealing
        var progress = Scalar[dtype](
            Float64(self.train_step_count) / Float64(max(1, self.beta_frames))
        )
        if progress > Scalar[dtype](1.0):
            progress = Scalar[dtype](1.0)
        cpu_state.buffer.anneal_beta(progress, Scalar[dtype](self.beta_start))

        # Sample from PER
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
        var b_weights = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var b_indices = InlineArray[Int, Self.BATCH](uninitialized=True)
        cpu_state.buffer.sample[Self.BATCH](
            b_obs, b_act1, b_rew, b_next, b_done, b_weights, b_indices
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

        # Dueling combine online
        var combined = InlineArray[Scalar[dtype], Self.BATCH * COMB](
            uninitialized=True
        )
        _dueling_dist_combine[Self.BATCH, Self.ACTIONS, ATOMS, Self.RAW_OUT](
            raw_arr, combined
        )

        # Target forward + dueling combine
        var target_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        var target_raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](target_raw_arr.unsafe_ptr())
        var p_target = cpu_state.target.params_view()
        Self.QNet.forward[Self.BATCH](next_obs_t, target_raw_t, p_target)
        var target_combined = InlineArray[Scalar[dtype], Self.BATCH * COMB](
            uninitialized=True
        )
        _dueling_dist_combine[Self.BATCH, Self.ACTIONS, ATOMS, Self.RAW_OUT](
            target_raw_arr, target_combined
        )

        # Online-next forward + dueling combine (Double DQN)
        var online_next_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        var online_next_raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](online_next_raw_arr.unsafe_ptr())
        Self.QNet.forward[Self.BATCH](next_obs_t, online_next_raw_t, p_online)
        var online_next_combined = InlineArray[
            Scalar[dtype], Self.BATCH * COMB
        ](uninitialized=True)
        _dueling_dist_combine[Self.BATCH, Self.ACTIONS, ATOMS, Self.RAW_OUT](
            online_next_raw_arr, online_next_combined
        )

        # Expected Q from online-next for action selection (Double DQN)
        var online_next_q = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        _rainbow_expected_q[Self.BATCH, Self.ACTIONS, ATOMS, COMB](
            online_next_combined, cpu_state.bins, online_next_q
        )

        # N-step discount factor
        var gamma_n = self.gamma
        for _ in range(Self.Config.n_step - 1):
            gamma_n *= self.gamma

        var v_min = Float64(Self.Config.v_min)
        var v_max = Float64(Self.Config.v_max)
        var dz = (v_max - v_min) / Float64(ATOMS - 1)

        # Compute gradient in combined space (IS-weighted CE)
        var grad_combined_arr = InlineArray[Scalar[dtype], Self.BATCH * COMB](
            fill=Scalar[dtype](0)
        )

        var total_loss: Float64 = 0.0
        var td_errors = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        for b in range(Self.BATCH):
            # 1. Best next action (Double DQN)
            var best_next_a = 0
            var best_next_q = online_next_q[b * Self.ACTIONS]
            for a in range(1, Self.ACTIONS):
                var q = online_next_q[b * Self.ACTIONS + a]
                if q > best_next_q:
                    best_next_q = q
                    best_next_a = a

            # 2. Target distribution softmax for best action
            var t_base = b * COMB + best_next_a * ATOMS
            var t_max = target_combined[t_base]
            for i in range(1, ATOMS):
                if target_combined[t_base + i] > t_max:
                    t_max = target_combined[t_base + i]
            var t_sum_exp = Scalar[dtype](0)
            for i in range(ATOMS):
                t_sum_exp += exp(target_combined[t_base + i] - t_max)
            var target_probs = InlineArray[Scalar[dtype], ATOMS](
                fill=Scalar[dtype](0)
            )
            for i in range(ATOMS):
                target_probs[i] = (
                    exp(target_combined[t_base + i] - t_max) / t_sum_exp
                )

            # 3. Bellman projection with γ^n
            var projected = InlineArray[Scalar[dtype], ATOMS](
                fill=Scalar[dtype](0)
            )
            var r = Float64(b_rew[b])
            var dm = Float64(Scalar[dtype](1.0) - b_done[b])

            for j in range(ATOMS):
                var tz = r + gamma_n * Float64(cpu_state.bins[j]) * dm
                if tz < v_min:
                    tz = v_min
                if tz > v_max:
                    tz = v_max
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

            # 4. IS-weighted CE gradient for taken action
            var action = Int(b_act1[b])
            var pred_base = b * COMB + action * ATOMS

            var pred_max = combined[pred_base]
            for i in range(1, ATOMS):
                if combined[pred_base + i] > pred_max:
                    pred_max = combined[pred_base + i]
            var pred_sum_exp = Scalar[dtype](0)
            for i in range(ATOMS):
                pred_sum_exp += exp(combined[pred_base + i] - pred_max)
            var log_sum_exp = pred_max + log(pred_sum_exp)

            # CE loss for this sample (used as priority)
            var sample_loss: Float64 = 0.0
            for i in range(ATOMS):
                var log_sm = Float64(combined[pred_base + i]) - Float64(
                    log_sum_exp
                )
                sample_loss -= Float64(projected[i]) * log_sm
            total_loss += sample_loss
            td_errors[b] = Scalar[dtype](sample_loss)

            # IS-weighted gradient: weight * (softmax - target) / BATCH
            var weight = b_weights[b]
            for i in range(ATOMS):
                var sm = exp(combined[pred_base + i] - pred_max) / pred_sum_exp
                grad_combined_arr[pred_base + i] = (
                    weight * (sm - projected[i]) / Scalar[dtype](Self.BATCH)
                )

        var loss = total_loss / Float64(Self.BATCH)

        # Reverse dueling gradient
        var grad_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](fill=Scalar[dtype](0))
        _dueling_dist_grad_reverse[
            Self.BATCH, Self.ACTIONS, ATOMS, Self.RAW_OUT
        ](grad_combined_arr, grad_raw_arr)

        # Backward + optimizer
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

        # Update PER priorities (using CE loss as TD error proxy)
        cpu_state.buffer.update_priorities[Self.BATCH](b_indices, td_errors)

        self.train_step_count += 1

        # ---- Diagnostic logging ----
        if self.logger and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count

                # Expected Q-value stats (from online combined distribution)
                var online_q = InlineArray[
                    Scalar[dtype], Self.BATCH * Self.ACTIONS
                ](uninitialized=True)
                _rainbow_expected_q[Self.BATCH, Self.ACTIONS, ATOMS, COMB](
                    combined, cpu_state.bins, online_q
                )

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

                # TD error stats (CE loss per sample, used as PER priority)
                var td_err_abs_sum: Float64 = 0.0
                var td_err_max_abs: Float64 = 0.0
                for b in range(Self.BATCH):
                    var abs_err = Float64(td_errors[b])
                    if abs_err < 0:
                        abs_err = -abs_err
                    td_err_abs_sum += abs_err
                    if abs_err > td_err_max_abs:
                        td_err_max_abs = abs_err
                self.logger[].log_scalar(
                    "td_error_abs_mean",
                    td_err_abs_sum / Float64(Self.BATCH),
                    step,
                )
                self.logger[].log_scalar("td_error_max", td_err_max_abs, step)

                # IS weight stats (importance sampling correction)
                var w_min = Float64(b_weights[0])
                var w_max = Float64(b_weights[0])
                var w_sum: Float64 = 0.0
                for b in range(Self.BATCH):
                    var w = Float64(b_weights[b])
                    w_sum += w
                    if w < w_min:
                        w_min = w
                    if w > w_max:
                        w_max = w
                self.logger[].log_scalar(
                    "is_weight_mean",
                    w_sum / Float64(Self.BATCH),
                    step,
                )
                self.logger[].log_scalar("is_weight_min", w_min, step)
                self.logger[].log_scalar("is_weight_max", w_max, step)

                # PER beta (IS correction annealing)
                var beta_val = Float64(self.beta_start) + (
                    1.0 - Float64(self.beta_start)
                ) * Float64(self.train_step_count) / Float64(
                    max(1, self.beta_frames)
                )
                if beta_val > 1.0:
                    beta_val = 1.0
                self.logger[].log_scalar("per_beta", beta_val, step)

                # Distribution entropy (how peaked/spread the predicted dist is)
                var entropy_sum: Float64 = 0.0
                for b in range(Self.BATCH):
                    var action = Int(b_act1[b])
                    var pred_base = b * COMB + action * ATOMS
                    var pred_max2 = combined[pred_base]
                    for i in range(1, ATOMS):
                        if combined[pred_base + i] > pred_max2:
                            pred_max2 = combined[pred_base + i]
                    var se2 = Scalar[dtype](0.0)
                    for i in range(ATOMS):
                        se2 += exp(combined[pred_base + i] - pred_max2)
                    var h: Float64 = 0.0
                    for i in range(ATOMS):
                        var p = Float64(
                            exp(combined[pred_base + i] - pred_max2) / se2
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

        # Target update
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
        pass  # No epsilon — noise provides exploration

    def get_explore_rate(self) -> Float64:
        return 0.0

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

        var combined = InlineArray[Scalar[dtype], Self.COMBINED](
            uninitialized=True
        )
        _dueling_dist_combine[1, Self.ACTIONS, Self.NUM_ATOMS, Self.RAW_OUT](
            raw_arr, combined
        )
        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        _rainbow_expected_q[1, Self.ACTIONS, Self.NUM_ATOMS, Self.COMBINED](
            combined, cpu_state.bins, q_arr
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
            "rainbow_agent", PARAM_SIZE, STATE_SIZE
        )
        content += self.state.online.write_sections("online_")
        content += self.state.target.write_sections("target_")
        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
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
        var step_str = get_metadata_value(metadata, "train_step_count")
        if len(step_str) > 0:
            self.train_step_count = Int(atol(step_str))

    # =========================================================================
    # CPU Training convenience
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
        from mojo_rl.deep_agents.core.training.offpolicy_train import (
            run_offpolicy_discrete_train,
        )

        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = Self.CPUStateType(gamma=self.gamma)
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
    # GPUOffPolicyAgent trait
    # =========================================================================

    def get_action_scale(self) -> Float64:
        return 1.0

    def get_total_steps(self) -> Int:
        return self.train_step_count

    def set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    def make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(ctx, gamma=self.gamma)

    def upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)
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
        """Noisy forward → dueling combine → expected Q → argmax (no epsilon).
        """
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.env_raw_buf.unsafe_ptr())
        var p = gpu_state.online.params_view()
        Self.QNet.forward_gpu[N_ENVS](ctx, obs_t, raw_t, p, gpu_state.inf_ws)

        # Dueling combine + expected Q + argmax in one kernel
        var q_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.env_q_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var bins_t = LayoutTensor[
            dtype, Layout.row_major(Self.NUM_ATOMS), MutAnyOrigin
        ](gpu_state.bins_buf.unsafe_ptr())

        @always_inline
        def rainbow_select_kernel(
            raw: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
            ],
            eq: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            bins: LayoutTensor[
                dtype, Layout.row_major(Self.NUM_ATOMS), MutAnyOrigin
            ],
            acts: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= N_ENVS:
                return

            # Dueling combine + expected Q for each action
            for a in range(Self.ACTIONS):
                # Combine V + A - mean(A) per atom, then compute expected Q
                var max_val = Scalar[dtype](-1e10)
                for i in range(Self.NUM_ATOMS):
                    var v_i = rebind[Scalar[dtype]](raw[e, i])
                    var mean_a = Scalar[dtype](0)
                    for a2 in range(Self.ACTIONS):
                        mean_a += rebind[Scalar[dtype]](
                            raw[e, Self.NUM_ATOMS + a2 * Self.NUM_ATOMS + i]
                        )
                    mean_a /= Scalar[dtype](Self.ACTIONS)
                    var q_ai = (
                        v_i
                        + rebind[Scalar[dtype]](
                            raw[e, Self.NUM_ATOMS + a * Self.NUM_ATOMS + i]
                        )
                        - mean_a
                    )
                    if q_ai > max_val:
                        max_val = q_ai

                var sum_exp = Scalar[dtype](0)
                var expected = Scalar[dtype](0)
                for i in range(Self.NUM_ATOMS):
                    var v_i = rebind[Scalar[dtype]](raw[e, i])
                    var mean_a = Scalar[dtype](0)
                    for a2 in range(Self.ACTIONS):
                        mean_a += rebind[Scalar[dtype]](
                            raw[e, Self.NUM_ATOMS + a2 * Self.NUM_ATOMS + i]
                        )
                    mean_a /= Scalar[dtype](Self.ACTIONS)
                    var q_ai = (
                        v_i
                        + rebind[Scalar[dtype]](
                            raw[e, Self.NUM_ATOMS + a * Self.NUM_ATOMS + i]
                        )
                        - mean_a
                    )
                    var e_val = exp(q_ai - max_val)
                    sum_exp += e_val
                    expected += e_val * rebind[Scalar[dtype]](bins[i])
                eq[e, a] = expected / sum_exp

            # Argmax (no epsilon)
            var best_q = rebind[Scalar[dtype]](eq[e, 0])
            var best_action = 0
            for a in range(1, Self.ACTIONS):
                var qv = rebind[Scalar[dtype]](eq[e, a])
                if qv > best_q:
                    best_q = qv
                    best_action = a
            acts[e] = Scalar[dtype](best_action)

        ctx.enqueue_function[rainbow_select_kernel, rainbow_select_kernel](
            raw_t,
            q_t,
            bins_t,
            actions_t,
            grid_dim=((N_ENVS + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    def do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Rainbow GPU training step."""
        comptime BATCH = Self.BATCH
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime ATOMS = Self.NUM_ATOMS
        comptime COMB = Self.COMBINED

        # Beta annealing
        var progress = Scalar[dtype](
            Float64(self.train_step_count) / Float64(max(1, self.beta_frames))
        )
        if progress > Scalar[dtype](1.0):
            progress = Scalar[dtype](1.0)
        gpu_state.buffer.anneal_beta(progress, Scalar[dtype](self.beta_start))

        # ---- Phase 1: PER sample ----
        gpu_state.buffer.sample[BATCH](
            ctx,
            gpu_state.s_obs,
            gpu_state.s_act,
            gpu_state.s_rew,
            gpu_state.s_nobs,
            gpu_state.s_done,
            gpu_state.s_idx,
            gpu_state.s_weights,
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_nobs.unsafe_ptr())
        var q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.q_raw.unsafe_ptr())
        var next_q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.next_q_raw.unsafe_ptr())
        var online_next_q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.online_next_q_raw.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Q_CS), MutAnyOrigin
        ](gpu_state.cache.unsafe_ptr())
        var grad_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.grad_raw.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.grad_input.unsafe_ptr())

        # Distributional tensors
        var q_comb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
        ](gpu_state.q_combined.unsafe_ptr())
        var next_comb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
        ](gpu_state.next_q_combined.unsafe_ptr())
        var online_next_comb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
        ](gpu_state.online_next_q_combined.unsafe_ptr())
        var expected_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.expected_q.unsafe_ptr())
        var grad_comb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
        ](gpu_state.grad_combined.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_act.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_rew.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_done.unsafe_ptr())
        var weights_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_weights.unsafe_ptr())
        var td_errors_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.td_errors.unsafe_ptr())
        var bins_t = LayoutTensor[dtype, Layout.row_major(ATOMS), MutAnyOrigin](
            gpu_state.bins_buf.unsafe_ptr()
        )

        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()

        # ---- Phase 2: Forward passes ----
        Self.QNet.forward_gpu_with_cache[BATCH](
            ctx, obs_t, q_raw_t, p_online, cache_t, gpu_state.train_ws
        )
        Self.QNet.forward_gpu[BATCH](
            ctx, next_obs_t, next_q_raw_t, p_target, gpu_state.train_ws
        )
        Self.QNet.forward_gpu[BATCH](
            ctx, next_obs_t, online_next_q_raw_t, p_online, gpu_state.train_ws
        )

        # ---- Phase 3: Dueling combine (3 kernels) ----
        @always_inline
        def dueling_combine_kernel(
            raw: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
            ],
            comb: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH:
                return
            var b = idx
            for i in range(ATOMS):
                var v_i = rebind[Scalar[dtype]](raw[b, i])
                var mean_a = Scalar[dtype](0)
                for a in range(Self.ACTIONS):
                    mean_a += rebind[Scalar[dtype]](
                        raw[b, ATOMS + a * ATOMS + i]
                    )
                mean_a /= Scalar[dtype](Self.ACTIONS)
                for a in range(Self.ACTIONS):
                    comb[b, a * ATOMS + i] = (
                        v_i
                        + rebind[Scalar[dtype]](raw[b, ATOMS + a * ATOMS + i])
                        - mean_a
                    )

        ctx.enqueue_function[dueling_combine_kernel, dueling_combine_kernel](
            q_raw_t,
            q_comb_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[dueling_combine_kernel, dueling_combine_kernel](
            next_q_raw_t,
            next_comb_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[dueling_combine_kernel, dueling_combine_kernel](
            online_next_q_raw_t,
            online_next_comb_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 4: Expected Q from online-next (for Double DQN action selection) ----
        @always_inline
        def expected_q_kernel(
            comb: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
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
                var max_val = rebind[Scalar[dtype]](comb[b, base])
                for i in range(1, ATOMS):
                    var v = rebind[Scalar[dtype]](comb[b, base + i])
                    if v > max_val:
                        max_val = v
                var sum_exp = Scalar[dtype](0)
                for i in range(ATOMS):
                    sum_exp += exp(
                        rebind[Scalar[dtype]](comb[b, base + i]) - max_val
                    )
                var expected = Scalar[dtype](0)
                for i in range(ATOMS):
                    var prob = (
                        exp(rebind[Scalar[dtype]](comb[b, base + i]) - max_val)
                        / sum_exp
                    )
                    expected += prob * rebind[Scalar[dtype]](bins[i])
                eq[b, a] = expected

        ctx.enqueue_function[expected_q_kernel, expected_q_kernel](
            online_next_comb_t,
            bins_t,
            expected_q_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 5: Bellman projection + IS-weighted CE grad + dueling reverse ----
        var gamma_n_s = Scalar[dtype](self.gamma)
        for _ in range(Self.Config.n_step - 1):
            gamma_n_s *= Scalar[dtype](self.gamma)
        var v_min_s = Scalar[dtype](Self.Config.v_min)
        var v_max_s = Scalar[dtype](Self.Config.v_max)
        var dz_s = Scalar[dtype](
            (Self.Config.v_max - Self.Config.v_min) / Float64(ATOMS - 1)
        )

        @always_inline
        def rainbow_project_grad_kernel(
            online_comb: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
            ],
            target_comb: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
            ],
            next_eq: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            bins: LayoutTensor[dtype, Layout.row_major(ATOMS), MutAnyOrigin],
            actions: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            weights: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            grad_c: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
            ],
            grad_r: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
            ],
            td_err: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            gamma_n: Scalar[dtype],
            vmin: Scalar[dtype],
            vmax: Scalar[dtype],
            dz: Scalar[dtype],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return

            # Zero gradients
            for i in range(COMB):
                grad_c[b, i] = Scalar[dtype](0)
            for i in range(Self.RAW_OUT):
                grad_r[b, i] = Scalar[dtype](0)

            # 1. Best next action (Double DQN)
            var best_a = 0
            var best_q = rebind[Scalar[dtype]](next_eq[b, 0])
            for a in range(1, Self.ACTIONS):
                var q = rebind[Scalar[dtype]](next_eq[b, a])
                if q > best_q:
                    best_q = q
                    best_a = a

            # 2. Target softmax for best action
            var t_base = best_a * ATOMS
            var t_max = rebind[Scalar[dtype]](target_comb[b, t_base])
            for i in range(1, ATOMS):
                var v = rebind[Scalar[dtype]](target_comb[b, t_base + i])
                if v > t_max:
                    t_max = v
            var t_sum_exp = Scalar[dtype](0)
            for i in range(ATOMS):
                t_sum_exp += exp(
                    rebind[Scalar[dtype]](target_comb[b, t_base + i]) - t_max
                )

            # 3. Bellman projection with γ^n
            var projected = InlineArray[Scalar[dtype], ATOMS](
                fill=Scalar[dtype](0)
            )
            var r = rebind[Scalar[dtype]](rewards[b])
            var dm = Scalar[dtype](1.0) - rebind[Scalar[dtype]](dones[b])
            for j in range(ATOMS):
                var t_prob = (
                    exp(
                        rebind[Scalar[dtype]](target_comb[b, t_base + j])
                        - t_max
                    )
                    / t_sum_exp
                )
                var tz = r + gamma_n * rebind[Scalar[dtype]](bins[j]) * dm
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
                    projected[l_idx] = projected[l_idx] + t_prob * (
                        Scalar[dtype](1.0) - u_w
                    )
                    projected[u_idx] = projected[u_idx] + t_prob * u_w

            # 4. IS-weighted CE gradient for taken action
            var action = Int(rebind[Scalar[dtype]](actions[b]))
            var p_base = action * ATOMS
            var p_max = rebind[Scalar[dtype]](online_comb[b, p_base])
            for i in range(1, ATOMS):
                var v = rebind[Scalar[dtype]](online_comb[b, p_base + i])
                if v > p_max:
                    p_max = v
            var p_sum_exp = Scalar[dtype](0)
            for i in range(ATOMS):
                p_sum_exp += exp(
                    rebind[Scalar[dtype]](online_comb[b, p_base + i]) - p_max
                )
            var log_sum_exp = p_max + log(p_sum_exp)

            # CE loss for priority
            var sample_loss = Scalar[dtype](0)
            for i in range(ATOMS):
                var log_sm = (
                    rebind[Scalar[dtype]](online_comb[b, p_base + i])
                    - log_sum_exp
                )
                sample_loss = sample_loss - projected[i] * log_sm
            td_err[b] = sample_loss

            var weight = rebind[Scalar[dtype]](weights[b])
            for i in range(ATOMS):
                var sm = (
                    exp(
                        rebind[Scalar[dtype]](online_comb[b, p_base + i])
                        - p_max
                    )
                    / p_sum_exp
                )
                grad_c[b, p_base + i] = (
                    weight * (sm - projected[i]) / Scalar[dtype](BATCH)
                )

            # 5. Dueling gradient reverse (combined → raw)
            for i in range(ATOMS):
                var sum_dq = Scalar[dtype](0)
                for a in range(Self.ACTIONS):
                    sum_dq += rebind[Scalar[dtype]](grad_c[b, a * ATOMS + i])
                grad_r[b, i] = sum_dq  # dV
                var one_over_n = Scalar[dtype](1.0) / Scalar[dtype](
                    Self.ACTIONS
                )
                for a in range(Self.ACTIONS):
                    grad_r[b, ATOMS + a * ATOMS + i] = (
                        rebind[Scalar[dtype]](grad_c[b, a * ATOMS + i])
                        - one_over_n * sum_dq
                    )

        ctx.enqueue_function[
            rainbow_project_grad_kernel, rainbow_project_grad_kernel
        ](
            q_comb_t,
            next_comb_t,
            expected_q_t,
            bins_t,
            actions_t,
            rewards_t,
            dones_t,
            weights_t,
            grad_comb_t,
            grad_raw_t,
            td_errors_t,
            gamma_n_s,
            v_min_s,
            v_max_s,
            dz_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 6: Backward + optimizer ----
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

        # ---- Phase 7: PER priority update ----
        gpu_state.buffer.update_priorities[BATCH](ctx, gpu_state.td_errors)

        # ---- GPU Diagnostic logging ----
        if (
            self.logger
            and self.diag_every > 0
            and self.train_step_count % self.diag_every == 0
        ):
            try:
                # Copy diagnostic data to host
                ctx.enqueue_copy(gpu_state.diag_comb_host, gpu_state.q_combined)
                ctx.enqueue_copy(gpu_state.diag_act_host, gpu_state.s_act)
                ctx.enqueue_copy(gpu_state.diag_rew_host, gpu_state.s_rew)
                ctx.enqueue_copy(gpu_state.diag_done_host, gpu_state.s_done)
                ctx.enqueue_copy(
                    gpu_state.diag_weights_host, gpu_state.s_weights
                )
                ctx.enqueue_copy(gpu_state.diag_td_host, gpu_state.td_errors)
                ctx.synchronize()

                var step = self.train_step_count

                # Compute expected Q from combined logits on host
                var comb_host_arr = InlineArray[Scalar[dtype], BATCH * COMB](
                    uninitialized=True
                )
                for i in range(BATCH * COMB):
                    comb_host_arr[i] = gpu_state.diag_comb_host[i]
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
                _rainbow_expected_q[BATCH, Self.ACTIONS, ATOMS, COMB](
                    comb_host_arr, bins_host, eq_host
                )

                # Q-value stats (expected Q from distributional)
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

                # Done fraction and reward stats
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

                # TD error stats (CE loss per sample, used as PER priority)
                var td_err_abs_sum: Float64 = 0.0
                var td_err_max_abs: Float64 = 0.0
                for b in range(BATCH):
                    var abs_err = Float64(gpu_state.diag_td_host[b])
                    if abs_err < 0:
                        abs_err = -abs_err
                    td_err_abs_sum += abs_err
                    if abs_err > td_err_max_abs:
                        td_err_max_abs = abs_err
                self.logger[].log_scalar(
                    "td_error_abs_mean",
                    td_err_abs_sum / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar("td_error_max", td_err_max_abs, step)

                # IS weight stats (importance sampling correction)
                var w_min = Float64(gpu_state.diag_weights_host[0])
                var w_max = Float64(gpu_state.diag_weights_host[0])
                var w_sum: Float64 = 0.0
                for b in range(BATCH):
                    var w = Float64(gpu_state.diag_weights_host[b])
                    w_sum += w
                    if w < w_min:
                        w_min = w
                    if w > w_max:
                        w_max = w
                self.logger[].log_scalar(
                    "is_weight_mean",
                    w_sum / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar("is_weight_min", w_min, step)
                self.logger[].log_scalar("is_weight_max", w_max, step)

                # PER beta
                var beta_val = Float64(self.beta_start) + (
                    1.0 - Float64(self.beta_start)
                ) * Float64(self.train_step_count) / Float64(
                    max(1, self.beta_frames)
                )
                if beta_val > 1.0:
                    beta_val = 1.0
                self.logger[].log_scalar("per_beta", beta_val, step)

                # Distribution entropy
                var entropy_sum: Float64 = 0.0
                for b in range(BATCH):
                    var action = Int(Float64(gpu_state.diag_act_host[b]))
                    var pred_base = b * COMB + action * ATOMS
                    var pred_max2 = Float64(gpu_state.diag_comb_host[pred_base])
                    for i in range(1, ATOMS):
                        var v = Float64(gpu_state.diag_comb_host[pred_base + i])
                        if v > pred_max2:
                            pred_max2 = v
                    var se2: Float64 = 0.0
                    for i in range(ATOMS):
                        se2 += exp(
                            Float64(gpu_state.diag_comb_host[pred_base + i])
                            - pred_max2
                        )
                    var h: Float64 = 0.0
                    for i in range(ATOMS):
                        var p = (
                            exp(
                                Float64(gpu_state.diag_comb_host[pred_base + i])
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
        pass  # No epsilon

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
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        target_total_steps: Int = 0,
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
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
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
            target_total_steps=target_total_steps,
        )
