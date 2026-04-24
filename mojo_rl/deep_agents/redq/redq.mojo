"""REDQ (Randomized Ensembled Double Q-learning) agent.

REDQ = SAC with:
  (1) N Q-networks instead of 2  (default N = 10)
  (2) Target Q = min over a random subset of M of N  (default M = 2)
  (3) UTD (update-to-data) ratio = 20
  (4) Policy + alpha updated only every POLICY_DELAY critic updates
  (5) Policy loss uses mean over all N online critics

The agent implements its own GPU training step (no delegation through
OffPolicyConfig strategies). CPU training is stubbed — v1 is GPU-only,
which matches the half_cheetah example.

Reference: Chen et al., "Randomized Ensembled Double Q-Learning" (ICLR 2021).
"""

from std.random import random_float64
from std.math import exp, log, sqrt
from layout import Layout, LayoutTensor
from std.memory import UnsafePointer
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import (
    Network,
    NetworkState,
    NetworkPair,
    GPUNetworkPair,
    GPUNetworkState,
)
from mojo_rl.nn.initializer import Kaiming, Xavier
from mojo_rl.nn.gpu.random import gaussian_noise
from mojo_rl.nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    read_metadata_section,
    save_checkpoint_file,
    read_checkpoint_file,
    set_metadata_value_float,
    set_metadata_value_int,
)

from mojo_rl.deep_agents.core.critic_group import CriticGroup, GPUCriticGroup
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.deep_agents.core.checkpoint_trait import Checkpointable
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.core import (
    BoxContinuousActionEnv,
    RenderableEnv,
    TrainingMetrics,
    GPUContinuousEnv,
)
from mojo_rl.deep_agents.core.kernels import (
    concat_obs_action_kernel,
    td_mse_grad_kernel,
    actor_grad_from_critic_kernel,
    sac_rsample_with_cache_kernel,
    sac_rsample_bwd_kernel,
    sac_sample_actions_counter_kernel,
    increment_rng_counter_kernel,
    alpha_adam_update_kernel,
    add_ci_grads_kernel,
    gradient_norm_kernel,
    gradient_reduce_apply_fused_kernel,
    fill_constant_kernel,
    accumulate_rewards_kernel,
    increment_steps_kernel,
    log_and_reset_completed_kernel,
    uniform_random_actions_kernel,
)
from .kernels import redq_ensemble_target_kernel
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
)

from .config import (
    REDQConfig,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
    REDQ_TARGET_REM,
)


# =============================================================================
# REDQGPUState — all GPU-side buffers for an N-critic REDQ agent
# =============================================================================


struct REDQGPUState[
    Config: REDQConfig,
    max_n_envs: Int = 64,
](Movable):
    """GPU state for REDQ. Owns actor + N-critic ensemble + replay buffer
    + all workspace buffers required by the training step.

    The per-critic forward/backward workspace (`critic_ws`) and activation
    cache (`q_cache`) are SHARED across the N critics — each critic does
    forward→backward back-to-back so the cache doesn't need to persist
    across other critics.
    """

    # Compile-time dims
    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.action_dim
    comptime ACTOR_OUT: Int = Self.Config.ActorModel.OUT_DIM
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime ACTOR_CS: Int = Self.Config.ActorModel.CACHE_SIZE
    comptime CRITIC_CS: Int = Self.Config.CriticModel.CACHE_SIZE
    comptime BATCH: Int = Self.Config.batch_size
    comptime N_ENS: Int = Self.Config.NUM_ENSEMBLE
    comptime N_MIN: Int = Self.Config.NUM_MIN
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[
        Self.Config.CriticModel, Self.Config.CriticOpt
    ]
    comptime ACTOR_WS = Self.ActorNet.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS = Self.CriticNet.WORKSPACE_SIZE_PER_SAMPLE

    # GPU-side agent scalar layout (same as SAC path)
    comptime GPU_ALPHA = 0
    comptime GPU_LOG_ALPHA = 1
    comptime GPU_ADAM_M = 2
    comptime GPU_ADAM_V = 3
    comptime GPU_ADAM_T = 4
    comptime GPU_TARGET_ENT = 5
    comptime GPU_ALPHA_LR = 6
    comptime GPU_SCALARS_SIZE = 7

    # --- Networks ---
    var actor: GPUNetworkPair[Self.Config.ActorModel, Self.Config.ActorOpt]
    var critics: GPUCriticGroup[
        Self.Config.CriticModel, Self.Config.CriticOpt, Self.N_ENS
    ]

    # --- Replay buffer ---
    var buffer: GPUReplayBuffer[
        Self.Config.buffer_capacity, Self.OBS, Self.ACTIONS
    ]

    # --- Exploration (for select_actions_gpu) ---
    var explore_raw: DeviceBuffer[dtype]  # [max_n_envs, ACTOR_OUT]
    var explore_ws: DeviceBuffer[dtype]   # [max_n_envs * ACTOR_WS]

    # --- Sampled batch ---
    var s_obs: DeviceBuffer[dtype]
    var s_act: DeviceBuffer[dtype]
    var s_rew: DeviceBuffer[dtype]
    var s_nobs: DeviceBuffer[dtype]
    var s_done: DeviceBuffer[dtype]
    var s_idx: DeviceBuffer[DType.int32]

    # --- Target-side (next state) ---
    var next_act: DeviceBuffer[dtype]     # [BS, ACTIONS]
    var next_lp: DeviceBuffer[dtype]      # [BS]
    var next_ci: DeviceBuffer[dtype]      # [BS, CRITIC_IN]
    # Stacked target-Q outputs: [N_ENS, BS] contiguous — one row per critic
    var next_q_stack: DeviceBuffer[dtype]
    var targets: DeviceBuffer[dtype]      # [BS]

    # --- Online critic update (shared across N critics, used serially) ---
    var ci: DeviceBuffer[dtype]           # [BS, CRITIC_IN]
    var q_out: DeviceBuffer[dtype]        # [BS, CRITIC_OUT]
    var q_cache: DeviceBuffer[dtype]      # [BS, CRITIC_CS]
    var q_grad: DeviceBuffer[dtype]       # [BS, CRITIC_OUT]
    var d_ci: DeviceBuffer[dtype]         # [BS, CRITIC_IN]
    var critic_ws: DeviceBuffer[dtype]    # forward/backward kernel workspace

    # --- Actor-side (policy update) ---
    var actor_out: DeviceBuffer[dtype]    # [BS, ACTOR_OUT]
    var actor_cache: DeviceBuffer[dtype]  # [BS, ACTOR_CS]
    var curr_act: DeviceBuffer[dtype]     # [BS, ACTIONS]
    var curr_lp: DeviceBuffer[dtype]      # [BS]
    var eps_cache: DeviceBuffer[dtype]    # [BS, ACTIONS]
    var new_ci: DeviceBuffer[dtype]       # [BS, CRITIC_IN]
    var dq_actor: DeviceBuffer[dtype]     # [BS, CRITIC_OUT]   (-1/(N*BS))
    var d_ci_per: DeviceBuffer[dtype]     # [BS, CRITIC_IN]    per-critic backward out
    var d_ci_sum: DeviceBuffer[dtype]     # [BS, CRITIC_IN]    accumulator across critics
    var grad_act: DeviceBuffer[dtype]     # [BS, ACTIONS]
    var actor_grad_buf: DeviceBuffer[dtype]  # [BS, ACTOR_OUT]
    var d_obs: DeviceBuffer[dtype]        # [BS, OBS]
    var actor_ws: DeviceBuffer[dtype]

    # --- Alpha / scalars / RNG ---
    var gpu_scalars: DeviceBuffer[dtype]
    var rng_counter: DeviceBuffer[DType.uint32]
    var explore_counter: DeviceBuffer[DType.uint32]
    # Subset indices uploaded from host each critic step (MODE=0)
    var subset_idxs: DeviceBuffer[DType.uint32]

    # --- Gradient clipping scratch ---
    var grad_clip_ps: DeviceBuffer[dtype]

    # --- Diagnostic host buffers (mirror SAC's _gpu_train_diagnostics) ---
    var diag_q_host: HostBuffer[dtype]      # [BS]   last critic's Q
    var diag_tgt_host: HostBuffer[dtype]    # [BS]   TD target
    var diag_rew_host: HostBuffer[dtype]    # [BS]   sampled reward
    var diag_done_host: HostBuffer[dtype]   # [BS]   sampled done
    var diag_act_host: HostBuffer[dtype]    # [BS*ACTIONS]  sampled action
    var diag_nq_host: HostBuffer[dtype]     # [BS]   first target critic's Q
    var diag_lp_host: HostBuffer[dtype]     # [BS]   last policy-step log_pi
    var diag_scalars_host: HostBuffer[dtype]  # [GPU_SCALARS_SIZE]

    def __init__(out self, ctx: DeviceContext) raises:
        self.actor = GPUNetworkPair[
            Self.Config.ActorModel, Self.Config.ActorOpt
        ](ctx)
        self.critics = GPUCriticGroup[
            Self.Config.CriticModel, Self.Config.CriticOpt, Self.N_ENS
        ](ctx)
        self.buffer = GPUReplayBuffer[
            Self.Config.buffer_capacity, Self.OBS, Self.ACTIONS
        ](ctx)

        comptime BS = Self.BATCH
        comptime MNE = Self.max_n_envs

        # Exploration
        self.explore_raw = ctx.enqueue_create_buffer[dtype](MNE * Self.ACTOR_OUT)
        self.explore_ws = ctx.enqueue_create_buffer[dtype](
            max(1, MNE * Self.ACTOR_WS)
        )

        # Sample batch
        self.s_obs = ctx.enqueue_create_buffer[dtype](BS * Self.OBS)
        self.s_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.s_rew = ctx.enqueue_create_buffer[dtype](BS)
        self.s_nobs = ctx.enqueue_create_buffer[dtype](BS * Self.OBS)
        self.s_done = ctx.enqueue_create_buffer[dtype](BS)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](BS)

        # Target side
        self.next_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.next_lp = ctx.enqueue_create_buffer[dtype](BS)
        self.next_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.next_q_stack = ctx.enqueue_create_buffer[dtype](
            Self.N_ENS * BS * Self.CRITIC_OUT
        )
        self.targets = ctx.enqueue_create_buffer[dtype](BS)

        # Online critic update
        self.ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.q_out = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.q_cache = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_CS)
        self.q_grad = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.d_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.critic_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.CRITIC_WS)
        )

        # Actor side
        self.actor_out = ctx.enqueue_create_buffer[dtype](BS * Self.ACTOR_OUT)
        self.actor_cache = ctx.enqueue_create_buffer[dtype](BS * Self.ACTOR_CS)
        self.curr_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.curr_lp = ctx.enqueue_create_buffer[dtype](BS)
        self.eps_cache = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.new_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.dq_actor = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.d_ci_per = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.d_ci_sum = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.grad_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.actor_grad_buf = ctx.enqueue_create_buffer[dtype](
            BS * Self.ACTOR_OUT
        )
        self.d_obs = ctx.enqueue_create_buffer[dtype](BS * Self.OBS)
        self.actor_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.ACTOR_WS)
        )

        # Alpha / RNG
        self.gpu_scalars = ctx.enqueue_create_buffer[dtype](
            Self.GPU_SCALARS_SIZE
        )
        self.gpu_scalars.enqueue_fill(Scalar[dtype](0.0))
        self.rng_counter = ctx.enqueue_create_buffer[DType.uint32](1)
        self.rng_counter.enqueue_fill(UInt32(0))
        self.explore_counter = ctx.enqueue_create_buffer[DType.uint32](1)
        self.explore_counter.enqueue_fill(UInt32(0))
        self.subset_idxs = ctx.enqueue_create_buffer[DType.uint32](
            max(1, Self.N_MIN)
        )

        # Gradient-clipping partial sums (sized for the larger of actor / critic)
        comptime A_PS = Self.Config.ActorModel.PARAM_SIZE
        comptime C_PS = Self.Config.CriticModel.PARAM_SIZE
        comptime MAX_PS = A_PS if A_PS > C_PS else C_PS
        comptime MAX_BLOCKS = (MAX_PS + TPB - 1) // TPB
        self.grad_clip_ps = ctx.enqueue_create_buffer[dtype](MAX_BLOCKS)

        # Diagnostic host buffers (pre-allocated, reused each diag call)
        self.diag_q_host = ctx.enqueue_create_host_buffer[dtype](BS)
        self.diag_tgt_host = ctx.enqueue_create_host_buffer[dtype](BS)
        self.diag_rew_host = ctx.enqueue_create_host_buffer[dtype](BS)
        self.diag_done_host = ctx.enqueue_create_host_buffer[dtype](BS)
        self.diag_act_host = ctx.enqueue_create_host_buffer[dtype](
            BS * Self.ACTIONS
        )
        self.diag_nq_host = ctx.enqueue_create_host_buffer[dtype](BS)
        self.diag_lp_host = ctx.enqueue_create_host_buffer[dtype](BS)
        self.diag_scalars_host = ctx.enqueue_create_host_buffer[dtype](
            Self.GPU_SCALARS_SIZE
        )

    def __init__(out self, *, deinit take: Self):
        self.actor = take.actor^
        self.critics = take.critics^
        self.buffer = take.buffer^
        self.explore_raw = take.explore_raw^
        self.explore_ws = take.explore_ws^
        self.s_obs = take.s_obs^
        self.s_act = take.s_act^
        self.s_rew = take.s_rew^
        self.s_nobs = take.s_nobs^
        self.s_done = take.s_done^
        self.s_idx = take.s_idx^
        self.next_act = take.next_act^
        self.next_lp = take.next_lp^
        self.next_ci = take.next_ci^
        self.next_q_stack = take.next_q_stack^
        self.targets = take.targets^
        self.ci = take.ci^
        self.q_out = take.q_out^
        self.q_cache = take.q_cache^
        self.q_grad = take.q_grad^
        self.d_ci = take.d_ci^
        self.critic_ws = take.critic_ws^
        self.actor_out = take.actor_out^
        self.actor_cache = take.actor_cache^
        self.curr_act = take.curr_act^
        self.curr_lp = take.curr_lp^
        self.eps_cache = take.eps_cache^
        self.new_ci = take.new_ci^
        self.dq_actor = take.dq_actor^
        self.d_ci_per = take.d_ci_per^
        self.d_ci_sum = take.d_ci_sum^
        self.grad_act = take.grad_act^
        self.actor_grad_buf = take.actor_grad_buf^
        self.d_obs = take.d_obs^
        self.actor_ws = take.actor_ws^
        self.gpu_scalars = take.gpu_scalars^
        self.rng_counter = take.rng_counter^
        self.explore_counter = take.explore_counter^
        self.subset_idxs = take.subset_idxs^
        self.grad_clip_ps = take.grad_clip_ps^
        self.diag_q_host = take.diag_q_host^
        self.diag_tgt_host = take.diag_tgt_host^
        self.diag_rew_host = take.diag_rew_host^
        self.diag_done_host = take.diag_done_host^
        self.diag_act_host = take.diag_act_host^
        self.diag_nq_host = take.diag_nq_host^
        self.diag_lp_host = take.diag_lp_host^
        self.diag_scalars_host = take.diag_scalars_host^

    # -------------------------------------------------------------------------
    # Tensor view helpers
    # -------------------------------------------------------------------------

    def s_obs_t(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](self.s_obs.unsafe_ptr())

    def s_nobs_t(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](self.s_nobs.unsafe_ptr())

    def s_act_t(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](self.s_act.unsafe_ptr())

    def s_rew_t(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.BATCH), MutAnyOrigin](
            self.s_rew.unsafe_ptr()
        )

    def s_done_t(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.BATCH), MutAnyOrigin](
            self.s_done.unsafe_ptr()
        )


# =============================================================================
# REDQAgent — config-driven ensemble SAC agent
# =============================================================================


struct REDQAgent[
    Config: REDQConfig,
    max_n_envs: Int = 64,
](Movable & Checkpointable):
    """REDQ agent. Owns an in-memory CPU CriticGroup for init / save-load
    and a GPU-side REDQGPUState for training.

    Training is driven by `train_gpu()` (defined below), which calls
    `select_actions_gpu`, `gpu_store`, `do_gpu_train_step`, and
    `soft_update_all` as appropriate.
    """

    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.action_dim
    comptime ACTOR_OUT: Int = Self.Config.ActorModel.OUT_DIM
    comptime BATCH: Int = Self.Config.batch_size
    comptime N_ENS: Int = Self.Config.NUM_ENSEMBLE
    comptime N_MIN: Int = Self.Config.NUM_MIN
    comptime UTD: Int = Self.Config.UTD_RATIO
    comptime POL_DELAY: Int = Self.Config.POLICY_DELAY
    comptime Q_MODE: Int = Self.Config.Q_TARGET_MODE
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime CRITIC_CS: Int = Self.Config.CriticModel.CACHE_SIZE
    comptime ACTOR_CS: Int = Self.Config.ActorModel.CACHE_SIZE
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[
        Self.Config.CriticModel, Self.Config.CriticOpt
    ]

    comptime GPUStateType = REDQGPUState[Self.Config, Self.max_n_envs]

    # CPU-side networks (used only for init / upload; CPU training is not
    # implemented in v1).
    var cpu_actor: NetworkPair[Self.Config.ActorModel, Self.Config.ActorOpt]
    var cpu_critics: CriticGroup[
        Self.Config.CriticModel, Self.Config.CriticOpt, Self.N_ENS
    ]

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var action_scale: Float64

    # Alpha state (SAC entropy temperature autotuning)
    var alpha: Float64
    var log_alpha: Float64
    var target_entropy: Float64
    var auto_alpha: Bool
    var alpha_lr: Float64
    var alpha_adam_m: Float64
    var alpha_adam_v: Float64
    var alpha_adam_t: Int

    # Gradient clipping (applied per critic + actor)
    var max_grad_norm: Float64

    # Training bookkeeping
    var total_steps: Int
    var critic_update_count: Int  # inner UTD-loop counter (total # critic steps)

    # Checkpointing
    var checkpoint_every: Int
    var checkpoint_path: String

    # Diagnostic logging frequency (in critic update steps; 0 = disabled).
    # Mirrors offpolicy_agent's diag_every — drives detailed metrics
    # (critic_loss, mean_q, etc.) emitted on every Nth gradient update,
    # independent of `print_every` (which is in env-transition units).
    var diag_every: Int

    def __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        action_scale: Float64 = 1.0,
        auto_alpha: Bool = True,
        alpha: Float64 = 0.2,
        alpha_lr: Float64 = 0.0003,
        target_entropy: Float64 = 0.0,
        max_grad_norm: Float64 = 0.0,  # REDQ paper does not clip; default off
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        diag_every: Int = 0,
    ):
        self.cpu_actor = NetworkPair[
            Self.Config.ActorModel, Self.Config.ActorOpt
        ]()
        self.cpu_actor.initialize[Xavier[]]()
        self.cpu_critics = CriticGroup[
            Self.Config.CriticModel, Self.Config.CriticOpt, Self.N_ENS
        ]()
        self.cpu_critics.initialize[Xavier[]]()

        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale

        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.log_alpha = log(alpha)
        self.target_entropy = (
            target_entropy if target_entropy != 0.0 else -Float64(Self.ACTIONS)
        )
        self.alpha_lr = alpha_lr
        self.alpha_adam_m = 0.0
        self.alpha_adam_v = 0.0
        self.alpha_adam_t = 0

        self.max_grad_norm = max_grad_norm
        self.total_steps = 0
        self.critic_update_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.diag_every = diag_every

    def __init__(out self, *, deinit take: Self):
        self.cpu_actor = take.cpu_actor^
        self.cpu_critics = take.cpu_critics^
        self.gamma = take.gamma
        self.tau = take.tau
        self.action_scale = take.action_scale
        self.alpha = take.alpha
        self.log_alpha = take.log_alpha
        self.target_entropy = take.target_entropy
        self.auto_alpha = take.auto_alpha
        self.alpha_lr = take.alpha_lr
        self.alpha_adam_m = take.alpha_adam_m
        self.alpha_adam_v = take.alpha_adam_v
        self.alpha_adam_t = take.alpha_adam_t
        self.max_grad_norm = take.max_grad_norm
        self.total_steps = take.total_steps
        self.critic_update_count = take.critic_update_count
        self.checkpoint_every = take.checkpoint_every
        self.checkpoint_path = take.checkpoint_path^
        self.diag_every = take.diag_every

    # -------------------------------------------------------------------------
    # Checkpointable — save/load CPU networks + scalar training state
    # -------------------------------------------------------------------------

    def save_checkpoint(self, path: String) raises -> None:
        """Save CPU actor + N-critic ensemble + agent scalars to a single file.

        Network weights are taken from `cpu_actor` / `cpu_critics`, so call
        `download_from_gpu` first if the GPU has fresher weights. The replay
        buffer is NOT serialized.
        """
        var content = write_checkpoint_header(
            "redq",
            Self.Config.ActorModel.PARAM_SIZE
            + Self.Config.CriticModel.PARAM_SIZE * Self.N_ENS,
            0,
        )
        content += self.cpu_actor.write_sections("actor_")
        # CriticGroup writes one section per critic with prefix
        # "<prefix>critic<i>_" — here that's "ensemble_critic0_" .. _critic9_.
        content += self.cpu_critics.write_sections("ensemble_")
        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("action_scale=" + String(self.action_scale))
        metadata.append("alpha=" + String(self.alpha))
        metadata.append("log_alpha=" + String(self.log_alpha))
        metadata.append("alpha_lr=" + String(self.alpha_lr))
        metadata.append("alpha_adam_m=" + String(self.alpha_adam_m))
        metadata.append("alpha_adam_v=" + String(self.alpha_adam_v))
        metadata.append("alpha_adam_t=" + String(self.alpha_adam_t))
        metadata.append("target_entropy=" + String(self.target_entropy))
        metadata.append("max_grad_norm=" + String(self.max_grad_norm))
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append(
            "critic_update_count=" + String(self.critic_update_count)
        )
        # Persist ensemble shape so a misconfigured reload errors via the
        # NUM_ENSEMBLE check below rather than reading garbage.
        metadata.append("num_ensemble=" + String(Self.N_ENS))
        metadata.append("num_min=" + String(Self.N_MIN))
        content += write_metadata_section(metadata)
        save_checkpoint_file(path, content)

    def load_checkpoint(mut self, path: String) raises -> None:
        """Restore CPU networks + scalar state. After this, call
        `upload_to_gpu` before resuming training so the GPU sees the
        reloaded weights and alpha state.

        Note: ensemble size N is fixed at compile time. A checkpoint saved
        with a different `NUM_ENSEMBLE` will read garbage into the extra
        critic sections — caller is responsible for matching configs.
        """
        var content = read_checkpoint_file(path)
        self.cpu_actor.read_sections(content, "actor_")
        self.cpu_critics.read_sections(content, "ensemble_")
        var metadata = read_metadata_section(content)
        set_metadata_value_float(metadata, "gamma", self.gamma)
        set_metadata_value_float(metadata, "tau", self.tau)
        set_metadata_value_float(metadata, "action_scale", self.action_scale)
        set_metadata_value_float(metadata, "alpha", self.alpha)
        set_metadata_value_float(metadata, "log_alpha", self.log_alpha)
        set_metadata_value_float(metadata, "alpha_lr", self.alpha_lr)
        set_metadata_value_float(metadata, "alpha_adam_m", self.alpha_adam_m)
        set_metadata_value_float(metadata, "alpha_adam_v", self.alpha_adam_v)
        set_metadata_value_int(metadata, "alpha_adam_t", self.alpha_adam_t)
        set_metadata_value_float(
            metadata, "target_entropy", self.target_entropy
        )
        set_metadata_value_float(
            metadata, "max_grad_norm", self.max_grad_norm
        )
        set_metadata_value_int(metadata, "total_steps", self.total_steps)
        set_metadata_value_int(
            metadata, "critic_update_count", self.critic_update_count
        )

    def make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(ctx)

    # -------------------------------------------------------------------------
    # CPU greedy inference (for evaluation)
    # -------------------------------------------------------------------------

    def select_greedy_action(
        self,
        obs: List[Float64],
    ) -> List[Float64]:
        """Deterministic action selection using the mean of the tanh-Gaussian
        actor (no sampling). Uses `cpu_actor` — call `download_from_gpu`
        first if GPU weights are fresher than CPU."""
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())

        var out_arr = InlineArray[Scalar[dtype], Self.Config.ActorModel.OUT_DIM](
            uninitialized=True
        )
        var out_t = LayoutTensor[
            dtype,
            Layout.row_major(1, Self.Config.ActorModel.OUT_DIM),
            MutAnyOrigin,
        ](out_arr.unsafe_ptr())
        var p = self.cpu_actor.online.params_view()
        Self.ActorNet.forward[1](obs_t, out_t, p)

        var result = List[Float64](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            var mean = Float64(out_arr[i])
            # Apply tanh squashing (matches training's tanh-Gaussian actor).
            var a = (exp(2.0 * mean) - 1.0) / (exp(2.0 * mean) + 1.0)
            a *= self.action_scale
            result.append(a)
        return result^

    # -------------------------------------------------------------------------
    # Evaluation (mirrors DeepSACAgent.evaluate)
    # -------------------------------------------------------------------------

    def evaluate[
        E: BoxContinuousActionEnv & RenderableEnv,
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent deterministically (mean action, no sampling).

        Uses `cpu_actor`; call `download_from_gpu` first if GPU weights are
        fresher than CPU. Returns average reward across `num_episodes`.
        """
        var quit_requested = False
        if render:
            _ = env.init_renderer()

        var total_reward: Float64 = 0.0
        var completed: Int = 0
        for ep in range(num_episodes):
            if quit_requested:
                break
            var obs_raw = env.reset_obs_list()
            var obs = List[Float64]()
            for i in range(len(obs_raw)):
                obs.append(Float64(obs_raw[i]))

            var episode_reward: Float64 = 0.0
            for _ in range(max_steps_per_episode):
                var action = self.select_greedy_action(obs)
                var result = env.step_continuous_vec(action)
                var next_obs = List[Float64]()
                for i in range(len(result[0])):
                    next_obs.append(Float64(result[0][i]))
                episode_reward += Float64(result[1])
                obs = next_obs^

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                if result[2]:
                    break

            total_reward += episode_reward
            completed += 1
            if verbose:
                print(
                    "  Episode "
                    + String(ep + 1)
                    + " | Reward: "
                    + String(episode_reward)[byte=:10]
                )

        if render:
            env.close_renderer()

        if completed == 0:
            return 0.0
        return total_reward / Float64(completed)

    # -------------------------------------------------------------------------
    # CPU ↔ GPU sync
    # -------------------------------------------------------------------------

    def upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.actor.upload_from(self.cpu_actor, ctx)
        gpu_state.critics.upload_from(self.cpu_critics, ctx)

        var scalars_host = ctx.enqueue_create_host_buffer[dtype](
            Self.GPUStateType.GPU_SCALARS_SIZE
        )
        scalars_host[Self.GPUStateType.GPU_ALPHA] = Scalar[dtype](self.alpha)
        scalars_host[Self.GPUStateType.GPU_LOG_ALPHA] = Scalar[dtype](
            self.log_alpha
        )
        scalars_host[Self.GPUStateType.GPU_ADAM_M] = Scalar[dtype](
            self.alpha_adam_m
        )
        scalars_host[Self.GPUStateType.GPU_ADAM_V] = Scalar[dtype](
            self.alpha_adam_v
        )
        scalars_host[Self.GPUStateType.GPU_ADAM_T] = Scalar[dtype](
            Float64(self.alpha_adam_t)
        )
        scalars_host[Self.GPUStateType.GPU_TARGET_ENT] = Scalar[dtype](
            self.target_entropy
        )
        scalars_host[Self.GPUStateType.GPU_ALPHA_LR] = Scalar[dtype](
            self.alpha_lr
        )
        ctx.enqueue_copy(gpu_state.gpu_scalars, scalars_host)

    def download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.actor.download_to(self.cpu_actor, ctx)
        gpu_state.critics.download_to(self.cpu_critics, ctx)

        var scalars_host = ctx.enqueue_create_host_buffer[dtype](
            Self.GPUStateType.GPU_SCALARS_SIZE
        )
        ctx.enqueue_copy(scalars_host, gpu_state.gpu_scalars)
        ctx.synchronize()
        self.alpha = Float64(scalars_host[Self.GPUStateType.GPU_ALPHA])
        self.log_alpha = Float64(scalars_host[Self.GPUStateType.GPU_LOG_ALPHA])

    # -------------------------------------------------------------------------
    # Exploration (stochastic SAC actor)
    # -------------------------------------------------------------------------

    def select_actions_gpu[
        N_ENVS: Int
    ](
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Stochastic action selection — forward actor, sample tanh-Gaussian.
        """
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.explore_raw.unsafe_ptr())
        var p = gpu_state.actor.online.params_view()

        Self.ActorNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, gpu_state.explore_ws
        )

        var act_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var explore_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.explore_counter.unsafe_ptr())

        comptime BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime sac_counter_k = sac_sample_actions_counter_kernel[
            dtype, N_ENVS, Self.ACTIONS, Self.ACTOR_OUT
        ]
        ctx.enqueue_function[sac_counter_k, sac_counter_k](
            act_t,
            raw_t,
            Scalar[dtype](self.action_scale),
            Scalar[dtype](-5.0),
            Scalar[dtype](2.0),
            explore_t,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    def sync_explore_counter(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        gpu_state.explore_counter.enqueue_fill(
            UInt32(self.total_steps * Self.ACTIONS)
        )

    # -------------------------------------------------------------------------
    # Store transitions (delegate to GPU replay buffer)
    # -------------------------------------------------------------------------

    def gpu_store[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        prev_obs_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        rewards_buf: DeviceBuffer[dtype],
        obs_buf: DeviceBuffer[dtype],
        dones_buf: DeviceBuffer[dtype],
    ) raises -> None:
        gpu_state.buffer.store[N_ENVS](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, dones_buf
        )
        self.total_steps += N_ENVS

    def gpu_buffer_is_ready(self, gpu_state: Self.GPUStateType) -> Bool:
        return gpu_state.buffer.is_ready[Self.BATCH]()

    # -------------------------------------------------------------------------
    # Subset index sampling (host-side) + upload
    # -------------------------------------------------------------------------

    def _sample_subset_and_upload(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Sample M distinct indices from [0, N_ENS) on the host, upload to
        `gpu_state.subset_idxs`. Called once per critic update.
        """
        var idx_host = ctx.enqueue_create_host_buffer[DType.uint32](
            max(1, Self.N_MIN)
        )
        # Reservoir-style sampling without replacement; N_MIN <= N_ENS.
        var picks = List[Int](capacity=Self.N_ENS)
        for i in range(Self.N_ENS):
            picks.append(i)
        # Fisher-Yates partial shuffle — first N_MIN positions are the sample.
        for i in range(Self.N_MIN):
            var j = i + Int(random_float64() * Float64(Self.N_ENS - i))
            if j >= Self.N_ENS:
                j = Self.N_ENS - 1
            var tmp = picks[i]
            picks[i] = picks[j]
            picks[j] = tmp
            idx_host[i] = UInt32(picks[i])
        ctx.enqueue_copy(gpu_state.subset_idxs, idx_host)

    # -------------------------------------------------------------------------
    # Soft update — all N target critics
    # -------------------------------------------------------------------------

    def soft_update_all(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        gpu_state.critics.soft_update_all(self.tau, ctx)

    # -------------------------------------------------------------------------
    # Diagnostics — fine-grained per-train-step logging.
    # Mirrors offpolicy_agent._gpu_train_diagnostics: copies GPU buffers to
    # pre-allocated host buffers on each diag_every boundary, computes mean
    # critic_loss / Q / target / reward / next_q / |action| / log_pi / alpha
    # on the host, and emits one log_scalar per metric. The step axis is
    # `critic_update_count` (gradient updates), not env transitions, so the
    # dashboard plots stay on the same axis as SAC/MBPO.
    # -------------------------------------------------------------------------

    def maybe_log_diagnostics[
        L: Logger
    ](
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        logger: UnsafePointer[L, MutAnyOrigin],
    ) raises -> None:
        """Emit fine-grained diagnostics every `diag_every` critic updates.

        No-op unless `diag_every > 0`, the logger pointer is non-null, and
        the current `critic_update_count` is on a `diag_every` boundary.
        Safe to call after every `do_gpu_train_step`.

        Notes:
          - The reported `critic_loss` / `mean_q` come from whichever online
            critic ran LAST in the critic-update phase (i.e. critic N-1).
            All N critics share the same TD target so any one is
            representative; this matches SAC's "Q1 only" reporting.
          - `mean_next_q` is taken from target critic 0 (first row of the
            stacked next_q buffer), again as a representative slice.
          - `mean_log_pi` reflects the most recent policy-update step. On
            iterations between policy updates, it stays at the previously
            written value — consistent with SAC, where log_pi is only
            recomputed during actor updates.
        """
        if not logger:
            return
        if self.diag_every <= 0:
            return
        if self.critic_update_count <= 0:
            return
        if self.critic_update_count % self.diag_every != 0:
            return

        comptime BS = Self.BATCH

        # GPU → host: q_out (last online critic), targets, sampled batch,
        # first target critic's Q (offset 0 of the stacked buffer), curr_lp,
        # and the alpha scalar bank.
        ctx.enqueue_copy(gpu_state.diag_q_host, gpu_state.q_out)
        ctx.enqueue_copy(gpu_state.diag_tgt_host, gpu_state.targets)
        ctx.enqueue_copy(gpu_state.diag_rew_host, gpu_state.s_rew)
        ctx.enqueue_copy(gpu_state.diag_done_host, gpu_state.s_done)
        ctx.enqueue_copy(gpu_state.diag_act_host, gpu_state.s_act)
        ctx.enqueue_copy(gpu_state.diag_lp_host, gpu_state.curr_lp)
        ctx.enqueue_copy(gpu_state.diag_scalars_host, gpu_state.gpu_scalars)

        # First target critic (slice 0 of next_q_stack [N_ENS, BS]).
        var nq0_subbuf = DeviceBuffer[dtype](
            ctx,
            gpu_state.next_q_stack.unsafe_ptr(),
            BS,
            owning=False,
        )
        ctx.enqueue_copy(gpu_state.diag_nq_host, nq0_subbuf)
        ctx.synchronize()

        var mean_q: Float64 = 0.0
        var mean_tgt: Float64 = 0.0
        var mean_rew: Float64 = 0.0
        var mean_done: Float64 = 0.0
        var mean_nq: Float64 = 0.0
        var mean_lp: Float64 = 0.0
        var critic_loss: Float64 = 0.0
        for b in range(BS):
            var q_val = Float64(gpu_state.diag_q_host[b])
            var tgt_val = Float64(gpu_state.diag_tgt_host[b])
            mean_q += q_val
            mean_tgt += tgt_val
            mean_rew += Float64(gpu_state.diag_rew_host[b])
            mean_done += Float64(gpu_state.diag_done_host[b])
            mean_nq += Float64(gpu_state.diag_nq_host[b])
            mean_lp += Float64(gpu_state.diag_lp_host[b])
            critic_loss += (q_val - tgt_val) * (q_val - tgt_val)

        var mean_abs_act: Float64 = 0.0
        for i in range(BS * Self.ACTIONS):
            var a = Float64(gpu_state.diag_act_host[i])
            mean_abs_act += a if a >= 0.0 else -a

        mean_q /= Float64(BS)
        mean_tgt /= Float64(BS)
        mean_rew /= Float64(BS)
        mean_done /= Float64(BS)
        mean_nq /= Float64(BS)
        mean_lp /= Float64(BS)
        critic_loss /= Float64(BS)
        mean_abs_act /= Float64(BS * Self.ACTIONS)

        var alpha = Float64(
            gpu_state.diag_scalars_host[Self.GPUStateType.GPU_ALPHA]
        )
        var log_alpha = Float64(
            gpu_state.diag_scalars_host[Self.GPUStateType.GPU_LOG_ALPHA]
        )

        var step = self.critic_update_count
        logger[].log_scalar("critic_loss", critic_loss, step)
        logger[].log_scalar("mean_q", mean_q, step)
        logger[].log_scalar("mean_target", mean_tgt, step)
        logger[].log_scalar("mean_reward", mean_rew, step)
        logger[].log_scalar("mean_next_q", mean_nq, step)
        logger[].log_scalar("mean_done", mean_done, step)
        logger[].log_scalar("mean_abs_action", mean_abs_act, step)
        logger[].log_scalar("mean_log_pi", mean_lp, step)
        logger[].log_scalar("alpha", alpha, step)
        logger[].log_scalar("log_alpha", log_alpha, step)

    # -------------------------------------------------------------------------
    # Single REDQ training iteration (one of UTD_RATIO inner steps)
    # -------------------------------------------------------------------------

    def do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """One inner REDQ step: critic update (always), policy+alpha update
        (every POLICY_DELAY steps), soft update of all target critics.

        Called `UTD_RATIO` times per env step by the outer training loop.
        """
        if not self.gpu_buffer_is_ready(gpu_state):
            return

        self.critic_update_count += 1
        var is_policy_step = (
            self.critic_update_count % Self.POL_DELAY == 0
        )

        # Sample a fresh subset of target-critic indices for min target.
        self._sample_subset_and_upload(ctx, gpu_state)

        # --- Increment RNG counter for this iteration ---
        comptime incr_k = increment_rng_counter_kernel
        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.rng_counter.unsafe_ptr())
        ctx.enqueue_function[incr_k, incr_k](
            rng_t, grid_dim=(1,), block_dim=(1,)
        )

        # --- Sample batch from replay buffer ---
        comptime BS = Self.BATCH
        gpu_state.buffer.sample[BS](
            ctx,
            rng_counter=gpu_state.rng_counter,
            sampled_obs=gpu_state.s_obs,
            sampled_actions=gpu_state.s_act,
            sampled_rewards=gpu_state.s_rew,
            sampled_next_obs=gpu_state.s_nobs,
            sampled_dones=gpu_state.s_done,
            indices=gpu_state.s_idx,
        )

        self._phase_critic_update(ctx, gpu_state)

        if is_policy_step:
            self._phase_actor_alpha_update(ctx, gpu_state)

        # Soft update all N target critics — paper-faithful (every step).
        self.soft_update_all(ctx, gpu_state)

    # -------------------------------------------------------------------------
    # Phase: critic update (N critics, subset-min target)
    # -------------------------------------------------------------------------

    def _phase_critic_update(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        comptime BS = Self.BATCH
        comptime ELEM_BLOCKS = (BS * Self.CRITIC_IN + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (BS + TPB - 1) // TPB
        comptime concat_k = concat_obs_action_kernel[
            dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN
        ]
        comptime mse_grad_k = td_mse_grad_kernel[dtype, BS, Self.CRITIC_OUT]

        var obs_t = gpu_state.s_obs_t()
        var nobs_t = gpu_state.s_nobs_t()
        var act_t = gpu_state.s_act_t()
        var rew_t = gpu_state.s_rew_t()
        var done_t = gpu_state.s_done_t()

        # --- 1. Reparameterized next action from ONLINE actor ---
        # Increment RNG counter before sampling.
        comptime incr_k = increment_rng_counter_kernel
        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.rng_counter.unsafe_ptr())
        ctx.enqueue_function[incr_k, incr_k](
            rng_t, grid_dim=(1,), block_dim=(1,)
        )

        var p_actor = gpu_state.actor.online.params_view()
        var nact_raw_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.actor_out.unsafe_ptr())
        Self.ActorNet.forward_gpu[BS](
            ctx, nobs_t, nact_raw_t, p_actor, gpu_state.actor_ws
        )
        var nact_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.next_act.unsafe_ptr())
        var nlp_t = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.next_lp.unsafe_ptr())
        var neps_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.eps_cache.unsafe_ptr())  # reused — not needed for target
        var ls_min = Scalar[dtype](-5.0)
        var ls_max = Scalar[dtype](2.0)
        var ascale = Scalar[dtype](self.action_scale)

        @parameter
        @always_inline
        def rsample_next(
            acts: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ],
            lp: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            eps: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
            ],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
            asc: Scalar[dtype],
            rng: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
        ):
            sac_rsample_with_cache_kernel[dtype, BS, Self.ACTIONS](
                acts, lp, eps, ao, lsmin, lsmax, asc, rng
            )

        ctx.enqueue_function[rsample_next, rsample_next](
            nact_t,
            nlp_t,
            neps_t,
            nact_raw_t,
            ls_min,
            ls_max,
            ascale,
            rng_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # --- 2. Concat [nobs, nact] → next_ci ---
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.next_ci.unsafe_ptr())
        ctx.enqueue_function[concat_k, concat_k](
            next_ci_t,
            nobs_t,
            nact_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        # --- 3. Forward all N target critics on next_ci — stacked output ---
        for n in range(Self.N_ENS):
            var p_t = gpu_state.critics.target_params_view(n)
            # Per-critic slice of the stacked buffer.
            var slice_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
            ](
                gpu_state.next_q_stack.unsafe_ptr() + n * BS * Self.CRITIC_OUT
            )
            Self.CriticNet.forward_gpu[BS](
                ctx, next_ci_t, slice_t, p_t, gpu_state.critic_ws
            )

        # --- 4. Compute TD targets from stacked Q + subset indices ---
        var nq_stack_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENS, BS), MutAnyOrigin
        ](gpu_state.next_q_stack.unsafe_ptr())
        var subset_t = LayoutTensor[
            DType.uint32, Layout.row_major(Self.N_MIN), MutAnyOrigin
        ](gpu_state.subset_idxs.unsafe_ptr())
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())
        var alpha_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.gpu_scalars.unsafe_ptr())
        var gamma_s = Scalar[dtype](self.gamma)

        comptime target_k = redq_ensemble_target_kernel[
            dtype, BS, Self.N_ENS, Self.N_MIN, Self.Q_MODE
        ]
        ctx.enqueue_function[target_k, target_k](
            targets_t,
            rew_t,
            nq_stack_t,
            done_t,
            nlp_t,
            subset_t,
            gamma_s,
            alpha_t,
            rng_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # --- 5. Concat [obs, act] → ci (input for ALL online critics) ---
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.ci.unsafe_ptr())
        ctx.enqueue_function[concat_k, concat_k](
            ci_t,
            obs_t,
            act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        # --- 6. For each online critic: forward → MSE grad → backward → step ---
        var q_out_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.q_out.unsafe_ptr())
        var q_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
        ](gpu_state.q_cache.unsafe_ptr())
        var q_grad_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.q_grad.unsafe_ptr())
        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci.unsafe_ptr())

        for n in range(Self.N_ENS):
            var p_o = gpu_state.critics.online_params_view(n)
            var g_o = gpu_state.critics.online_grads_view(n)
            Self.CriticNet.forward_gpu_with_cache[BS](
                ctx,
                ci_t,
                q_out_t,
                p_o,
                q_cache_t,
                gpu_state.critic_ws,
            )
            ctx.enqueue_function[mse_grad_k, mse_grad_k](
                q_grad_t,
                q_out_t,
                targets_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            gpu_state.critics.pairs[n].online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BS](
                ctx,
                q_grad_t,
                d_ci_t,
                p_o,
                q_cache_t,
                g_o,
                gpu_state.critic_ws,
            )
            if self.max_grad_norm > 0.0:
                self._clip_critic_grads(ctx, gpu_state, g_o)
            gpu_state.critics.pairs[n].online.optimizer_step(ctx)

    # -------------------------------------------------------------------------
    # Phase: actor + alpha update (mean-Q over all N online critics)
    # -------------------------------------------------------------------------

    def _phase_actor_alpha_update(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        comptime BS = Self.BATCH
        comptime ELEM_BLOCKS = (BS * Self.CRITIC_IN + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (BS + TPB - 1) // TPB
        comptime ACT_BLOCKS = (BS * Self.ACTIONS + TPB - 1) // TPB
        comptime incr_k = increment_rng_counter_kernel
        comptime concat_k = concat_obs_action_kernel[
            dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN
        ]

        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.rng_counter.unsafe_ptr())

        var obs_t = gpu_state.s_obs_t()
        var p_actor = gpu_state.actor.online.params_view()
        var a_grads = gpu_state.actor.online.grads_view()

        # --- 1. Increment RNG counter before rsample ---
        ctx.enqueue_function[incr_k, incr_k](
            rng_t, grid_dim=(1,), block_dim=(1,)
        )

        # --- 2. Forward actor with cache → raw [mean || log_std_raw] ---
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.actor_out.unsafe_ptr())
        var actor_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTOR_CS), MutAnyOrigin
        ](gpu_state.actor_cache.unsafe_ptr())
        Self.ActorNet.forward_gpu_with_cache[BS](
            ctx, obs_t, raw_t, p_actor, actor_cache_t, gpu_state.actor_ws
        )

        # --- 3. sac_rsample → curr_act, curr_lp, eps_cache ---
        var curr_act_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.curr_act.unsafe_ptr())
        var curr_lp_t = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.curr_lp.unsafe_ptr())
        var eps_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.eps_cache.unsafe_ptr())
        var ls_min = Scalar[dtype](-5.0)
        var ls_max = Scalar[dtype](2.0)
        var ascale = Scalar[dtype](self.action_scale)

        @parameter
        @always_inline
        def rsample_curr(
            acts: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ],
            lp: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            eps: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
            ],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
            asc: Scalar[dtype],
            rng: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
        ):
            sac_rsample_with_cache_kernel[dtype, BS, Self.ACTIONS](
                acts, lp, eps, ao, lsmin, lsmax, asc, rng
            )

        ctx.enqueue_function[rsample_curr, rsample_curr](
            curr_act_t,
            curr_lp_t,
            eps_t,
            raw_t,
            ls_min,
            ls_max,
            ascale,
            rng_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # --- 4. Concat [obs, curr_act] → new_ci ---
        var new_ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.new_ci.unsafe_ptr())
        ctx.enqueue_function[concat_k, concat_k](
            new_ci_t,
            obs_t,
            curr_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        # --- 5. For each online critic: forward(cache) → backward with
        #         seed -1/(N·BS), accumulate d_ci_sum = Σ d_ci[n] ---
        var q_out_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.q_out.unsafe_ptr())
        var q_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
        ](gpu_state.q_cache.unsafe_ptr())
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.dq_actor.unsafe_ptr())
        var d_ci_per_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci_per.unsafe_ptr())
        var d_ci_sum_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci_sum.unsafe_ptr())

        # Pre-fill dq_actor with -1/(N·BS) so backward seeds mean-Q gradient.
        var seed_val = Scalar[dtype](
            -1.0 / (Float64(Self.N_ENS) * Float64(BS))
        )
        comptime fill_dq_k = fill_constant_kernel[
            dtype, BS * Self.CRITIC_OUT
        ]
        var dq_flat_t = LayoutTensor[
            dtype, Layout.row_major(BS * Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.dq_actor.unsafe_ptr())
        comptime DQ_BLOCKS = (BS * Self.CRITIC_OUT + TPB - 1) // TPB
        ctx.enqueue_function[fill_dq_k, fill_dq_k](
            dq_flat_t,
            seed_val,
            grid_dim=(DQ_BLOCKS,),
            block_dim=(TPB,),
        )

        # Zero d_ci_sum.
        comptime fill_sum_k = fill_constant_kernel[
            dtype, BS * Self.CRITIC_IN
        ]
        var d_ci_sum_flat_t = LayoutTensor[
            dtype, Layout.row_major(BS * Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci_sum.unsafe_ptr())
        ctx.enqueue_function[fill_sum_k, fill_sum_k](
            d_ci_sum_flat_t,
            Scalar[dtype](0.0),
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        comptime add_k = add_ci_grads_kernel[
            dtype, BS, Self.CRITIC_IN
        ]

        for n in range(Self.N_ENS):
            var p_o = gpu_state.critics.online_params_view(n)
            # Note: we write backward into the critic's grad buffer to be
            # accurate, but REDQ doesn't need those grads here (policy update
            # doesn't step the critics). We zero them so stale state can't
            # leak back into the next critic loss step.
            var g_o = gpu_state.critics.online_grads_view(n)
            Self.CriticNet.forward_gpu_with_cache[BS](
                ctx,
                new_ci_t,
                q_out_t,
                p_o,
                q_cache_t,
                gpu_state.critic_ws,
            )
            gpu_state.critics.pairs[n].online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BS](
                ctx,
                dq_t,
                d_ci_per_t,
                p_o,
                q_cache_t,
                g_o,
                gpu_state.critic_ws,
            )
            # Zero the critic grads again — policy update must not leave them
            # populated for the next critic-update phase.
            gpu_state.critics.pairs[n].online.zero_grads(ctx)
            # Accumulate d_ci_sum += d_ci_per.
            ctx.enqueue_function[add_k, add_k](
                d_ci_sum_t,
                d_ci_per_t,
                grid_dim=(ELEM_BLOCKS,),
                block_dim=(TPB,),
            )

        # --- 6. Extract grad_act from d_ci_sum ---
        var grad_act_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.grad_act.unsafe_ptr())
        comptime ext_k = actor_grad_from_critic_kernel[
            dtype, BS, Self.OBS, Self.ACTIONS
        ]
        ctx.enqueue_function[ext_k, ext_k](
            grad_act_t,
            d_ci_sum_t,
            grid_dim=(ACT_BLOCKS,),
            block_dim=(TPB,),
        )

        # --- 7. sac_rsample backward → actor_grad ---
        var actor_grad_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.actor_grad_buf.unsafe_ptr())
        var alpha_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.gpu_scalars.unsafe_ptr())

        @parameter
        @always_inline
        def rsample_bwd(
            agrad: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
            ],
            ga: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ],
            ab: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
            ca: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ],
            eps: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
            ],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
            asc: Scalar[dtype],
        ):
            sac_rsample_bwd_kernel[dtype, BS, Self.ACTIONS](
                agrad, ga, ab, ca, eps, ao, lsmin, lsmax, asc
            )

        ctx.enqueue_function[rsample_bwd, rsample_bwd](
            actor_grad_t,
            grad_act_t,
            alpha_t,
            curr_act_t,
            eps_t,
            raw_t,
            ls_min,
            ls_max,
            ascale,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # --- 8. Actor backward ---
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
        ](gpu_state.d_obs.unsafe_ptr())
        gpu_state.actor.online.zero_grads(ctx)
        Self.ActorNet.backward_gpu[BS](
            ctx,
            actor_grad_t,
            d_obs_t,
            p_actor,
            actor_cache_t,
            a_grads,
            gpu_state.actor_ws,
        )

        # --- 9. Clip actor grads + step ---
        if self.max_grad_norm > 0.0:
            self._clip_actor_grads(ctx, gpu_state, a_grads)
        gpu_state.actor.online.optimizer_step(ctx)

        # --- 10. Alpha auto-tuning (SAC-style Adam on log_alpha) ---
        if self.auto_alpha:
            comptime GS = Self.GPUStateType
            comptime alpha_k = alpha_adam_update_kernel[
                dtype,
                BS,
                GS.GPU_ALPHA,
                GS.GPU_LOG_ALPHA,
                GS.GPU_ADAM_M,
                GS.GPU_ADAM_V,
                GS.GPU_ADAM_T,
                GS.GPU_TARGET_ENT,
                GS.GPU_ALPHA_LR,
            ]
            var scalars_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_state.gpu_scalars.unsafe_ptr())

            @parameter
            @always_inline
            def alpha_wrapper(
                sc: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
                lp: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                la_max: Scalar[dtype],
                la_min: Scalar[dtype],
                lp_clip: Scalar[dtype],
            ):
                alpha_k(sc, lp, la_max, la_min, lp_clip)

            ctx.enqueue_function[alpha_wrapper, alpha_wrapper](
                scalars_t,
                curr_lp_t,
                Scalar[dtype](2.0),
                Scalar[dtype](-10.0),
                Scalar[dtype](50.0),
                grid_dim=(1,),
                block_dim=(1,),
            )

    # -------------------------------------------------------------------------
    # Gradient clipping helpers
    # -------------------------------------------------------------------------

    def _clip_critic_grads(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        grads: LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.CriticModel.PARAM_SIZE),
            MutAnyOrigin,
        ],
    ) raises -> None:
        comptime C_PS = Self.Config.CriticModel.PARAM_SIZE
        comptime C_BLOCKS = (C_PS + TPB - 1) // TPB
        comptime c_norm_k = gradient_norm_kernel[
            dtype, C_PS, C_BLOCKS, TPB
        ]
        comptime c_clip_k = gradient_reduce_apply_fused_kernel[
            dtype, C_PS, C_BLOCKS, TPB
        ]
        var ps_t = LayoutTensor[
            dtype, Layout.row_major(C_BLOCKS), MutAnyOrigin
        ](gpu_state.grad_clip_ps.unsafe_ptr())
        ctx.enqueue_function[c_norm_k, c_norm_k](
            ps_t,
            grads,
            grid_dim=(C_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[c_clip_k, c_clip_k](
            grads,
            ps_t,
            Scalar[dtype](self.max_grad_norm),
            grid_dim=(C_BLOCKS,),
            block_dim=(TPB,),
        )

    def _clip_actor_grads(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        grads: LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.ActorModel.PARAM_SIZE),
            MutAnyOrigin,
        ],
    ) raises -> None:
        comptime A_PS = Self.Config.ActorModel.PARAM_SIZE
        comptime A_BLOCKS = (A_PS + TPB - 1) // TPB
        comptime a_norm_k = gradient_norm_kernel[
            dtype, A_PS, A_BLOCKS, TPB
        ]
        comptime a_clip_k = gradient_reduce_apply_fused_kernel[
            dtype, A_PS, A_BLOCKS, TPB
        ]
        var ps_t = LayoutTensor[
            dtype, Layout.row_major(A_BLOCKS), MutAnyOrigin
        ](gpu_state.grad_clip_ps.unsafe_ptr())
        ctx.enqueue_function[a_norm_k, a_norm_k](
            ps_t,
            grads,
            grid_dim=(A_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[a_clip_k, a_clip_k](
            grads,
            ps_t,
            Scalar[dtype](self.max_grad_norm),
            grid_dim=(A_BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # High-level training loop implementation
    #
    # Lives on the struct so `train_gpu()` below can invoke it via method
    # dispatch on `self`. See MBPOAgent for the rationale (Mojo nightly
    # dev2026042305 L-value unification bug on `mut self` →
    # `mut agent: REDQAgent[Config, max_n_envs=n_envs]` — kept on the
    # struct to sidestep the bug).
    # =========================================================================

    def _run_train_gpu_impl[
        E: GPUContinuousEnv,
        L: Logger = NoOpLogger,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 5_000,
        verbose: Bool = False,
        print_every: Int = 10_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[
            L, MutAnyOrigin
        ](),
        rng_seed: UInt64 = 42,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ) raises -> TrainingMetrics:
        """REDQ GPU training loop body. See `train_gpu()` for the public wrapper."""
        comptime n_envs = Self.max_n_envs
        var metrics = TrainingMetrics(
            algorithm_name="REDQ",
            environment_name=environment_name,
        )

        var gpu_state = self.make_gpu_state(ctx)
        self.upload_to_gpu(gpu_state, ctx)

        var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
        var prev_obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
        var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.ACTION_DIM)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

        var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var episode_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var gpu_reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
        var gpu_episode_count_buf = ctx.enqueue_create_buffer[dtype](1)
        var host_reward_sum = ctx.enqueue_create_host_buffer[dtype](1)
        var host_episode_count = ctx.enqueue_create_host_buffer[dtype](1)

        var ws_size = E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV
        if ws_size == 0:
            ws_size = 1
        var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)
        if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
            E.init_step_workspace_gpu[n_envs](ctx, workspace_buf)

        E.reset_kernel_gpu[n_envs, E.STATE_SIZE](
            ctx, states_buf, rng_seed=rng_seed
        )
        E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=rng_seed,
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )
        ctx.enqueue_memset(episode_rewards_buf, 0)
        ctx.enqueue_memset(episode_steps_buf, 0)
        ctx.enqueue_memset(gpu_reward_sum_buf, 0)
        ctx.enqueue_memset(gpu_episode_count_buf, 0)

        comptime tpb = 256
        comptime env_blocks = (n_envs + tpb - 1) // tpb
        comptime accum_k = accumulate_rewards_kernel[dtype, n_envs]
        comptime incr_steps_k = increment_steps_kernel[dtype, n_envs]
        comptime log_reset_k = log_and_reset_completed_kernel[dtype, n_envs]
        comptime act_tpb = 256
        comptime act_blocks = (n_envs * E.ACTION_DIM + act_tpb - 1) // act_tpb
        comptime warmup_kernel = uniform_random_actions_kernel[
            dtype, n_envs, E.ACTION_DIM
        ]

        var total_steps = 0
        var total_train_steps = 0
        var completed_episodes = 0
        var last_avg_reward: Float64 = 0.0
        var step_seed: UInt32 = UInt32(rng_seed)
        var next_print = print_every

        var progress_interval = print_every // 20
        if progress_interval < n_envs:
            progress_interval = n_envs
        var next_progress = progress_interval

        var ckpt_every = checkpoint_every
        var ckpt_path = checkpoint_path
        if self.checkpoint_every > 0 and self.checkpoint_path.byte_length() > 0:
            ckpt_every = self.checkpoint_every
            ckpt_path = self.checkpoint_path
        var next_checkpoint = ckpt_every

        var action_scale_val = Scalar[dtype](self.action_scale)
        var alg_name = String("REDQ")
        var alpha_host = ctx.enqueue_create_host_buffer[dtype](1)

        while total_steps < num_steps:
            ctx.enqueue_copy(prev_obs_buf, obs_buf)
            if total_steps < warmup_steps:
                var act_t = LayoutTensor[
                    dtype,
                    Layout.row_major(n_envs, E.ACTION_DIM),
                    MutAnyOrigin,
                ](actions_buf.unsafe_ptr())
                ctx.enqueue_function[warmup_kernel, warmup_kernel](
                    act_t,
                    action_scale_val,
                    Scalar[DType.uint32](step_seed),
                    grid_dim=(act_blocks,),
                    block_dim=(act_tpb,),
                )
            else:
                self.sync_explore_counter(ctx, gpu_state)
                self.select_actions_gpu[n_envs](
                    ctx, gpu_state, obs_buf, actions_buf
                )

            step_seed += 1
            E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                terminated_buf,
                obs_buf,
                rng_seed=UInt64(step_seed),
                workspace_ptr=workspace_buf.unsafe_ptr(),
            )

            self.gpu_store[n_envs](
                ctx,
                gpu_state,
                prev_obs_buf,
                actions_buf,
                rewards_buf,
                obs_buf,
                terminated_buf,
            )

            var er_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](episode_rewards_buf.unsafe_ptr())
            var rw_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](rewards_buf.unsafe_ptr())
            var es_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](episode_steps_buf.unsafe_ptr())
            var dn_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](dones_buf.unsafe_ptr())
            var rs_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_reward_sum_buf.unsafe_ptr())
            var ec_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_episode_count_buf.unsafe_ptr())
            ctx.enqueue_function[accum_k, accum_k](
                er_t, rw_t, grid_dim=(env_blocks,), block_dim=(tpb,)
            )
            ctx.enqueue_function[incr_steps_k, incr_steps_k](
                es_t, grid_dim=(env_blocks,), block_dim=(tpb,)
            )
            ctx.enqueue_function[log_reset_k, log_reset_k](
                dn_t,
                er_t,
                es_t,
                rs_t,
                ec_t,
                grid_dim=(1,),
                block_dim=(1,),
            )

            step_seed += 1
            E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
                ctx,
                states_buf,
                dones_buf,
                rng_seed=UInt64(step_seed),
                workspace_ptr=workspace_buf.unsafe_ptr(),
            )
            E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
                ctx, states_buf, obs_buf
            )

            total_steps += n_envs

            if total_steps >= warmup_steps:
                comptime UTD = Self.Config.UTD_RATIO
                var n_updates = UTD * n_envs
                for _ in range(n_updates):
                    self.do_gpu_train_step(ctx, gpu_state)
                    self.maybe_log_diagnostics[L](ctx, gpu_state, logger)
                total_train_steps += n_updates

            if verbose and total_steps >= next_progress:
                var interval_start = next_print - print_every
                print_progress_bar(
                    total_steps - interval_start,
                    print_every,
                    total_train_steps,
                    alg_name,
                )
                next_progress += progress_interval

            if (
                verbose or (logger and logger[].is_active())
            ) and total_steps >= next_print:
                ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
                ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
                ctx.enqueue_copy(alpha_host, gpu_state.gpu_scalars)
                ctx.synchronize()

                var recent_count = Int(host_episode_count[0])
                var recent_sum = Float64(host_reward_sum[0])
                var cur_alpha = Float64(alpha_host[0])
                completed_episodes += recent_count
                if recent_count > 0:
                    last_avg_reward = recent_sum / Float64(recent_count)
                    for _ in range(recent_count):
                        metrics.log_episode(
                            completed_episodes, last_avg_reward, 0, 0.0
                        )

                ctx.enqueue_memset(gpu_reward_sum_buf, 0)
                ctx.enqueue_memset(gpu_episode_count_buf, 0)

                if logger:
                    logger[].log_scalar(
                        "avg_reward", last_avg_reward, total_steps
                    )
                    logger[].log_scalar(
                        "episodes", Float64(completed_episodes), total_steps
                    )
                    logger[].log_scalar(
                        "train_steps", Float64(total_train_steps), total_steps
                    )
                    logger[].log_scalar("alpha", cur_alpha, total_steps)

                if verbose:
                    clear_progress_bar()
                    var status_line = (
                        alg_name
                        + " | Step "
                        + String(total_steps)
                        + " / "
                        + String(num_steps)
                        + " | Ep: "
                        + String(completed_episodes)
                        + " | AvgR: "
                        + String(last_avg_reward)[byte=:7]
                        + " | Train: "
                        + String(total_train_steps)
                        + " | α: "
                        + String(cur_alpha)[byte=:6]
                    )
                    print(status_line)
                next_print += print_every

            if ckpt_every > 0 and total_steps >= next_checkpoint:
                self.download_from_gpu(gpu_state, ctx)
                ctx.synchronize()
                self.save_checkpoint(ckpt_path)
                if verbose:
                    clear_progress_bar()
                    print(
                        alg_name
                        + " | checkpoint @ step "
                        + String(total_steps)
                        + " -> "
                        + ckpt_path
                    )
                next_checkpoint += ckpt_every

        ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
        ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
        ctx.enqueue_copy(alpha_host, gpu_state.gpu_scalars)
        ctx.synchronize()

        var final_count = Int(host_episode_count[0])
        var final_sum = Float64(host_reward_sum[0])
        completed_episodes += final_count
        if final_count > 0:
            last_avg_reward = final_sum / Float64(final_count)
            for _ in range(final_count):
                metrics.log_episode(completed_episodes, last_avg_reward, 0, 0.0)

        self.download_from_gpu(gpu_state, ctx)
        ctx.synchronize()

        if ckpt_every > 0 and ckpt_path.byte_length() > 0:
            self.save_checkpoint(ckpt_path)

        if logger and logger[].is_active():
            logger[].log_scalar(
                "avg_reward", last_avg_reward, total_steps
            )
            logger[].log_scalar(
                "episodes", Float64(completed_episodes), total_steps
            )
            logger[].log_scalar(
                "train_steps", Float64(total_train_steps), total_steps
            )
            logger[].log_scalar("alpha", Float64(alpha_host[0]), total_steps)
            logger[].flush()

        if verbose:
            clear_progress_bar()
            print(
                alg_name
                + " | Step "
                + String(total_steps)
                + " / "
                + String(num_steps)
                + " | Ep: "
                + String(completed_episodes)
                + " | AvgR: "
                + String(last_avg_reward)[byte=:7]
                + " | Train: "
                + String(total_train_steps)
                + " [DONE]"
            )

        return metrics^

    # =========================================================================
    # High-level training — convenience wrapper
    # =========================================================================

    def train_gpu[
        E: GPUContinuousEnv,
        L: Logger = NoOpLogger,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 5_000,
        verbose: Bool = False,
        print_every: Int = 10_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[
            L, MutAnyOrigin
        ](),
        rng_seed: UInt64 = 42,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ) raises -> TrainingMetrics:
        """GPU REDQ training convenience wrapper.

        Runs the GPU training loop in-place on `self`. Final trained weights
        are downloaded back onto `self.cpu_actor` / `self.cpu_critics`.
        """
        return self._run_train_gpu_impl[E, L](
            ctx,
            num_steps,
            warmup_steps=warmup_steps,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            logger=logger,
            rng_seed=rng_seed,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
        )
