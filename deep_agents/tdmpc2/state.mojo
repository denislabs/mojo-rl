"""TDMPC2GPUState — GPU buffer container for TD-MPC2 training.

Holds all device buffers, host buffers, and LayoutTensor views needed for
one TD-MPC2 GPU training loop. Replaces 80+ local `DeviceBuffer` allocations
in `train_gpu[]` with a single struct construction.

Usage:
    var gpu_state = TDMPC2GPUState[...](ctx)
    # upload CPU weights → gpu_state
    # training loop uses gpu_state.enc.params_buf, gpu_state.z_buf, etc.
"""

from nn.model import Model
from nn.optimizer import Optimizer
from nn.training import Network, NetworkState, GPUNetworkState
from nn.constants import dtype, TPB
from deep_agents.core.replay.sequence_replay_buffer import SequenceReplayBuffer
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from .world_model import WorldModel


# =============================================================================
# TDMPC2GPUState
# =============================================================================


struct TDMPC2GPUState[
    # Network model / optimizer types (6 sub-networks)
    EncModel: Model,
    EncOpt: Optimizer,
    DynModel: Model,
    DynOpt: Optimizer,
    RewModel: Model,
    RewOpt: Optimizer,
    TermModel: Model,
    TermOpt: Optimizer,
    PolModel: Model,
    PolOpt: Optimizer,
    QModel: Model,
    QOpt: Optimizer,
    # Dimension parameters
    obs_dim: Int,
    action_dim: Int,
    latent_dim: Int,
    num_bins: Int,
    batch_size: Int,
    horizon: Int,
    max_n_envs: Int,
    env_state_size: Int,
]:
    """GPU-resident state for TD-MPC2 training.

    Holds all device buffers needed for one TD-MPC2 training loop:
      - 11 sub-network GPU states (enc, dyn, rew, term, pol, q1..q5)
      - 5 target Q param-only buffers
      - Training workspaces, intermediate buffers, gradient buffers
      - Batch data upload/download buffers
      - Environment-sized buffers for data collection

    Created once at the start of GPU training.

    Parameters:
        EncModel..QOpt: Network model and optimizer types for each sub-network.
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        latent_dim: Latent state dimension.
        num_bins: Distributional RL bins.
        batch_size: Training batch size.
        horizon: Planning horizon H.
        max_n_envs: Max parallel environments (sizes exploration/env buffers).
        env_state_size: GPU environment state size per env.
    """

    # ── Derived compile-time constants ──────────────────────────────────────
    comptime BATCH = Self.batch_size
    comptime H = Self.horizon
    comptime OBS = Self.obs_dim
    comptime ACT = Self.action_dim
    comptime LATENT = Self.latent_dim
    comptime BINS = Self.num_bins
    comptime ZA = Self.LATENT + Self.ACT

    # Flat batch buffer sizes
    comptime B_OBS = Self.BATCH * Self.OBS
    comptime B_ACT = Self.BATCH * Self.ACT
    comptime B_LATENT = Self.BATCH * Self.LATENT
    comptime B_ZA = Self.BATCH * Self.ZA
    comptime B_BINS = Self.BATCH * Self.BINS
    comptime BATCH_OBS_FLAT = Self.BATCH * (Self.H + 1) * Self.OBS
    comptime BATCH_ACT_FLAT = Self.BATCH * Self.H * Self.ACT
    comptime BATCH_SCALAR_FLAT = Self.BATCH * Self.H
    comptime BATCH_TGTS_FLAT = Self.H * Self.BATCH * Self.BINS

    # Network parameter sizes
    comptime ENC_P = Self.EncModel.PARAM_SIZE
    comptime DYN_P = Self.DynModel.PARAM_SIZE
    comptime REW_P = Self.RewModel.PARAM_SIZE
    comptime TERM_P = Self.TermModel.PARAM_SIZE
    comptime POL_P = Self.PolModel.PARAM_SIZE
    comptime Q_P = Self.QModel.PARAM_SIZE

    # Cache sizes
    comptime ENC_C = Self.EncModel.CACHE_SIZE
    comptime DYN_C = Self.DynModel.CACHE_SIZE
    comptime REW_C = Self.RewModel.CACHE_SIZE
    comptime TERM_C = Self.TermModel.CACHE_SIZE
    comptime POL_C = Self.PolModel.CACHE_SIZE
    comptime Q_C = Self.QModel.CACHE_SIZE

    # Workspace sizes per sample
    comptime ENC_W = Network[
        Self.EncModel, Self.EncOpt
    ].WORKSPACE_SIZE_PER_SAMPLE
    comptime DYN_W = Network[
        Self.DynModel, Self.DynOpt
    ].WORKSPACE_SIZE_PER_SAMPLE
    comptime REW_W = Network[
        Self.RewModel, Self.RewOpt
    ].WORKSPACE_SIZE_PER_SAMPLE
    comptime TERM_W = Network[
        Self.TermModel, Self.TermOpt
    ].WORKSPACE_SIZE_PER_SAMPLE
    comptime POL_W = Network[
        Self.PolModel, Self.PolOpt
    ].WORKSPACE_SIZE_PER_SAMPLE
    comptime Q_W = Network[Self.QModel, Self.QOpt].WORKSPACE_SIZE_PER_SAMPLE

    # Batch workspace sizes
    comptime ENC_BATCH_WS = Self.BATCH * Self.ENC_W
    comptime DYN_BATCH_WS = Self.BATCH * Self.DYN_W
    comptime REW_BATCH_WS = Self.BATCH * Self.REW_W
    comptime TERM_BATCH_WS = Self.BATCH * Self.TERM_W
    comptime POL_BATCH_WS = Self.BATCH * Self.POL_W
    comptime Q_BATCH_WS = Self.BATCH * Self.Q_W

    # Gradient block sizes
    comptime ENC_GRAD_BLOCKS = (Self.ENC_P + TPB - 1) // TPB
    comptime DYN_GRAD_BLOCKS = (Self.DYN_P + TPB - 1) // TPB
    comptime REW_GRAD_BLOCKS = (Self.REW_P + TPB - 1) // TPB
    comptime TERM_GRAD_BLOCKS = (Self.TERM_P + TPB - 1) // TPB
    comptime POL_GRAD_BLOCKS = (Self.POL_P + TPB - 1) // TPB
    comptime Q_GRAD_BLOCKS = (Self.Q_P + TPB - 1) // TPB
    comptime BATCH_BLOCKS = (Self.BATCH + TPB - 1) // TPB
    comptime DUMMY_SIZE = max(Self.B_ZA, Self.B_OBS)

    # Env-sized constants
    comptime ENV_STATE = Self.max_n_envs * Self.env_state_size
    comptime ENV_OBS = Self.max_n_envs * Self.OBS
    comptime ENV_ACT = Self.max_n_envs * Self.ACT
    comptime ENV_LATENT = Self.max_n_envs * Self.LATENT
    comptime ENV_PI_OUT = Self.max_n_envs * 2 * Self.ACT
    comptime ENC_ENV_WS = Self.max_n_envs * Self.ENC_W
    comptime POL_ENV_WS = Self.max_n_envs * Self.POL_W

    # ── GPU Network states (GPUNetworkState: params + grads + optimizer state) ──
    var enc: GPUNetworkState[Self.EncModel, Self.EncOpt]
    var dyn: GPUNetworkState[Self.DynModel, Self.DynOpt]
    var rew: GPUNetworkState[Self.RewModel, Self.RewOpt]
    var term: GPUNetworkState[Self.TermModel, Self.TermOpt]
    var pol: GPUNetworkState[Self.PolModel, Self.PolOpt]
    var q1: GPUNetworkState[Self.QModel, Self.QOpt]
    var q2: GPUNetworkState[Self.QModel, Self.QOpt]
    var q3: GPUNetworkState[Self.QModel, Self.QOpt]
    var q4: GPUNetworkState[Self.QModel, Self.QOpt]
    var q5: GPUNetworkState[Self.QModel, Self.QOpt]

    # ── Target Q networks (params only, no grads/state needed) ──
    var q1t_params_buf: DeviceBuffer[dtype]
    var q2t_params_buf: DeviceBuffer[dtype]
    var q3t_params_buf: DeviceBuffer[dtype]
    var q4t_params_buf: DeviceBuffer[dtype]
    var q5t_params_buf: DeviceBuffer[dtype]

    # ── Training workspace buffers (batch-sized) ──
    var enc_cache_buf: DeviceBuffer[dtype]
    var enc_batch_ws_buf: DeviceBuffer[dtype]
    var dyn_cache_buf: DeviceBuffer[dtype]
    var dyn_batch_ws_buf: DeviceBuffer[dtype]
    var rew_cache_buf: DeviceBuffer[dtype]
    var rew_batch_ws_buf: DeviceBuffer[dtype]
    var term_cache_buf: DeviceBuffer[dtype]
    var term_batch_ws_buf: DeviceBuffer[dtype]
    var pol_cache_buf: DeviceBuffer[dtype]
    var pol_batch_ws_buf: DeviceBuffer[dtype]
    var q1_cache_buf: DeviceBuffer[dtype]
    var q1_batch_ws_buf: DeviceBuffer[dtype]
    var q2_cache_buf: DeviceBuffer[dtype]
    var q2_batch_ws_buf: DeviceBuffer[dtype]
    var q3_cache_buf: DeviceBuffer[dtype]
    var q3_batch_ws_buf: DeviceBuffer[dtype]
    var q4_cache_buf: DeviceBuffer[dtype]
    var q4_batch_ws_buf: DeviceBuffer[dtype]
    var q5_cache_buf: DeviceBuffer[dtype]
    var q5_batch_ws_buf: DeviceBuffer[dtype]
    var qt_batch_ws_buf: DeviceBuffer[
        dtype
    ]  # shared for target Q no-grad passes

    # ── Inference workspace buffers (env-sized) ──
    var enc_env_ws_buf: DeviceBuffer[dtype]
    var pol_env_ws_buf: DeviceBuffer[dtype]

    # ── Gradient norm partial-sum buffers ──
    var enc_grad_ps_buf: DeviceBuffer[dtype]
    var dyn_grad_ps_buf: DeviceBuffer[dtype]
    var rew_grad_ps_buf: DeviceBuffer[dtype]
    var term_grad_ps_buf: DeviceBuffer[dtype]
    var pol_grad_ps_buf: DeviceBuffer[dtype]
    var q_grad_ps_buf: DeviceBuffer[
        dtype
    ]  # 5 * Q_GRAD_BLOCKS for fused 5Q norm

    # ── Intermediate training buffers ──
    var z_buf: DeviceBuffer[dtype]  # [B_LATENT] current z_t
    var z_next_buf: DeviceBuffer[dtype]  # [B_LATENT] enc(obs_{t+1}) stop-grad
    var z_pred_buf: DeviceBuffer[dtype]  # [B_LATENT] dynamics(za_t)
    var za_buf: DeviceBuffer[dtype]  # [B_ZA] [z_t, a_t]
    var pi_out_buf: DeviceBuffer[dtype]  # [BATCH * 2 * ACT]
    var pi_act_buf: DeviceBuffer[dtype]  # [B_ACT] tanh(mean) actions
    var logits_buf: DeviceBuffer[dtype]  # [B_BINS] shared Q/rew logits
    var term_prob_buf: DeviceBuffer[dtype]  # [BATCH]
    var q_min_buf: DeviceBuffer[dtype]  # [BATCH]

    # ── Per-step extraction buffers ──
    var obs_step_buf: DeviceBuffer[dtype]  # [B_OBS]
    var obs_next_step_buf: DeviceBuffer[dtype]  # [B_OBS]
    var act_step_buf: DeviceBuffer[dtype]  # [B_ACT]
    var rew_step_buf: DeviceBuffer[dtype]  # [BATCH]
    var done_step_buf: DeviceBuffer[dtype]  # [BATCH]
    var tgt_step_buf: DeviceBuffer[dtype]  # [B_BINS]

    # ── Gradient buffers ──
    var grad_z_pred_buf: DeviceBuffer[dtype]  # [B_LATENT]
    var grad_za_buf: DeviceBuffer[dtype]  # [B_ZA]
    var grad_z_dyn_buf: DeviceBuffer[dtype]  # [B_LATENT]
    var grad_z_term_buf: DeviceBuffer[dtype]  # [B_LATENT]
    var grad_enc_out_buf: DeviceBuffer[dtype]  # [B_LATENT]
    var grad_logits_buf: DeviceBuffer[dtype]  # [B_BINS]
    var grad_term_prob_buf: DeviceBuffer[dtype]  # [BATCH]
    var grad_pi_out_buf: DeviceBuffer[dtype]  # [BATCH * 2 * ACT]
    var dummy_grad_buf: DeviceBuffer[dtype]  # [max(B_ZA, B_OBS)]

    # ── TD targets + bins ──
    var td_targets_buf: DeviceBuffer[dtype]  # [H * BATCH * BINS]
    var bins_buf: DeviceBuffer[dtype]  # [BINS]

    # ── Batch data GPU buffers (CPU→GPU per training step) ──
    var batch_obs_buf: DeviceBuffer[dtype]  # [BATCH_OBS_FLAT]
    var batch_act_buf: DeviceBuffer[dtype]  # [BATCH_ACT_FLAT]
    var batch_rew_buf: DeviceBuffer[dtype]  # [BATCH_SCALAR_FLAT]
    var batch_done_buf: DeviceBuffer[dtype]  # [BATCH_SCALAR_FLAT]

    # ── Batch data host buffers (CPU-side, for CPU→GPU transfers) ──
    var batch_obs_host: HostBuffer[dtype]
    var batch_act_host: HostBuffer[dtype]
    var batch_rew_host: HostBuffer[dtype]
    var batch_done_host: HostBuffer[dtype]

    # ── Environment GPU buffers ──
    var states_buf: DeviceBuffer[dtype]  # [ENV_STATE]
    var env_obs_buf: DeviceBuffer[dtype]  # [ENV_OBS]
    var env_act_buf: DeviceBuffer[dtype]  # [ENV_ACT]
    var env_rew_buf: DeviceBuffer[dtype]  # [max_n_envs]
    var env_done_buf: DeviceBuffer[dtype]  # [max_n_envs]
    var env_z_buf: DeviceBuffer[dtype]  # [ENV_LATENT]
    var env_pi_out_buf: DeviceBuffer[dtype]  # [ENV_PI_OUT]

    # ── Episode tracking (GPU) ──
    var ep_rew_buf: DeviceBuffer[dtype]  # [max_n_envs]
    var ep_steps_buf: DeviceBuffer[dtype]  # [max_n_envs]
    var completed_rew_buf: DeviceBuffer[dtype]  # [max_n_envs]
    var completed_steps_buf: DeviceBuffer[dtype]  # [max_n_envs]
    var completed_mask_buf: DeviceBuffer[dtype]  # [max_n_envs]

    # ── Environment host buffers ──
    var env_obs_host: HostBuffer[dtype]
    var env_act_host: HostBuffer[dtype]
    var env_rew_host: HostBuffer[dtype]
    var env_done_host: HostBuffer[dtype]
    var completed_rew_host: HostBuffer[dtype]
    var completed_steps_host: HostBuffer[dtype]
    var completed_mask_host: HostBuffer[dtype]

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU and host buffers."""

        # ── Network states ──
        self.enc = GPUNetworkState[Self.EncModel, Self.EncOpt](ctx)
        self.dyn = GPUNetworkState[Self.DynModel, Self.DynOpt](ctx)
        self.rew = GPUNetworkState[Self.RewModel, Self.RewOpt](ctx)
        self.term = GPUNetworkState[Self.TermModel, Self.TermOpt](ctx)
        self.pol = GPUNetworkState[Self.PolModel, Self.PolOpt](ctx)
        self.q1 = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.q2 = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.q3 = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.q4 = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.q5 = GPUNetworkState[Self.QModel, Self.QOpt](ctx)

        # ── Target Q params ──
        self.q1t_params_buf = ctx.enqueue_create_buffer[dtype](Self.Q_P)
        self.q2t_params_buf = ctx.enqueue_create_buffer[dtype](Self.Q_P)
        self.q3t_params_buf = ctx.enqueue_create_buffer[dtype](Self.Q_P)
        self.q4t_params_buf = ctx.enqueue_create_buffer[dtype](Self.Q_P)
        self.q5t_params_buf = ctx.enqueue_create_buffer[dtype](Self.Q_P)

        # ── Training cache + workspace buffers (batch-sized) ──
        self.enc_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ENC_C
        )
        self.enc_batch_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.ENC_BATCH_WS
        )
        self.dyn_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.DYN_C
        )
        self.dyn_batch_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.DYN_BATCH_WS
        )
        self.rew_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.REW_C
        )
        self.rew_batch_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.REW_BATCH_WS
        )
        self.term_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.TERM_C
        )
        self.term_batch_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.TERM_BATCH_WS
        )
        self.pol_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.POL_C
        )
        self.pol_batch_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.POL_BATCH_WS
        )
        self.q1_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.Q_C
        )
        self.q1_batch_ws_buf = ctx.enqueue_create_buffer[dtype](Self.Q_BATCH_WS)
        self.q2_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.Q_C
        )
        self.q2_batch_ws_buf = ctx.enqueue_create_buffer[dtype](Self.Q_BATCH_WS)
        self.q3_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.Q_C
        )
        self.q3_batch_ws_buf = ctx.enqueue_create_buffer[dtype](Self.Q_BATCH_WS)
        self.q4_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.Q_C
        )
        self.q4_batch_ws_buf = ctx.enqueue_create_buffer[dtype](Self.Q_BATCH_WS)
        self.q5_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.Q_C
        )
        self.q5_batch_ws_buf = ctx.enqueue_create_buffer[dtype](Self.Q_BATCH_WS)
        self.qt_batch_ws_buf = ctx.enqueue_create_buffer[dtype](Self.Q_BATCH_WS)

        # ── Inference workspace buffers (env-sized) ──
        self.enc_env_ws_buf = ctx.enqueue_create_buffer[dtype](Self.ENC_ENV_WS)
        self.pol_env_ws_buf = ctx.enqueue_create_buffer[dtype](Self.POL_ENV_WS)

        # ── Gradient norm partial-sum buffers ──
        self.enc_grad_ps_buf = ctx.enqueue_create_buffer[dtype](
            Self.ENC_GRAD_BLOCKS
        )
        self.dyn_grad_ps_buf = ctx.enqueue_create_buffer[dtype](
            Self.DYN_GRAD_BLOCKS
        )
        self.rew_grad_ps_buf = ctx.enqueue_create_buffer[dtype](
            Self.REW_GRAD_BLOCKS
        )
        self.term_grad_ps_buf = ctx.enqueue_create_buffer[dtype](
            Self.TERM_GRAD_BLOCKS
        )
        self.pol_grad_ps_buf = ctx.enqueue_create_buffer[dtype](
            Self.POL_GRAD_BLOCKS
        )
        self.q_grad_ps_buf = ctx.enqueue_create_buffer[dtype](
            Self.Q_GRAD_BLOCKS * 5
        )

        # ── Intermediate training buffers ──
        self.z_buf = ctx.enqueue_create_buffer[dtype](Self.B_LATENT)
        self.z_next_buf = ctx.enqueue_create_buffer[dtype](Self.B_LATENT)
        self.z_pred_buf = ctx.enqueue_create_buffer[dtype](Self.B_LATENT)
        self.za_buf = ctx.enqueue_create_buffer[dtype](Self.B_ZA)
        self.pi_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * 2 * Self.ACT
        )
        self.pi_act_buf = ctx.enqueue_create_buffer[dtype](Self.B_ACT)
        self.logits_buf = ctx.enqueue_create_buffer[dtype](Self.B_BINS)
        self.term_prob_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.q_min_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH)

        # ── Per-step extraction buffers ──
        self.obs_step_buf = ctx.enqueue_create_buffer[dtype](Self.B_OBS)
        self.obs_next_step_buf = ctx.enqueue_create_buffer[dtype](Self.B_OBS)
        self.act_step_buf = ctx.enqueue_create_buffer[dtype](Self.B_ACT)
        self.rew_step_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.done_step_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.tgt_step_buf = ctx.enqueue_create_buffer[dtype](Self.B_BINS)

        # ── Gradient buffers ──
        self.grad_z_pred_buf = ctx.enqueue_create_buffer[dtype](Self.B_LATENT)
        self.grad_za_buf = ctx.enqueue_create_buffer[dtype](Self.B_ZA)
        self.grad_z_dyn_buf = ctx.enqueue_create_buffer[dtype](Self.B_LATENT)
        self.grad_z_term_buf = ctx.enqueue_create_buffer[dtype](Self.B_LATENT)
        self.grad_enc_out_buf = ctx.enqueue_create_buffer[dtype](Self.B_LATENT)
        self.grad_logits_buf = ctx.enqueue_create_buffer[dtype](Self.B_BINS)
        self.grad_term_prob_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.grad_pi_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * 2 * Self.ACT
        )
        self.dummy_grad_buf = ctx.enqueue_create_buffer[dtype](Self.DUMMY_SIZE)

        # ── TD targets + bins ──
        self.td_targets_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_TGTS_FLAT
        )
        self.bins_buf = ctx.enqueue_create_buffer[dtype](Self.BINS)

        # ── Batch data GPU buffers ──
        self.batch_obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_OBS_FLAT
        )
        self.batch_act_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_ACT_FLAT
        )
        self.batch_rew_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SCALAR_FLAT
        )
        self.batch_done_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SCALAR_FLAT
        )

        # ── Batch data host buffers ──
        self.batch_obs_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_OBS_FLAT
        )
        self.batch_act_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_ACT_FLAT
        )
        self.batch_rew_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_SCALAR_FLAT
        )
        self.batch_done_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_SCALAR_FLAT
        )

        # ── Environment GPU buffers ──
        self.states_buf = ctx.enqueue_create_buffer[dtype](Self.ENV_STATE)
        self.env_obs_buf = ctx.enqueue_create_buffer[dtype](Self.ENV_OBS)
        self.env_act_buf = ctx.enqueue_create_buffer[dtype](Self.ENV_ACT)
        self.env_rew_buf = ctx.enqueue_create_buffer[dtype](Self.max_n_envs)
        self.env_done_buf = ctx.enqueue_create_buffer[dtype](Self.max_n_envs)
        self.env_z_buf = ctx.enqueue_create_buffer[dtype](Self.ENV_LATENT)
        self.env_pi_out_buf = ctx.enqueue_create_buffer[dtype](Self.ENV_PI_OUT)

        # ── Episode tracking ──
        self.ep_rew_buf = ctx.enqueue_create_buffer[dtype](Self.max_n_envs)
        self.ep_steps_buf = ctx.enqueue_create_buffer[dtype](Self.max_n_envs)
        self.completed_rew_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs
        )
        self.completed_steps_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs
        )
        self.completed_mask_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs
        )

        # Zero episode tracking
        ctx.enqueue_memset(self.ep_rew_buf, 0)
        ctx.enqueue_memset(self.ep_steps_buf, 0)

        # ── Environment host buffers ──
        self.env_obs_host = ctx.enqueue_create_host_buffer[dtype](Self.ENV_OBS)
        self.env_act_host = ctx.enqueue_create_host_buffer[dtype](Self.ENV_ACT)
        self.env_rew_host = ctx.enqueue_create_host_buffer[dtype](
            Self.max_n_envs
        )
        self.env_done_host = ctx.enqueue_create_host_buffer[dtype](
            Self.max_n_envs
        )
        self.completed_rew_host = ctx.enqueue_create_host_buffer[dtype](
            Self.max_n_envs
        )
        self.completed_steps_host = ctx.enqueue_create_host_buffer[dtype](
            Self.max_n_envs
        )
        self.completed_mask_host = ctx.enqueue_create_host_buffer[dtype](
            Self.max_n_envs
        )


# =============================================================================
# TDMPC2CPUState — CPU buffer container for TD-MPC2 training
# =============================================================================


struct TDMPC2CPUState[
    obs_dim: Int,
    action_dim: Int,
    latent_dim: Int = 256,
    mlp_dim: Int = 256,
    num_bins: Int = 101,
    num_q: Int = 5,
    simplex_dim: Int = 8,
    v_min: Float64 = -10.0,
    v_max: Float64 = 10.0,
    enc_lr: Float64 = 9e-5,
    wm_lr: Float64 = 3e-4,
    pi_lr: Float64 = 3e-4,
    buffer_capacity: Int = 1_000_000,
    batch_size: Int = 256,
    horizon: Int = 3,
](Movable):
    """CPU-resident state for TD-MPC2 training.

    Holds all heap-allocated data needed for one TD-MPC2 training loop:
      - WorldModel with all 15 NetworkStates + bins
      - SequenceReplayBuffer (streaming obs/act/rew/done, NOT tuple-based)
      - Pre-allocated scratch Lists for update() (avoids per-call allocation)

    Created once in TDMPC2Agent.__init__.

    Does NOT conform to OffPolicyState because TDMPC2 uses
    SequenceReplayBuffer (streaming obs/act/rew/done) instead of the
    standard (s, a, r, s', done) tuple-based ReplayBuffer. Custom
    observe()/is_ready() methods are provided instead.

    Parameters:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        latent_dim: Latent state dimension (default: 256).
        mlp_dim: Hidden layer width (default: 256).
        num_bins: Number of bins for distributional RL (default: 101).
        num_q: Number of Q-networks in the ensemble (default: 5).
        simplex_dim: SimNorm group size for dynamics head (default: 8).
        v_min: Minimum value for distribution bins (default: -10.0).
        v_max: Maximum value for distribution bins (default: 10.0).
        enc_lr: Encoder learning rate (default: 9e-5).
        wm_lr: World model learning rate (default: 3e-4).
        pi_lr: Policy learning rate (default: 3e-4).
        buffer_capacity: Replay buffer capacity (default: 1M).
        batch_size: Training batch size (default: 256).
        horizon: Planning horizon H (default: 3).
    """

    # ── WorldModel type alias ─────────────────────────────────────────────
    comptime WM = WorldModel[
        Self.obs_dim,
        Self.action_dim,
        Self.latent_dim,
        Self.mlp_dim,
        Self.num_bins,
        Self.num_q,
        Self.simplex_dim,
        Self.v_min,
        Self.v_max,
        Self.enc_lr,
        Self.wm_lr,
        Self.pi_lr,
    ]

    # ── Shorthand dimension constants ─────────────────────────────────────
    comptime BATCH = Self.batch_size
    comptime H = Self.horizon
    comptime OBS = Self.obs_dim
    comptime ACT = Self.action_dim
    comptime LATENT = Self.latent_dim
    comptime BINS = Self.num_bins
    comptime ZA = Self.LATENT + Self.ACT

    # Flat batch sizes
    comptime B_OBS = Self.BATCH * Self.OBS
    comptime B_ACT = Self.BATCH * Self.ACT
    comptime B_LATENT = Self.BATCH * Self.LATENT
    comptime B_ZA = Self.BATCH * Self.ZA
    comptime B_BINS = Self.BATCH * Self.BINS
    comptime BATCH_OBS_FLAT = Self.BATCH * (Self.H + 1) * Self.OBS
    comptime BATCH_ACT_FLAT = Self.BATCH * Self.H * Self.ACT
    comptime BATCH_SCALAR_FLAT = Self.BATCH * Self.H
    comptime BATCH_TGTS_FLAT = Self.H * Self.BATCH * Self.BINS

    # Cache sizes (per sample)
    comptime ENC_CACHE_SIZE = Self.WM.EncModel.CACHE_SIZE
    comptime DYN_CACHE_SIZE = Self.WM.DynModel.CACHE_SIZE
    comptime REW_CACHE_SIZE = Self.WM.RewModel.CACHE_SIZE
    comptime TERM_CACHE_SIZE = Self.WM.TermModel.CACHE_SIZE
    comptime POL_CACHE_SIZE = Self.WM.PolModel.CACHE_SIZE
    comptime Q_CACHE_SIZE = Self.WM.QModel.CACHE_SIZE

    # ── Core state ────────────────────────────────────────────────────────
    var world_model: Self.WM
    var buffer: SequenceReplayBuffer[
        Self.buffer_capacity, Self.OBS, Self.ACT, dtype
    ]

    # ── Batch data (filled by buffer.sample_sequences) ────────────────────
    var _batch_obs: List[Scalar[dtype]]  # [BATCH*(H+1)*OBS]
    var _batch_actions: List[Scalar[dtype]]  # [BATCH*H*ACT]
    var _batch_rewards: List[Scalar[dtype]]  # [BATCH*H]
    var _batch_dones: List[Scalar[dtype]]  # [BATCH*H]

    # ── World model update scratch (reused each horizon step) ─────────────
    var _obs_0: List[Scalar[dtype]]  # [B_OBS] first-step observations
    var _next_obs: List[Scalar[dtype]]  # [B_OBS] next observations per step
    var _enc_cache: List[Scalar[dtype]]  # [BATCH * ENC_CACHE_SIZE]
    var _z_current: List[Scalar[dtype]]  # [B_LATENT] rolling latent state
    var _z_pred: List[Scalar[dtype]]  # [B_LATENT] dynamics prediction
    var _z_enc_next: List[
        Scalar[dtype]
    ]  # [B_LATENT] encoded next obs (stop-grad)
    var _za: List[Scalar[dtype]]  # [B_ZA] concatenated z+a (shared w/ policy)
    var _dyn_cache: List[Scalar[dtype]]  # [BATCH * DYN_CACHE_SIZE]
    var _rew_logits: List[Scalar[dtype]]  # [B_BINS]
    var _rew_cache: List[Scalar[dtype]]  # [BATCH * REW_CACHE_SIZE]
    var _term_cache: List[Scalar[dtype]]  # [BATCH * TERM_CACHE_SIZE]
    var _q_logits: List[
        Scalar[dtype]
    ]  # [B_BINS] single Q logits (reused per Q)
    var _q_cache: List[Scalar[dtype]]  # [BATCH * Q_CACHE_SIZE]
    var _a_next_mean: List[Scalar[dtype]]  # [B_ACT]
    var _a_next_log_std: List[Scalar[dtype]]  # [B_ACT]
    var _td_targets: List[Scalar[dtype]]  # [H * BATCH * BINS]

    # ── Policy update scratch ─────────────────────────────────────────────
    var _pi_cache: List[Scalar[dtype]]  # [BATCH * POL_CACHE_SIZE]
    var _pi_out: List[Scalar[dtype]]  # [BATCH * 2 * ACT]
    var _a_pi: List[Scalar[dtype]]  # [B_ACT]
    var _q_logits2: List[Scalar[dtype]]  # [B_BINS] second Q for min(Q1, Q2)

    fn __init__(out self):
        """Allocate world model, replay buffer, and all scratch buffers."""

        # ── Core state ────────────────────────────────────────────────────
        self.world_model = Self.WM()
        self.buffer = SequenceReplayBuffer[
            Self.buffer_capacity, Self.OBS, Self.ACT, dtype
        ]()

        # ── Allocate scratch with capacity ────────────────────────────────

        # Batch data
        self._batch_obs = List[Scalar[dtype]](capacity=Self.BATCH_OBS_FLAT)
        self._batch_actions = List[Scalar[dtype]](capacity=Self.BATCH_ACT_FLAT)
        self._batch_rewards = List[Scalar[dtype]](
            capacity=Self.BATCH_SCALAR_FLAT
        )
        self._batch_dones = List[Scalar[dtype]](capacity=Self.BATCH_SCALAR_FLAT)

        # WM update scratch
        self._obs_0 = List[Scalar[dtype]](capacity=Self.B_OBS)
        self._next_obs = List[Scalar[dtype]](capacity=Self.B_OBS)
        self._enc_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ENC_CACHE_SIZE
        )
        self._z_current = List[Scalar[dtype]](capacity=Self.B_LATENT)
        self._z_pred = List[Scalar[dtype]](capacity=Self.B_LATENT)
        self._z_enc_next = List[Scalar[dtype]](capacity=Self.B_LATENT)
        self._za = List[Scalar[dtype]](capacity=Self.B_ZA)
        self._dyn_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.DYN_CACHE_SIZE
        )
        self._rew_logits = List[Scalar[dtype]](capacity=Self.B_BINS)
        self._rew_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.REW_CACHE_SIZE
        )
        self._term_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.TERM_CACHE_SIZE
        )
        self._q_logits = List[Scalar[dtype]](capacity=Self.B_BINS)
        self._q_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.Q_CACHE_SIZE
        )
        self._a_next_mean = List[Scalar[dtype]](capacity=Self.B_ACT)
        self._a_next_log_std = List[Scalar[dtype]](capacity=Self.B_ACT)
        self._td_targets = List[Scalar[dtype]](capacity=Self.BATCH_TGTS_FLAT)

        # Policy update scratch
        self._pi_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.POL_CACHE_SIZE
        )
        self._pi_out = List[Scalar[dtype]](capacity=Self.BATCH * 2 * Self.ACT)
        self._a_pi = List[Scalar[dtype]](capacity=Self.B_ACT)
        self._q_logits2 = List[Scalar[dtype]](capacity=Self.B_BINS)

        # ── Zero-fill all scratch buffers ─────────────────────────────────
        # Grouped by size to minimise loop count (matching TD3CPUState style)

        # BATCH_OBS_FLAT sized
        for _ in range(Self.BATCH_OBS_FLAT):
            self._batch_obs.append(Scalar[dtype](0))

        # BATCH_ACT_FLAT sized
        for _ in range(Self.BATCH_ACT_FLAT):
            self._batch_actions.append(Scalar[dtype](0))

        # BATCH_SCALAR_FLAT sized
        for _ in range(Self.BATCH_SCALAR_FLAT):
            self._batch_rewards.append(Scalar[dtype](0))
            self._batch_dones.append(Scalar[dtype](0))

        # B_OBS sized
        for _ in range(Self.B_OBS):
            self._obs_0.append(Scalar[dtype](0))
            self._next_obs.append(Scalar[dtype](0))

        # B_LATENT sized
        for _ in range(Self.B_LATENT):
            self._z_current.append(Scalar[dtype](0))
            self._z_pred.append(Scalar[dtype](0))
            self._z_enc_next.append(Scalar[dtype](0))

        # B_ZA sized
        for _ in range(Self.B_ZA):
            self._za.append(Scalar[dtype](0))

        # B_ACT sized
        for _ in range(Self.B_ACT):
            self._a_next_mean.append(Scalar[dtype](0))
            self._a_next_log_std.append(Scalar[dtype](0))
            self._a_pi.append(Scalar[dtype](0))

        # B_BINS sized
        for _ in range(Self.B_BINS):
            self._rew_logits.append(Scalar[dtype](0))
            self._q_logits.append(Scalar[dtype](0))
            self._q_logits2.append(Scalar[dtype](0))

        # BATCH * cache sized
        for _ in range(Self.BATCH * Self.ENC_CACHE_SIZE):
            self._enc_cache.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.DYN_CACHE_SIZE):
            self._dyn_cache.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.REW_CACHE_SIZE):
            self._rew_cache.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.TERM_CACHE_SIZE):
            self._term_cache.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.Q_CACHE_SIZE):
            self._q_cache.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.POL_CACHE_SIZE):
            self._pi_cache.append(Scalar[dtype](0))

        # Policy output: BATCH * 2 * ACT
        for _ in range(Self.BATCH * 2 * Self.ACT):
            self._pi_out.append(Scalar[dtype](0))

        # TD targets: H * BATCH * BINS
        for _ in range(Self.BATCH_TGTS_FLAT):
            self._td_targets.append(Scalar[dtype](0))

    # ── Helper methods ────────────────────────────────────────────────────

    fn observe(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        action: InlineArray[Scalar[dtype], Self.ACT],
        reward: Scalar[dtype],
        done: Bool,
    ):
        """Push one streaming transition into the sequence replay buffer.

        Args:
            obs: Observation at this timestep.
            action: Action taken.
            reward: Reward received.
            done: Whether episode terminated.
        """
        self.buffer.add(obs, action, reward, done)

    fn is_ready(self) -> Bool:
        """Return True if buffer has enough samples for one training batch.

        Requires at least BATCH + H + 1 samples so that
        sample_sequences[BATCH, H] can find enough valid sequences.
        """
        return self.buffer.is_ready[Self.BATCH + Self.H + 1]()
