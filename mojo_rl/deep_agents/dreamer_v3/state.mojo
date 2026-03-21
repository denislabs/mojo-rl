"""DreamerV3 State Containers — CPU and GPU state for DreamerV3 training.

CPU state (DreamerV3CPUState):
  - RSSM world model (encoder, dynamics, decoder, reward/continue heads)
  - Actor-Critic networks with slow critic EMA
  - SequenceReplayBuffer for streaming obs/act/rew/done
  - Pre-allocated scratch buffers for BPTT, imagination, and actor-critic

GPU state (DreamerV3GPUState):
  - GPUNetworkState for all 13 networks (11 RSSM + actor + critic)
  - Slow critic params on device
  - DeviceBuffer scratch for observe, imagination, and training
  - Symlog bins on device

Created once in DreamerV3Agent.__init__ / make_gpu_state.
"""

from std.memory import alloc, memset
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearMish, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.deep_agents.core.replay.sequence_replay_buffer import (
    SequenceReplayBuffer,
)
from mojo_rl.nn.autodiff.composite_params import CompositeParams
from .rssm import RSSM


struct DreamerV3CPUState[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    DETER_DIM: Int = 512,
    HIDDEN: Int = 128,
    STOCH_DIM: Int = 8,
    CLASSES: Int = 8,
    UNITS: Int = 128,
    NUM_BINS: Int = 255,
    BLOCKS: Int = 4,
    WM_LR: Float64 = 1e-4,
    ACTOR_LR: Float64 = 1e-4,
    CRITIC_LR: Float64 = 3e-5,
    UNIMIX: Float64 = 0.01,
    FREE_NATS: Float64 = 1.0,
    BUFFER_CAPACITY: Int = 1000000,
    BATCH_SIZE: Int = 16,
    BATCH_LENGTH: Int = 64,
    IMAGINE_HORIZON: Int = 15,
](Movable):
    """CPU-resident state for DreamerV3 training.

    Holds all heap-allocated data needed for one DreamerV3 training loop:
      - RSSM world model with encoder, GRU dynamics, decoder, reward/continue heads
      - Actor (tanh-normal) and distributional critic with slow EMA target
      - SequenceReplayBuffer (streaming obs/act/rew/done)
      - Pre-allocated scratch buffers for BPTT observe, imagination, and actor-critic

    Parameters:
        OBS_DIM: Observation space dimension.
        ACTION_DIM: Action space dimension.
        DETER_DIM: GRU deterministic state dimension (default: 512).
        HIDDEN: GRU projection hidden dimension (default: 128).
        STOCH_DIM: Number of categorical stochastic variables (default: 8).
        CLASSES: Number of classes per stochastic variable (default: 8).
        UNITS: Actor/critic hidden layer width (default: 128).
        NUM_BINS: Number of bins for distributional critic (default: 255).
        BLOCKS: Number of blocks in RSSM (default: 4).
        WM_LR: World model learning rate (default: 1e-4).
        ACTOR_LR: Actor learning rate (default: 3e-5).
        CRITIC_LR: Critic learning rate (default: 3e-5).
        UNIMIX: Uniform mixture coefficient for categorical (default: 0.01).
        FREE_NATS: Free nats threshold for KL loss (default: 1.0).
        BUFFER_CAPACITY: Replay buffer capacity (default: 1M).
        BATCH_SIZE: Training batch size (default: 16).
        BATCH_LENGTH: Sequence length for BPTT (default: 64).
        IMAGINE_HORIZON: Imagination rollout horizon (default: 15).
    """

    # ── Derived compile-time constants ─────────────────────────────────────
    comptime STOCH_FLAT: Int = Self.STOCH_DIM * Self.CLASSES
    comptime FEAT_DIM: Int = Self.DETER_DIM + Self.STOCH_FLAT
    comptime IMAG_BATCH: Int = Self.BATCH_SIZE * Self.BATCH_LENGTH

    # ── RSSM type alias ────────────────────────────────────────────────────
    comptime RSSMType = RSSM[
        Self.OBS_DIM,
        Self.ACTION_DIM,
        Self.DETER_DIM,
        Self.HIDDEN,
        Self.STOCH_DIM,
        Self.CLASSES,
        Self.UNITS,
        Self.NUM_BINS,
        Self.BLOCKS,
        Self.WM_LR,
        Self.UNIMIX,
        Self.FREE_NATS,
    ]

    # ── Actor: feat -> tanh-normal (mean, log_std) ─────────────────────────
    comptime ActorModel = Sequential[
        LinearMish[Self.FEAT_DIM, Self.UNITS],
        LinearMish[Self.UNITS, Self.UNITS],
        Parallel[
            Linear[Self.UNITS, Self.ACTION_DIM],
            Linear[Self.UNITS, Self.ACTION_DIM],
        ],
    ]

    # ── Critic: feat -> NUM_BINS logits (distributional) ───────────────────
    comptime CriticModel = Sequential[
        LinearMish[Self.FEAT_DIM, Self.UNITS],
        LinearMish[Self.UNITS, Self.UNITS],
        Linear[Self.UNITS, Self.NUM_BINS],
    ]

    # ── Shorthand dimension constants ──────────────────────────────────────
    comptime BATCH: Int = Self.BATCH_SIZE
    comptime BL: Int = Self.BATCH_LENGTH
    comptime OBS: Int = Self.OBS_DIM
    comptime ACT: Int = Self.ACTION_DIM
    comptime DETER: Int = Self.DETER_DIM
    comptime STOCH: Int = Self.STOCH_FLAT
    comptime FEAT: Int = Self.FEAT_DIM
    comptime BINS: Int = Self.NUM_BINS
    comptime HORIZON: Int = Self.IMAGINE_HORIZON
    comptime IB: Int = Self.IMAG_BATCH

    # ── Core state ─────────────────────────────────────────────────────────

    # World model
    var rssm: Self.RSSMType

    # Actor-Critic
    var actor: NetworkState[Self.ActorModel, Adam[LR=Self.ACTOR_LR]]
    var critic: NetworkState[Self.CriticModel, Adam[LR=Self.CRITIC_LR]]
    var slow_critic_params: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    # Replay buffer
    var buffer: SequenceReplayBuffer[Self.BUFFER_CAPACITY, Self.OBS, Self.ACT]

    # ── Pre-allocated scratch buffers for sequence batch ───────────────────
    var _batch_obs: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*(BL+1)*OBS]
    var _batch_actions: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*BL*ACT]
    var _batch_rewards: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*BL]
    var _batch_dones: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*BL]

    # ── RSSM observe scratch (cached for BPTT) ────────────────────────────
    var _all_deter: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BL*BATCH*DETER]
    var _all_stoch: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BL*BATCH*STOCH]
    var _all_post_probs: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BL*BATCH*STOCH]
    var _all_prior_probs: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BL*BATCH*STOCH]
    var _all_feats: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BL*BATCH*FEAT]

    # ── GRU core scratch buffers (reused per timestep) ────────────────────
    var _proj_d: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*HIDDEN]
    var _proj_s: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*HIDDEN]
    var _proj_a: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*HIDDEN]
    var _concat_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*(DETER+3*HIDDEN)]
    var _hidden_out: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*DETER]
    var _gate_out: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*3*DETER]
    var _embed: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*STOCH]
    var _post_logits: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*STOCH]
    var _prior_logits: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*STOCH]
    var _deter_embed_concat: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*(DETER+STOCH)]
    var _symlog_obs: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*OBS]

    # ── Decoder/head scratch ──────────────────────────────────────────────
    var _dec_out: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*OBS]
    var _rew_logits: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*NUM_BINS]
    var _cont_logits: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH*1]

    # ── Imagination scratch [IMAG_BATCH x HORIZON] ────────────────────────
    var _imag_deter: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [HORIZON*IB*DETER]
    var _imag_stoch: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [HORIZON*IB*STOCH]
    var _imag_feat: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [HORIZON*IB*FEAT]
    var _imag_actions: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [HORIZON*IB*ACT]
    var _imag_log_probs: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [HORIZON*IB]
    var _imag_rewards: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [HORIZON*IB]
    var _imag_values: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [HORIZON*IB]
    var _imag_continues: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [HORIZON*IB]
    var _imag_returns: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [HORIZON*IB]

    # ── Actor scratch ─────────────────────────────────────────────────────
    var _actor_out: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [IB*2*ACT]
    var _actor_feat: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [IB*FEAT]

    # ── Critic scratch ────────────────────────────────────────────────────
    var _critic_logits: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [IB*NUM_BINS]
    var _slow_critic_logits: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [IB*NUM_BINS]

    # ── Return normalization (percentile-based EMA) ───────────────────────
    var return_ema_lo: Float64  # 5th percentile
    var return_ema_hi: Float64  # 95th percentile

    # ══════════════════════════════════════════════════════════════════════
    # Constructors
    # ══════════════════════════════════════════════════════════════════════

    fn __init__(out self):
        """Allocate RSSM, actor-critic, replay buffer, and all scratch buffers."""

        # ── Core state ────────────────────────────────────────────────────
        self.rssm = Self.RSSMType()

        # Actor
        self.actor = NetworkState[Self.ActorModel, Adam[LR=Self.ACTOR_LR]]()
        self.actor.initialize[Kaiming[]]()

        # Critic
        self.critic = NetworkState[Self.CriticModel, Adam[LR=Self.CRITIC_LR]]()
        self.critic.initialize[Kaiming[]]()

        # Slow critic (EMA copy of critic params)
        self.slow_critic_params = alloc[Scalar[dtype]](Self.CriticModel.PARAM_SIZE)
        var cp = self.critic.params
        for i in range(Self.CriticModel.PARAM_SIZE):
            self.slow_critic_params[i] = cp[i]

        # Replay buffer
        self.buffer = SequenceReplayBuffer[
            Self.BUFFER_CAPACITY, Self.OBS, Self.ACT
        ]()

        # ── Batch data scratch ────────────────────────────────────────────
        comptime BATCH_OBS_SIZE = Self.BATCH * (Self.BL + 1) * Self.OBS
        self._batch_obs = alloc[Scalar[dtype]](BATCH_OBS_SIZE)
        memset(self._batch_obs, 0, BATCH_OBS_SIZE)

        comptime BATCH_ACT_SIZE = Self.BATCH * Self.BL * Self.ACT
        self._batch_actions = alloc[Scalar[dtype]](BATCH_ACT_SIZE)
        memset(self._batch_actions, 0, BATCH_ACT_SIZE)

        comptime BATCH_SCALAR_SIZE = Self.BATCH * Self.BL
        self._batch_rewards = alloc[Scalar[dtype]](BATCH_SCALAR_SIZE)
        memset(self._batch_rewards, 0, BATCH_SCALAR_SIZE)

        self._batch_dones = alloc[Scalar[dtype]](BATCH_SCALAR_SIZE)
        memset(self._batch_dones, 0, BATCH_SCALAR_SIZE)

        # ── RSSM observe scratch ──────────────────────────────────────────
        comptime ALL_DETER_SIZE = Self.BL * Self.BATCH * Self.DETER
        self._all_deter = alloc[Scalar[dtype]](ALL_DETER_SIZE)
        memset(self._all_deter, 0, ALL_DETER_SIZE)

        comptime ALL_STOCH_SIZE = Self.BL * Self.BATCH * Self.STOCH
        self._all_stoch = alloc[Scalar[dtype]](ALL_STOCH_SIZE)
        memset(self._all_stoch, 0, ALL_STOCH_SIZE)

        self._all_post_probs = alloc[Scalar[dtype]](ALL_STOCH_SIZE)
        memset(self._all_post_probs, 0, ALL_STOCH_SIZE)

        self._all_prior_probs = alloc[Scalar[dtype]](ALL_STOCH_SIZE)
        memset(self._all_prior_probs, 0, ALL_STOCH_SIZE)

        comptime ALL_FEAT_SIZE = Self.BL * Self.BATCH * Self.FEAT
        self._all_feats = alloc[Scalar[dtype]](ALL_FEAT_SIZE)
        memset(self._all_feats, 0, ALL_FEAT_SIZE)

        # ── GRU core scratch ──────────────────────────────────────────────
        comptime PROJ_SIZE = Self.BATCH * Self.HIDDEN
        self._proj_d = alloc[Scalar[dtype]](PROJ_SIZE)
        memset(self._proj_d, 0, PROJ_SIZE)

        self._proj_s = alloc[Scalar[dtype]](PROJ_SIZE)
        memset(self._proj_s, 0, PROJ_SIZE)

        self._proj_a = alloc[Scalar[dtype]](PROJ_SIZE)
        memset(self._proj_a, 0, PROJ_SIZE)

        comptime CONCAT_SIZE = Self.BATCH * (Self.DETER + 3 * Self.HIDDEN)
        self._concat_buf = alloc[Scalar[dtype]](CONCAT_SIZE)
        memset(self._concat_buf, 0, CONCAT_SIZE)

        comptime HIDDEN_OUT_SIZE = Self.BATCH * Self.DETER
        self._hidden_out = alloc[Scalar[dtype]](HIDDEN_OUT_SIZE)
        memset(self._hidden_out, 0, HIDDEN_OUT_SIZE)

        comptime GATE_SIZE = Self.BATCH * 3 * Self.DETER
        self._gate_out = alloc[Scalar[dtype]](GATE_SIZE)
        memset(self._gate_out, 0, GATE_SIZE)

        comptime EMBED_SIZE = Self.BATCH * Self.STOCH
        self._embed = alloc[Scalar[dtype]](EMBED_SIZE)
        memset(self._embed, 0, EMBED_SIZE)

        self._post_logits = alloc[Scalar[dtype]](EMBED_SIZE)
        memset(self._post_logits, 0, EMBED_SIZE)

        self._prior_logits = alloc[Scalar[dtype]](EMBED_SIZE)
        memset(self._prior_logits, 0, EMBED_SIZE)

        comptime DETER_EMBED_SIZE = Self.BATCH * (Self.DETER + Self.STOCH)
        self._deter_embed_concat = alloc[Scalar[dtype]](DETER_EMBED_SIZE)
        memset(self._deter_embed_concat, 0, DETER_EMBED_SIZE)

        comptime SYMLOG_SIZE = Self.BATCH * Self.OBS
        self._symlog_obs = alloc[Scalar[dtype]](SYMLOG_SIZE)
        memset(self._symlog_obs, 0, SYMLOG_SIZE)

        # ── Decoder/head scratch ──────────────────────────────────────────
        comptime DEC_SIZE = Self.BATCH * Self.OBS
        self._dec_out = alloc[Scalar[dtype]](DEC_SIZE)
        memset(self._dec_out, 0, DEC_SIZE)

        comptime REW_LOGITS_SIZE = Self.BATCH * Self.BINS
        self._rew_logits = alloc[Scalar[dtype]](REW_LOGITS_SIZE)
        memset(self._rew_logits, 0, REW_LOGITS_SIZE)

        comptime CONT_SIZE = Self.BATCH * 1
        self._cont_logits = alloc[Scalar[dtype]](CONT_SIZE)
        memset(self._cont_logits, 0, CONT_SIZE)

        # ── Imagination scratch ───────────────────────────────────────────
        comptime IMAG_DETER_SIZE = Self.HORIZON * Self.IB * Self.DETER
        self._imag_deter = alloc[Scalar[dtype]](IMAG_DETER_SIZE)
        memset(self._imag_deter, 0, IMAG_DETER_SIZE)

        comptime IMAG_STOCH_SIZE = Self.HORIZON * Self.IB * Self.STOCH
        self._imag_stoch = alloc[Scalar[dtype]](IMAG_STOCH_SIZE)
        memset(self._imag_stoch, 0, IMAG_STOCH_SIZE)

        comptime IMAG_FEAT_SIZE = Self.HORIZON * Self.IB * Self.FEAT
        self._imag_feat = alloc[Scalar[dtype]](IMAG_FEAT_SIZE)
        memset(self._imag_feat, 0, IMAG_FEAT_SIZE)

        comptime IMAG_ACT_SIZE = Self.HORIZON * Self.IB * Self.ACT
        self._imag_actions = alloc[Scalar[dtype]](IMAG_ACT_SIZE)
        memset(self._imag_actions, 0, IMAG_ACT_SIZE)

        comptime IMAG_SCALAR_SIZE = Self.HORIZON * Self.IB
        self._imag_log_probs = alloc[Scalar[dtype]](IMAG_SCALAR_SIZE)
        memset(self._imag_log_probs, 0, IMAG_SCALAR_SIZE)

        self._imag_rewards = alloc[Scalar[dtype]](IMAG_SCALAR_SIZE)
        memset(self._imag_rewards, 0, IMAG_SCALAR_SIZE)

        self._imag_values = alloc[Scalar[dtype]](IMAG_SCALAR_SIZE)
        memset(self._imag_values, 0, IMAG_SCALAR_SIZE)

        self._imag_continues = alloc[Scalar[dtype]](IMAG_SCALAR_SIZE)
        memset(self._imag_continues, 0, IMAG_SCALAR_SIZE)

        self._imag_returns = alloc[Scalar[dtype]](IMAG_SCALAR_SIZE)
        memset(self._imag_returns, 0, IMAG_SCALAR_SIZE)

        # ── Actor scratch ─────────────────────────────────────────────────
        comptime ACTOR_OUT_SIZE = Self.IB * 2 * Self.ACT
        self._actor_out = alloc[Scalar[dtype]](ACTOR_OUT_SIZE)
        memset(self._actor_out, 0, ACTOR_OUT_SIZE)

        comptime ACTOR_FEAT_SIZE = Self.IB * Self.FEAT
        self._actor_feat = alloc[Scalar[dtype]](ACTOR_FEAT_SIZE)
        memset(self._actor_feat, 0, ACTOR_FEAT_SIZE)

        # ── Critic scratch ────────────────────────────────────────────────
        comptime CRITIC_SIZE = Self.IB * Self.BINS
        self._critic_logits = alloc[Scalar[dtype]](CRITIC_SIZE)
        memset(self._critic_logits, 0, CRITIC_SIZE)

        self._slow_critic_logits = alloc[Scalar[dtype]](CRITIC_SIZE)
        memset(self._slow_critic_logits, 0, CRITIC_SIZE)

        # ── Return normalization ──────────────────────────────────────────
        self.return_ema_lo = 0.0
        self.return_ema_hi = 1.0

    fn __init__(out self, *, deinit take: Self):
        """Move constructor — transfer ownership of all fields."""
        self.rssm = take.rssm^
        self.actor = take.actor^
        self.critic = take.critic^
        self.slow_critic_params = take.slow_critic_params
        self.buffer = take.buffer^

        # Batch data
        self._batch_obs = take._batch_obs
        self._batch_actions = take._batch_actions
        self._batch_rewards = take._batch_rewards
        self._batch_dones = take._batch_dones

        # RSSM observe scratch
        self._all_deter = take._all_deter
        self._all_stoch = take._all_stoch
        self._all_post_probs = take._all_post_probs
        self._all_prior_probs = take._all_prior_probs
        self._all_feats = take._all_feats

        # GRU core scratch
        self._proj_d = take._proj_d
        self._proj_s = take._proj_s
        self._proj_a = take._proj_a
        self._concat_buf = take._concat_buf
        self._hidden_out = take._hidden_out
        self._gate_out = take._gate_out
        self._embed = take._embed
        self._post_logits = take._post_logits
        self._prior_logits = take._prior_logits
        self._deter_embed_concat = take._deter_embed_concat
        self._symlog_obs = take._symlog_obs

        # Decoder/head scratch
        self._dec_out = take._dec_out
        self._rew_logits = take._rew_logits
        self._cont_logits = take._cont_logits

        # Imagination scratch
        self._imag_deter = take._imag_deter
        self._imag_stoch = take._imag_stoch
        self._imag_feat = take._imag_feat
        self._imag_actions = take._imag_actions
        self._imag_log_probs = take._imag_log_probs
        self._imag_rewards = take._imag_rewards
        self._imag_values = take._imag_values
        self._imag_continues = take._imag_continues
        self._imag_returns = take._imag_returns

        # Actor scratch
        self._actor_out = take._actor_out
        self._actor_feat = take._actor_feat

        # Critic scratch
        self._critic_logits = take._critic_logits
        self._slow_critic_logits = take._slow_critic_logits

        # Return normalization
        self.return_ema_lo = take.return_ema_lo
        self.return_ema_hi = take.return_ema_hi

    # ══════════════════════════════════════════════════════════════════════
    # Helper methods
    # ══════════════════════════════════════════════════════════════════════

    fn is_ready(self) -> Bool:
        """Return True if buffer has enough samples for one training batch.

        Requires at least BATCH_SIZE + BATCH_LENGTH + 1 samples so that
        sample_sequences can find enough valid sequences.
        """
        return self.buffer.is_ready[Self.BATCH_SIZE + Self.BATCH_LENGTH + 1]()

    fn slow_critic_update(mut self, tau: Float64):
        """EMA update: slow_params = (1-tau) * slow_params + tau * critic_params.

        Args:
            tau: Interpolation coefficient (typically 0.02).
        """
        var cp = self.critic.params
        for i in range(Self.CriticModel.PARAM_SIZE):
            var s = Float64(self.slow_critic_params[i])
            var c = Float64(cp[i])
            self.slow_critic_params[i] = Scalar[dtype]((1.0 - tau) * s + tau * c)


# =============================================================================
# GPU State
# =============================================================================


struct DreamerV3GPUState[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    DETER_DIM: Int = 512,
    HIDDEN: Int = 128,
    STOCH_DIM: Int = 8,
    CLASSES: Int = 8,
    UNITS: Int = 128,
    NUM_BINS: Int = 255,
    BLOCKS: Int = 4,
    WM_LR: Float64 = 1e-4,
    ACTOR_LR: Float64 = 1e-4,
    CRITIC_LR: Float64 = 3e-5,
    UNIMIX: Float64 = 0.01,
    FREE_NATS: Float64 = 1.0,
    BATCH_SIZE: Int = 16,
    BATCH_LENGTH: Int = 64,
    IMAGINE_HORIZON: Int = 15,
    MAX_N_ENVS: Int = 64,
](Movable):
    """GPU-resident state for DreamerV3 training.

    Holds all device buffers for GPU training: network states (params, grads,
    optimizer state), scratch buffers for observe/imagination phases, and
    symlog bins. SequenceReplayBuffer stays on CPU — sampled batches are
    uploaded to GPU per training step.
    """

    # ── Derived compile-time constants ─────────────────────────────────────
    comptime STOCH_FLAT: Int = Self.STOCH_DIM * Self.CLASSES
    comptime FEAT_DIM: Int = Self.DETER_DIM + Self.STOCH_FLAT
    comptime IMAG_BATCH: Int = Self.BATCH_SIZE * Self.BATCH_LENGTH

    # ── Shorthand ────────────────────────────────────────────────────────
    comptime BATCH: Int = Self.BATCH_SIZE
    comptime BL: Int = Self.BATCH_LENGTH
    comptime OBS: Int = Self.OBS_DIM
    comptime ACT: Int = Self.ACTION_DIM
    comptime DETER: Int = Self.DETER_DIM
    comptime STOCH: Int = Self.STOCH_FLAT
    comptime FEAT: Int = Self.FEAT_DIM
    comptime BINS: Int = Self.NUM_BINS
    comptime HORIZON: Int = Self.IMAGINE_HORIZON
    comptime IB: Int = Self.IMAG_BATCH
    comptime MAX_N: Int = Self.MAX_N_ENVS

    # ── RSSM type alias (same as CPUState) ────────────────────────────────
    comptime RSSMType = RSSM[
        Self.OBS_DIM, Self.ACTION_DIM, Self.DETER_DIM, Self.HIDDEN,
        Self.STOCH_DIM, Self.CLASSES, Self.UNITS, Self.NUM_BINS,
        Self.BLOCKS, Self.WM_LR, Self.UNIMIX, Self.FREE_NATS,
    ]

    # ── Model type aliases (from RSSM + CPUState) ────────────────────────
    comptime ActorModel = Sequential[
        LinearMish[Self.FEAT_DIM, Self.UNITS],
        LinearMish[Self.UNITS, Self.UNITS],
        Parallel[
            Linear[Self.UNITS, Self.ACTION_DIM],
            Linear[Self.UNITS, Self.ACTION_DIM],
        ],
    ]
    comptime CriticModel = Sequential[
        LinearMish[Self.FEAT_DIM, Self.UNITS],
        LinearMish[Self.UNITS, Self.UNITS],
        Linear[Self.UNITS, Self.NUM_BINS],
    ]

    # Network wrapper type aliases
    comptime WMOpt = Adam[LR=Self.WM_LR]
    comptime ActorOpt = Adam[LR=Self.ACTOR_LR]
    comptime CriticOpt = Adam[LR=Self.CRITIC_LR]

    # ── RSSM GPU network states (11 networks) ────────────────────────────
    var encoder: GPUNetworkState[Self.RSSMType.EncModel, Self.WMOpt]
    var posterior: GPUNetworkState[Self.RSSMType.PostModel, Self.WMOpt]
    var prior: GPUNetworkState[Self.RSSMType.PriorModel, Self.WMOpt]
    var decoder: GPUNetworkState[Self.RSSMType.DecModel, Self.WMOpt]
    var reward_head: GPUNetworkState[Self.RSSMType.RewModel, Self.WMOpt]
    var continue_head: GPUNetworkState[Self.RSSMType.ContModel, Self.WMOpt]
    var deter_proj: GPUNetworkState[Self.RSSMType.DeterProj, Self.WMOpt]
    var stoch_proj: GPUNetworkState[Self.RSSMType.StochProj, Self.WMOpt]
    var action_proj: GPUNetworkState[Self.RSSMType.ActionProj, Self.WMOpt]
    var gru_hidden: GPUNetworkState[Self.RSSMType.GRUHiddenModel, Self.WMOpt]
    var gru_gates: GPUNetworkState[Self.RSSMType.GRUGateModel, Self.WMOpt]

    # ── Actor-Critic GPU network states ──────────────────────────────────
    var actor: GPUNetworkState[Self.ActorModel, Self.ActorOpt]
    var critic: GPUNetworkState[Self.CriticModel, Self.CriticOpt]
    var slow_critic: GPUNetworkState[Self.CriticModel, Self.CriticOpt]

    # ── Symlog bins on device ────────────────────────────────────────────
    var bins_buf: DeviceBuffer[dtype]  # [NUM_BINS]

    # ── Batch data (uploaded from CPU per train step) ────────────────────
    var batch_obs: DeviceBuffer[dtype]  # [BATCH * (BL+1) * OBS]
    var batch_actions: DeviceBuffer[dtype]  # [BATCH * BL * ACT]
    var batch_rewards: DeviceBuffer[dtype]  # [BATCH * BL]
    var batch_dones: DeviceBuffer[dtype]  # [BATCH * BL]

    # ── RSSM observe scratch (per-timestep, reused) ──────────────────────
    var deter_buf: DeviceBuffer[dtype]  # [BATCH * DETER]
    var stoch_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]
    var new_deter_buf: DeviceBuffer[dtype]  # [BATCH * DETER]
    var new_stoch_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]
    var post_probs_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]
    var prior_probs_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]
    var feat_buf: DeviceBuffer[dtype]  # [BATCH * FEAT]
    var obs_step_buf: DeviceBuffer[dtype]  # [BATCH * OBS]
    var act_step_buf: DeviceBuffer[dtype]  # [BATCH * ACT]
    var symlog_obs_buf: DeviceBuffer[dtype]  # [BATCH * OBS]
    var embed_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]
    var post_in_buf: DeviceBuffer[dtype]  # [BATCH * (DETER + STOCH)]
    var post_logits_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]
    var prior_logits_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]
    var norm_action_buf: DeviceBuffer[dtype]  # [BATCH * ACT]

    # GRU scratch
    var proj_d_buf: DeviceBuffer[dtype]  # [BATCH * HIDDEN]
    var proj_s_buf: DeviceBuffer[dtype]  # [BATCH * HIDDEN]
    var proj_a_buf: DeviceBuffer[dtype]  # [BATCH * HIDDEN]
    var concat_buf: DeviceBuffer[dtype]  # [BATCH * (DETER + 3*HIDDEN)]
    var hidden_out_buf: DeviceBuffer[dtype]  # [BATCH * DETER]
    var gate_out_buf: DeviceBuffer[dtype]  # [BATCH * 3*DETER]

    # Dummy stoch for prior sampling
    var dummy_stoch_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]

    # ── Cached observe results (all timesteps for imagination) ───────────
    var all_deter_buf: DeviceBuffer[dtype]  # [BL * BATCH * DETER]
    var all_stoch_buf: DeviceBuffer[dtype]  # [BL * BATCH * STOCH]
    var all_post_probs_buf: DeviceBuffer[dtype]  # [BL * BATCH * STOCH]
    var all_prior_probs_buf: DeviceBuffer[dtype]  # [BL * BATCH * STOCH]
    var all_feats_buf: DeviceBuffer[dtype]  # [BL * BATCH * FEAT]
    var all_embed_buf: DeviceBuffer[dtype]  # [BL * BATCH * STOCH]

    # ── Decoder/head scratch ─────────────────────────────────────────────
    var dec_out_buf: DeviceBuffer[dtype]  # [BATCH * OBS]
    var rew_logits_buf: DeviceBuffer[dtype]  # [max(BATCH, IB) * BINS]
    var cont_out_buf: DeviceBuffer[dtype]  # [max(BATCH, IB) * 1]
    var kl_buf: DeviceBuffer[dtype]  # [BATCH] (per-element KL)

    # ── Imagination GRU scratch (IB-sized) ───────────────────────────────
    var imag_proj_d_buf: DeviceBuffer[dtype]  # [IB * HIDDEN]
    var imag_proj_s_buf: DeviceBuffer[dtype]  # [IB * HIDDEN]
    var imag_proj_a_buf: DeviceBuffer[dtype]  # [IB * HIDDEN]
    var imag_concat_buf: DeviceBuffer[dtype]  # [IB * (DETER + 3*HIDDEN)]
    var imag_hidden_buf: DeviceBuffer[dtype]  # [IB * DETER]
    var imag_gate_buf: DeviceBuffer[dtype]  # [IB * 3*DETER]
    var imag_norm_act_buf: DeviceBuffer[dtype]  # [IB * ACT]
    var imag_prior_logits_buf: DeviceBuffer[dtype]  # [IB * STOCH]
    var imag_prior_probs_buf: DeviceBuffer[dtype]  # [IB * STOCH]

    # ── Imagination scratch [HORIZON x IB] ───────────────────────────────
    # Using IB-sized buffers for all imagination steps
    var imag_deter_buf: DeviceBuffer[dtype]  # [2 * IB * DETER] (ping-pong)
    var imag_stoch_buf: DeviceBuffer[dtype]  # [2 * IB * STOCH]
    var imag_feat_buf: DeviceBuffer[dtype]  # [IB * FEAT]

    # Imagination scalars
    var imag_rewards_buf: DeviceBuffer[dtype]  # [HORIZON * IB]
    var imag_values_buf: DeviceBuffer[dtype]  # [HORIZON * IB]
    var imag_continues_buf: DeviceBuffer[dtype]  # [HORIZON * IB]
    var imag_returns_buf: DeviceBuffer[dtype]  # [HORIZON * IB]
    var imag_actions_buf: DeviceBuffer[dtype]  # [IB * ACT]
    var imag_log_probs_buf: DeviceBuffer[dtype]  # [IB]
    var imag_advantages_buf: DeviceBuffer[dtype]  # [IB]

    # Per-horizon deter/stoch/actions (for multi-step actor-critic training)
    var imag_all_deter_buf: DeviceBuffer[dtype]  # [HORIZON * IB * DETER]
    var imag_all_stoch_buf: DeviceBuffer[dtype]  # [HORIZON * IB * STOCH]
    var imag_all_actions_buf: DeviceBuffer[dtype]  # [HORIZON * IB * ACT]

    # Per-horizon imagination caches (for dynamics backprop through actor)
    var imag_actor_cache_buf: DeviceBuffer[dtype]  # [HORIZON * IB * ActorModel.CACHE_SIZE]
    var imag_aproj_cache_buf: DeviceBuffer[dtype]  # [HORIZON * IB * ActionProj.CACHE_SIZE]
    var imag_gh_cache_buf: DeviceBuffer[dtype]  # [HORIZON * IB * GRUHiddenModel.CACHE_SIZE]
    var imag_gg_cache_buf: DeviceBuffer[dtype]  # [HORIZON * IB * GRUGateModel.CACHE_SIZE]
    var imag_prior_cache_buf: DeviceBuffer[dtype]  # [HORIZON * IB * PriorModel.CACHE_SIZE]
    var imag_rew_cache_buf: DeviceBuffer[dtype]  # [HORIZON * IB * RewModel.CACHE_SIZE]
    # Per-horizon saved gate_out for GRU backward
    var imag_gate_out_save_buf: DeviceBuffer[dtype]  # [HORIZON * IB * 3*DETER]
    # Per-horizon actor_out for reparameterization backward
    var imag_actor_out_save_buf: DeviceBuffer[dtype]  # [HORIZON * IB * 2*ACT]

    # ── Actor/Critic GPU scratch (IB-sized for imagination) ──────────────
    var actor_out_buf: DeviceBuffer[dtype]  # [IB * 2*ACT]
    var actor_cache_buf: DeviceBuffer[dtype]  # [IB * ActorModel.CACHE_SIZE]
    var actor_grad_buf: DeviceBuffer[dtype]  # [IB * 2*ACT]
    var actor_grad_in_buf: DeviceBuffer[dtype]  # [IB * FEAT]
    var critic_logits_buf: DeviceBuffer[dtype]  # [IB * BINS]
    var critic_cache_buf: DeviceBuffer[dtype]  # [IB * CriticModel.CACHE_SIZE]
    var critic_grad_buf: DeviceBuffer[dtype]  # [IB * BINS]
    var critic_grad_in_buf: DeviceBuffer[dtype]  # [IB * FEAT]
    var two_hot_targets_buf: DeviceBuffer[dtype]  # [IB * BINS]
    var symlog_returns_buf: DeviceBuffer[dtype]  # [IB]
    var returns_minmax_buf: DeviceBuffer[dtype]  # [2] (min, max)

    # ── Per-timestep host buffers (pre-allocated for observe loop) ────────
    var host_obs_step_buf: HostBuffer[dtype]  # [BATCH * OBS]
    var host_act_step_buf: HostBuffer[dtype]  # [BATCH * ACT]
    var host_target_buf: HostBuffer[dtype]  # [BATCH * OBS]
    var host_rew_symlog_step_buf: HostBuffer[dtype]  # [BATCH]
    var host_cont_target_step_buf: HostBuffer[dtype]  # [BATCH]

    # ── Pre-allocated host buffers for training diagnostics ────────────
    var host_dec_diag_buf: HostBuffer[dtype]  # [BATCH * OBS]
    var host_rew_diag_buf: HostBuffer[dtype]  # [BATCH * NUM_BINS]
    var host_cont_diag_buf: HostBuffer[dtype]  # [BATCH]
    var host_kl_diag_buf: HostBuffer[dtype]  # [BATCH]
    var host_minmax_buf: HostBuffer[dtype]  # [2]

    # ── Pre-allocated host buffers for batch upload ────────────────────
    var host_upload_obs_buf: HostBuffer[dtype]  # [BATCH * (BL+1) * OBS]
    var host_upload_act_buf: HostBuffer[dtype]  # [BATCH * BL * ACT]
    var host_upload_rew_buf: HostBuffer[dtype]  # [BATCH * BL]
    var host_upload_done_buf: HostBuffer[dtype]  # [BATCH * BL]

    # ── Pre-allocated host buffers for imagination diagnostics ──────
    var host_diag_imag_buf: HostBuffer[dtype]  # [HORIZON * IB] (reused for rew/ret/val)
    var host_diag_actor_buf: HostBuffer[dtype]  # [ActorModel.PARAM_SIZE] (reused for grads/params)

    # ── Pre-allocated host buffer for bins upload ─────────────────────
    var host_bins_buf: HostBuffer[dtype]  # [NUM_BINS]

    # ── Decoder backward scratch ─────────────────────────────────────────
    var dec_cache_buf: DeviceBuffer[dtype]  # [BATCH * DecModel.CACHE_SIZE]
    var dec_grad_out_buf: DeviceBuffer[dtype]  # [BATCH * OBS]
    var dec_grad_in_buf: DeviceBuffer[dtype]  # [BATCH * FEAT]
    var dec_target_buf: DeviceBuffer[dtype]  # [BATCH * OBS]

    # ── Continue backward scratch ────────────────────────────────────────
    var cont_target_buf: DeviceBuffer[dtype]  # [BATCH * 1]
    var cont_grad_buf: DeviceBuffer[dtype]  # [BATCH * 1]

    # ── Reward backward scratch ───────────────────────────────────────────
    var rew_cache_buf: DeviceBuffer[dtype]  # [BATCH * RewModel.CACHE_SIZE]
    var rew_target_buf: DeviceBuffer[dtype]  # [BATCH * NUM_BINS]
    var rew_grad_out_buf: DeviceBuffer[dtype]  # [BATCH * NUM_BINS]
    var rew_grad_in_buf: DeviceBuffer[dtype]  # [BATCH * FEAT]
    var rew_symlog_buf: DeviceBuffer[dtype]  # [BATCH]

    # ── Continue backward cache ───────────────────────────────────────────
    var cont_cache_buf: DeviceBuffer[dtype]  # [BATCH * ContModel.CACHE_SIZE]
    var cont_grad_in_buf: DeviceBuffer[dtype]  # [BATCH * FEAT]

    # ── Posterior backward scratch ────────────────────────────────────────
    var post_cache_buf: DeviceBuffer[dtype]  # [BATCH * PostModel.CACHE_SIZE]
    var post_grad_out_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]
    var post_grad_in_buf: DeviceBuffer[dtype]  # [BATCH * (DETER + STOCH)]

    # ── Prior backward scratch ────────────────────────────────────────────
    var prior_cache_buf: DeviceBuffer[dtype]  # [BATCH * PriorModel.CACHE_SIZE]
    var prior_grad_out_buf: DeviceBuffer[dtype]  # [BATCH * STOCH]
    var prior_grad_in_buf: DeviceBuffer[dtype]  # [BATCH * DETER]

    # ── BPTT per-timestep caches (BL * B * CACHE_SIZE) ────────────────────
    var all_enc_cache_buf: DeviceBuffer[dtype]
    var all_dproj_cache_buf: DeviceBuffer[dtype]
    var all_sproj_cache_buf: DeviceBuffer[dtype]
    var all_aproj_cache_buf: DeviceBuffer[dtype]
    var all_gru_hidden_cache_buf: DeviceBuffer[dtype]
    var all_gru_gates_cache_buf: DeviceBuffer[dtype]
    var all_post_cache_buf: DeviceBuffer[dtype]
    var all_prior_cache_buf: DeviceBuffer[dtype]

    # ── BPTT per-timestep saved activations ───────────────────────────────
    var all_gate_out_buf: DeviceBuffer[dtype]  # [BL * B * 3*DETER]
    var all_prev_deter_buf: DeviceBuffer[dtype]  # [BL * B * DETER]
    var all_symlog_obs_buf: DeviceBuffer[dtype]  # [BL * B * OBS]
    var all_norm_action_buf: DeviceBuffer[dtype]  # [BL * B * ACT]
    var all_d_feat_buf: DeviceBuffer[dtype]  # [BL * B * FEAT]

    # ── BPTT gradient scratch (reused across backward timesteps) ──────────
    var d_feat_buf: DeviceBuffer[dtype]  # [B * FEAT]
    var d_deter_total_buf: DeviceBuffer[dtype]  # [B * DETER]
    var d_stoch_feat_buf: DeviceBuffer[dtype]  # [B * STOCH]
    var d_gate_out_bwd_buf: DeviceBuffer[dtype]  # [B * 3*DETER]
    var d_prev_deter_gru_buf: DeviceBuffer[dtype]  # [B * DETER]
    var d_concat_bwd_buf: DeviceBuffer[dtype]  # [B * (DETER+3*HIDDEN)]
    var d_proj_d_bwd_buf: DeviceBuffer[dtype]  # [B * HIDDEN]
    var d_proj_s_bwd_buf: DeviceBuffer[dtype]  # [B * HIDDEN]
    var d_proj_a_bwd_buf: DeviceBuffer[dtype]  # [B * HIDDEN]
    var d_hidden_out_bwd_buf: DeviceBuffer[dtype]  # [B * DETER]
    var d_embed_bwd_buf: DeviceBuffer[dtype]  # [B * STOCH]
    var d_symlog_obs_bwd_buf: DeviceBuffer[dtype]  # [B * OBS]
    var d_prev_deter_dproj_buf: DeviceBuffer[dtype]  # [B * DETER]
    var d_prev_stoch_bwd_buf: DeviceBuffer[dtype]  # [B * STOCH]
    var d_prev_action_bwd_buf: DeviceBuffer[dtype]  # [B * ACT]
    var d_recurrent_deter_buf: DeviceBuffer[dtype]  # [B * DETER]
    var d_recurrent_stoch_buf: DeviceBuffer[dtype]  # [B * STOCH]
    var d_post_logits_total_buf: DeviceBuffer[dtype]  # [B * STOCH]
    var d_deter_from_post_buf: DeviceBuffer[dtype]  # [B * DETER]

    # ── Combined prediction heads (ComputeGraph) ─────────────────────────
    # HeadsGraph = ComputeGraph[decoder, reward, continue] with 3-way fan-out
    # Used in BPTT backward instead of 3 separate forward/backward calls
    comptime HeadsGraph = Self.RSSMType.HeadsGraph
    comptime HeadsCP = Self.RSSMType.HeadsCP
    comptime HEADS_OUT_DIM: Int = Self.RSSMType.HEADS_OUT_DIM
    comptime HEADS_CACHE_SIZE: Int = Self.RSSMType.HEADS_CACHE_SIZE
    var heads_params_buf: DeviceBuffer[dtype]  # [HeadsGraph.PARAM_SIZE]
    var heads_grads_buf: DeviceBuffer[dtype]  # [HeadsGraph.PARAM_SIZE]
    var heads_cache_buf: DeviceBuffer[dtype]  # [BATCH * HEADS_CACHE_SIZE]
    var heads_out_buf: DeviceBuffer[dtype]  # [BATCH * HEADS_OUT_DIM]
    var heads_grad_out_buf: DeviceBuffer[dtype]  # [BATCH * HEADS_OUT_DIM]
    var ws_heads: DeviceBuffer[dtype]

    # ── Network workspace buffers ────────────────────────────────────────
    # Sized for the maximum batch dimension (IB for imagination phase)
    var ws_encoder: DeviceBuffer[dtype]
    var ws_posterior: DeviceBuffer[dtype]
    var ws_prior: DeviceBuffer[dtype]
    var ws_decoder: DeviceBuffer[dtype]
    var ws_reward: DeviceBuffer[dtype]
    var ws_continue: DeviceBuffer[dtype]
    var ws_deter_proj: DeviceBuffer[dtype]
    var ws_stoch_proj: DeviceBuffer[dtype]
    var ws_action_proj: DeviceBuffer[dtype]
    var ws_gru_hidden: DeviceBuffer[dtype]
    var ws_gru_gates: DeviceBuffer[dtype]
    var ws_actor: DeviceBuffer[dtype]
    var ws_critic: DeviceBuffer[dtype]

    # ── Inference buffers (for select_actions_gpu) ───────────────────────
    var inf_obs_buf: DeviceBuffer[dtype]  # [MAX_N * OBS]
    var inf_deter_buf: DeviceBuffer[dtype]  # [MAX_N * DETER]
    var inf_stoch_buf: DeviceBuffer[dtype]  # [MAX_N * STOCH]
    var inf_action_buf: DeviceBuffer[dtype]  # [MAX_N * ACT]
    var inf_feat_buf: DeviceBuffer[dtype]  # [MAX_N * FEAT]
    var inf_actor_out_buf: DeviceBuffer[dtype]  # [MAX_N * 2*ACT]

    # ── Gradient clipping scratch ─────────────────────────────────────────
    var grad_partial_sums_buf: DeviceBuffer[dtype]  # [1024] (shared across nets)

    # ── Pinned host buffers for batch upload ─────────────────────────────
    var host_batch_obs: DeviceBuffer[dtype]  # pinned [BATCH * (BL+1) * OBS]
    var host_batch_actions: DeviceBuffer[dtype]  # pinned [BATCH * BL * ACT]
    var host_batch_rewards: DeviceBuffer[dtype]  # pinned [BATCH * BL]
    var host_batch_dones: DeviceBuffer[dtype]  # pinned [BATCH * BL]

    # ══════════════════════════════════════════════════════════════════════
    # Constructor
    # ══════════════════════════════════════════════════════════════════════

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers."""
        # ── RSSM network states ──────────────────────────────────────────
        self.encoder = GPUNetworkState[Self.RSSMType.EncModel, Self.WMOpt](ctx)
        self.posterior = GPUNetworkState[Self.RSSMType.PostModel, Self.WMOpt](ctx)
        self.prior = GPUNetworkState[Self.RSSMType.PriorModel, Self.WMOpt](ctx)
        self.decoder = GPUNetworkState[Self.RSSMType.DecModel, Self.WMOpt](ctx)
        self.reward_head = GPUNetworkState[Self.RSSMType.RewModel, Self.WMOpt](ctx)
        self.continue_head = GPUNetworkState[Self.RSSMType.ContModel, Self.WMOpt](ctx)
        self.deter_proj = GPUNetworkState[Self.RSSMType.DeterProj, Self.WMOpt](ctx)
        self.stoch_proj = GPUNetworkState[Self.RSSMType.StochProj, Self.WMOpt](ctx)
        self.action_proj = GPUNetworkState[Self.RSSMType.ActionProj, Self.WMOpt](ctx)
        self.gru_hidden = GPUNetworkState[Self.RSSMType.GRUHiddenModel, Self.WMOpt](ctx)
        self.gru_gates = GPUNetworkState[Self.RSSMType.GRUGateModel, Self.WMOpt](ctx)

        # ── Actor-Critic ─────────────────────────────────────────────────
        self.actor = GPUNetworkState[Self.ActorModel, Self.ActorOpt](ctx)
        self.critic = GPUNetworkState[Self.CriticModel, Self.CriticOpt](ctx)
        self.slow_critic = GPUNetworkState[Self.CriticModel, Self.CriticOpt](ctx)

        # ── Symlog bins ──────────────────────────────────────────────────
        self.bins_buf = ctx.enqueue_create_buffer[dtype](Self.BINS)

        # ── Batch data ───────────────────────────────────────────────────
        self.batch_obs = ctx.enqueue_create_buffer[dtype](Self.BATCH * (Self.BL + 1) * Self.OBS)
        self.batch_actions = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.BL * Self.ACT)
        self.batch_rewards = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.BL)
        self.batch_dones = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.BL)

        # ── Observe scratch ──────────────────────────────────────────────
        self.deter_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)
        self.stoch_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.new_deter_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)
        self.new_stoch_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.post_probs_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.prior_probs_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.feat_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.FEAT)
        self.obs_step_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.OBS)
        self.act_step_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.ACT)
        self.symlog_obs_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.OBS)
        self.embed_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.post_in_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * (Self.DETER + Self.STOCH))
        self.post_logits_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.prior_logits_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.norm_action_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.ACT)

        # GRU scratch
        self.proj_d_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.HIDDEN)
        self.proj_s_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.HIDDEN)
        self.proj_a_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.HIDDEN)
        self.concat_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * (Self.DETER + 3 * Self.HIDDEN))
        self.hidden_out_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)
        self.gate_out_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * 3 * Self.DETER)
        self.dummy_stoch_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)

        # ── Cached observe ───────────────────────────────────────────────
        self.all_deter_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.DETER)
        self.all_stoch_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.STOCH)
        self.all_post_probs_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.STOCH)
        self.all_prior_probs_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.STOCH)
        self.all_feats_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.FEAT)
        self.all_embed_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.STOCH)

        # ── Decoder/head scratch ─────────────────────────────────────────
        self.dec_out_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.OBS)
        comptime MAX_BATCH_BINS = Self.IB * Self.BINS  # IB > BATCH, so IB is max
        self.rew_logits_buf = ctx.enqueue_create_buffer[dtype](MAX_BATCH_BINS)
        comptime MAX_BATCH_1 = Self.IB
        self.cont_out_buf = ctx.enqueue_create_buffer[dtype](MAX_BATCH_1)
        self.kl_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH)

        # ── Imagination GRU scratch (IB-sized) ──────────────────────────
        self.imag_proj_d_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.HIDDEN)
        self.imag_proj_s_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.HIDDEN)
        self.imag_proj_a_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.HIDDEN)
        self.imag_concat_buf = ctx.enqueue_create_buffer[dtype](Self.IB * (Self.DETER + 3 * Self.HIDDEN))
        self.imag_hidden_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.DETER)
        self.imag_gate_buf = ctx.enqueue_create_buffer[dtype](Self.IB * 3 * Self.DETER)
        self.imag_norm_act_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.ACT)
        self.imag_prior_logits_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.STOCH)
        self.imag_prior_probs_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.STOCH)

        # ── Imagination scratch ──────────────────────────────────────────
        self.imag_deter_buf = ctx.enqueue_create_buffer[dtype](2 * Self.IB * Self.DETER)
        self.imag_stoch_buf = ctx.enqueue_create_buffer[dtype](2 * Self.IB * Self.STOCH)
        self.imag_feat_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.FEAT)

        comptime IMAG_SCALAR = Self.HORIZON * Self.IB
        self.imag_rewards_buf = ctx.enqueue_create_buffer[dtype](IMAG_SCALAR)
        self.imag_values_buf = ctx.enqueue_create_buffer[dtype](IMAG_SCALAR)
        self.imag_continues_buf = ctx.enqueue_create_buffer[dtype](IMAG_SCALAR)
        self.imag_returns_buf = ctx.enqueue_create_buffer[dtype](IMAG_SCALAR)
        self.imag_actions_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.ACT)
        self.imag_log_probs_buf = ctx.enqueue_create_buffer[dtype](Self.IB)
        self.imag_advantages_buf = ctx.enqueue_create_buffer[dtype](Self.IB)
        self.imag_all_deter_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * Self.DETER)
        self.imag_all_stoch_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * Self.STOCH)
        self.imag_all_actions_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * Self.ACT)
        self.imag_actor_cache_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * Self.ActorModel.CACHE_SIZE)
        self.imag_aproj_cache_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * Self.RSSMType.ActionProj.CACHE_SIZE)
        self.imag_gh_cache_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * Self.RSSMType.GRUHiddenModel.CACHE_SIZE)
        self.imag_gg_cache_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * Self.RSSMType.GRUGateModel.CACHE_SIZE)
        self.imag_prior_cache_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * Self.RSSMType.PriorModel.CACHE_SIZE)
        self.imag_rew_cache_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * Self.RSSMType.RewModel.CACHE_SIZE)
        self.imag_gate_out_save_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * 3 * Self.DETER)
        self.imag_actor_out_save_buf = ctx.enqueue_create_buffer[dtype](Self.HORIZON * Self.IB * 2 * Self.ACT)

        # ── Actor/Critic scratch ─────────────────────────────────────────
        comptime ACTOR_OUT_DIM = Self.ActorModel.OUT_DIM
        self.actor_out_buf = ctx.enqueue_create_buffer[dtype](Self.IB * ACTOR_OUT_DIM)
        self.actor_cache_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.ActorModel.CACHE_SIZE)
        self.actor_grad_buf = ctx.enqueue_create_buffer[dtype](Self.IB * ACTOR_OUT_DIM)
        self.actor_grad_in_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.FEAT)
        self.critic_logits_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.BINS)
        self.critic_cache_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.CriticModel.CACHE_SIZE)
        self.critic_grad_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.BINS)
        self.critic_grad_in_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.FEAT)
        self.two_hot_targets_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.BINS)
        self.symlog_returns_buf = ctx.enqueue_create_buffer[dtype](Self.IB)
        self.returns_minmax_buf = ctx.enqueue_create_buffer[dtype](2)

        # ── Pre-allocated host buffers for observe loop ──────────────────
        self.host_obs_step_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.OBS)
        self.host_act_step_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.ACT)
        self.host_target_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.OBS)
        self.host_rew_symlog_step_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH)
        self.host_cont_target_step_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH)

        # ── Pre-allocated host buffers for training diagnostics ───────
        self.host_dec_diag_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.OBS)
        self.host_rew_diag_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.BINS)
        self.host_cont_diag_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH)
        self.host_kl_diag_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH)
        self.host_minmax_buf = ctx.enqueue_create_host_buffer[dtype](2)

        # ── Pre-allocated host buffers for batch upload ───────────────
        self.host_upload_obs_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * (Self.BL + 1) * Self.OBS)
        self.host_upload_act_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.BL * Self.ACT)
        self.host_upload_rew_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.BL)
        self.host_upload_done_buf = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.BL)

        # ── Pre-allocated host buffers for imagination diagnostics ────
        self.host_diag_imag_buf = ctx.enqueue_create_host_buffer[dtype](Self.HORIZON * Self.IB)
        self.host_diag_actor_buf = ctx.enqueue_create_host_buffer[dtype](Self.ActorModel.PARAM_SIZE)

        # ── Pre-allocated host buffer for bins upload ─────────────────
        self.host_bins_buf = ctx.enqueue_create_host_buffer[dtype](Self.BINS)

        # ── Decoder backward scratch ─────────────────────────────────────
        self.dec_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.RSSMType.DecModel.CACHE_SIZE)
        self.dec_grad_out_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.OBS)
        self.dec_grad_in_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.FEAT)
        self.dec_target_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.OBS)

        # ── Continue backward scratch ────────────────────────────────────
        self.cont_target_buf = ctx.enqueue_create_buffer[dtype](Self.IB)
        self.cont_grad_buf = ctx.enqueue_create_buffer[dtype](Self.IB)

        # ── Reward backward scratch ──────────────────────────────────────
        self.rew_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.RSSMType.RewModel.CACHE_SIZE)
        self.rew_target_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.BINS)
        self.rew_grad_out_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.BINS)
        self.rew_grad_in_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.FEAT)
        self.rew_symlog_buf = ctx.enqueue_create_buffer[dtype](Self.IB)

        # ── Continue backward cache ──────────────────────────────────────
        self.cont_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.RSSMType.ContModel.CACHE_SIZE)
        self.cont_grad_in_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.FEAT)

        # ── Posterior backward scratch ───────────────────────────────────
        self.post_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.RSSMType.PostModel.CACHE_SIZE)
        self.post_grad_out_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.post_grad_in_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * (Self.DETER + Self.STOCH))

        # ── Prior backward scratch ───────────────────────────────────────
        self.prior_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.RSSMType.PriorModel.CACHE_SIZE)
        self.prior_grad_out_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.prior_grad_in_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)

        # ── BPTT per-timestep caches ──────────────────────────────────────
        self.all_enc_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.RSSMType.EncModel.CACHE_SIZE)
        self.all_dproj_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.RSSMType.DeterProj.CACHE_SIZE)
        self.all_sproj_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.RSSMType.StochProj.CACHE_SIZE)
        self.all_aproj_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.RSSMType.ActionProj.CACHE_SIZE)
        self.all_gru_hidden_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.RSSMType.GRUHiddenModel.CACHE_SIZE)
        self.all_gru_gates_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.RSSMType.GRUGateModel.CACHE_SIZE)
        self.all_post_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.RSSMType.PostModel.CACHE_SIZE)
        self.all_prior_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.RSSMType.PriorModel.CACHE_SIZE)

        # ── BPTT per-timestep saved activations ──────────────────────────
        self.all_gate_out_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * 3 * Self.DETER)
        self.all_prev_deter_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.DETER)
        self.all_symlog_obs_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.OBS)
        self.all_norm_action_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.ACT)
        self.all_d_feat_buf = ctx.enqueue_create_buffer[dtype](Self.BL * Self.BATCH * Self.FEAT)

        # ── BPTT gradient scratch ─────────────────────────────────────────
        self.d_feat_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.FEAT)
        self.d_deter_total_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)
        self.d_stoch_feat_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.d_gate_out_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * 3 * Self.DETER)
        self.d_prev_deter_gru_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)
        self.d_concat_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * (Self.DETER + 3 * Self.HIDDEN))
        self.d_proj_d_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.HIDDEN)
        self.d_proj_s_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.HIDDEN)
        self.d_proj_a_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.HIDDEN)
        self.d_hidden_out_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)
        self.d_embed_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.d_symlog_obs_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.OBS)
        self.d_prev_deter_dproj_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)
        self.d_prev_stoch_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.d_prev_action_bwd_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.ACT)
        self.d_recurrent_deter_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)
        self.d_recurrent_stoch_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.d_post_logits_total_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.STOCH)
        self.d_deter_from_post_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.DETER)

        # ── Combined prediction heads (ComputeGraph) ─────────────────────
        self.heads_params_buf = ctx.enqueue_create_buffer[dtype](Self.HeadsGraph.PARAM_SIZE)
        self.heads_grads_buf = ctx.enqueue_create_buffer[dtype](Self.HeadsGraph.PARAM_SIZE)
        self.heads_cache_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.HEADS_CACHE_SIZE)
        self.heads_out_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.HEADS_OUT_DIM)
        self.heads_grad_out_buf = ctx.enqueue_create_buffer[dtype](Self.IB * Self.HEADS_OUT_DIM)

        # ── Network workspace buffers ────────────────────────────────────
        # Use IB (imag batch) as max batch size for workspace allocation
        comptime EncNet = Network[Self.RSSMType.EncModel, Self.WMOpt]
        comptime PostNet = Network[Self.RSSMType.PostModel, Self.WMOpt]
        comptime PriorNet = Network[Self.RSSMType.PriorModel, Self.WMOpt]
        comptime DecNet = Network[Self.RSSMType.DecModel, Self.WMOpt]
        comptime RewNet = Network[Self.RSSMType.RewModel, Self.WMOpt]
        comptime ContNet = Network[Self.RSSMType.ContModel, Self.WMOpt]
        comptime DProjNet = Network[Self.RSSMType.DeterProj, Self.WMOpt]
        comptime SProjNet = Network[Self.RSSMType.StochProj, Self.WMOpt]
        comptime AProjNet = Network[Self.RSSMType.ActionProj, Self.WMOpt]
        comptime GHNet = Network[Self.RSSMType.GRUHiddenModel, Self.WMOpt]
        comptime GGNet = Network[Self.RSSMType.GRUGateModel, Self.WMOpt]
        comptime ActNet = Network[Self.ActorModel, Self.ActorOpt]
        comptime CritNet = Network[Self.CriticModel, Self.CriticOpt]

        # Workspace sizes — use max(BATCH, IB) for networks used in both phases
        comptime WS_IB_ACTOR = Self.IB * ActNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_IB_CRITIC = Self.IB * CritNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_IB_REWARD = Self.IB * RewNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_IB_CONT = Self.IB * ContNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_IB_PRIOR = Self.IB * PriorNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_IB_DPROJ = Self.IB * DProjNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_IB_SPROJ = Self.IB * SProjNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_IB_APROJ = Self.IB * AProjNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_IB_GH = Self.IB * GHNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_IB_GG = Self.IB * GGNet.WORKSPACE_SIZE_PER_SAMPLE
        # Observe-phase (encoder batched across BL timesteps)
        comptime WS_B_ENC = Self.IB * EncNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_B_POST = Self.BATCH * PostNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_B_DEC = Self.BATCH * DecNet.WORKSPACE_SIZE_PER_SAMPLE

        # Use max of observe and imagination sizes, minimum 1
        fn max_ws(a: Int, b: Int) -> Int:
            return a if a > b else b

        self.ws_encoder = ctx.enqueue_create_buffer[dtype](max_ws(WS_B_ENC, 1))
        self.ws_posterior = ctx.enqueue_create_buffer[dtype](max_ws(WS_B_POST, 1))
        self.ws_prior = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_PRIOR, 1))
        self.ws_decoder = ctx.enqueue_create_buffer[dtype](max_ws(WS_B_DEC, 1))
        self.ws_reward = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_REWARD, 1))
        self.ws_continue = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_CONT, 1))
        self.ws_deter_proj = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_DPROJ, 1))
        self.ws_stoch_proj = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_SPROJ, 1))
        self.ws_action_proj = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_APROJ, 1))
        self.ws_gru_hidden = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_GH, 1))
        self.ws_gru_gates = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_GG, 1))
        self.ws_actor = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_ACTOR, 1))
        self.ws_critic = ctx.enqueue_create_buffer[dtype](max_ws(WS_IB_CRITIC, 1))
        comptime WS_B_HEADS = Self.IB * Self.HeadsGraph.WORKSPACE_SIZE_PER_SAMPLE
        self.ws_heads = ctx.enqueue_create_buffer[dtype](max_ws(WS_B_HEADS, 1))

        # ── Inference buffers ────────────────────────────────────────────
        self.inf_obs_buf = ctx.enqueue_create_buffer[dtype](Self.MAX_N * Self.OBS)
        self.inf_deter_buf = ctx.enqueue_create_buffer[dtype](Self.MAX_N * Self.DETER)
        self.inf_stoch_buf = ctx.enqueue_create_buffer[dtype](Self.MAX_N * Self.STOCH)
        self.inf_action_buf = ctx.enqueue_create_buffer[dtype](Self.MAX_N * Self.ACT)
        self.inf_feat_buf = ctx.enqueue_create_buffer[dtype](Self.MAX_N * Self.FEAT)
        self.inf_actor_out_buf = ctx.enqueue_create_buffer[dtype](Self.MAX_N * ACTOR_OUT_DIM)

        # ── Gradient clipping scratch ──────────────────────────────────
        self.grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](1024)

        # ── Pinned host buffers ──────────────────────────────────────────
        self.host_batch_obs = ctx.enqueue_create_buffer[dtype](Self.BATCH * (Self.BL + 1) * Self.OBS)
        self.host_batch_actions = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.BL * Self.ACT)
        self.host_batch_rewards = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.BL)
        self.host_batch_dones = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.BL)

        # Zero-initialize key buffers
        ctx.enqueue_memset(self.deter_buf, 0)
        ctx.enqueue_memset(self.stoch_buf, 0)
        ctx.enqueue_memset(self.inf_deter_buf, 0)
        ctx.enqueue_memset(self.inf_stoch_buf, 0)
        ctx.enqueue_memset(self.inf_action_buf, 0)

    fn __init__(out self, *, deinit take: Self):
        """Move constructor."""
        self.encoder = take.encoder^
        self.posterior = take.posterior^
        self.prior = take.prior^
        self.decoder = take.decoder^
        self.reward_head = take.reward_head^
        self.continue_head = take.continue_head^
        self.deter_proj = take.deter_proj^
        self.stoch_proj = take.stoch_proj^
        self.action_proj = take.action_proj^
        self.gru_hidden = take.gru_hidden^
        self.gru_gates = take.gru_gates^
        self.actor = take.actor^
        self.critic = take.critic^
        self.slow_critic = take.slow_critic^
        self.bins_buf = take.bins_buf^
        self.batch_obs = take.batch_obs^
        self.batch_actions = take.batch_actions^
        self.batch_rewards = take.batch_rewards^
        self.batch_dones = take.batch_dones^
        self.deter_buf = take.deter_buf^
        self.stoch_buf = take.stoch_buf^
        self.new_deter_buf = take.new_deter_buf^
        self.new_stoch_buf = take.new_stoch_buf^
        self.post_probs_buf = take.post_probs_buf^
        self.prior_probs_buf = take.prior_probs_buf^
        self.feat_buf = take.feat_buf^
        self.obs_step_buf = take.obs_step_buf^
        self.act_step_buf = take.act_step_buf^
        self.symlog_obs_buf = take.symlog_obs_buf^
        self.embed_buf = take.embed_buf^
        self.post_in_buf = take.post_in_buf^
        self.post_logits_buf = take.post_logits_buf^
        self.prior_logits_buf = take.prior_logits_buf^
        self.norm_action_buf = take.norm_action_buf^
        self.proj_d_buf = take.proj_d_buf^
        self.proj_s_buf = take.proj_s_buf^
        self.proj_a_buf = take.proj_a_buf^
        self.concat_buf = take.concat_buf^
        self.hidden_out_buf = take.hidden_out_buf^
        self.gate_out_buf = take.gate_out_buf^
        self.dummy_stoch_buf = take.dummy_stoch_buf^
        self.all_deter_buf = take.all_deter_buf^
        self.all_stoch_buf = take.all_stoch_buf^
        self.all_post_probs_buf = take.all_post_probs_buf^
        self.all_prior_probs_buf = take.all_prior_probs_buf^
        self.all_feats_buf = take.all_feats_buf^
        self.all_embed_buf = take.all_embed_buf^
        self.dec_out_buf = take.dec_out_buf^
        self.rew_logits_buf = take.rew_logits_buf^
        self.cont_out_buf = take.cont_out_buf^
        self.kl_buf = take.kl_buf^
        self.imag_proj_d_buf = take.imag_proj_d_buf^
        self.imag_proj_s_buf = take.imag_proj_s_buf^
        self.imag_proj_a_buf = take.imag_proj_a_buf^
        self.imag_concat_buf = take.imag_concat_buf^
        self.imag_hidden_buf = take.imag_hidden_buf^
        self.imag_gate_buf = take.imag_gate_buf^
        self.imag_norm_act_buf = take.imag_norm_act_buf^
        self.imag_prior_logits_buf = take.imag_prior_logits_buf^
        self.imag_prior_probs_buf = take.imag_prior_probs_buf^
        self.imag_deter_buf = take.imag_deter_buf^
        self.imag_stoch_buf = take.imag_stoch_buf^
        self.imag_feat_buf = take.imag_feat_buf^
        self.imag_rewards_buf = take.imag_rewards_buf^
        self.imag_values_buf = take.imag_values_buf^
        self.imag_continues_buf = take.imag_continues_buf^
        self.imag_returns_buf = take.imag_returns_buf^
        self.imag_actions_buf = take.imag_actions_buf^
        self.imag_log_probs_buf = take.imag_log_probs_buf^
        self.imag_advantages_buf = take.imag_advantages_buf^
        self.imag_all_deter_buf = take.imag_all_deter_buf^
        self.imag_all_stoch_buf = take.imag_all_stoch_buf^
        self.imag_all_actions_buf = take.imag_all_actions_buf^
        self.imag_actor_cache_buf = take.imag_actor_cache_buf^
        self.imag_aproj_cache_buf = take.imag_aproj_cache_buf^
        self.imag_gh_cache_buf = take.imag_gh_cache_buf^
        self.imag_gg_cache_buf = take.imag_gg_cache_buf^
        self.imag_prior_cache_buf = take.imag_prior_cache_buf^
        self.imag_rew_cache_buf = take.imag_rew_cache_buf^
        self.imag_gate_out_save_buf = take.imag_gate_out_save_buf^
        self.imag_actor_out_save_buf = take.imag_actor_out_save_buf^
        self.actor_out_buf = take.actor_out_buf^
        self.actor_cache_buf = take.actor_cache_buf^
        self.actor_grad_buf = take.actor_grad_buf^
        self.actor_grad_in_buf = take.actor_grad_in_buf^
        self.critic_logits_buf = take.critic_logits_buf^
        self.critic_cache_buf = take.critic_cache_buf^
        self.critic_grad_buf = take.critic_grad_buf^
        self.critic_grad_in_buf = take.critic_grad_in_buf^
        self.two_hot_targets_buf = take.two_hot_targets_buf^
        self.symlog_returns_buf = take.symlog_returns_buf^
        self.returns_minmax_buf = take.returns_minmax_buf^
        self.host_obs_step_buf = take.host_obs_step_buf^
        self.host_act_step_buf = take.host_act_step_buf^
        self.host_target_buf = take.host_target_buf^
        self.host_rew_symlog_step_buf = take.host_rew_symlog_step_buf^
        self.host_cont_target_step_buf = take.host_cont_target_step_buf^
        self.host_dec_diag_buf = take.host_dec_diag_buf^
        self.host_rew_diag_buf = take.host_rew_diag_buf^
        self.host_cont_diag_buf = take.host_cont_diag_buf^
        self.host_kl_diag_buf = take.host_kl_diag_buf^
        self.host_minmax_buf = take.host_minmax_buf^
        self.host_upload_obs_buf = take.host_upload_obs_buf^
        self.host_upload_act_buf = take.host_upload_act_buf^
        self.host_upload_rew_buf = take.host_upload_rew_buf^
        self.host_upload_done_buf = take.host_upload_done_buf^
        self.host_diag_imag_buf = take.host_diag_imag_buf^
        self.host_diag_actor_buf = take.host_diag_actor_buf^
        self.host_bins_buf = take.host_bins_buf^
        self.dec_cache_buf = take.dec_cache_buf^
        self.dec_grad_out_buf = take.dec_grad_out_buf^
        self.dec_grad_in_buf = take.dec_grad_in_buf^
        self.dec_target_buf = take.dec_target_buf^
        self.cont_target_buf = take.cont_target_buf^
        self.cont_grad_buf = take.cont_grad_buf^
        self.rew_cache_buf = take.rew_cache_buf^
        self.rew_target_buf = take.rew_target_buf^
        self.rew_grad_out_buf = take.rew_grad_out_buf^
        self.rew_grad_in_buf = take.rew_grad_in_buf^
        self.rew_symlog_buf = take.rew_symlog_buf^
        self.cont_cache_buf = take.cont_cache_buf^
        self.cont_grad_in_buf = take.cont_grad_in_buf^
        self.post_cache_buf = take.post_cache_buf^
        self.post_grad_out_buf = take.post_grad_out_buf^
        self.post_grad_in_buf = take.post_grad_in_buf^
        self.prior_cache_buf = take.prior_cache_buf^
        self.prior_grad_out_buf = take.prior_grad_out_buf^
        self.prior_grad_in_buf = take.prior_grad_in_buf^
        self.all_enc_cache_buf = take.all_enc_cache_buf^
        self.all_dproj_cache_buf = take.all_dproj_cache_buf^
        self.all_sproj_cache_buf = take.all_sproj_cache_buf^
        self.all_aproj_cache_buf = take.all_aproj_cache_buf^
        self.all_gru_hidden_cache_buf = take.all_gru_hidden_cache_buf^
        self.all_gru_gates_cache_buf = take.all_gru_gates_cache_buf^
        self.all_post_cache_buf = take.all_post_cache_buf^
        self.all_prior_cache_buf = take.all_prior_cache_buf^
        self.all_gate_out_buf = take.all_gate_out_buf^
        self.all_prev_deter_buf = take.all_prev_deter_buf^
        self.all_symlog_obs_buf = take.all_symlog_obs_buf^
        self.all_norm_action_buf = take.all_norm_action_buf^
        self.all_d_feat_buf = take.all_d_feat_buf^
        self.d_feat_buf = take.d_feat_buf^
        self.d_deter_total_buf = take.d_deter_total_buf^
        self.d_stoch_feat_buf = take.d_stoch_feat_buf^
        self.d_gate_out_bwd_buf = take.d_gate_out_bwd_buf^
        self.d_prev_deter_gru_buf = take.d_prev_deter_gru_buf^
        self.d_concat_bwd_buf = take.d_concat_bwd_buf^
        self.d_proj_d_bwd_buf = take.d_proj_d_bwd_buf^
        self.d_proj_s_bwd_buf = take.d_proj_s_bwd_buf^
        self.d_proj_a_bwd_buf = take.d_proj_a_bwd_buf^
        self.d_hidden_out_bwd_buf = take.d_hidden_out_bwd_buf^
        self.d_embed_bwd_buf = take.d_embed_bwd_buf^
        self.d_symlog_obs_bwd_buf = take.d_symlog_obs_bwd_buf^
        self.d_prev_deter_dproj_buf = take.d_prev_deter_dproj_buf^
        self.d_prev_stoch_bwd_buf = take.d_prev_stoch_bwd_buf^
        self.d_prev_action_bwd_buf = take.d_prev_action_bwd_buf^
        self.d_recurrent_deter_buf = take.d_recurrent_deter_buf^
        self.d_recurrent_stoch_buf = take.d_recurrent_stoch_buf^
        self.d_post_logits_total_buf = take.d_post_logits_total_buf^
        self.d_deter_from_post_buf = take.d_deter_from_post_buf^
        self.heads_params_buf = take.heads_params_buf^
        self.heads_grads_buf = take.heads_grads_buf^
        self.heads_cache_buf = take.heads_cache_buf^
        self.heads_out_buf = take.heads_out_buf^
        self.heads_grad_out_buf = take.heads_grad_out_buf^
        self.ws_heads = take.ws_heads^
        self.ws_encoder = take.ws_encoder^
        self.ws_posterior = take.ws_posterior^
        self.ws_prior = take.ws_prior^
        self.ws_decoder = take.ws_decoder^
        self.ws_reward = take.ws_reward^
        self.ws_continue = take.ws_continue^
        self.ws_deter_proj = take.ws_deter_proj^
        self.ws_stoch_proj = take.ws_stoch_proj^
        self.ws_action_proj = take.ws_action_proj^
        self.ws_gru_hidden = take.ws_gru_hidden^
        self.ws_gru_gates = take.ws_gru_gates^
        self.ws_actor = take.ws_actor^
        self.ws_critic = take.ws_critic^
        self.inf_obs_buf = take.inf_obs_buf^
        self.inf_deter_buf = take.inf_deter_buf^
        self.inf_stoch_buf = take.inf_stoch_buf^
        self.inf_action_buf = take.inf_action_buf^
        self.inf_feat_buf = take.inf_feat_buf^
        self.inf_actor_out_buf = take.inf_actor_out_buf^
        self.grad_partial_sums_buf = take.grad_partial_sums_buf^
        self.host_batch_obs = take.host_batch_obs^
        self.host_batch_actions = take.host_batch_actions^
        self.host_batch_rewards = take.host_batch_rewards^
        self.host_batch_dones = take.host_batch_dones^
