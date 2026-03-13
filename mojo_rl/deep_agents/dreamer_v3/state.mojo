"""DreamerV3CPUState — CPU state container for DreamerV3 training.

Holds all heap-allocated data needed for one DreamerV3 training loop:
  - RSSM world model (encoder, dynamics, decoder, reward/continue heads)
  - Actor-Critic networks with slow critic EMA
  - SequenceReplayBuffer for streaming obs/act/rew/done
  - Pre-allocated scratch buffers for BPTT, imagination, and actor-critic

Created once in DreamerV3Agent.__init__.
"""

from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearMish, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.deep_agents.core.replay.sequence_replay_buffer import (
    SequenceReplayBuffer,
)
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
    ACTOR_LR: Float64 = 3e-5,
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
