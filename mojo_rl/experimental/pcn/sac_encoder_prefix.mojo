"""EncoderPrefixSACConfig + weight injection — Phase-2b end-to-end fine-tuning.

Custom SAC config that prepends a 2-layer LinearTanh -> Linear prefix to
both actor and critic. The prefix matches the PCEncoder's structure
(tanh-hidden + identity-output), so a trained encoder's parameters can
be stamped directly into the first 2 layers of SAC's networks. SAC's
gradient then flows back through the prefix during training — i.e. the
encoder is fine-tuned as part of the RL objective rather than held frozen.

Architecture (per SACConfig but with a 2-layer prefix added):
    Actor:  LinearTanh[OBS, HID] -> Linear[HID, HID] -> LinearReLU[HID, HID]
            -> Parallel[Linear[HID, ACT], LinearTanh[HID, ACT]]
    Critic: LinearTanh[OBS+ACT, HID] -> Linear[HID, HID] -> LinearReLU[HID, HID]
            -> Linear[HID, 1]

This is one hidden layer deeper than vanilla SACConfig. The appropriate
control is "same architecture, random Xavier init" (variant #2 in the
Phase-2b plan), not raw-obs vanilla SAC.

Encoder layout assumption (matches PCEncoder used in the Phase-2 tests):
    encoder input  = [prev_z (HID) | prev_action (ACTION_DIM) | raw_obs (OBS_DIM)]
    encoder output = z (HID)

For the non-recurrent fine-tune experiment, only the obs-slice of the
encoder's W1 is stamped into the SAC actor and critic first layer. The
critic's action-input rows (and the prev_z + prev_action rows of W1) are
discarded. The encoder's full second layer (W2 + b2) is stamped into the
SAC second layer.

See `docs/PCN_MBRL_DESIGN.md` Phase-2b for context.
"""

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype as default_dtype, gpu_align
from mojo_rl.nn.model import (
    Linear,
    LinearReLU,
    LinearTanh,
    Sequential,
    Parallel,
)
from mojo_rl.nn.optimizer import Adam

from mojo_rl.deep_agents.core.configs.offpolicy_config import OffPolicyConfig
from mojo_rl.deep_agents.core.strategies.exploration import StochasticSample
from mojo_rl.deep_agents.core.strategies.update_schedule import DelayedActorOnly
from mojo_rl.deep_agents.core.strategies.target_value import EntropicTwinQTarget
from mojo_rl.deep_agents.core.strategies.target_action import ReparamTarget
from mojo_rl.deep_agents.core.strategies.actor_loss import AutodiffMaxEntLoss


# =============================================================================
# Config
# =============================================================================


struct EncoderPrefixSACConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 64,
    CAP: Int = 100000,
    BS: Int = 64,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
    action_scale: Float64 = 1.0,
](OffPolicyConfig):
    """SAC with a 2-layer LinearTanh -> Linear prefix on actor and critic.

    Drop-in for SACConfig with everything else identical: same optimizers,
    same strategies, same buffer / batch sizes. The only differences are
    the network architectures (one extra hidden layer, tanh+linear prefix
    instead of relu+relu).
    """

    comptime NAME: String = "EncoderPrefixSAC"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime ActorModel = Sequential[
        LinearTanh[Self.OBS, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            LinearTanh[Self.HIDDEN, Self.ACT],
        ],
    ]
    comptime CriticModel = Sequential[
        LinearTanh[Self.OBS + Self.ACT, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]

    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = 2
    comptime HAS_TARGET_ACTOR: Bool = False

    comptime Explore = StochasticSample
    comptime Schedule = DelayedActorOnly
    comptime TargetAction = ReparamTarget
    comptime TargetValue = EntropicTwinQTarget
    comptime ActorLoss = AutodiffMaxEntLoss[action_scale=Self.action_scale]


# =============================================================================
# Encoder weight injection
# =============================================================================
#
# A trained PCEncoder has params laid out as:
#
#     [ W1 (ENC_IN x HID) | b1 (HID) | W2 (HID x HID) | b2 (HID) ]
#
# row-major, with ENC_IN = HID + ACTION_DIM + OBS_DIM. The obs rows of
# W1 start at index `HID + ACTION_DIM`.
#
# Sequential lays out actor/critic params with `gpu_align` (4-element)
# padding between layers. For Pendulum-scale dims (OBS=3, ACT=1, HID=64)
# the first-layer sizes (256 actor / 320 critic) are already 4-aligned,
# but we still call `gpu_align` so the helper is correct for other dims.


def _enc_b1_offset(enc_in: Int, hid: Int) -> Int:
    return enc_in * hid


def _enc_w2_offset(enc_in: Int, hid: Int) -> Int:
    return enc_in * hid + hid


def _enc_b2_offset(enc_in: Int, hid: Int) -> Int:
    return enc_in * hid + hid + hid * hid


def inject_encoder_into_actor[
    OBS: Int,
    HID: Int,
    ENC_IN: Int,
    dtype: DType = default_dtype,
](
    actor_params: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    enc_params: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    obs_slice_offset: Int,
):
    """Stamp encoder W1[obs_slice], b1, W2, b2 into the first 2 actor layers.

    Layout:
      Actor layer 0 (LinearTanh[OBS, HID]) = [W (OBS*HID) | b (HID)]
        starts at offset 0, occupies (OBS*HID + HID) elements.
      Actor layer 1 (Linear[HID, HID])     = [W (HID*HID) | b (HID)]
        starts at offset gpu_align(layer-0 size).

    Args:
      actor_params:      Raw pointer to the actor's flat param buffer
                         (e.g. agent.state.actor.online.params).
      enc_params:        Raw pointer to the encoder's flat param buffer.
      obs_slice_offset:  Row in encoder W1 where obs columns begin
                         (typically ENC_IN - OBS for `[prev_z, action, obs]`).
    """
    var l0_size = OBS * HID + HID
    var l1_offset = gpu_align(l0_size)

    # Inject W1[obs_rows, :] into actor layer 0 W.
    # Encoder W1 row-major (ENC_IN, HID): W1[r, j] = enc_params[r*HID + j].
    # Actor layer 0 W row-major (OBS, HID): W[i, j] = actor_params[i*HID + j].
    for i in range(OBS):
        var enc_row_offset = (obs_slice_offset + i) * HID
        var act_row_offset = i * HID
        for j in range(HID):
            actor_params[act_row_offset + j] = enc_params[enc_row_offset + j]

    # Inject b1 into actor layer 0 bias.
    var enc_b1 = _enc_b1_offset(ENC_IN, HID)
    var act_l0_bias = OBS * HID
    for j in range(HID):
        actor_params[act_l0_bias + j] = enc_params[enc_b1 + j]

    # Inject W2 into actor layer 1 W (HID*HID weights).
    var enc_w2 = _enc_w2_offset(ENC_IN, HID)
    var act_l1_w = l1_offset
    for k in range(HID * HID):
        actor_params[act_l1_w + k] = enc_params[enc_w2 + k]

    # Inject b2 into actor layer 1 bias.
    var enc_b2 = _enc_b2_offset(ENC_IN, HID)
    var act_l1_bias = l1_offset + HID * HID
    for j in range(HID):
        actor_params[act_l1_bias + j] = enc_params[enc_b2 + j]


def inject_encoder_into_critic[
    OBS: Int,
    ACT: Int,
    HID: Int,
    ENC_IN: Int,
    dtype: DType = default_dtype,
](
    critic_params: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    enc_params: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    obs_slice_offset: Int,
):
    """Stamp encoder W1[obs_slice], b1, W2, b2 into the first 2 critic layers.

    Layout:
      Critic layer 0 (LinearTanh[OBS+ACT, HID]) = [W ((OBS+ACT)*HID) | b (HID)]
        starts at offset 0, occupies ((OBS+ACT)*HID + HID) elements.
      Critic layer 1 (Linear[HID, HID])         = [W (HID*HID) | b (HID)]
        starts at offset gpu_align(layer-0 size).

    Critic input order is `[obs, action]` (see `concat_obs_action_batch`),
    so the OBS rows of the critic's W are at indices [0, OBS) and the
    ACT rows at [OBS, OBS+ACT). Only the OBS rows are stamped; the ACT
    rows are left at whatever the prior Xavier init wrote.
    """
    var in_dim = OBS + ACT
    var l0_size = in_dim * HID + HID
    var l1_offset = gpu_align(l0_size)

    # Inject W1[obs_rows, :] into critic layer 0 W (OBS rows only).
    for i in range(OBS):
        var enc_row_offset = (obs_slice_offset + i) * HID
        var crit_row_offset = i * HID
        for j in range(HID):
            critic_params[crit_row_offset + j] = enc_params[enc_row_offset + j]

    # Inject b1 into critic layer 0 bias.
    var enc_b1 = _enc_b1_offset(ENC_IN, HID)
    var crit_l0_bias = in_dim * HID
    for j in range(HID):
        critic_params[crit_l0_bias + j] = enc_params[enc_b1 + j]

    # Inject W2 into critic layer 1 W.
    var enc_w2 = _enc_w2_offset(ENC_IN, HID)
    var crit_l1_w = l1_offset
    for k in range(HID * HID):
        critic_params[crit_l1_w + k] = enc_params[enc_w2 + k]

    # Inject b2 into critic layer 1 bias.
    var enc_b2 = _enc_b2_offset(ENC_IN, HID)
    var crit_l1_bias = l1_offset + HID * HID
    for j in range(HID):
        critic_params[crit_l1_bias + j] = enc_params[enc_b2 + j]
