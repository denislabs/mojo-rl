"""Recurrent State-Space Model (RSSM) for DreamerV3.

The RSSM is the core world model of DreamerV3. It maintains a recurrent
deterministic state (deter) and a stochastic categorical latent (stoch),
learning to predict observations, rewards, and episode continuation.

Architecture:
  encoder:      obs -> embed (symlog preprocessing)
  posterior:    concat(deter, embed) -> stoch logits (observation-conditioned)
  prior:        deter -> stoch logits (prediction without observation)
  decoder:      feat -> obs reconstruction
  reward_head:  feat -> NUM_BINS logits (distributional with symlog bins)
  continue_head: feat -> 1 (sigmoid applied manually)
  GRU core:     (prev_deter, prev_stoch, prev_action) -> new_deter

where feat = concat(deter, stoch_flat).

The stochastic latent uses categorical distributions with uniform mixture
(unimix) for exploration, and straight-through gradient estimation.

Reference: Hafner et al., 2023 — Mastering Diverse Domains through
World Models (DreamerV3)
"""

from std.math import exp, log, sqrt, abs
from std.random import random_float64
from std.memory import alloc, memset
from std.gpu.host import DeviceContext, DeviceBuffer

from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Linear,
    LinearMish,
    Sequential,
    Parallel,
    Identity,
    Model,
)
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.nn.autodiff.compute_graph import ComputeGraph, GNode
from mojo_rl.nn.autodiff.composite_params import CompositeParams
from mojo_rl.nn.loss.two_hot import (
    compute_symlog_bins,
    two_hot_encode,
    two_hot_encode_batch,
    decode_value,
    decode_value_batch,
    symlog,
    symexp,
)


# =============================================================================
# Standalone Utility Functions
# =============================================================================


def categorical_sample[
    BATCH: Int, STOCH_DIM: Int, CLASSES: Int, UNIMIX: Float64 = 0.01
](
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    mut output: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    mut probs: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    training: Bool = True,
):
    """Sample from categorical distributions with uniform mixture.

    For each batch element and each of STOCH_DIM categories:
    1. Compute softmax over CLASSES logits
    2. Mix with uniform: p = (1 - UNIMIX) * softmax + UNIMIX / CLASSES
    3. Sample one-hot via Gumbel-max (training) or argmax (inference)
    4. Store probs for KL computation

    Output is one-hot encoded (straight-through for gradients).

    Args:
        logits: Raw logits [BATCH, STOCH_DIM * CLASSES].
        output: Sampled one-hot vectors [BATCH, STOCH_DIM * CLASSES] (written).
        probs: Mixed probabilities [BATCH, STOCH_DIM * CLASSES] (written).
        training: If True, use Gumbel-max sampling; if False, use argmax.
    """
    var unimix = Scalar[dtype](UNIMIX)
    var one_minus_unimix = Scalar[dtype](1.0 - UNIMIX)
    var uniform_prob = Scalar[dtype](1.0 / Float64(CLASSES))

    for b in range(BATCH):
        for s in range(STOCH_DIM):
            var base = s * CLASSES

            # --- Softmax over CLASSES ---
            # Find max for numerical stability
            var max_val = rebind[Scalar[dtype]](logits[b, base])
            for c in range(1, CLASSES):
                var v = rebind[Scalar[dtype]](logits[b, base + c])
                if v > max_val:
                    max_val = v

            # Compute exp and sum
            var sum_exp = Scalar[dtype](0.0)
            for c in range(CLASSES):
                var e = exp(
                    rebind[Scalar[dtype]](logits[b, base + c]) - max_val
                )
                probs[b, base + c] = e  # temporary storage
                sum_exp += e

            # Normalize to softmax, then apply unimix
            for c in range(CLASSES):
                var softmax_p = (
                    rebind[Scalar[dtype]](probs[b, base + c]) / sum_exp
                )
                probs[b, base + c] = (
                    one_minus_unimix * softmax_p + unimix * uniform_prob
                )

            # --- Sampling ---
            var best_idx = 0

            if training:
                # Gumbel-max trick: argmax(log(p) + gumbel_noise)
                var best_score = Scalar[dtype](-1e10)
                for c in range(CLASSES):
                    var p = rebind[Scalar[dtype]](probs[b, base + c])
                    # Gumbel noise: -log(-log(u)), u ~ Uniform(0, 1)
                    var u = Scalar[dtype](random_float64(0.0001, 0.9999))
                    var gumbel = -log(-log(u))
                    var score = log(p + Scalar[dtype](1e-8)) + gumbel
                    if score > best_score:
                        best_score = score
                        best_idx = Int(c)
            else:
                # Argmax (greedy mode)
                var best_p = rebind[Scalar[dtype]](probs[b, base])
                for c in range(1, CLASSES):
                    var p = rebind[Scalar[dtype]](probs[b, base + c])
                    if p > best_p:
                        best_p = p
                        best_idx = Int(c)

            # Write one-hot output
            for c in range(CLASSES):
                output[b, base + c] = Scalar[dtype](0.0)
            output[b, base + best_idx] = Scalar[dtype](1.0)


def kl_divergence[
    BATCH: Int, STOCH_DIM: Int, CLASSES: Int
](
    post_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    prior_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
) -> Float64:
    """Compute KL divergence KL(posterior || prior) over categorical distributions.

    Args:
        post_probs: Posterior probabilities [BATCH, STOCH_DIM * CLASSES].
        prior_probs: Prior probabilities [BATCH, STOCH_DIM * CLASSES].

    Returns:
        Mean KL divergence over the batch (sum over STOCH_DIM and CLASSES).
    """
    var total = Float64(0.0)
    var eps = Float64(1e-8)
    for b in range(BATCH):
        for s in range(STOCH_DIM):
            for c in range(CLASSES):
                var idx = s * CLASSES + c
                var p = Float64(rebind[Scalar[dtype]](post_probs[b, idx]))
                var q = Float64(rebind[Scalar[dtype]](prior_probs[b, idx]))
                if p > eps:
                    total += p * (log(p + eps) - log(q + eps))
    return total / Float64(BATCH)


# =============================================================================
# RSSM World Model
# =============================================================================


struct RSSM[
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
    UNIMIX: Float64 = 0.01,
    FREE_NATS: Float64 = 1.0,
](Movable):
    """Recurrent State-Space Model for DreamerV3.

    The RSSM maintains a deterministic recurrent state (deter) and a
    stochastic categorical latent (stoch). The posterior uses observations
    to refine the stochastic state, while the prior predicts it from
    the deterministic state alone (used during imagination).

    Parameters:
        OBS_DIM: Observation dimension.
        ACTION_DIM: Action dimension.
        DETER_DIM: Deterministic recurrent state dimension (default: 512).
        HIDDEN: Projection hidden dimension for GRU inputs (default: 128).
        STOCH_DIM: Number of categorical distributions (default: 8).
        CLASSES: Number of classes per categorical distribution (default: 8).
        UNITS: MLP hidden layer width (default: 128).
        NUM_BINS: Number of bins for distributional reward head (default: 255).
        BLOCKS: Number of GRU blocks (unused in simplified version) (default: 4).
        WM_LR: World model learning rate (default: 1e-4).
        UNIMIX: Uniform mixture coefficient for categorical sampling (default: 0.01).
        FREE_NATS: Free nats threshold for KL balancing (default: 1.0).
    """

    # -------------------------------------------------------------------------
    # Compile-time constants
    # -------------------------------------------------------------------------

    comptime STOCH_FLAT: Int = Self.STOCH_DIM * Self.CLASSES  # 64
    comptime FEAT_DIM: Int = Self.DETER_DIM + Self.STOCH_FLAT  # 576
    comptime BLOCK_SIZE: Int = Self.DETER_DIM // Self.BLOCKS  # 128
    # BlockLinear input: BLOCK_SIZE (grouped deter) + 3*HIDDEN (tiled projections)
    comptime BL_IN: Int = Self.BLOCK_SIZE + 3 * Self.HIDDEN  # 512
    # Gate output per block: 3 * BLOCK_SIZE (reset, cand, update)
    comptime GATE_OUT: Int = 3 * Self.BLOCK_SIZE  # 384

    # -------------------------------------------------------------------------
    # Model type definitions
    # -------------------------------------------------------------------------

    # Encoder: obs -> stoch_flat embedding
    comptime EncModel = Sequential[
        LinearMish[Self.OBS_DIM, Self.UNITS],
        Linear[Self.UNITS, Self.STOCH_FLAT],
    ]

    # Posterior: concat(deter, embed) -> stoch logits
    comptime PostModel = Sequential[
        LinearMish[Self.DETER_DIM + Self.STOCH_FLAT, Self.UNITS],
        Linear[Self.UNITS, Self.STOCH_DIM * Self.CLASSES],
    ]

    # Prior: deter -> stoch logits
    comptime PriorModel = Sequential[
        LinearMish[Self.DETER_DIM, Self.UNITS],
        Linear[Self.UNITS, Self.STOCH_DIM * Self.CLASSES],
    ]

    # Decoder: feat -> obs reconstruction
    comptime DecModel = Sequential[
        LinearMish[Self.FEAT_DIM, Self.UNITS],
        Linear[Self.UNITS, Self.OBS_DIM],
    ]

    # Reward head: feat -> NUM_BINS logits (distributional with symlog bins)
    comptime RewModel = Sequential[
        LinearMish[Self.FEAT_DIM, Self.UNITS],
        LinearMish[Self.UNITS, Self.UNITS],
        Linear[Self.UNITS, Self.NUM_BINS],
    ]

    # Continue head: feat -> 1 (sigmoid applied manually after forward)
    comptime ContModel = Sequential[
        LinearMish[Self.FEAT_DIM, Self.UNITS],
        LinearMish[Self.UNITS, Self.UNITS],
        Linear[Self.UNITS, 1],
    ]

    # GRU input projections: deter -> hidden, stoch -> hidden, action -> hidden
    comptime DeterProj = LinearMish[Self.DETER_DIM, Self.HIDDEN]
    comptime StochProj = LinearMish[Self.STOCH_FLAT, Self.HIDDEN]
    comptime ActionProj = LinearMish[Self.ACTION_DIM, Self.HIDDEN]

    # GRU core layers (simplified — avoids BlockLinear for CPU-first impl)
    # Hidden: concat(deter, proj_d, proj_s, proj_a) -> DETER_DIM
    comptime GRUHiddenModel = LinearMish[
        Self.DETER_DIM + 3 * Self.HIDDEN, Self.DETER_DIM
    ]
    # Gates: DETER_DIM -> 3 * DETER_DIM (reset, candidate, update)
    comptime GRUGateModel = Linear[Self.DETER_DIM, 3 * Self.DETER_DIM]

    # Network wrapper type aliases (for external access to PARAM_SIZE, etc.)
    comptime EncNet = Network[Self.EncModel, Adam[LR=Self.WM_LR]]
    comptime PostNet = Network[Self.PostModel, Adam[LR=Self.WM_LR]]
    comptime PriorNet = Network[Self.PriorModel, Adam[LR=Self.WM_LR]]
    comptime DecNet = Network[Self.DecModel, Adam[LR=Self.WM_LR]]
    comptime RewNet = Network[Self.RewModel, Adam[LR=Self.WM_LR]]
    comptime ContNet = Network[Self.ContModel, Adam[LR=Self.WM_LR]]
    comptime DeterProjNet = Network[Self.DeterProj, Adam[LR=Self.WM_LR]]
    comptime StochProjNet = Network[Self.StochProj, Adam[LR=Self.WM_LR]]
    comptime ActionProjNet = Network[Self.ActionProj, Adam[LR=Self.WM_LR]]
    comptime GRUHiddenNet = Network[Self.GRUHiddenModel, Adam[LR=Self.WM_LR]]
    comptime GRUGateNet = Network[Self.GRUGateModel, Adam[LR=Self.WM_LR]]

    # -------------------------------------------------------------------------
    # Sub-network states
    # -------------------------------------------------------------------------

    var encoder: NetworkState[Self.EncModel, Adam[LR=Self.WM_LR]]
    var posterior: NetworkState[Self.PostModel, Adam[LR=Self.WM_LR]]
    var prior: NetworkState[Self.PriorModel, Adam[LR=Self.WM_LR]]
    var decoder: NetworkState[Self.DecModel, Adam[LR=Self.WM_LR]]
    var reward_head: NetworkState[Self.RewModel, Adam[LR=Self.WM_LR]]
    var continue_head: NetworkState[Self.ContModel, Adam[LR=Self.WM_LR]]

    # GRU core networks
    var deter_proj: NetworkState[Self.DeterProj, Adam[LR=Self.WM_LR]]
    var stoch_proj: NetworkState[Self.StochProj, Adam[LR=Self.WM_LR]]
    var action_proj: NetworkState[Self.ActionProj, Adam[LR=Self.WM_LR]]
    var gru_hidden: NetworkState[Self.GRUHiddenModel, Adam[LR=Self.WM_LR]]
    var gru_gates: NetworkState[Self.GRUGateModel, Adam[LR=Self.WM_LR]]

    # Symlog bins for distributional reward/value
    var bins: InlineArray[Float32, Self.NUM_BINS]

    # =========================================================================
    # Constructors
    # =========================================================================

    def __init__(out self):
        """Initialize all sub-networks with Kaiming initialization and
        compute symlog bins for distributional reward prediction."""
        self.encoder = NetworkState[Self.EncModel, Adam[LR=Self.WM_LR]]()
        self.encoder.initialize[Kaiming[]]()

        self.posterior = NetworkState[Self.PostModel, Adam[LR=Self.WM_LR]]()
        self.posterior.initialize[Kaiming[]]()

        self.prior = NetworkState[Self.PriorModel, Adam[LR=Self.WM_LR]]()
        self.prior.initialize[Kaiming[]]()

        self.decoder = NetworkState[Self.DecModel, Adam[LR=Self.WM_LR]]()
        self.decoder.initialize[Kaiming[]]()

        self.reward_head = NetworkState[Self.RewModel, Adam[LR=Self.WM_LR]]()
        self.reward_head.initialize[Kaiming[]]()

        self.continue_head = NetworkState[Self.ContModel, Adam[LR=Self.WM_LR]]()
        self.continue_head.initialize[Kaiming[]]()

        self.deter_proj = NetworkState[Self.DeterProj, Adam[LR=Self.WM_LR]]()
        self.deter_proj.initialize[Kaiming[]]()

        self.stoch_proj = NetworkState[Self.StochProj, Adam[LR=Self.WM_LR]]()
        self.stoch_proj.initialize[Kaiming[]]()

        self.action_proj = NetworkState[Self.ActionProj, Adam[LR=Self.WM_LR]]()
        self.action_proj.initialize[Kaiming[]]()

        self.gru_hidden = NetworkState[
            Self.GRUHiddenModel, Adam[LR=Self.WM_LR]
        ]()
        self.gru_hidden.initialize[Kaiming[]]()

        self.gru_gates = NetworkState[Self.GRUGateModel, Adam[LR=Self.WM_LR]]()
        self.gru_gates.initialize[Kaiming[]]()

        # Compute symlog-spaced bins for distributional reward head
        self.bins = compute_symlog_bins[Self.NUM_BINS]()

    def __init__(out self, *, deinit take: Self):
        """Move constructor — transfers ownership of all fields."""
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
        self.bins = take.bins

    # =========================================================================
    # GRU Core Forward
    # =========================================================================

    def core_forward[
        BATCH: Int
    ](
        self,
        prev_deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        prev_stoch: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        prev_action: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTION_DIM), MutAnyOrigin
        ],
        mut new_deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        # Scratch buffers
        mut proj_d: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut proj_s: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut proj_a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ],
        mut concat_buf: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.DETER_DIM + 3 * Self.HIDDEN),
            MutAnyOrigin,
        ],
        mut hidden_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        mut gate_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, 3 * Self.DETER_DIM), MutAnyOrigin
        ],
    ):
        """GRU core forward pass.

        Computes the next deterministic state from previous deter, stoch,
        and action using a GRU-like gating mechanism:

        1. Three input projections with Mish activation
        2. Concatenate [deter, proj_d, proj_s, proj_a]
        3. Hidden layer (LinearMish)
        4. Gate layer -> split into reset, candidate, update
        5. Apply gating: new_deter = update * cand + (1 - update) * prev_deter

        The update gate is biased toward keeping the old state (sigmoid(x - 1)).

        Args:
            prev_deter: Previous deterministic state [BATCH, DETER_DIM].
            prev_stoch: Previous stochastic state (flat) [BATCH, STOCH_FLAT].
            prev_action: Previous action [BATCH, ACTION_DIM].
            new_deter: Output deterministic state [BATCH, DETER_DIM] (written).
            proj_d: Scratch for deter projection [BATCH, HIDDEN].
            proj_s: Scratch for stoch projection [BATCH, HIDDEN].
            proj_a: Scratch for action projection [BATCH, HIDDEN].
            concat_buf: Scratch for concatenation [BATCH, DETER_DIM + 3*HIDDEN].
            hidden_out: Scratch for hidden layer output [BATCH, DETER_DIM].
            gate_out: Scratch for gate layer output [BATCH, 3*DETER_DIM].
        """
        # Normalize action: a /= max(1, |a|) elementwise
        # We apply this in-place to a scratch copy within the action projection.
        # For simplicity, we create a normalized action buffer on the stack
        # and pass it to the action projection.
        var norm_action_ptr = alloc[Scalar[dtype]](BATCH * Self.ACTION_DIM)
        for b in range(BATCH):
            for i in range(Self.ACTION_DIM):
                var a = rebind[Scalar[dtype]](prev_action[b, i])
                var abs_a = abs(a)
                var one = Scalar[dtype](1.0)
                if abs_a > one:
                    (norm_action_ptr + b * Self.ACTION_DIM + i)[] = a / abs_a
                else:
                    (norm_action_ptr + b * Self.ACTION_DIM + i)[] = a
        var norm_action = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTION_DIM), MutAnyOrigin
        ](norm_action_ptr)

        # 1. Input projections (each with Mish activation)
        Self.DeterProjNet.forward[BATCH](
            prev_deter, proj_d, self.deter_proj.params_view()
        )
        Self.StochProjNet.forward[BATCH](
            prev_stoch, proj_s, self.stoch_proj.params_view()
        )
        Self.ActionProjNet.forward[BATCH](
            norm_action, proj_a, self.action_proj.params_view()
        )

        # 2. Concatenate [deter, proj_d, proj_s, proj_a]
        for b in range(BATCH):
            for i in range(Self.DETER_DIM):
                concat_buf[b, i] = prev_deter[b, i]
            for i in range(Self.HIDDEN):
                concat_buf[b, Self.DETER_DIM + i] = proj_d[b, i]
                concat_buf[b, Self.DETER_DIM + Self.HIDDEN + i] = proj_s[b, i]
                concat_buf[b, Self.DETER_DIM + 2 * Self.HIDDEN + i] = proj_a[
                    b, i
                ]

        # 3. Hidden layer (LinearMish)
        Self.GRUHiddenNet.forward[BATCH](
            concat_buf, hidden_out, self.gru_hidden.params_view()
        )

        # 4. Gate layer (Linear, no activation)
        Self.GRUGateNet.forward[BATCH](
            hidden_out, gate_out, self.gru_gates.params_view()
        )

        # 5. Apply GRU-style gating
        for b in range(BATCH):
            for i in range(Self.DETER_DIM):
                var reset_logit = rebind[Scalar[dtype]](gate_out[b, i])
                var cand_logit = rebind[Scalar[dtype]](
                    gate_out[b, Self.DETER_DIM + i]
                )
                var update_logit = rebind[Scalar[dtype]](
                    gate_out[b, 2 * Self.DETER_DIM + i]
                )

                var one = Scalar[dtype](1.0)

                # reset = sigmoid(reset_logit)
                var reset_val = one / (one + exp(-reset_logit))

                # cand = tanh(reset * cand_logit)
                var rc = reset_val * cand_logit
                var exp_rc = exp(rc)
                var exp_neg_rc = exp(-rc)
                var cand_val = (exp_rc - exp_neg_rc) / (exp_rc + exp_neg_rc)

                # update = sigmoid(update_logit - 1) — bias toward keeping old state
                var update_val = one / (one + exp(-(update_logit - one)))

                var old_d = rebind[Scalar[dtype]](prev_deter[b, i])
                new_deter[b, i] = (
                    update_val * cand_val + (one - update_val) * old_d
                )

        # Free normalized action scratch
        norm_action_ptr.free()

    # =========================================================================
    # Observe Step (Posterior — uses observations)
    # =========================================================================

    def observe_step[
        BATCH: Int
    ](
        self,
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ],
        prev_deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        prev_stoch: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        prev_action: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTION_DIM), MutAnyOrigin
        ],
        # Outputs
        mut new_deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        mut new_stoch: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        mut post_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        mut prior_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        mut feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        training: Bool = True,
    ):
        """Single observe step: use observation to compute posterior.

        1. Apply symlog to observations and encode to embedding
        2. Compute new deterministic state via GRU core
        3. Compute posterior logits from concat(deter, embed)
        4. Compute prior logits from deter alone
        5. Sample stochastic state from posterior
        6. Build feature vector feat = concat(deter, stoch)

        Args:
            obs: Current observation [BATCH, OBS_DIM].
            prev_deter: Previous deterministic state [BATCH, DETER_DIM].
            prev_stoch: Previous stochastic state [BATCH, STOCH_FLAT].
            prev_action: Previous action [BATCH, ACTION_DIM].
            new_deter: Output deterministic state [BATCH, DETER_DIM] (written).
            new_stoch: Output stochastic state [BATCH, STOCH_FLAT] (written).
            post_probs: Posterior probabilities [BATCH, STOCH_FLAT] (written).
            prior_probs: Prior probabilities [BATCH, STOCH_FLAT] (written).
            feat: Output feature vector [BATCH, FEAT_DIM] (written).
            training: If True, sample stochastic; if False, use mode.
        """
        # --- Allocate scratch buffers ---
        comptime GRU_CONCAT_DIM = Self.DETER_DIM + 3 * Self.HIDDEN
        comptime POST_IN_DIM = Self.DETER_DIM + Self.STOCH_FLAT

        var scratch_proj_d = alloc[Scalar[dtype]](BATCH * Self.HIDDEN)
        memset(scratch_proj_d, 0, BATCH * Self.HIDDEN)
        var scratch_proj_s = alloc[Scalar[dtype]](BATCH * Self.HIDDEN)
        memset(scratch_proj_s, 0, BATCH * Self.HIDDEN)
        var scratch_proj_a = alloc[Scalar[dtype]](BATCH * Self.HIDDEN)
        memset(scratch_proj_a, 0, BATCH * Self.HIDDEN)
        var scratch_concat = alloc[Scalar[dtype]](BATCH * GRU_CONCAT_DIM)
        memset(scratch_concat, 0, BATCH * GRU_CONCAT_DIM)
        var scratch_hidden = alloc[Scalar[dtype]](BATCH * Self.DETER_DIM)
        memset(scratch_hidden, 0, BATCH * Self.DETER_DIM)
        var scratch_gate = alloc[Scalar[dtype]](BATCH * 3 * Self.DETER_DIM)
        memset(scratch_gate, 0, BATCH * 3 * Self.DETER_DIM)
        var scratch_symlog_obs = alloc[Scalar[dtype]](BATCH * Self.OBS_DIM)
        var scratch_embed = alloc[Scalar[dtype]](BATCH * Self.STOCH_FLAT)
        memset(scratch_embed, 0, BATCH * Self.STOCH_FLAT)
        var scratch_post_in = alloc[Scalar[dtype]](BATCH * POST_IN_DIM)
        memset(scratch_post_in, 0, BATCH * POST_IN_DIM)
        var scratch_post_logits = alloc[Scalar[dtype]](BATCH * Self.STOCH_FLAT)
        memset(scratch_post_logits, 0, BATCH * Self.STOCH_FLAT)
        var scratch_prior_logits = alloc[Scalar[dtype]](BATCH * Self.STOCH_FLAT)
        memset(scratch_prior_logits, 0, BATCH * Self.STOCH_FLAT)

        # Create LayoutTensor views
        var proj_d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](scratch_proj_d)
        var proj_s_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](scratch_proj_s)
        var proj_a_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](scratch_proj_a)
        var concat_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GRU_CONCAT_DIM), MutAnyOrigin
        ](scratch_concat)
        var hidden_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ](scratch_hidden)
        var gate_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 3 * Self.DETER_DIM), MutAnyOrigin
        ](scratch_gate)
        var symlog_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ](scratch_symlog_obs)
        var embed_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](scratch_embed)
        var post_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, POST_IN_DIM), MutAnyOrigin
        ](scratch_post_in)
        var post_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](scratch_post_logits)
        var prior_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](scratch_prior_logits)

        # --- 1. Symlog preprocessing + encode ---
        for b in range(BATCH):
            for i in range(Self.OBS_DIM):
                var val = Float32(rebind[Scalar[dtype]](obs[b, i]))
                symlog_obs_t[b, i] = Scalar[dtype](symlog(val))

        Self.EncNet.forward[BATCH](
            symlog_obs_t, embed_t, self.encoder.params_view()
        )

        # --- 2. GRU core forward ---
        self.core_forward[BATCH](
            prev_deter,
            prev_stoch,
            prev_action,
            new_deter,
            proj_d_t,
            proj_s_t,
            proj_a_t,
            concat_t,
            hidden_t,
            gate_t,
        )

        # --- 3. Posterior: concat(deter, embed) -> logits ---
        for b in range(BATCH):
            for i in range(Self.DETER_DIM):
                post_in_t[b, i] = new_deter[b, i]
            for i in range(Self.STOCH_FLAT):
                post_in_t[b, Self.DETER_DIM + i] = embed_t[b, i]

        Self.PostNet.forward[BATCH](
            post_in_t, post_logits_t, self.posterior.params_view()
        )

        # --- 4. Prior: deter -> logits ---
        Self.PriorNet.forward[BATCH](
            new_deter, prior_logits_t, self.prior.params_view()
        )

        # --- 5. Sample from posterior (with probs for KL) ---
        # Also compute prior probs for KL
        categorical_sample[BATCH, Self.STOCH_DIM, Self.CLASSES, Self.UNIMIX](
            post_logits_t, new_stoch, post_probs, training
        )

        # Compute prior probs (we need a dummy output for sampling)
        var dummy_stoch_ptr = alloc[Scalar[dtype]](BATCH * Self.STOCH_FLAT)
        memset(dummy_stoch_ptr, 0, BATCH * Self.STOCH_FLAT)
        var dummy_stoch_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](dummy_stoch_ptr)
        categorical_sample[BATCH, Self.STOCH_DIM, Self.CLASSES, Self.UNIMIX](
            prior_logits_t, dummy_stoch_t, prior_probs, False
        )
        dummy_stoch_ptr.free()

        # --- 6. Build feature vector: feat = concat(deter, stoch) ---
        for b in range(BATCH):
            for i in range(Self.DETER_DIM):
                feat[b, i] = new_deter[b, i]
            for i in range(Self.STOCH_FLAT):
                feat[b, Self.DETER_DIM + i] = new_stoch[b, i]

        # --- Free scratch buffers ---
        scratch_proj_d.free()
        scratch_proj_s.free()
        scratch_proj_a.free()
        scratch_concat.free()
        scratch_hidden.free()
        scratch_gate.free()
        scratch_symlog_obs.free()
        scratch_embed.free()
        scratch_post_in.free()
        scratch_post_logits.free()
        scratch_prior_logits.free()

    # =========================================================================
    # Imagine Step (Prior — no observations)
    # =========================================================================

    def imagine_step[
        BATCH: Int
    ](
        self,
        prev_deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        prev_stoch: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        action: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTION_DIM), MutAnyOrigin
        ],
        # Outputs
        mut new_deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        mut new_stoch: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        mut feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        training: Bool = True,
    ):
        """Single imagination step: predict without observations using the prior.

        Used during latent imagination (actor-critic training in dream space).

        1. Compute new deterministic state via GRU core
        2. Compute prior logits from deter alone
        3. Sample stochastic state from prior
        4. Build feature vector feat = concat(deter, stoch)

        Args:
            prev_deter: Previous deterministic state [BATCH, DETER_DIM].
            prev_stoch: Previous stochastic state [BATCH, STOCH_FLAT].
            action: Action to take [BATCH, ACTION_DIM].
            new_deter: Output deterministic state [BATCH, DETER_DIM] (written).
            new_stoch: Output stochastic state [BATCH, STOCH_FLAT] (written).
            feat: Output feature vector [BATCH, FEAT_DIM] (written).
            training: If True, sample stochastic; if False, use mode.
        """
        # --- Allocate scratch buffers ---
        comptime GRU_CONCAT_DIM = Self.DETER_DIM + 3 * Self.HIDDEN

        var scratch_proj_d = alloc[Scalar[dtype]](BATCH * Self.HIDDEN)
        memset(scratch_proj_d, 0, BATCH * Self.HIDDEN)
        var scratch_proj_s = alloc[Scalar[dtype]](BATCH * Self.HIDDEN)
        memset(scratch_proj_s, 0, BATCH * Self.HIDDEN)
        var scratch_proj_a = alloc[Scalar[dtype]](BATCH * Self.HIDDEN)
        memset(scratch_proj_a, 0, BATCH * Self.HIDDEN)
        var scratch_concat = alloc[Scalar[dtype]](BATCH * GRU_CONCAT_DIM)
        memset(scratch_concat, 0, BATCH * GRU_CONCAT_DIM)
        var scratch_hidden = alloc[Scalar[dtype]](BATCH * Self.DETER_DIM)
        memset(scratch_hidden, 0, BATCH * Self.DETER_DIM)
        var scratch_gate = alloc[Scalar[dtype]](BATCH * 3 * Self.DETER_DIM)
        memset(scratch_gate, 0, BATCH * 3 * Self.DETER_DIM)
        var scratch_prior_logits = alloc[Scalar[dtype]](BATCH * Self.STOCH_FLAT)
        memset(scratch_prior_logits, 0, BATCH * Self.STOCH_FLAT)
        var scratch_probs = alloc[Scalar[dtype]](BATCH * Self.STOCH_FLAT)
        memset(scratch_probs, 0, BATCH * Self.STOCH_FLAT)

        # Create LayoutTensor views
        var proj_d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](scratch_proj_d)
        var proj_s_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](scratch_proj_s)
        var proj_a_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](scratch_proj_a)
        var concat_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GRU_CONCAT_DIM), MutAnyOrigin
        ](scratch_concat)
        var hidden_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ](scratch_hidden)
        var gate_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 3 * Self.DETER_DIM), MutAnyOrigin
        ](scratch_gate)
        var prior_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](scratch_prior_logits)
        var probs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](scratch_probs)

        # --- 1. GRU core forward ---
        self.core_forward[BATCH](
            prev_deter,
            prev_stoch,
            action,
            new_deter,
            proj_d_t,
            proj_s_t,
            proj_a_t,
            concat_t,
            hidden_t,
            gate_t,
        )

        # --- 2. Prior: deter -> logits ---
        Self.PriorNet.forward[BATCH](
            new_deter, prior_logits_t, self.prior.params_view()
        )

        # --- 3. Sample from prior ---
        categorical_sample[BATCH, Self.STOCH_DIM, Self.CLASSES, Self.UNIMIX](
            prior_logits_t, new_stoch, probs_t, training
        )

        # --- 4. Build feature vector: feat = concat(deter, stoch) ---
        for b in range(BATCH):
            for i in range(Self.DETER_DIM):
                feat[b, i] = new_deter[b, i]
            for i in range(Self.STOCH_FLAT):
                feat[b, Self.DETER_DIM + i] = new_stoch[b, i]

        # --- Free scratch buffers ---
        scratch_proj_d.free()
        scratch_proj_s.free()
        scratch_proj_a.free()
        scratch_concat.free()
        scratch_hidden.free()
        scratch_gate.free()
        scratch_prior_logits.free()
        scratch_probs.free()

    # =========================================================================
    # Decoder / Reward / Continue Heads
    # =========================================================================

    def decode[
        BATCH: Int
    ](
        self,
        feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        mut obs_pred: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ],
    ):
        """Decode feature vector to observation prediction.

        The output is in symlog space — apply symexp to get actual values.

        Args:
            feat: Feature vector [BATCH, FEAT_DIM].
            obs_pred: Predicted observation [BATCH, OBS_DIM] (written).
        """
        Self.DecNet.forward[BATCH](feat, obs_pred, self.decoder.params_view())

    def predict_reward[
        BATCH: Int
    ](
        self,
        feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        mut reward_logits: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.NUM_BINS), MutAnyOrigin
        ],
    ):
        """Predict reward distribution logits from feature vector.

        Use decode_value / decode_value_batch with self.bins to convert
        logits to scalar reward predictions.

        Args:
            feat: Feature vector [BATCH, FEAT_DIM].
            reward_logits: Output reward logits [BATCH, NUM_BINS] (written).
        """
        Self.RewNet.forward[BATCH](
            feat, reward_logits, self.reward_head.params_view()
        )

    def predict_continue[
        BATCH: Int
    ](
        self,
        feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        mut cont_prob: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ],
    ):
        """Predict episode continuation probability from feature vector.

        Applies sigmoid manually to the raw output logit.

        Args:
            feat: Feature vector [BATCH, FEAT_DIM].
            cont_prob: Output continuation probability [BATCH, 1] (written).
        """
        Self.ContNet.forward[BATCH](
            feat, cont_prob, self.continue_head.params_view()
        )

        # Apply sigmoid manually
        for b in range(BATCH):
            var logit = rebind[Scalar[dtype]](cont_prob[b, 0])
            var one = Scalar[dtype](1.0)
            cont_prob[b, 0] = one / (one + exp(-logit))

    # =========================================================================
    # Gradient Management
    # =========================================================================

    def zero_all_grads(mut self):
        """Zero gradients for all world model sub-networks."""
        self.encoder.zero_grads()
        self.posterior.zero_grads()
        self.prior.zero_grads()
        self.decoder.zero_grads()
        self.reward_head.zero_grads()
        self.continue_head.zero_grads()
        self.deter_proj.zero_grads()
        self.stoch_proj.zero_grads()
        self.action_proj.zero_grads()
        self.gru_hidden.zero_grads()
        self.gru_gates.zero_grads()

    # =========================================================================
    # Parameter Updates
    # =========================================================================

    def update_all_params(mut self):
        """Apply gradient updates to all world model parameters."""
        self.encoder.optimizer_step()
        self.posterior.optimizer_step()
        self.prior.optimizer_step()
        self.decoder.optimizer_step()
        self.reward_head.optimizer_step()
        self.continue_head.optimizer_step()
        self.deter_proj.optimizer_step()
        self.stoch_proj.optimizer_step()
        self.action_proj.optimizer_step()
        self.gru_hidden.optimizer_step()
        self.gru_gates.optimizer_step()

    # =========================================================================
    # Autodiff Prediction Heads (ComputeGraph-based)
    # =========================================================================
    #
    # The prediction heads share feat as input with a 3-way fan-out:
    #
    #   feat ──→ Decoder     → obs_hat       (OBS_DIM)
    #        ├─→ RewardHead  → rew_logits    (NUM_BINS)
    #        └─→ ContHead    → cont_logit    (1, pre-sigmoid)
    #
    # ComputeGraph handles the fan-out topology and automatic gradient
    # accumulation at the feat fan-out point during backward. This replaces
    # the manual backward that only had decoder and discarded grad_feat.
    #
    # Output: [obs_hat(OBS_DIM), rew_logits(NUM_BINS), cont_logit(1)]
    # =========================================================================

    comptime _DEC_REW_DIM: Int = Self.OBS_DIM + Self.NUM_BINS
    comptime HEADS_OUT_DIM: Int = Self._DEC_REW_DIM + 1

    comptime HeadsGraph = ComputeGraph[
        GNode["decoder", Self.DecModel],
        GNode["rew_head", Self.RewModel],
        GNode["cont_head", Self.ContModel],
        GNode["cat_dr", Identity[Self._DEC_REW_DIM], "decoder", "rew_head"],
        GNode["output", Identity[Self.HEADS_OUT_DIM], "cat_dr", "cont_head"],
    ]

    comptime HeadsCP = CompositeParams[
        Self.DecModel, Self.RewModel, Self.ContModel
    ]

    comptime HEADS_CACHE_SIZE: Int = Self.HeadsGraph.CACHE_SIZE

    def predict_all_heads[
        BATCH: Int
    ](
        self,
        feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HEADS_OUT_DIM), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.HEADS_CACHE_SIZE),
            MutAnyOrigin,
        ],
    ):
        """Forward all prediction heads via ComputeGraph.

        Output layout: [obs_hat(OBS_DIM), rew_logits(NUM_BINS), cont_logit(1)]

        The cache must be preserved for the subsequent backward call.
        Allocate as: BATCH * HEADS_CACHE_SIZE elements.
        """
        # Assemble params from separate network states
        var combined = InlineArray[Scalar[dtype], Self.HeadsCP.TOTAL_SIZE](
            uninitialized=True
        )
        Self.HeadsCP.assemble(
            combined.unsafe_ptr(),
            self.decoder.params_view().ptr,
            self.reward_head.params_view().ptr,
            self.continue_head.params_view().ptr,
        )
        var params_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.HeadsGraph.PARAM_SIZE),
            MutAnyOrigin,
        ](combined.unsafe_ptr())

        Self.HeadsGraph.forward[BATCH](feat, output, params_t, cache)

    def backward_all_heads[
        BATCH: Int
    ](
        mut self,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HEADS_OUT_DIM), MutAnyOrigin
        ],
        mut grad_feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.HEADS_CACHE_SIZE),
            MutAnyOrigin,
        ],
    ):
        """Backward all prediction heads via ComputeGraph.

        Computes grad_feat = d(loss)/d(feat) from ALL three head losses,
        and accumulates parameter gradients into decoder, reward_head,
        and continue_head gradient buffers.

        Args:
            grad_output: Gradient seeds for each head output, concatenated as
                [d_obs_hat(OBS_DIM), d_rew_logits(NUM_BINS), d_cont_logit(1)].
            grad_feat: Output gradient w.r.t. feat [BATCH, FEAT_DIM] (written).
                This should be backpropagated through encoder/GRU for full BPTT.
            cache: Cache from predict_all_heads forward pass.
        """
        # Re-assemble params (needed for backward)
        var combined = InlineArray[Scalar[dtype], Self.HeadsCP.TOTAL_SIZE](
            uninitialized=True
        )
        Self.HeadsCP.assemble(
            combined.unsafe_ptr(),
            self.decoder.params_view().ptr,
            self.reward_head.params_view().ptr,
            self.continue_head.params_view().ptr,
        )
        var params_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.HeadsGraph.PARAM_SIZE),
            MutAnyOrigin,
        ](combined.unsafe_ptr())

        # Combined grads buffer
        var combined_grads = InlineArray[
            Scalar[dtype], Self.HeadsCP.TOTAL_SIZE
        ](uninitialized=True)
        for i in range(Self.HeadsCP.TOTAL_SIZE):
            combined_grads[i] = Scalar[dtype](0.0)
        var grads_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.HeadsGraph.PARAM_SIZE),
            MutAnyOrigin,
        ](combined_grads.unsafe_ptr())

        # Backward: computes grad_feat + param grads for all heads
        Self.HeadsGraph.backward[BATCH](
            grad_output, grad_feat, params_t, cache, grads_t
        )

        # Scatter param grads back to individual networks (accumulate)
        Self.HeadsCP.scatter_add(
            combined_grads.unsafe_ptr(),
            self.decoder.grads_view().ptr,
            self.reward_head.grads_view().ptr,
            self.continue_head.grads_view().ptr,
        )

    # =========================================================================
    # GPU Prediction Heads (ComputeGraph-based)
    # =========================================================================
    #
    # GPU versions of predict_all_heads / backward_all_heads.
    # Uses HeadsGraph.forward_gpu / backward_gpu with pre-assembled combined
    # params/grads DeviceBuffers. The caller manages buffer assembly/scatter
    # via HeadsCP.assemble_gpu / scatter_add_gpu.
    #
    # Output layout: [obs_hat(OBS_DIM), rew_logits(NUM_BINS), cont_logit(1)]
    # =========================================================================

    @staticmethod
    def predict_all_heads_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HEADS_OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.HeadsGraph.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.HEADS_CACHE_SIZE),
            MutAnyOrigin,
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU forward all prediction heads via ComputeGraph.

        Output layout: [obs_hat(OBS_DIM), rew_logits(NUM_BINS), cont_logit(1)]
        The cache must be preserved for the subsequent backward call.
        """
        Self.HeadsGraph.forward_gpu[BATCH](
            ctx,
            output,
            feat,
            params,
            cache,
            workspace,
        )

    @staticmethod
    def backward_all_heads_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HEADS_OUT_DIM), MutAnyOrigin
        ],
        mut grad_feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.HeadsGraph.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.HEADS_CACHE_SIZE),
            MutAnyOrigin,
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.HeadsGraph.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU backward all prediction heads via ComputeGraph.

        Computes grad_feat = d(loss)/d(feat) from ALL three head losses
        (automatic fan-out gradient accumulation), and accumulates parameter
        gradients into the combined grads buffer.

        Use HeadsCP.scatter_add_gpu to distribute grads to individual networks.
        """
        Self.HeadsGraph.backward_gpu[BATCH](
            ctx,
            grad_feat,
            grad_output,
            params,
            cache,
            grads,
            workspace,
        )

    # =========================================================================
    # Backward from feat through encoder (Step 1 of BPTT)
    # =========================================================================
    #
    # Gradient path:
    #   grad_feat → split [grad_deter, grad_stoch]
    #            → straight-through categorical → grad_post_logits
    #            → Posterior.backward → [grad_deter_from_post, grad_embed]
    #            → Encoder.backward → encoder param grads
    #   grad_deter_total = grad_deter + grad_deter_from_post
    #
    # This method re-forwards encoder and posterior with cache (since
    # observe_step used inference-mode forward without caching).
    # =========================================================================

    def backward_feat_to_encoder[
        BATCH: Int
    ](
        mut self,
        grad_feat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.FEAT_DIM), MutAnyOrigin
        ],
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ],
        deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        post_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        mut grad_deter_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
    ):
        """Backpropagate from grad_feat through posterior and encoder.

        Computes:
        1. grad_deter (direct from feat split)
        2. grad_stoch → straight-through → grad_post_logits
        3. Posterior backward → grad_post_input = [grad_deter_from_post, grad_embed]
        4. Encoder backward → encoder param grads
        5. grad_deter_out = grad_deter + grad_deter_from_post

        Accumulates param grads for posterior and encoder networks.

        Args:
            grad_feat: Gradient w.r.t. feat from backward_all_heads.
            obs: Original observation (for encoder re-forward with cache).
            deter: Deterministic state at this timestep.
            post_probs: Posterior probabilities (from observe_step, for straight-through).
            grad_deter_out: Output total gradient w.r.t. deter (for GRU BPTT).
        """
        comptime POST_IN = Self.DETER_DIM + Self.STOCH_FLAT

        # =====================================================================
        # 1. Split grad_feat → grad_deter + grad_stoch
        # =====================================================================
        # feat = concat(deter, stoch), so:
        #   grad_deter = grad_feat[:, :DETER_DIM]
        #   grad_stoch = grad_feat[:, DETER_DIM:]

        # =====================================================================
        # 2. Straight-through gradient through categorical sampling
        # =====================================================================
        # Forward: logits → softmax → probs (with unimix) → sample → one_hot
        # Straight-through: treat output = probs, so grad_probs = grad_stoch
        # Then: grad_softmax = (1 - UNIMIX) * grad_probs
        # Then: grad_logits = softmax_vjp(grad_softmax)
        #   where softmax_vjp(g, s) = s * (g - sum(g * s)) per group
        var grad_logits = InlineArray[Scalar[dtype], BATCH * Self.STOCH_FLAT](
            uninitialized=True
        )

        comptime ONE_MINUS_UNIMIX = 1.0 - Self.UNIMIX

        for b in range(BATCH):
            for s in range(Self.STOCH_DIM):
                var base = s * Self.CLASSES

                # grad_probs = grad_stoch (straight-through)
                # grad_softmax = (1 - UNIMIX) * grad_probs
                # Compute dot(grad_softmax, softmax_p) per group
                # softmax_p = (probs - UNIMIX/CLASSES) / (1 - UNIMIX)
                var dot_gs: Float64 = 0.0
                for c in range(Self.CLASSES):
                    var grad_stoch_c = Float64(
                        grad_feat.ptr[
                            b * Self.FEAT_DIM + Self.DETER_DIM + base + c
                        ]
                    )
                    var prob_c = Float64(
                        rebind[Scalar[dtype]](post_probs[b, base + c])
                    )
                    # softmax_c = (prob_c - UNIMIX/CLASSES) / (1-UNIMIX)
                    var softmax_c = (
                        prob_c - Self.UNIMIX / Float64(Self.CLASSES)
                    ) / ONE_MINUS_UNIMIX
                    if softmax_c < 0.0:
                        softmax_c = 0.0
                    var grad_sm_c = ONE_MINUS_UNIMIX * grad_stoch_c
                    dot_gs += grad_sm_c * softmax_c

                # grad_logits[c] = softmax_c * (grad_softmax_c - dot_gs)
                for c in range(Self.CLASSES):
                    var grad_stoch_c = Float64(
                        grad_feat.ptr[
                            b * Self.FEAT_DIM + Self.DETER_DIM + base + c
                        ]
                    )
                    var prob_c = Float64(
                        rebind[Scalar[dtype]](post_probs[b, base + c])
                    )
                    var softmax_c = (
                        prob_c - Self.UNIMIX / Float64(Self.CLASSES)
                    ) / ONE_MINUS_UNIMIX
                    if softmax_c < 0.0:
                        softmax_c = 0.0
                    var grad_sm_c = ONE_MINUS_UNIMIX * grad_stoch_c
                    grad_logits[b * Self.STOCH_FLAT + base + c] = Scalar[dtype](
                        softmax_c * (grad_sm_c - dot_gs)
                    )

        var grad_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](grad_logits.unsafe_ptr())

        # =====================================================================
        # 3. Re-forward encoder with cache (for backward)
        # =====================================================================
        var symlog_obs = InlineArray[Scalar[dtype], BATCH * Self.OBS_DIM](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(Self.OBS_DIM):
                var val = Float32(rebind[Scalar[dtype]](obs[b, i]))
                symlog_obs[b * Self.OBS_DIM + i] = Scalar[dtype](symlog(val))
        var symlog_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ](symlog_obs.unsafe_ptr())

        var embed = InlineArray[Scalar[dtype], BATCH * Self.STOCH_FLAT](
            uninitialized=True
        )
        var embed_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](embed.unsafe_ptr())

        comptime ENC_CS = Self.EncModel.CACHE_SIZE
        var enc_cache = InlineArray[Scalar[dtype], BATCH * ENC_CS](
            uninitialized=True
        )
        var enc_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ENC_CS), MutAnyOrigin
        ](enc_cache.unsafe_ptr())

        Self.EncNet.forward_with_cache[BATCH](
            symlog_obs_t, embed_t, self.encoder.params_view(), enc_cache_t
        )

        # =====================================================================
        # 4. Re-forward posterior with cache (for backward)
        # =====================================================================
        var post_in = InlineArray[Scalar[dtype], BATCH * POST_IN](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(Self.DETER_DIM):
                post_in[b * POST_IN + i] = rebind[Scalar[dtype]](deter[b, i])
            for i in range(Self.STOCH_FLAT):
                post_in[b * POST_IN + Self.DETER_DIM + i] = embed[
                    b * Self.STOCH_FLAT + i
                ]
        var post_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, POST_IN), MutAnyOrigin
        ](post_in.unsafe_ptr())

        var post_logits = InlineArray[Scalar[dtype], BATCH * Self.STOCH_FLAT](
            uninitialized=True
        )
        var post_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](post_logits.unsafe_ptr())

        comptime POST_CS = Self.PostModel.CACHE_SIZE
        var post_cache = InlineArray[Scalar[dtype], BATCH * POST_CS](
            uninitialized=True
        )
        var post_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, POST_CS), MutAnyOrigin
        ](post_cache.unsafe_ptr())

        Self.PostNet.forward_with_cache[BATCH](
            post_in_t,
            post_logits_t,
            self.posterior.params_view(),
            post_cache_t,
        )

        # =====================================================================
        # 5. Posterior backward: grad_logits → grad_post_input
        # =====================================================================
        var grad_post_in = InlineArray[Scalar[dtype], BATCH * POST_IN](
            uninitialized=True
        )
        for i in range(BATCH * POST_IN):
            grad_post_in[i] = Scalar[dtype](0.0)
        var grad_post_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, POST_IN), MutAnyOrigin
        ](grad_post_in.unsafe_ptr())

        var post_grads = self.posterior.grads_view()
        Self.PostNet.backward[BATCH](
            grad_logits_t,
            grad_post_in_t,
            self.posterior.params_view(),
            post_cache_t,
            post_grads,
        )

        # =====================================================================
        # 6. Extract grad_embed from grad_post_input and encoder backward
        # =====================================================================
        # grad_post_input = [grad_deter_from_post, grad_embed]
        var grad_embed = InlineArray[Scalar[dtype], BATCH * Self.STOCH_FLAT](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(Self.STOCH_FLAT):
                grad_embed[b * Self.STOCH_FLAT + i] = grad_post_in[
                    b * POST_IN + Self.DETER_DIM + i
                ]
        var grad_embed_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ](grad_embed.unsafe_ptr())

        # Encoder backward: grad_embed → encoder param grads
        var grad_obs = InlineArray[Scalar[dtype], BATCH * Self.OBS_DIM](
            uninitialized=True
        )
        var grad_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ](grad_obs.unsafe_ptr())

        var enc_grads = self.encoder.grads_view()
        Self.EncNet.backward[BATCH](
            grad_embed_t,
            grad_obs_t,
            self.encoder.params_view(),
            enc_cache_t,
            enc_grads,
        )

        # =====================================================================
        # 7. Combine grad_deter contributions
        # =====================================================================
        # grad_deter_out = grad_deter (from feat split) + grad_deter_from_post
        for b in range(BATCH):
            for i in range(Self.DETER_DIM):
                # Direct contribution from feat split
                var gd_feat = Float64(grad_feat.ptr[b * Self.FEAT_DIM + i])
                # Contribution from posterior backward
                var gd_post = Float64(grad_post_in[b * POST_IN + i])
                grad_deter_out.ptr[b * Self.DETER_DIM + i] = Scalar[dtype](
                    gd_feat + gd_post
                )

    # =========================================================================
    # KL Loss Backward (Step 2 of BPTT)
    # =========================================================================
    #
    # DreamerV3 dual KL balancing:
    #   dyn_kl (weight 0.5): stop-grad on posterior → trains PRIOR
    #     d_KL/d_prior_probs = -p/q → softmax_vjp → Prior.backward
    #   rep_kl (weight 0.1): stop-grad on prior → trains POSTERIOR
    #     d_KL/d_post_probs = log(p) - log(q) + 1 → straight-through → Posterior.backward → Encoder.backward
    #
    # Free nats: if KL < FREE_NATS, gradient is zero (clamped).
    #
    # Gradient contributions:
    #   - Prior params (from dyn_kl)
    #   - Posterior params (from rep_kl)
    #   - Encoder params (from rep_kl → posterior → embed → encoder)
    #   - grad_deter += contributions from both KL terms through posterior/prior backward
    # =========================================================================

    def backward_kl_loss[
        BATCH: Int
    ](
        mut self,
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ],
        deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        post_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        prior_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        dyn_scale: Float64,
        rep_scale: Float64,
        mut grad_deter_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
    ):
        """Backward pass for KL loss with dual KL balancing.

        Implements DreamerV3's KL loss gradients:
        - dyn_kl (dyn_scale, typically 0.5): trains prior to match posterior
        - rep_kl (rep_scale, typically 0.1): trains posterior to match prior

        Free nats: if KL < FREE_NATS, no gradient is applied.

        ACCUMULATES into grad_deter_out (add to existing values).
        ACCUMULATES into prior/posterior/encoder param grads.

        Args:
            obs: Original observation (for encoder re-forward if rep_kl > 0).
            deter: Deterministic state at this timestep.
            post_probs: Posterior probabilities (from observe_step).
            prior_probs: Prior probabilities (from observe_step).
            dyn_scale: Weight for dynamics KL (typically 0.5).
            rep_scale: Weight for representation KL (typically 0.1).
            grad_deter_out: Accumulated gradient w.r.t. deter (ADDED to).
        """
        var eps = Float64(1e-8)

        # Compute KL to check free nats threshold
        var kl_val = Float64(0.0)
        for b in range(BATCH):
            for k in range(Self.STOCH_FLAT):
                var p = Float64(rebind[Scalar[dtype]](post_probs[b, k]))
                var q = Float64(rebind[Scalar[dtype]](prior_probs[b, k]))
                if p > eps:
                    kl_val += p * (log(p + eps) - log(q + eps))
        kl_val /= Float64(BATCH)

        # Free nats: if KL below threshold, no gradient
        if kl_val < Self.FREE_NATS:
            return

        # Normalization factor
        var inv_batch = 1.0 / Float64(BATCH)

        # =================================================================
        # dyn_kl: trains PRIOR (stop-grad on posterior)
        # d_KL/d_prior_probs[k] = -p[k] / q[k] (per sample, then /BATCH)
        # =================================================================
        if dyn_scale > 0.0:
            # Compute d_KL/d_prior_probs
            var grad_prior_probs = InlineArray[
                Scalar[dtype], BATCH * Self.STOCH_FLAT
            ](uninitialized=True)
            for b in range(BATCH):
                for k in range(Self.STOCH_FLAT):
                    var p = Float64(rebind[Scalar[dtype]](post_probs[b, k]))
                    var q = Float64(rebind[Scalar[dtype]](prior_probs[b, k]))
                    # d_KL/d_q = -p/q, scaled by dyn_scale / BATCH
                    var dkl_dq = -p / (q + eps) * dyn_scale * inv_batch
                    grad_prior_probs[b * Self.STOCH_FLAT + k] = Scalar[dtype](
                        dkl_dq
                    )

            # Softmax VJP through unimix to get grad_prior_logits
            var grad_prior_logits = InlineArray[
                Scalar[dtype], BATCH * Self.STOCH_FLAT
            ](uninitialized=True)

            comptime ONE_MINUS_UNIMIX = 1.0 - Self.UNIMIX

            for b in range(BATCH):
                for s in range(Self.STOCH_DIM):
                    var base = s * Self.CLASSES
                    # dot(grad_softmax, softmax) per group
                    var dot_gs: Float64 = 0.0
                    for c in range(Self.CLASSES):
                        var gp = Float64(
                            grad_prior_probs[b * Self.STOCH_FLAT + base + c]
                        )
                        var prob_c = Float64(
                            rebind[Scalar[dtype]](prior_probs[b, base + c])
                        )
                        var sm_c = (
                            prob_c - Self.UNIMIX / Float64(Self.CLASSES)
                        ) / ONE_MINUS_UNIMIX
                        if sm_c < 0.0:
                            sm_c = 0.0
                        var g_sm = ONE_MINUS_UNIMIX * gp
                        dot_gs += g_sm * sm_c

                    for c in range(Self.CLASSES):
                        var gp = Float64(
                            grad_prior_probs[b * Self.STOCH_FLAT + base + c]
                        )
                        var prob_c = Float64(
                            rebind[Scalar[dtype]](prior_probs[b, base + c])
                        )
                        var sm_c = (
                            prob_c - Self.UNIMIX / Float64(Self.CLASSES)
                        ) / ONE_MINUS_UNIMIX
                        if sm_c < 0.0:
                            sm_c = 0.0
                        var g_sm = ONE_MINUS_UNIMIX * gp
                        grad_prior_logits[
                            b * Self.STOCH_FLAT + base + c
                        ] = Scalar[dtype](sm_c * (g_sm - dot_gs))

            var grad_prior_logits_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.STOCH_FLAT),
                MutAnyOrigin,
            ](grad_prior_logits.unsafe_ptr())

            # Re-forward prior with cache
            comptime PRIOR_CS = Self.PriorModel.CACHE_SIZE
            var prior_cache = InlineArray[Scalar[dtype], BATCH * PRIOR_CS](
                uninitialized=True
            )
            var prior_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRIOR_CS), MutAnyOrigin
            ](prior_cache.unsafe_ptr())

            var prior_logits_tmp = InlineArray[
                Scalar[dtype], BATCH * Self.STOCH_FLAT
            ](uninitialized=True)
            var prior_logits_tmp_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.STOCH_FLAT),
                MutAnyOrigin,
            ](prior_logits_tmp.unsafe_ptr())

            Self.PriorNet.forward_with_cache[BATCH](
                deter,
                prior_logits_tmp_t,
                self.prior.params_view(),
                prior_cache_t,
            )

            # Prior backward → grad_deter_from_prior
            var grad_deter_prior = InlineArray[
                Scalar[dtype], BATCH * Self.DETER_DIM
            ](uninitialized=True)
            for i in range(BATCH * Self.DETER_DIM):
                grad_deter_prior[i] = Scalar[dtype](0.0)
            var grad_deter_prior_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.DETER_DIM),
                MutAnyOrigin,
            ](grad_deter_prior.unsafe_ptr())

            var prior_grads = self.prior.grads_view()
            Self.PriorNet.backward[BATCH](
                grad_prior_logits_t,
                grad_deter_prior_t,
                self.prior.params_view(),
                prior_cache_t,
                prior_grads,
            )

            # Accumulate into grad_deter_out
            for i in range(BATCH * Self.DETER_DIM):
                grad_deter_out.ptr[i] = (
                    grad_deter_out.ptr[i] + grad_deter_prior[i]
                )

        # =================================================================
        # rep_kl: trains POSTERIOR (stop-grad on prior)
        # d_KL/d_post_probs[k] = log(p[k]) - log(q[k]) + 1
        # =================================================================
        if rep_scale > 0.0:
            comptime POST_IN = Self.DETER_DIM + Self.STOCH_FLAT
            comptime ONE_MINUS_UNIMIX_R = 1.0 - Self.UNIMIX

            # Compute d_KL/d_post_probs
            var grad_post_probs = InlineArray[
                Scalar[dtype], BATCH * Self.STOCH_FLAT
            ](uninitialized=True)
            for b in range(BATCH):
                for k in range(Self.STOCH_FLAT):
                    var p = Float64(rebind[Scalar[dtype]](post_probs[b, k]))
                    var q = Float64(rebind[Scalar[dtype]](prior_probs[b, k]))
                    # d_KL/d_p = log(p) - log(q) + 1, scaled
                    var dkl_dp: Float64 = 0.0
                    if p > eps:
                        dkl_dp = (
                            (log(p + eps) - log(q + eps) + 1.0)
                            * rep_scale
                            * inv_batch
                        )
                    grad_post_probs[b * Self.STOCH_FLAT + k] = Scalar[dtype](
                        dkl_dp
                    )

            # Softmax VJP through unimix → grad_post_logits
            var grad_post_logits = InlineArray[
                Scalar[dtype], BATCH * Self.STOCH_FLAT
            ](uninitialized=True)

            for b in range(BATCH):
                for s in range(Self.STOCH_DIM):
                    var base = s * Self.CLASSES
                    var dot_gs: Float64 = 0.0
                    for c in range(Self.CLASSES):
                        var gp = Float64(
                            grad_post_probs[b * Self.STOCH_FLAT + base + c]
                        )
                        var prob_c = Float64(
                            rebind[Scalar[dtype]](post_probs[b, base + c])
                        )
                        var sm_c = (
                            prob_c - Self.UNIMIX / Float64(Self.CLASSES)
                        ) / ONE_MINUS_UNIMIX_R
                        if sm_c < 0.0:
                            sm_c = 0.0
                        var g_sm = ONE_MINUS_UNIMIX_R * gp
                        dot_gs += g_sm * sm_c

                    for c in range(Self.CLASSES):
                        var gp = Float64(
                            grad_post_probs[b * Self.STOCH_FLAT + base + c]
                        )
                        var prob_c = Float64(
                            rebind[Scalar[dtype]](post_probs[b, base + c])
                        )
                        var sm_c = (
                            prob_c - Self.UNIMIX / Float64(Self.CLASSES)
                        ) / ONE_MINUS_UNIMIX_R
                        if sm_c < 0.0:
                            sm_c = 0.0
                        var g_sm = ONE_MINUS_UNIMIX_R * gp
                        grad_post_logits[
                            b * Self.STOCH_FLAT + base + c
                        ] = Scalar[dtype](sm_c * (g_sm - dot_gs))

            var grad_post_logits_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.STOCH_FLAT),
                MutAnyOrigin,
            ](grad_post_logits.unsafe_ptr())

            # Re-forward encoder + posterior with cache
            var symlog_obs = InlineArray[Scalar[dtype], BATCH * Self.OBS_DIM](
                uninitialized=True
            )
            for b in range(BATCH):
                for i in range(Self.OBS_DIM):
                    var val = Float32(rebind[Scalar[dtype]](obs[b, i]))
                    symlog_obs[b * Self.OBS_DIM + i] = Scalar[dtype](
                        symlog(val)
                    )
            var symlog_obs_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
            ](symlog_obs.unsafe_ptr())

            var embed = InlineArray[Scalar[dtype], BATCH * Self.STOCH_FLAT](
                uninitialized=True
            )
            var embed_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.STOCH_FLAT),
                MutAnyOrigin,
            ](embed.unsafe_ptr())

            comptime ENC_CS = Self.EncModel.CACHE_SIZE
            var enc_cache = InlineArray[Scalar[dtype], BATCH * ENC_CS](
                uninitialized=True
            )
            var enc_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, ENC_CS), MutAnyOrigin
            ](enc_cache.unsafe_ptr())

            Self.EncNet.forward_with_cache[BATCH](
                symlog_obs_t,
                embed_t,
                self.encoder.params_view(),
                enc_cache_t,
            )

            # Build posterior input: concat(deter, embed)
            var post_in = InlineArray[Scalar[dtype], BATCH * POST_IN](
                uninitialized=True
            )
            for b in range(BATCH):
                for i in range(Self.DETER_DIM):
                    post_in[b * POST_IN + i] = rebind[Scalar[dtype]](
                        deter[b, i]
                    )
                for i in range(Self.STOCH_FLAT):
                    post_in[b * POST_IN + Self.DETER_DIM + i] = embed[
                        b * Self.STOCH_FLAT + i
                    ]
            var post_in_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, POST_IN), MutAnyOrigin
            ](post_in.unsafe_ptr())

            var post_logits_tmp = InlineArray[
                Scalar[dtype], BATCH * Self.STOCH_FLAT
            ](uninitialized=True)
            var post_logits_tmp_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.STOCH_FLAT),
                MutAnyOrigin,
            ](post_logits_tmp.unsafe_ptr())

            comptime POST_CS = Self.PostModel.CACHE_SIZE
            var post_cache = InlineArray[Scalar[dtype], BATCH * POST_CS](
                uninitialized=True
            )
            var post_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, POST_CS), MutAnyOrigin
            ](post_cache.unsafe_ptr())

            Self.PostNet.forward_with_cache[BATCH](
                post_in_t,
                post_logits_tmp_t,
                self.posterior.params_view(),
                post_cache_t,
            )

            # Posterior backward → grad_post_input
            var grad_post_in = InlineArray[Scalar[dtype], BATCH * POST_IN](
                uninitialized=True
            )
            for i in range(BATCH * POST_IN):
                grad_post_in[i] = Scalar[dtype](0.0)
            var grad_post_in_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, POST_IN), MutAnyOrigin
            ](grad_post_in.unsafe_ptr())

            var post_grads = self.posterior.grads_view()
            Self.PostNet.backward[BATCH](
                grad_post_logits_t,
                grad_post_in_t,
                self.posterior.params_view(),
                post_cache_t,
                post_grads,
            )

            # Extract grad_embed and encoder backward
            var grad_embed = InlineArray[
                Scalar[dtype], BATCH * Self.STOCH_FLAT
            ](uninitialized=True)
            for b in range(BATCH):
                for i in range(Self.STOCH_FLAT):
                    grad_embed[b * Self.STOCH_FLAT + i] = grad_post_in[
                        b * POST_IN + Self.DETER_DIM + i
                    ]
            var grad_embed_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.STOCH_FLAT),
                MutAnyOrigin,
            ](grad_embed.unsafe_ptr())

            var grad_obs = InlineArray[Scalar[dtype], BATCH * Self.OBS_DIM](
                uninitialized=True
            )
            var grad_obs_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.OBS_DIM),
                MutAnyOrigin,
            ](grad_obs.unsafe_ptr())

            var enc_grads = self.encoder.grads_view()
            Self.EncNet.backward[BATCH](
                grad_embed_t,
                grad_obs_t,
                self.encoder.params_view(),
                enc_cache_t,
                enc_grads,
            )

            # Accumulate grad_deter from posterior input
            for b in range(BATCH):
                for i in range(Self.DETER_DIM):
                    grad_deter_out.ptr[b * Self.DETER_DIM + i] = (
                        grad_deter_out.ptr[b * Self.DETER_DIM + i]
                        + grad_post_in[b * POST_IN + i]
                    )

    # =========================================================================
    # GRU Core Backward (Step 3 of BPTT)
    # =========================================================================
    #
    # Backpropagates grad_new_deter through the GRU core to produce
    # grad_prev_deter and grad_prev_stoch for the previous timestep.
    #
    # Forward was:
    #   proj_d = DeterProj(prev_deter)
    #   proj_s = StochProj(prev_stoch)
    #   proj_a = ActionProj(norm_action)
    #   concat = [prev_deter, proj_d, proj_s, proj_a]
    #   hidden = GRUHidden(concat)
    #   gates = GRUGates(hidden) → [reset, cand, update] logits
    #   new_deter = update * tanh(reset * cand) + (1-update) * prev_deter
    #
    # Backward:
    #   7. Gate application backward → d_gates, d_prev_deter_direct
    #   6. GRUGates.backward(d_gates) → d_hidden
    #   5. GRUHidden.backward(d_hidden) → d_concat
    #   4. Split d_concat → d_prev_deter_concat, d_proj_d, d_proj_s, d_proj_a
    #   3. ActionProj.backward(d_proj_a) → action param grads
    #   2. StochProj.backward(d_proj_s) → d_prev_stoch + stoch param grads
    #   1. DeterProj.backward(d_proj_d) → d_prev_deter_proj + deter param grads
    #   Total: d_prev_deter = direct + concat + proj contributions
    # =========================================================================

    def backward_gru_core[
        BATCH: Int
    ](
        mut self,
        grad_new_deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        prev_deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        prev_stoch: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
        prev_action: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTION_DIM), MutAnyOrigin
        ],
        mut grad_prev_deter: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DETER_DIM), MutAnyOrigin
        ],
        mut grad_prev_stoch: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STOCH_FLAT), MutAnyOrigin
        ],
    ):
        """Backward through GRU core for BPTT.

        Re-forwards all GRU sub-networks with cache, then backward through
        gate application and all sub-networks.

        Accumulates param grads for DeterProj, StochProj, ActionProj,
        GRUHidden, GRUGates.

        Args:
            grad_new_deter: Gradient w.r.t. new_deter (from prediction heads + KL).
            prev_deter: Previous deterministic state (input to GRU).
            prev_stoch: Previous stochastic state (input to GRU).
            prev_action: Previous action (input to GRU).
            grad_prev_deter: Output gradient w.r.t. prev_deter (WRITTEN, for t-1).
            grad_prev_stoch: Output gradient w.r.t. prev_stoch (WRITTEN, for t-1).
        """
        comptime DETER = Self.DETER_DIM
        comptime GRU_CONCAT = DETER + 3 * Self.HIDDEN

        # =================================================================
        # 1. Normalize action (same as forward)
        # =================================================================
        var norm_action = InlineArray[Scalar[dtype], BATCH * Self.ACTION_DIM](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(Self.ACTION_DIM):
                var a = rebind[Scalar[dtype]](prev_action[b, i])
                var abs_a = abs(a)
                var one = Scalar[dtype](1.0)
                if abs_a > one:
                    norm_action[b * Self.ACTION_DIM + i] = a / abs_a
                else:
                    norm_action[b * Self.ACTION_DIM + i] = a
        var norm_action_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTION_DIM), MutAnyOrigin
        ](norm_action.unsafe_ptr())

        # =================================================================
        # 2. Re-forward projections with cache
        # =================================================================
        comptime DP_CS = Self.DeterProj.CACHE_SIZE
        comptime SP_CS = Self.StochProj.CACHE_SIZE
        comptime AP_CS = Self.ActionProj.CACHE_SIZE
        comptime GH_CS = Self.GRUHiddenModel.CACHE_SIZE
        comptime GG_CS = Self.GRUGateModel.CACHE_SIZE

        var proj_d = InlineArray[Scalar[dtype], BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var proj_d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](proj_d.unsafe_ptr())
        var dp_cache = InlineArray[Scalar[dtype], BATCH * DP_CS](
            uninitialized=True
        )
        var dp_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DP_CS), MutAnyOrigin
        ](dp_cache.unsafe_ptr())
        Self.DeterProjNet.forward_with_cache[BATCH](
            prev_deter, proj_d_t, self.deter_proj.params_view(), dp_cache_t
        )

        var proj_s = InlineArray[Scalar[dtype], BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var proj_s_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](proj_s.unsafe_ptr())
        var sp_cache = InlineArray[Scalar[dtype], BATCH * SP_CS](
            uninitialized=True
        )
        var sp_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, SP_CS), MutAnyOrigin
        ](sp_cache.unsafe_ptr())
        Self.StochProjNet.forward_with_cache[BATCH](
            prev_stoch, proj_s_t, self.stoch_proj.params_view(), sp_cache_t
        )

        var proj_a = InlineArray[Scalar[dtype], BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var proj_a_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](proj_a.unsafe_ptr())
        var ap_cache = InlineArray[Scalar[dtype], BATCH * AP_CS](
            uninitialized=True
        )
        var ap_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, AP_CS), MutAnyOrigin
        ](ap_cache.unsafe_ptr())
        Self.ActionProjNet.forward_with_cache[BATCH](
            norm_action_t, proj_a_t, self.action_proj.params_view(), ap_cache_t
        )

        # =================================================================
        # 3. Re-forward GRU concat → hidden → gates with cache
        # =================================================================
        var concat_buf = InlineArray[Scalar[dtype], BATCH * GRU_CONCAT](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(DETER):
                concat_buf[b * GRU_CONCAT + i] = rebind[Scalar[dtype]](
                    prev_deter[b, i]
                )
            for i in range(Self.HIDDEN):
                concat_buf[b * GRU_CONCAT + DETER + i] = proj_d[
                    b * Self.HIDDEN + i
                ]
                concat_buf[b * GRU_CONCAT + DETER + Self.HIDDEN + i] = proj_s[
                    b * Self.HIDDEN + i
                ]
                concat_buf[
                    b * GRU_CONCAT + DETER + 2 * Self.HIDDEN + i
                ] = proj_a[b * Self.HIDDEN + i]
        var concat_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GRU_CONCAT), MutAnyOrigin
        ](concat_buf.unsafe_ptr())

        var hidden_out = InlineArray[Scalar[dtype], BATCH * DETER](
            uninitialized=True
        )
        var hidden_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
        ](hidden_out.unsafe_ptr())
        var gh_cache = InlineArray[Scalar[dtype], BATCH * GH_CS](
            uninitialized=True
        )
        var gh_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GH_CS), MutAnyOrigin
        ](gh_cache.unsafe_ptr())
        Self.GRUHiddenNet.forward_with_cache[BATCH](
            concat_t, hidden_t, self.gru_hidden.params_view(), gh_cache_t
        )

        var gate_out = InlineArray[Scalar[dtype], BATCH * 3 * DETER](
            uninitialized=True
        )
        var gate_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 3 * DETER), MutAnyOrigin
        ](gate_out.unsafe_ptr())
        var gg_cache = InlineArray[Scalar[dtype], BATCH * GG_CS](
            uninitialized=True
        )
        var gg_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GG_CS), MutAnyOrigin
        ](gg_cache.unsafe_ptr())
        Self.GRUGateNet.forward_with_cache[BATCH](
            hidden_t, gate_t, self.gru_gates.params_view(), gg_cache_t
        )

        # =================================================================
        # 4. Gate application backward
        # =================================================================
        # Forward was:
        #   reset = sigmoid(gate[i])
        #   cand = tanh(reset * gate[DETER+i])
        #   update = sigmoid(gate[2*DETER+i] - 1)
        #   new_deter[i] = update * cand + (1 - update) * old_d[i]
        var d_gate_out = InlineArray[Scalar[dtype], BATCH * 3 * DETER](
            uninitialized=True
        )
        for i in range(BATCH * 3 * DETER):
            d_gate_out[i] = Scalar[dtype](0.0)

        # Initialize grad_prev_deter with direct contribution
        for i in range(BATCH * DETER):
            grad_prev_deter.ptr[i] = Scalar[dtype](0.0)

        for b in range(BATCH):
            for i in range(DETER):
                var d_nd = Float64(grad_new_deter.ptr[b * DETER + i])

                var reset_logit = Float64(gate_out[b * 3 * DETER + i])
                var cand_logit = Float64(gate_out[b * 3 * DETER + DETER + i])
                var update_logit = Float64(
                    gate_out[b * 3 * DETER + 2 * DETER + i]
                )

                var one = 1.0

                # Recompute gate values
                var reset_val = one / (one + exp(-reset_logit))
                var rc = reset_val * cand_logit
                var cand_val = (exp(rc) - exp(-rc)) / (
                    exp(rc) + exp(-rc)
                )  # tanh(rc)
                var update_val = one / (one + exp(-(update_logit - one)))
                var old_d = Float64(rebind[Scalar[dtype]](prev_deter[b, i]))

                # Backward through: new_d = update * cand + (1-update) * old_d
                var d_update = d_nd * (cand_val - old_d)
                var d_cand = d_nd * update_val
                var d_old_d = d_nd * (one - update_val)

                # update = sigmoid(update_logit - 1)
                var d_update_logit = d_update * update_val * (one - update_val)

                # cand = tanh(rc), d_tanh = 1 - tanh^2
                var d_rc = d_cand * (one - cand_val * cand_val)

                # rc = reset * cand_logit
                var d_reset = d_rc * cand_logit
                var d_cand_logit = d_rc * reset_val

                # reset = sigmoid(reset_logit)
                var d_reset_logit = d_reset * reset_val * (one - reset_val)

                d_gate_out[b * 3 * DETER + i] = Scalar[dtype](d_reset_logit)
                d_gate_out[b * 3 * DETER + DETER + i] = Scalar[dtype](
                    d_cand_logit
                )
                d_gate_out[b * 3 * DETER + 2 * DETER + i] = Scalar[dtype](
                    d_update_logit
                )

                # Direct contribution to prev_deter
                grad_prev_deter.ptr[b * DETER + i] = Scalar[dtype](d_old_d)

        var d_gate_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 3 * DETER), MutAnyOrigin
        ](d_gate_out.unsafe_ptr())

        # =================================================================
        # 5. GRUGates backward: d_gate_out → d_hidden
        # =================================================================
        var d_hidden = InlineArray[Scalar[dtype], BATCH * DETER](
            uninitialized=True
        )
        for i in range(BATCH * DETER):
            d_hidden[i] = Scalar[dtype](0.0)
        var d_hidden_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
        ](d_hidden.unsafe_ptr())

        var gg_grads = self.gru_gates.grads_view()
        Self.GRUGateNet.backward[BATCH](
            d_gate_t,
            d_hidden_t,
            self.gru_gates.params_view(),
            gg_cache_t,
            gg_grads,
        )

        # =================================================================
        # 6. GRUHidden backward: d_hidden → d_concat
        # =================================================================
        var d_concat = InlineArray[Scalar[dtype], BATCH * GRU_CONCAT](
            uninitialized=True
        )
        for i in range(BATCH * GRU_CONCAT):
            d_concat[i] = Scalar[dtype](0.0)
        var d_concat_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, GRU_CONCAT), MutAnyOrigin
        ](d_concat.unsafe_ptr())

        var gh_grads = self.gru_hidden.grads_view()
        Self.GRUHiddenNet.backward[BATCH](
            d_hidden_t,
            d_concat_t,
            self.gru_hidden.params_view(),
            gh_cache_t,
            gh_grads,
        )

        # =================================================================
        # 7. Split d_concat → d_prev_deter_concat, d_proj_d/s/a
        # =================================================================
        # concat = [prev_deter, proj_d, proj_s, proj_a]
        # Accumulate d_prev_deter from concat portion
        for b in range(BATCH):
            for i in range(DETER):
                grad_prev_deter.ptr[b * DETER + i] = (
                    grad_prev_deter.ptr[b * DETER + i]
                    + d_concat[b * GRU_CONCAT + i]
                )

        # Extract projection gradients
        var d_proj_d = InlineArray[Scalar[dtype], BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var d_proj_s = InlineArray[Scalar[dtype], BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var d_proj_a = InlineArray[Scalar[dtype], BATCH * Self.HIDDEN](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(Self.HIDDEN):
                d_proj_d[b * Self.HIDDEN + i] = d_concat[
                    b * GRU_CONCAT + DETER + i
                ]
                d_proj_s[b * Self.HIDDEN + i] = d_concat[
                    b * GRU_CONCAT + DETER + Self.HIDDEN + i
                ]
                d_proj_a[b * Self.HIDDEN + i] = d_concat[
                    b * GRU_CONCAT + DETER + 2 * Self.HIDDEN + i
                ]

        # =================================================================
        # 8. Projection backwards → param grads + grad_prev_deter/stoch
        # =================================================================
        var d_proj_d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](d_proj_d.unsafe_ptr())
        var d_proj_s_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](d_proj_s.unsafe_ptr())
        var d_proj_a_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN), MutAnyOrigin
        ](d_proj_a.unsafe_ptr())

        # DeterProj backward → d_prev_deter_proj
        var d_prev_deter_proj = InlineArray[Scalar[dtype], BATCH * DETER](
            uninitialized=True
        )
        for i in range(BATCH * DETER):
            d_prev_deter_proj[i] = Scalar[dtype](0.0)
        var d_prev_deter_proj_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
        ](d_prev_deter_proj.unsafe_ptr())

        var dp_grads = self.deter_proj.grads_view()
        Self.DeterProjNet.backward[BATCH](
            d_proj_d_t,
            d_prev_deter_proj_t,
            self.deter_proj.params_view(),
            dp_cache_t,
            dp_grads,
        )

        # Accumulate into grad_prev_deter
        for i in range(BATCH * DETER):
            grad_prev_deter.ptr[i] = (
                grad_prev_deter.ptr[i] + d_prev_deter_proj[i]
            )

        # StochProj backward → grad_prev_stoch
        for i in range(BATCH * Self.STOCH_FLAT):
            grad_prev_stoch.ptr[i] = Scalar[dtype](0.0)

        var sp_grads = self.stoch_proj.grads_view()
        Self.StochProjNet.backward[BATCH](
            d_proj_s_t,
            grad_prev_stoch,
            self.stoch_proj.params_view(),
            sp_cache_t,
            sp_grads,
        )

        # ActionProj backward → d_action (not needed for training, but
        # accumulates param grads for ActionProj)
        var d_action = InlineArray[Scalar[dtype], BATCH * Self.ACTION_DIM](
            uninitialized=True
        )
        for i in range(BATCH * Self.ACTION_DIM):
            d_action[i] = Scalar[dtype](0.0)
        var d_action_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTION_DIM), MutAnyOrigin
        ](d_action.unsafe_ptr())

        var ap_grads = self.action_proj.grads_view()
        Self.ActionProjNet.backward[BATCH](
            d_proj_a_t,
            d_action_t,
            self.action_proj.params_view(),
            ap_cache_t,
            ap_grads,
        )
