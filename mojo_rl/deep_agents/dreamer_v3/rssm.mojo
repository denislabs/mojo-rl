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

from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearMish, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState
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


fn categorical_sample[
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
                var e = exp(rebind[Scalar[dtype]](logits[b, base + c]) - max_val)
                probs[b, base + c] = e  # temporary storage
                sum_exp += e

            # Normalize to softmax, then apply unimix
            for c in range(CLASSES):
                var softmax_p = rebind[Scalar[dtype]](probs[b, base + c]) / sum_exp
                probs[b, base + c] = one_minus_unimix * softmax_p + unimix * uniform_prob

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


fn kl_divergence[
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

    fn __init__(out self):
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

        self.gru_hidden = NetworkState[Self.GRUHiddenModel, Adam[LR=Self.WM_LR]]()
        self.gru_hidden.initialize[Kaiming[]]()

        self.gru_gates = NetworkState[Self.GRUGateModel, Adam[LR=Self.WM_LR]]()
        self.gru_gates.initialize[Kaiming[]]()

        # Compute symlog-spaced bins for distributional reward head
        self.bins = compute_symlog_bins[Self.NUM_BINS]()

    fn __init__(out self, *, deinit take: Self):
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

    fn core_forward[
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
                var reset_logit = rebind[Scalar[dtype]](
                    gate_out[b, i]
                )
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
                new_deter[b, i] = update_val * cand_val + (
                    one - update_val
                ) * old_d

        # Free normalized action scratch
        norm_action_ptr.free()

    # =========================================================================
    # Observe Step (Posterior — uses observations)
    # =========================================================================

    fn observe_step[
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

    fn imagine_step[
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

    fn decode[
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
        Self.DecNet.forward[BATCH](
            feat, obs_pred, self.decoder.params_view()
        )

    fn predict_reward[
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

    fn predict_continue[
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

    fn zero_all_grads(mut self):
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

    fn update_all_params(mut self):
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
