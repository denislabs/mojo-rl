"""World Model for TDMPC2.

The world model simultaneously learns:
  - encoder:     observation → latent state z
  - dynamics:    (z, a) → z' (next latent state with SimNorm output)
  - reward:      (z, a) → logits[NUM_BINS] (distributional)
  - termination: z → scalar in (0,1) (episode termination probability)
  - policy:      z → (mean, log_std) for Gaussian prior used in MPPI
  - Q-ensemble:  NUM_Q × (z, a) → logits[NUM_BINS] (distributional Q-values)

Architecture (all MLPs use NormedLinear blocks):
  encoder:     [NL(OBS, MLP), NL(MLP, LATENT)]
  dynamics:    [NL(LATENT+ACT, MLP), NL(MLP, LATENT), Linear(LATENT, LATENT) + SimNorm]
  reward:      [NL(LATENT+ACT, MLP), NL(MLP, MLP), Linear(MLP, NUM_BINS)]
  termination: [NL(LATENT, MLP), NL(MLP, MLP), Linear(MLP, 1) + Sigmoid]
  policy:      [NL(LATENT, MLP), NL(MLP, MLP), Linear(MLP, 2*ACT)]
  Q_i:         [NL(LATENT+ACT, MLP), NL(MLP, MLP), Linear(MLP, NUM_BINS)]

Reference: Hansen et al., 2023 — TD-MPC2
"""

from std.math import exp, log, sqrt
from std.randomndom import random_float64

from nn.constants import dtype
from nn.model import (
    Linear,
    Sequential,
    Sigmoid,
    NormedLinear,
    SimNorm,
)
from nn.optimizer import Adam
from nn.initializer import Kaiming
from nn.training import Network
from nn.loss.two_hot import (
    compute_bins,
    two_hot_encode_batch,
    decode_value_batch,
)


struct WorldModel[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    LATENT_DIM: Int = 256,
    MLP_DIM: Int = 256,
    NUM_BINS: Int = 101,
    NUM_Q: Int = 5,
    SIMPLEX_DIM: Int = 8,
    V_MIN: Float64 = -10.0,
    V_MAX: Float64 = 10.0,
]:
    """World model for TDMPC2 with encoder, dynamics, reward, termination,
    policy, and Q-function ensemble.

    Parameters:
        OBS_DIM: Observation dimension.
        ACTION_DIM: Action dimension.
        LATENT_DIM: Latent state dimension (default: 256).
        MLP_DIM: Hidden layer width (default: 256).
        NUM_BINS: Number of bins for distributional RL (default: 101).
        NUM_Q: Number of Q-networks in the ensemble (default: 5).
        SIMPLEX_DIM: SimNorm group size for dynamics head (default: 8).
        V_MIN: Minimum value for distribution bins (default: -10.0).
        V_MAX: Maximum value for distribution bins (default: 10.0).

    Note: LATENT_DIM must be divisible by SIMPLEX_DIM.
    """

    # Concatenated input dimensions
    comptime ZA_DIM: Int = Self.LATENT_DIM + Self.ACTION_DIM

    # -------------------------------------------------------------------------
    # Network type definitions
    # -------------------------------------------------------------------------

    # Encoder: OBS_DIM → LATENT_DIM
    comptime EncModel = Sequential[
        NormedLinear[Self.OBS_DIM, Self.MLP_DIM],
        NormedLinear[Self.MLP_DIM, Self.LATENT_DIM],
    ]

    # Dynamics: (LATENT + ACTION) → LATENT with SimNorm output

    comptime DynModel = Sequential[
        NormedLinear[Self.ZA_DIM, Self.MLP_DIM],
        NormedLinear[Self.MLP_DIM, Self.LATENT_DIM],
        Linear[Self.LATENT_DIM, Self.LATENT_DIM],
        SimNorm[Self.LATENT_DIM, Self.SIMPLEX_DIM],
    ]

    # Reward: (LATENT + ACTION) → NUM_BINS logits
    comptime RewModel = Sequential[
        NormedLinear[Self.ZA_DIM, Self.MLP_DIM],
        NormedLinear[Self.MLP_DIM, Self.MLP_DIM],
        Linear[Self.MLP_DIM, Self.NUM_BINS],
    ]

    # Termination: LATENT → scalar in (0,1)

    comptime TermModel = Sequential[
        NormedLinear[Self.LATENT_DIM, Self.MLP_DIM],
        NormedLinear[Self.MLP_DIM, Self.MLP_DIM],
        Linear[Self.MLP_DIM, 1],
        Sigmoid[1],
    ]

    # Policy: LATENT → (mean, log_std) for Gaussian actions
    comptime PolModel = Sequential[
        NormedLinear[Self.LATENT_DIM, Self.MLP_DIM],
        NormedLinear[Self.MLP_DIM, Self.MLP_DIM],
        Linear[Self.MLP_DIM, 2 * Self.ACTION_DIM],
    ]

    # Q-network: (LATENT + ACTION) → NUM_BINS logits
    comptime QModel = Sequential[
        NormedLinear[Self.ZA_DIM, Self.MLP_DIM],
        NormedLinear[Self.MLP_DIM, Self.MLP_DIM],
        Linear[Self.MLP_DIM, Self.NUM_BINS],
    ]

    # Network wrapper types
    comptime EncoderNet = Network[Self.EncModel, Adam, Kaiming]
    comptime DynamicsNet = Network[Self.DynModel, Adam, Kaiming]
    comptime RewardNet = Network[Self.RewModel, Adam, Kaiming]
    comptime TermNet = Network[Self.TermModel, Adam, Kaiming]
    comptime PolicyNet = Network[Self.PolModel, Adam, Kaiming]
    comptime QNet = Network[Self.QModel, Adam, Kaiming]

    # -------------------------------------------------------------------------
    # Sub-networks
    # -------------------------------------------------------------------------
    var encoder: Self.EncoderNet
    var dynamics: Self.DynamicsNet
    var reward_head: Self.RewardNet
    var termination: Self.TermNet
    var policy: Self.PolicyNet

    # Q-ensemble (NUM_Q=5 networks)
    var q1: Self.QNet
    var q2: Self.QNet
    var q3: Self.QNet
    var q4: Self.QNet
    var q5: Self.QNet

    # Target Q-networks (no gradient, soft-updated from live Qs)
    var q1_target: Self.QNet
    var q2_target: Self.QNet
    var q3_target: Self.QNet
    var q4_target: Self.QNet
    var q5_target: Self.QNet

    # Fixed bin values for distributional RL
    var bins: InlineArray[Float32, Self.NUM_BINS]

    fn __init__(
        out self,
        enc_lr: Float64 = 9e-5,  # encoder LR = 0.3 * world_model_lr
        wm_lr: Float64 = 3e-4,  # world model (non-encoder) LR
        pi_lr: Float64 = 3e-4,  # policy LR
    ):
        """Initialize WorldModel with all sub-networks.

        Args:
            enc_lr: Encoder learning rate (typically 0.3 * wm_lr).
            wm_lr: World model learning rate.
            pi_lr: Policy learning rate.
        """
        self.encoder = Self.EncoderNet(Adam(lr=enc_lr), Kaiming())
        self.dynamics = Self.DynamicsNet(Adam(lr=wm_lr), Kaiming())
        self.reward_head = Self.RewardNet(Adam(lr=wm_lr), Kaiming())
        self.termination = Self.TermNet(Adam(lr=wm_lr), Kaiming())
        self.policy = Self.PolicyNet(Adam(lr=pi_lr), Kaiming())

        self.q1 = Self.QNet(Adam(lr=wm_lr), Kaiming())
        self.q2 = Self.QNet(Adam(lr=wm_lr), Kaiming())
        self.q3 = Self.QNet(Adam(lr=wm_lr), Kaiming())
        self.q4 = Self.QNet(Adam(lr=wm_lr), Kaiming())
        self.q5 = Self.QNet(Adam(lr=wm_lr), Kaiming())

        self.q1_target = Self.QNet(Adam(lr=wm_lr), Kaiming())
        self.q2_target = Self.QNet(Adam(lr=wm_lr), Kaiming())
        self.q3_target = Self.QNet(Adam(lr=wm_lr), Kaiming())
        self.q4_target = Self.QNet(Adam(lr=wm_lr), Kaiming())
        self.q5_target = Self.QNet(Adam(lr=wm_lr), Kaiming())

        # Initialize target Q networks with same weights as live Q networks
        self.q1_target.copy_params_from(self.q1)
        self.q2_target.copy_params_from(self.q2)
        self.q3_target.copy_params_from(self.q3)
        self.q4_target.copy_params_from(self.q4)
        self.q5_target.copy_params_from(self.q5)

        # Compute bin values
        self.bins = compute_bins[Self.NUM_BINS](
            Float32(Self.V_MIN), Float32(Self.V_MAX)
        )

    # =========================================================================
    # Forward Methods
    # =========================================================================

    fn encode[
        BATCH: Int
    ](
        self,
        obs_ptr: UnsafePointer[Scalar[dtype]],
        z_ptr: UnsafePointer[Scalar[dtype]],
    ):
        """Encode observations to latent states (no cache, stop-gradient).

        Args:
            obs_ptr: Pointer to input observations [BATCH * OBS_DIM].
            z_ptr: Pointer to output latent states [BATCH * LATENT_DIM] (written).
        """
        self.encoder.forward_ptr[BATCH](obs_ptr, z_ptr)

    fn encode_with_cache[
        BATCH: Int
    ](
        self,
        obs_ptr: UnsafePointer[Scalar[dtype]],
        z_ptr: UnsafePointer[Scalar[dtype]],
        mut cache: List[Scalar[dtype]],
    ):
        """Encode with cache for backpropagation.

        Args:
            obs_ptr: Pointer to input observations [BATCH * OBS_DIM].
            z_ptr: Pointer to output latent states [BATCH * LATENT_DIM] (written).
            cache: Pre-allocated cache [BATCH * EncModel.CACHE_SIZE] (written).
        """
        self.encoder.forward_with_cache_ptr[BATCH](obs_ptr, z_ptr, cache)

    fn dynamics_forward[
        BATCH: Int
    ](
        self,
        z_a_ptr: UnsafePointer[Scalar[dtype]],
        z_next_ptr: UnsafePointer[Scalar[dtype]],
    ):
        """Predict next latent state (no cache).

        Args:
            z_a_ptr: Pointer to concatenated (latent, action) [BATCH * ZA_DIM].
            z_next_ptr: Pointer to output next latent state [BATCH * LATENT_DIM] (written).
        """
        self.dynamics.forward_ptr[BATCH](z_a_ptr, z_next_ptr)

    fn dynamics_forward_with_cache[
        BATCH: Int
    ](
        self,
        z_a_ptr: UnsafePointer[Scalar[dtype]],
        z_next_ptr: UnsafePointer[Scalar[dtype]],
        mut cache: List[Scalar[dtype]],
    ):
        """Predict next latent state with cache for backprop."""
        self.dynamics.forward_with_cache_ptr[BATCH](z_a_ptr, z_next_ptr, cache)

    fn reward_forward[
        BATCH: Int
    ](
        self,
        z_a_ptr: UnsafePointer[Scalar[dtype]],
        logits_ptr: UnsafePointer[Scalar[dtype]],
    ):
        """Predict reward distribution logits (no cache)."""
        self.reward_head.forward_ptr[BATCH](z_a_ptr, logits_ptr)

    fn reward_forward_with_cache[
        BATCH: Int
    ](
        self,
        z_a_ptr: UnsafePointer[Scalar[dtype]],
        logits_ptr: UnsafePointer[Scalar[dtype]],
        mut cache: List[Scalar[dtype]],
    ):
        """Predict reward distribution logits with cache."""
        self.reward_head.forward_with_cache_ptr[BATCH](
            z_a_ptr, logits_ptr, cache
        )

    fn termination_forward[
        BATCH: Int
    ](
        self,
        z_ptr: UnsafePointer[Scalar[dtype]],
        term_prob_ptr: UnsafePointer[Scalar[dtype]],
    ):
        """Predict termination probability (no cache).

        Args:
            z_ptr: Pointer to latent states [BATCH * LATENT_DIM].
            term_prob_ptr: Pointer to output termination probabilities [BATCH] (written).
        """
        var out = List[Scalar[dtype]](capacity=BATCH)
        for _ in range(BATCH):
            out.append(Scalar[dtype](0))
        self.termination.forward_ptr[BATCH](z_ptr, out.unsafe_ptr())
        for b in range(BATCH):
            term_prob_ptr[b] = out[b]

    fn policy_forward[
        BATCH: Int
    ](
        self,
        z_ptr: UnsafePointer[Scalar[dtype]],
        mean_ptr: UnsafePointer[Scalar[dtype]],
        log_std_ptr: UnsafePointer[Scalar[dtype]],
    ):
        """Predict Gaussian policy parameters (no cache).

        Args:
            z_ptr: Pointer to latent states [BATCH * LATENT_DIM].
            mean_ptr: Pointer to output action mean [BATCH * ACTION_DIM] (written).
            log_std_ptr: Pointer to output log std [BATCH * ACTION_DIM] (written).
        """
        var out = List[Scalar[dtype]](capacity=BATCH * 2 * Self.ACTION_DIM)
        for _ in range(BATCH * 2 * Self.ACTION_DIM):
            out.append(Scalar[dtype](0))
        self.policy.forward_ptr[BATCH](z_ptr, out.unsafe_ptr())
        for b in range(BATCH):
            for i in range(Self.ACTION_DIM):
                mean_ptr[b * Self.ACTION_DIM + i] = out[
                    b * 2 * Self.ACTION_DIM + i
                ]
                # Clamp log_std to [-10, 2] for numerical stability
                var ls = Float64(
                    out[b * 2 * Self.ACTION_DIM + Self.ACTION_DIM + i]
                )
                if ls < -10.0:
                    ls = -10.0
                if ls > 2.0:
                    ls = 2.0
                log_std_ptr[b * Self.ACTION_DIM + i] = Scalar[dtype](ls)

    fn policy_forward_with_cache[
        BATCH: Int
    ](
        self,
        z_ptr: UnsafePointer[Scalar[dtype]],
        out_ptr: UnsafePointer[Scalar[dtype]],
        mut cache: List[Scalar[dtype]],
    ):
        """Predict policy output with cache for backprop."""
        self.policy.forward_with_cache_ptr[BATCH](z_ptr, out_ptr, cache)

    fn q_forward[
        BATCH: Int
    ](
        self,
        z_a_ptr: UnsafePointer[Scalar[dtype]],
        q_logits_ptr: UnsafePointer[Scalar[dtype]],
        use_target: Bool = False,
    ):
        """Forward pass through all Q-networks.

        Args:
            z_a_ptr: Pointer to concatenated (latent, action) [BATCH * ZA_DIM].
            q_logits_ptr: Pointer to output logits [NUM_Q * BATCH * NUM_BINS] (written).
            use_target: If True, use target Q-networks (default: False).
        """
        var logits1 = List[Scalar[dtype]](capacity=BATCH * Self.NUM_BINS)
        var logits2 = List[Scalar[dtype]](capacity=BATCH * Self.NUM_BINS)
        var logits3 = List[Scalar[dtype]](capacity=BATCH * Self.NUM_BINS)
        var logits4 = List[Scalar[dtype]](capacity=BATCH * Self.NUM_BINS)
        var logits5 = List[Scalar[dtype]](capacity=BATCH * Self.NUM_BINS)
        for _ in range(BATCH * Self.NUM_BINS):
            logits1.append(Scalar[dtype](0))
            logits2.append(Scalar[dtype](0))
            logits3.append(Scalar[dtype](0))
            logits4.append(Scalar[dtype](0))
            logits5.append(Scalar[dtype](0))

        if use_target:
            self.q1_target.forward_ptr[BATCH](z_a_ptr, logits1.unsafe_ptr())
            self.q2_target.forward_ptr[BATCH](z_a_ptr, logits2.unsafe_ptr())
            self.q3_target.forward_ptr[BATCH](z_a_ptr, logits3.unsafe_ptr())
            self.q4_target.forward_ptr[BATCH](z_a_ptr, logits4.unsafe_ptr())
            self.q5_target.forward_ptr[BATCH](z_a_ptr, logits5.unsafe_ptr())
        else:
            self.q1.forward_ptr[BATCH](z_a_ptr, logits1.unsafe_ptr())
            self.q2.forward_ptr[BATCH](z_a_ptr, logits2.unsafe_ptr())
            self.q3.forward_ptr[BATCH](z_a_ptr, logits3.unsafe_ptr())
            self.q4.forward_ptr[BATCH](z_a_ptr, logits4.unsafe_ptr())
            self.q5.forward_ptr[BATCH](z_a_ptr, logits5.unsafe_ptr())

        for b in range(BATCH * Self.NUM_BINS):
            q_logits_ptr[0 * BATCH * Self.NUM_BINS + b] = logits1[b]
            q_logits_ptr[1 * BATCH * Self.NUM_BINS + b] = logits2[b]
            q_logits_ptr[2 * BATCH * Self.NUM_BINS + b] = logits3[b]
            q_logits_ptr[3 * BATCH * Self.NUM_BINS + b] = logits4[b]
            q_logits_ptr[4 * BATCH * Self.NUM_BINS + b] = logits5[b]

    fn q_min_forward[
        BATCH: Int
    ](
        self,
        z_a_ptr: UnsafePointer[Scalar[dtype]],
        values_ptr: UnsafePointer[Scalar[dtype]],
        use_target: Bool = False,
    ):
        """Compute min Q-value across ensemble for each sample.

        Args:
            z_a_ptr: Pointer to concatenated (latent, action) [BATCH * ZA_DIM].
            values_ptr: Pointer to output min Q-values [BATCH] (written).
            use_target: If True, use target Q-networks.
        """
        var all_logits = List[Scalar[dtype]](
            capacity=Self.NUM_Q * BATCH * Self.NUM_BINS
        )
        for _ in range(Self.NUM_Q * BATCH * Self.NUM_BINS):
            all_logits.append(Scalar[dtype](0))
        self.q_forward[BATCH](z_a_ptr, all_logits.unsafe_ptr(), use_target)

        # Decode scalar values and take min across ensemble
        for b in range(BATCH):
            var min_val = Float32(1e10)
            for q_idx in range(Self.NUM_Q):
                var base = q_idx * BATCH * Self.NUM_BINS + b * Self.NUM_BINS
                var logits_b = InlineArray[Float32, Self.NUM_BINS](
                    uninitialized=True
                )
                for i in range(Self.NUM_BINS):
                    logits_b[i] = Float32(all_logits[base + i])
                var val = decode_value_batch_scalar[Self.NUM_BINS](
                    logits_b, self.bins
                )
                if val < min_val:
                    min_val = val
            values_ptr[b] = Scalar[dtype](min_val)

    # =========================================================================
    # Soft Update for Target Networks
    # =========================================================================

    fn soft_update_q_targets(mut self, tau: Float64):
        """Soft update target Q-networks: θ_target ← τ*θ + (1-τ)*θ_target.

        Args:
            tau: Interpolation coefficient (default: 0.01 in TDMPC2).
        """
        self.q1_target.soft_update_from(self.q1, tau)
        self.q2_target.soft_update_from(self.q2, tau)
        self.q3_target.soft_update_from(self.q3, tau)
        self.q4_target.soft_update_from(self.q4, tau)
        self.q5_target.soft_update_from(self.q5, tau)

    # =========================================================================
    # Gradient Zeroing
    # =========================================================================

    fn zero_all_grads(mut self):
        """Zero gradients for all world model sub-networks."""
        self.encoder.zero_grads()
        self.dynamics.zero_grads()
        self.reward_head.zero_grads()
        self.termination.zero_grads()
        self.q1.zero_grads()
        self.q2.zero_grads()
        self.q3.zero_grads()
        self.q4.zero_grads()
        self.q5.zero_grads()

    fn zero_policy_grads(mut self):
        """Zero gradients for the policy network."""
        self.policy.zero_grads()

    # =========================================================================
    # Parameter Updates
    # =========================================================================

    fn update_world_model_params(mut self):
        """Apply gradient updates to all world model parameters (exc. policy).
        """
        self.encoder.update()
        self.dynamics.update()
        self.reward_head.update()
        self.termination.update()
        self.q1.update()
        self.q2.update()
        self.q3.update()
        self.q4.update()
        self.q5.update()

    fn update_policy_params(mut self):
        """Apply gradient updates to policy parameters."""
        self.policy.update()


fn decode_value_batch_scalar[
    NUM_BINS: Int
](
    logits: InlineArray[Float32, NUM_BINS],
    bins: InlineArray[Float32, NUM_BINS],
) -> Float32:
    """Decode a single distributional value from logits.

    Args:
        logits: Raw logits over bins [NUM_BINS].
        bins: Bin values [NUM_BINS].

    Returns:
        Expected value under the softmax distribution.
    """
    var max_val = logits[0]
    for i in range(1, NUM_BINS):
        if logits[i] > max_val:
            max_val = logits[i]

    var sum_exp = Float32(0.0)
    for i in range(NUM_BINS):
        sum_exp += exp(logits[i] - max_val)

    var value = Float32(0.0)
    for i in range(NUM_BINS):
        var prob = exp(logits[i] - max_val) / sum_exp
        value += prob * bins[i]

    return value
