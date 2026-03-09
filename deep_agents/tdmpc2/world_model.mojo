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
from std.random import random_float64

from layout import Layout, LayoutTensor
from nn.constants import dtype
from nn.model import (
    Linear,
    Sequential,
    Parallel,
    Sigmoid,
    NormedLinear,
    SimNorm,
)
from nn.optimizer import Adam
from nn.initializer import Kaiming
from nn.training import Network, NetworkState
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
    ENC_LR: Float64 = 9e-5,  # encoder LR = 0.3 * world_model_lr
    WM_LR: Float64 = 3e-4,  # world model (non-encoder) LR
    PI_LR: Float64 = 3e-4,  # policy LR
](Movable):
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
        ENC_LR: Encoder learning rate (default: 9e-5).
        WM_LR: World model learning rate (default: 3e-4).
        PI_LR: Policy learning rate (default: 3e-4).

    Note: LATENT_DIM must be divisible by SIMPLEX_DIM.
    """

    # Concatenated input dimensions
    comptime ZA_DIM: Int = Self.LATENT_DIM + Self.ACTION_DIM

    # -------------------------------------------------------------------------
    # Model type definitions
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
        Parallel[
            Linear[Self.MLP_DIM, Self.ACTION_DIM],   # mean head
            Linear[Self.MLP_DIM, Self.ACTION_DIM],   # log_std head
        ],
    ]

    # Q-network: (LATENT + ACTION) → NUM_BINS logits
    comptime QModel = Sequential[
        NormedLinear[Self.ZA_DIM, Self.MLP_DIM],
        NormedLinear[Self.MLP_DIM, Self.MLP_DIM],
        Linear[Self.MLP_DIM, Self.NUM_BINS],
    ]

    # Network wrapper type aliases (kept for TDMPC2Agent constant access:
    # PARAM_SIZE, CACHE_SIZE, WORKSPACE_SIZE_PER_SAMPLE)
    comptime EncoderNet = Network[Self.EncModel, Adam[LR = Self.ENC_LR]]
    comptime DynamicsNet = Network[
        Self.DynModel,
        Adam[LR = Self.WM_LR],
    ]
    comptime RewardNet = Network[
        Self.RewModel,
        Adam[LR = Self.WM_LR],
    ]
    comptime TermNet = Network[
        Self.TermModel,
        Adam[LR = Self.WM_LR],
    ]
    comptime PolicyNet = Network[
        Self.PolModel,
        Adam[LR = Self.PI_LR],
    ]
    comptime QNet = Network[
        Self.QModel,
        Adam[LR = Self.WM_LR],
    ]

    # -------------------------------------------------------------------------
    # Sub-network states (NetworkState replaces old stateful Network instances)
    # -------------------------------------------------------------------------
    var encoder: NetworkState[Self.EncModel, Adam[LR = Self.ENC_LR]]
    var dynamics: NetworkState[Self.DynModel, Adam[LR = Self.WM_LR]]
    var reward_head: NetworkState[Self.RewModel, Adam[LR = Self.WM_LR]]
    var termination: NetworkState[Self.TermModel, Adam[LR = Self.WM_LR]]
    var policy: NetworkState[Self.PolModel, Adam[LR = Self.PI_LR]]

    # Q-ensemble (NUM_Q=5 networks)
    var q1: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]
    var q2: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]
    var q3: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]
    var q4: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]
    var q5: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]

    # Target Q-networks (no gradient, soft-updated from live Qs)
    var q1_target: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]
    var q2_target: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]
    var q3_target: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]
    var q4_target: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]
    var q5_target: NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]

    # Fixed bin values for distributional RL
    var bins: InlineArray[Float32, Self.NUM_BINS]

    fn __init__(
        out self,
    ):
        """Initialize WorldModel with all sub-networks."""
        self.encoder = NetworkState[Self.EncModel, Adam[LR = Self.ENC_LR]]()
        self.encoder.initialize[Kaiming]()

        self.dynamics = NetworkState[Self.DynModel, Adam[LR = Self.WM_LR]]()
        self.dynamics.initialize[Kaiming]()

        self.reward_head = NetworkState[Self.RewModel, Adam[LR = Self.WM_LR]]()
        self.reward_head.initialize[Kaiming]()

        self.termination = NetworkState[Self.TermModel, Adam[LR = Self.WM_LR]]()
        self.termination.initialize[Kaiming]()

        self.policy = NetworkState[Self.PolModel, Adam[LR = Self.PI_LR]]()
        self.policy.initialize[Kaiming]()

        self.q1 = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
        self.q1.initialize[Kaiming]()
        self.q2 = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
        self.q2.initialize[Kaiming]()
        self.q3 = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
        self.q3.initialize[Kaiming]()
        self.q4 = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
        self.q4.initialize[Kaiming]()
        self.q5 = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
        self.q5.initialize[Kaiming]()

        # Initialize target Q networks with same weights as live Q networks
        self.q1_target = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
        self.q1_target.copy_params_from(self.q1)
        self.q2_target = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
        self.q2_target.copy_params_from(self.q2)
        self.q3_target = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
        self.q3_target.copy_params_from(self.q3)
        self.q4_target = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
        self.q4_target.copy_params_from(self.q4)
        self.q5_target = NetworkState[Self.QModel, Adam[LR = Self.WM_LR]]()
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
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ],
        mut z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LATENT_DIM), MutAnyOrigin
        ],
    ):
        """Encode observations to latent states (no cache, stop-gradient).

        Args:
            obs: Input observations [BATCH * OBS_DIM].
            z: Output latent states [BATCH * LATENT_DIM] (written).
        """

        Self.EncoderNet.forward[BATCH](obs, z, self.encoder.params_view())

    fn encode_with_cache[
        BATCH: Int
    ](
        self,
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ],
        mut z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.EncModel.CACHE_SIZE),
            MutAnyOrigin,
        ],
    ):
        """Encode with cache for backpropagation.

        Args:
            obs: Input observations [BATCH * OBS_DIM].
            z: Output latent states [BATCH * LATENT_DIM] (written).
            cache: Pre-allocated cache [BATCH * EncModel.CACHE_SIZE] (written).
        """

        Self.EncoderNet.forward_with_cache[BATCH](
            obs, z, self.encoder.params_view(), cache
        )

    fn dynamics_forward[
        BATCH: Int
    ](
        self,
        z_a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ZA_DIM), MutAnyOrigin
        ],
        mut z_next: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LATENT_DIM), MutAnyOrigin
        ],
    ):
        """Predict next latent state (no cache).

        Args:
            z_a: Concatenated (latent, action) [BATCH * ZA_DIM].
            z_next: Output next latent state [BATCH * LATENT_DIM] (written).
        """
        Self.DynamicsNet.forward[BATCH](
            z_a, z_next, self.dynamics.params_view()
        )

    fn dynamics_forward_with_cache[
        BATCH: Int
    ](
        self,
        z_a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ZA_DIM), MutAnyOrigin
        ],
        mut z_next: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ],
    ):
        """Predict next latent state with cache for backprop."""
        Self.DynamicsNet.forward_with_cache[BATCH](
            z_a, z_next, self.dynamics.params_view(), cache
        )

    fn reward_forward[
        BATCH: Int
    ](
        self,
        z_a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ZA_DIM), MutAnyOrigin
        ],
        mut logits: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.NUM_BINS), MutAnyOrigin
        ],
    ):
        """Predict reward distribution logits (no cache)."""
        Self.RewardNet.forward[BATCH](
            z_a, logits, self.reward_head.params_view()
        )

    fn reward_forward_with_cache[
        BATCH: Int
    ](
        self,
        z_a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ZA_DIM), MutAnyOrigin
        ],
        mut logits: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.NUM_BINS), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.RewModel.CACHE_SIZE),
            MutAnyOrigin,
        ],
    ):
        """Predict reward distribution logits with cache."""
        Self.RewardNet.forward_with_cache[BATCH](
            z_a, logits, self.reward_head.params_view(), cache
        )

    fn termination_forward[
        BATCH: Int
    ](
        self,
        z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut term_prob: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ],
    ):
        """Predict termination probability (no cache).

        Args:
            z: Latent states [BATCH * LATENT_DIM].
            term_prob: Output termination probabilities [BATCH] (written).
        """

        # TermModel.OUT_DIM = 1, output is [BATCH, 1]
        var out = List[Scalar[dtype]](capacity=BATCH)
        for _ in range(BATCH):
            out.append(Scalar[dtype](0))
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](out.unsafe_ptr())
        Self.TermNet.forward[BATCH](z, out_t, self.termination.params_view())
        for b in range(BATCH):
            term_prob[b, 0] = out[b]

    fn policy_forward[
        BATCH: Int
    ](
        self,
        z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut mean: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTION_DIM), MutAnyOrigin
        ],
        mut log_std: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTION_DIM), MutAnyOrigin
        ],
    ):
        """Predict Gaussian policy parameters (no cache).

        Args:
            z: Latent states [BATCH * LATENT_DIM].
            mean: Output action mean [BATCH * ACTION_DIM] (written).
            log_std: Output log std [BATCH * ACTION_DIM] (written).
        """

        comptime POL_OUT = Self.PolModel.OUT_DIM
        var out = List[Scalar[dtype]](capacity=BATCH * POL_OUT)
        for _ in range(BATCH * POL_OUT):
            out.append(Scalar[dtype](0))
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, POL_OUT), MutAnyOrigin
        ](out.unsafe_ptr())
        Self.PolicyNet.forward[BATCH](z, out_t, self.policy.params_view())
        for b in range(BATCH):
            for i in range(Self.ACTION_DIM):
                mean[b, i] = out[b * POL_OUT + i]
                # Clamp log_std to [-10, 2] for numerical stability
                var ls = Float64(
                    out[b * POL_OUT + Self.ACTION_DIM + i]
                )
                if ls < -10.0:
                    ls = -10.0
                if ls > 2.0:
                    ls = 2.0
                log_std[b, i] = Scalar[dtype](ls)

    fn policy_forward_with_cache[
        BATCH: Int
    ](
        self,
        z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.PolModel.OUT_DIM), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.PolModel.CACHE_SIZE),
            MutAnyOrigin,
        ],
    ):
        """Predict policy output with cache for backprop."""
        Self.PolicyNet.forward_with_cache[BATCH](
            z, out, self.policy.params_view(), cache
        )

    fn q_forward[
        BATCH: Int
    ](
        self,
        z_a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ZA_DIM), MutAnyOrigin
        ],
        mut q_logits: LayoutTensor[
            dtype,
            Layout.row_major(Self.NUM_Q * BATCH * Self.NUM_BINS),
            MutAnyOrigin,
        ],
        use_target: Bool = False,
    ):
        """Forward pass through all Q-networks.

        Args:
            z_a: Concatenated (latent, action) [BATCH * ZA_DIM].
            q_logits: Output logits [NUM_Q * BATCH * NUM_BINS] (written).
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

        var l1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.NUM_BINS), MutAnyOrigin
        ](logits1.unsafe_ptr())
        var l2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.NUM_BINS), MutAnyOrigin
        ](logits2.unsafe_ptr())
        var l3_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.NUM_BINS), MutAnyOrigin
        ](logits3.unsafe_ptr())
        var l4_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.NUM_BINS), MutAnyOrigin
        ](logits4.unsafe_ptr())
        var l5_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.NUM_BINS), MutAnyOrigin
        ](logits5.unsafe_ptr())

        if use_target:
            Self.QNet.forward[BATCH](z_a, l1_t, self.q1_target.params_view())
            Self.QNet.forward[BATCH](z_a, l2_t, self.q2_target.params_view())
            Self.QNet.forward[BATCH](z_a, l3_t, self.q3_target.params_view())
            Self.QNet.forward[BATCH](z_a, l4_t, self.q4_target.params_view())
            Self.QNet.forward[BATCH](z_a, l5_t, self.q5_target.params_view())
        else:
            Self.QNet.forward[BATCH](z_a, l1_t, self.q1.params_view())
            Self.QNet.forward[BATCH](z_a, l2_t, self.q2.params_view())
            Self.QNet.forward[BATCH](z_a, l3_t, self.q3.params_view())
            Self.QNet.forward[BATCH](z_a, l4_t, self.q4.params_view())
            Self.QNet.forward[BATCH](z_a, l5_t, self.q5.params_view())

        for b in range(BATCH * Self.NUM_BINS):
            q_logits[0 * BATCH * Self.NUM_BINS + b] = logits1[b]
            q_logits[1 * BATCH * Self.NUM_BINS + b] = logits2[b]
            q_logits[2 * BATCH * Self.NUM_BINS + b] = logits3[b]
            q_logits[3 * BATCH * Self.NUM_BINS + b] = logits4[b]
            q_logits[4 * BATCH * Self.NUM_BINS + b] = logits5[b]

    fn q_forward_single_no_cache[
        BATCH: Int
    ](
        self,
        q_idx: Int,
        z_a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ZA_DIM), MutAnyOrigin
        ],
        mut logits: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.NUM_BINS), MutAnyOrigin
        ],
    ):
        """Forward pass through a single Q-network (no cache).

        Args:
            q_idx: Index of Q-network (0..4).
            z_a: Concatenated (latent, action) [BATCH * ZA_DIM].
            logits: Output logits [BATCH * NUM_BINS] (written).
        """
        if q_idx == 0:
            Self.QNet.forward[BATCH](z_a, logits, self.q1.params_view())
        elif q_idx == 1:
            Self.QNet.forward[BATCH](z_a, logits, self.q2.params_view())
        elif q_idx == 2:
            Self.QNet.forward[BATCH](z_a, logits, self.q3.params_view())
        elif q_idx == 3:
            Self.QNet.forward[BATCH](z_a, logits, self.q4.params_view())
        else:
            Self.QNet.forward[BATCH](z_a, logits, self.q5.params_view())

    fn q_min_forward[
        BATCH: Int
    ](
        self,
        z_a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ZA_DIM), MutAnyOrigin
        ],
        mut values: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ],
        use_target: Bool = False,
    ):
        """Compute min Q-value across ensemble for each sample.

        Args:
            z_a: Concatenated (latent, action) [BATCH * ZA_DIM].
            values: Output min Q-values [BATCH] (written).
            use_target: If True, use target Q-networks.
        """
        var all_logits = List[Scalar[dtype]](
            capacity=Self.NUM_Q * BATCH * Self.NUM_BINS
        )
        for _ in range(Self.NUM_Q * BATCH * Self.NUM_BINS):
            all_logits.append(Scalar[dtype](0))
        var all_logits_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.NUM_Q * BATCH * Self.NUM_BINS),
            MutAnyOrigin,
        ](all_logits.unsafe_ptr())
        self.q_forward[BATCH](z_a, all_logits_t, use_target)

        # Decode scalar values and take min across ensemble
        for b in range(BATCH):
            var min_val: values.element_type = 1e10
            for q_idx in range(Self.NUM_Q):
                var base = q_idx * BATCH * Self.NUM_BINS + b * Self.NUM_BINS
                var logits_b = InlineArray[Float32, Self.NUM_BINS](
                    uninitialized=True
                )
                for i in range(Self.NUM_BINS):
                    logits_b[i] = Float32(all_logits[base + i])
                var val: values.element_type = decode_value_batch_scalar[
                    Self.NUM_BINS
                ](logits_b, self.bins)
                if val < min_val:
                    min_val = val
            values[b, 0] = min_val

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
        self.encoder.optimizer_step()
        self.dynamics.optimizer_step()
        self.reward_head.optimizer_step()
        self.termination.optimizer_step()
        self.q1.optimizer_step()
        self.q2.optimizer_step()
        self.q3.optimizer_step()
        self.q4.optimizer_step()
        self.q5.optimizer_step()

    fn update_policy_params(mut self):
        """Apply gradient updates to policy parameters."""
        self.policy.optimizer_step()


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
