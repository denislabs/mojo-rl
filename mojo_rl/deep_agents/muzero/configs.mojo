"""MuZero Configs — Compile-time configuration for MuZero family agents.

Follows the same composable pattern as DQN/SAC configs:
  - MuZeroConfig trait defines the interface
  - Concrete configs (MuZeroMLPConfig, MuZeroCNNConfig, MuZeroResNetConfig)
    bundle network architectures + hyperparameters
  - GenericMuZeroAgent[Config] works with any config

Usage:
    # Standard MuZero on CartPole
    var agent = GenericMuZeroAgent[MuZeroMLPConfig[4, 2]]()

    # MuZero CNN on Pong pixel observations
    var agent = GenericMuZeroAgent[MuZeroCNNConfig[3]]()

    # MuZero with ResNet on complex environments
    var agent = GenericMuZeroAgent[MuZeroResNetConfig[17, 6]]()
"""

from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearMish,
    LayerNorm,
    MinMaxNorm,
    Sequential,
    Parallel,
)
from mojo_rl.nn.model import Conv2DReLU, FlattenLayer
from mojo_rl.nn.optimizer import Optimizer, Adam, AdamW
from mojo_rl.nn.autodiff.combinators import Residual
from .strategies import (
    SearchMode,
    LearnedDynamics,
    ValueEncoding,
    CategoricalEncoding,
    HiddenScaling,
    MinMaxScale,
    ExplorationNoise,
    DirichletNoise,
    PUCTFormula,
    MuZeroPUCT,
    AlphaGoPUCT,
    BackupMode,
    NStepBootstrap,
    MonteCarloReturn,
    LambdaReturn,
    PlayerMode,
    SinglePlayer,
    SelfPlay,
)


# ═══════════════════════════════════════════════════════════════════════════
# MuZero Config Trait
# ═══════════════════════════════════════════════════════════════════════════


trait MuZeroConfig:
    """Compile-time configuration for MuZero family agents.

    Bundles network architectures, optimizer, dimensions, and training
    hyperparameters. Concrete implementations define the full algorithm
    variant (MLP, CNN, ResNet, etc.).

    All MuZero variants share the same training loop (K-step unrolled
    forward/backward with MCTS planning). Only the networks differ.
    """

    comptime NAME: String

    # ── Dimensions ────────────────────────────────────────────────
    comptime obs_dim: Int         # Observation space dimension
    comptime action_dim: Int      # Number of discrete actions
    comptime latent_dim: Int      # Hidden state dimension
    comptime num_bins: Int        # Distributional value/reward bins

    # ── Derived dimensions ────────────────────────────────────────
    comptime DYN_IN: Int          # = latent_dim + action_dim
    comptime DYN_OUT: Int         # = latent_dim + num_bins
    comptime PRED_OUT: Int        # = action_dim + num_bins

    # ── Network Architectures ─────────────────────────────────────
    comptime RepModel: Model      # h(obs) → hidden_state
    comptime DynModel: Model      # g(hidden || one_hot_action) → (next_hidden || reward_logits)
    comptime PredModel: Model     # f(hidden) → (policy_logits || value_logits)
    comptime OptType: Optimizer   # Shared optimizer for all three networks

    # ── Training Hyperparameters ──────────────────────────────────
    comptime batch_size: Int      # Training batch size
    comptime buffer_capacity: Int # Replay buffer capacity
    comptime unroll_steps: Int    # K-step unroll depth
    comptime td_steps: Int        # N-step bootstrap horizon

    # ── MCTS Hyperparameters ──────────────────────────────────────
    comptime num_simulations: Int # MCTS simulations per action
    comptime max_nodes: Int       # Maximum tree nodes per search

    # ── Strategy Types ────────────────────────────────────────────
    comptime Search: SearchMode           # Learned dynamics vs true game rules
    comptime Encoding: ValueEncoding      # Categorical vs scalar value encoding
    comptime Scaling: HiddenScaling       # Hidden state normalization
    comptime Noise: ExplorationNoise      # Root exploration noise
    comptime PUCT: PUCTFormula            # UCB exploration formula
    comptime Backup: BackupMode           # Return computation strategy
    comptime Players: PlayerMode          # Single-player vs self-play

    # ── Flags ─────────────────────────────────────────────────────
    comptime USE_REANALYZE: Bool  # Enable MuZero Reanalyze


# ═══════════════════════════════════════════════════════════════════════════
# MuZero MLP Config (standard, clean observations)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroMLPConfig[
    OBS: Int,
    ACT: Int,
    LATENT: Int = 256,
    HIDDEN: Int = 256,
    BINS: Int = 101,
    LR: Float64 = 3e-4,
    CAP: Int = 100000,
    BS: Int = 128,
    K: Int = 5,
    N: Int = 10,
    SIMS: Int = 50,
    NODES: Int = 64,
](MuZeroConfig):
    """Standard MuZero with MLP networks for clean observations.

    Three-layer MLP for representation, dynamics, and prediction.
    Suitable for CartPole (4D), Pong clean obs (6D), etc.
    """

    comptime NAME: String = "MuZero-MLP"

    # Dimensions
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

    # Networks
    # MinMaxNorm at end of rep — Phase G post-mortem 2026-05-05: muzero-general's
    # `representation()` does min-max via PyTorch tensor ops (`.min`/`.max`/
    # subtract/divide) which PyTorch tracks in autograd, so gradient flows
    # through min-max during training. Our previous post-hoc
    # `scale_hidden_kernel` was OUTSIDE the autodiff graph, leaving the rep
    # network with no gradient signal about its raw output magnitudes →
    # activations exploded to 10⁶ and direction collapsed (sign-symmetric).
    # MinMaxNorm is a proper Model with forward + backward, so gradient now
    # flows through it like in the reference. The post-hoc kernel call after
    # rep_net forward is redundant on already-normalized output (idempotent
    # on [0,1]).
    comptime RepModel = Sequential[
        LinearMish[Self.OBS, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.LATENT],
        MinMaxNorm[Self.LATENT],
    ]

    # DynModel — split output into [hidden + reward_logits] via Parallel so
    # MinMaxNorm only normalizes the hidden portion (BINS reward bins are
    # categorical logits and should NOT be min-max-normalized). Matches
    # muzero-general/models.py:147-170 where `next_encoded_state` is
    # min-max'd via PyTorch tensor ops while `reward` is left unnormalized.
    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Sequential[
                Linear[Self.HIDDEN, Self.LATENT],
                MinMaxNorm[Self.LATENT],
            ],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    # AdamW with weight_decay=1e-4 — matches muzero-general's
    # `torch.optim.Adam(weight_decay=1e-4)` (cartpole.py:86, trainer.py:48).
    # Our `Adam` struct does not support weight_decay (only `AdamW` does);
    # using bare `Adam` previously meant the agent's `weight_decay` field
    # was a dead knob and rep weights drifted unbounded under the K-step
    # unroll, causing pre-min-max activations to reach 10⁶ magnitude and
    # post-scale hidden state to saturate uniform across obs (state-blind).
    # See docs/MUZERO_AUDIT.md Phase G post-mortem 2026-05-04.
    # Adam with PyTorch-style L2-in-gradient weight decay (WEIGHT_DECAY=1e-4
    # to match muzero-general/games/cartpole.py:86 exactly). Switched from
    # AdamW (decoupled decay) on 2026-05-05: AdamW's `param *= (1 - LR*W)`
    # continues shrinking weights at rate LR·W·param indefinitely, which
    # over-decays small late-training gradients. PyTorch's L2-in-gradient
    # adds `W·param` to grad before m/v update — when grad → 0 in late
    # training, v_hat → (W·param)² and the per-step update caps at
    # LR·sign(param), avoiding the "bleed-to-zero" weight collapse we
    # observed for MuZero CartPole through the AdamW phase. See
    # docs/MUZERO_AUDIT.md for the full chain of evidence.
    comptime OptType = Adam[LR=Self.LR, WEIGHT_DECAY=1e-4]

    # Training
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K
    comptime td_steps: Int = Self.N

    # MCTS
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    # Strategy types
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    # Dirichlet fraction reverted 0.5 → 0.25 (2026-05-05, Phase H12) to align
    # with muzero-general's CartPole config (root_exploration_fraction=0.25).
    # The 0.5 bump was a band-aid for a different root cause (pre-Bug-F
    # representation collapse). After the pipeline-level fixes (Bug F, value
    # encoding, network sizing per H12), reference fraction=0.25 should
    # suffice. Alpha=0.25 is the standard small-game default.
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    # Backup: NStepBootstrap. We tried MonteCarloReturn (2026-05-05,
    # following AZ-CartPole's adaptation) and it was strictly worse — full
    # representation collapse returned (post-train hiddens bit-identical
    # across LEFT/CENTER/RIGHT, |rep| shrank to 0.27). Root cause:
    # `nstep_value_targets_kernel` truncates at N steps, and with N=10 +
    # CartPole avg episode ~22, the majority of sample positions don't
    # terminate within the window, so MC target = Σ₀⁹ γⁱ ≈ 9.56 —
    # state-independent constant. The bootstrap γⁿV(s_{t+N}), even from an
    # undertrained value head, was the only state-dependent signal in
    # late-trajectory targets. AZ-CartPole's MC works because their replay
    # accesses full episodes, not an N-step window. Re-enabling MC for our
    # path would require N ≥ episode length (~50 instead of 10), which is
    # a separate experiment from the backup-mode change itself.
    comptime Backup = NStepBootstrap
    comptime Players = SinglePlayer

    comptime USE_REANALYZE: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# MuZero CNN Config (pixel observations, 4x84x84)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroCNNConfig[
    ACT: Int,
    LATENT: Int = 512,
    BINS: Int = 101,
    LR: Float64 = 2.5e-4,
    CAP: Int = 100000,
    BS: Int = 64,
    K: Int = 5,
    N: Int = 10,
    SIMS: Int = 50,
    NODES: Int = 64,
](MuZeroConfig):
    """MuZero with CNN representation for 4x84x84 pixel observations.

    NatureDQN-style Conv2D representation network (Mnih et al., 2015).
    Dynamics and prediction use MLP on the latent space.
    Suitable for Pong, Breakout, Space Invaders pixel modes.
    """

    comptime NAME: String = "MuZero-CNN"

    # Dimensions
    comptime obs_dim: Int = 4 * 84 * 84  # 28224
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

    # CNN Representation (NatureDQN downsampling → latent) + MinMaxNorm.
    # See MuZeroMLPConfig (configs.mojo:150-183) for latent-norm rationale.
    comptime RepModel = Sequential[
        Conv2DReLU[4, 32, 8, 4, 0, 84, 84],     # → 20x20
        Conv2DReLU[32, 64, 4, 2, 0, 20, 20],    # → 9x9
        Conv2DReLU[64, 64, 3, 1, 0, 9, 9],      # → 7x7
        FlattenLayer[64 * 7 * 7],                 # → 3136
        LinearReLU[64 * 7 * 7, 512],
        Linear[512, Self.LATENT],
        MinMaxNorm[Self.LATENT],
    ]

    # Dynamics in latent space; MinMaxNorm only on hidden split, reward raw.
    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, 256],
        LinearMish[256, 256],
        Parallel[
            Sequential[
                Linear[256, Self.LATENT],
                MinMaxNorm[Self.LATENT],
            ],
            Linear[256, Self.BINS],
        ],
    ]

    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, 256],
        Parallel[
            Linear[256, Self.ACT],
            Linear[256, Self.BINS],
        ],
    ]

    # AdamW with weight_decay=1e-4 — matches muzero-general's
    # `torch.optim.Adam(weight_decay=1e-4)` (cartpole.py:86, trainer.py:48).
    # Our `Adam` struct does not support weight_decay (only `AdamW` does);
    # using bare `Adam` previously meant the agent's `weight_decay` field
    # was a dead knob and rep weights drifted unbounded under the K-step
    # unroll, causing pre-min-max activations to reach 10⁶ magnitude and
    # post-scale hidden state to saturate uniform across obs (state-blind).
    # See docs/MUZERO_AUDIT.md Phase G post-mortem 2026-05-04.
    # Adam with PyTorch-style L2-in-gradient weight decay (WEIGHT_DECAY=1e-4
    # to match muzero-general/games/cartpole.py:86 exactly). Switched from
    # AdamW (decoupled decay) on 2026-05-05: AdamW's `param *= (1 - LR*W)`
    # continues shrinking weights at rate LR·W·param indefinitely, which
    # over-decays small late-training gradients. PyTorch's L2-in-gradient
    # adds `W·param` to grad before m/v update — when grad → 0 in late
    # training, v_hat → (W·param)² and the per-step update caps at
    # LR·sign(param), avoiding the "bleed-to-zero" weight collapse we
    # observed for MuZero CartPole through the AdamW phase. See
    # docs/MUZERO_AUDIT.md for the full chain of evidence.
    comptime OptType = Adam[LR=Self.LR, WEIGHT_DECAY=1e-4]

    # Training
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K
    comptime td_steps: Int = Self.N

    # MCTS
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    # Strategy types
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = NStepBootstrap
    comptime Players = SinglePlayer

    comptime USE_REANALYZE: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# MuZero ResNet Config (deeper networks for harder tasks)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroResNetConfig[
    OBS: Int,
    ACT: Int,
    LATENT: Int = 256,
    HIDDEN: Int = 256,
    BINS: Int = 101,
    LR: Float64 = 3e-4,
    CAP: Int = 100000,
    BS: Int = 128,
    K: Int = 5,
    N: Int = 10,
    SIMS: Int = 50,
    NODES: Int = 128,
](MuZeroConfig):
    """MuZero with ResNet blocks for deeper representation.

    Uses Residual connections in representation and dynamics networks
    for better gradient flow in deeper models. Suitable for more
    complex environments where the standard MLP underfits.
    """

    comptime NAME: String = "MuZero-ResNet"

    # Dimensions
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

    # Representation with ResBlocks + MinMaxNorm. See MuZeroMLPConfig
    # (configs.mojo:150-183) for the latent-norm rationale.
    comptime RepModel = Sequential[
        LinearMish[Self.OBS, Self.HIDDEN],
        Residual[Sequential[
            LinearMish[Self.HIDDEN, Self.HIDDEN],
            Linear[Self.HIDDEN, Self.HIDDEN],
        ]],
        Residual[Sequential[
            LinearMish[Self.HIDDEN, Self.HIDDEN],
            Linear[Self.HIDDEN, Self.HIDDEN],
        ]],
        Linear[Self.HIDDEN, Self.LATENT],
        MinMaxNorm[Self.LATENT],
    ]

    # Dynamics with ResBlock; MinMaxNorm only on hidden split, reward raw.
    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, Self.HIDDEN],
        Residual[Sequential[
            LinearMish[Self.HIDDEN, Self.HIDDEN],
            Linear[Self.HIDDEN, Self.HIDDEN],
        ]],
        Parallel[
            Sequential[
                Linear[Self.HIDDEN, Self.LATENT],
                MinMaxNorm[Self.LATENT],
            ],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    # Prediction with deeper heads
    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    # AdamW with weight_decay=1e-4 — matches muzero-general's
    # `torch.optim.Adam(weight_decay=1e-4)` (cartpole.py:86, trainer.py:48).
    # Our `Adam` struct does not support weight_decay (only `AdamW` does);
    # using bare `Adam` previously meant the agent's `weight_decay` field
    # was a dead knob and rep weights drifted unbounded under the K-step
    # unroll, causing pre-min-max activations to reach 10⁶ magnitude and
    # post-scale hidden state to saturate uniform across obs (state-blind).
    # See docs/MUZERO_AUDIT.md Phase G post-mortem 2026-05-04.
    # Adam with PyTorch-style L2-in-gradient weight decay (WEIGHT_DECAY=1e-4
    # to match muzero-general/games/cartpole.py:86 exactly). Switched from
    # AdamW (decoupled decay) on 2026-05-05: AdamW's `param *= (1 - LR*W)`
    # continues shrinking weights at rate LR·W·param indefinitely, which
    # over-decays small late-training gradients. PyTorch's L2-in-gradient
    # adds `W·param` to grad before m/v update — when grad → 0 in late
    # training, v_hat → (W·param)² and the per-step update caps at
    # LR·sign(param), avoiding the "bleed-to-zero" weight collapse we
    # observed for MuZero CartPole through the AdamW phase. See
    # docs/MUZERO_AUDIT.md for the full chain of evidence.
    comptime OptType = Adam[LR=Self.LR, WEIGHT_DECAY=1e-4]

    # Training
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K
    comptime td_steps: Int = Self.N

    # MCTS
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    # Strategy types
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = NStepBootstrap
    comptime Players = SinglePlayer

    comptime USE_REANALYZE: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# MuZero Large Config (for competitive performance)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroLargeConfig[
    OBS: Int,
    ACT: Int,
    LR: Float64 = 1e-4,
    SIMS: Int = 100,
](MuZeroConfig):
    """MuZero with large networks for competitive performance.

    512-dim latent, 301-bin distributional, 100 MCTS simulations,
    larger buffer and batch for more stable training.
    """

    comptime NAME: String = "MuZero-Large"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = 512
    comptime num_bins: Int = 301
    comptime DYN_IN: Int = 512 + Self.ACT
    comptime DYN_OUT: Int = 512 + 301
    comptime PRED_OUT: Int = Self.ACT + 301

    # MinMaxNorm appended to rep + on dynamics hidden split. See
    # MuZeroMLPConfig (configs.mojo:150-183) for the latent-norm rationale.
    comptime RepModel = Sequential[
        LinearMish[Self.OBS, 512],
        LinearMish[512, 512],
        Residual[Sequential[LinearMish[512, 512], Linear[512, 512]]],
        Linear[512, 512],
        MinMaxNorm[512],
    ]

    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, 512],
        Residual[Sequential[LinearMish[512, 512], Linear[512, 512]]],
        Parallel[
            Sequential[
                Linear[512, 512],
                MinMaxNorm[512],
            ],
            Linear[512, 301],
        ],
    ]

    comptime PredModel = Sequential[
        LinearMish[512, 512],
        LinearMish[512, 512],
        Parallel[Linear[512, Self.ACT], Linear[512, 301]],
    ]

    # AdamW with weight_decay=1e-4 — matches muzero-general's
    # `torch.optim.Adam(weight_decay=1e-4)` (cartpole.py:86, trainer.py:48).
    # Our `Adam` struct does not support weight_decay (only `AdamW` does);
    # using bare `Adam` previously meant the agent's `weight_decay` field
    # was a dead knob and rep weights drifted unbounded under the K-step
    # unroll, causing pre-min-max activations to reach 10⁶ magnitude and
    # post-scale hidden state to saturate uniform across obs (state-blind).
    # See docs/MUZERO_AUDIT.md Phase G post-mortem 2026-05-04.
    # Adam with PyTorch-style L2-in-gradient weight decay (WEIGHT_DECAY=1e-4
    # to match muzero-general/games/cartpole.py:86 exactly). Switched from
    # AdamW (decoupled decay) on 2026-05-05: AdamW's `param *= (1 - LR*W)`
    # continues shrinking weights at rate LR·W·param indefinitely, which
    # over-decays small late-training gradients. PyTorch's L2-in-gradient
    # adds `W·param` to grad before m/v update — when grad → 0 in late
    # training, v_hat → (W·param)² and the per-step update caps at
    # LR·sign(param), avoiding the "bleed-to-zero" weight collapse we
    # observed for MuZero CartPole through the AdamW phase. See
    # docs/MUZERO_AUDIT.md for the full chain of evidence.
    comptime OptType = Adam[LR=Self.LR, WEIGHT_DECAY=1e-4]

    comptime batch_size: Int = 256
    comptime buffer_capacity: Int = 500000
    comptime unroll_steps: Int = 5
    comptime td_steps: Int = 10
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = 256

    # Strategy types
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = NStepBootstrap
    comptime Players = SinglePlayer

    comptime USE_REANALYZE: Bool = True


# ═══════════════════════════════════════════════════════════════════════════
# EfficientZero Config (MuZero + self-supervised consistency)
# ═══════════════════════════════════════════════════════════════════════════


struct EfficientZeroConfig[
    OBS: Int,
    ACT: Int,
    LATENT: Int = 256,
    HIDDEN: Int = 256,
    BINS: Int = 101,
    LR: Float64 = 3e-4,
    SIMS: Int = 50,
](MuZeroConfig):
    """EfficientZero-style config (Ye et al., 2021).

    Extends MuZero with:
    - Self-supervised consistency loss (not yet implemented in training)
    - Symlog value encoding for better sample efficiency
    - Lambda returns for smoother value targets

    Network architectures same as MuZero MLP.
    Strategy choices optimize for sample efficiency.
    """

    comptime NAME: String = "EfficientZero"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

    # MinMaxNorm at end of rep + on dynamics hidden split. See
    # MuZeroMLPConfig (configs.mojo:150-183) for the latent-norm rationale.
    comptime RepModel = Sequential[
        LinearMish[Self.OBS, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.LATENT],
        MinMaxNorm[Self.LATENT],
    ]
    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Sequential[
                Linear[Self.HIDDEN, Self.LATENT],
                MinMaxNorm[Self.LATENT],
            ],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]
    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]
    # AdamW with weight_decay=1e-4 — matches muzero-general's
    # `torch.optim.Adam(weight_decay=1e-4)` (cartpole.py:86, trainer.py:48).
    # Our `Adam` struct does not support weight_decay (only `AdamW` does);
    # using bare `Adam` previously meant the agent's `weight_decay` field
    # was a dead knob and rep weights drifted unbounded under the K-step
    # unroll, causing pre-min-max activations to reach 10⁶ magnitude and
    # post-scale hidden state to saturate uniform across obs (state-blind).
    # See docs/MUZERO_AUDIT.md Phase G post-mortem 2026-05-04.
    # Adam with PyTorch-style L2-in-gradient weight decay (WEIGHT_DECAY=1e-4
    # to match muzero-general/games/cartpole.py:86 exactly). Switched from
    # AdamW (decoupled decay) on 2026-05-05: AdamW's `param *= (1 - LR*W)`
    # continues shrinking weights at rate LR·W·param indefinitely, which
    # over-decays small late-training gradients. PyTorch's L2-in-gradient
    # adds `W·param` to grad before m/v update — when grad → 0 in late
    # training, v_hat → (W·param)² and the per-step update caps at
    # LR·sign(param), avoiding the "bleed-to-zero" weight collapse we
    # observed for MuZero CartPole through the AdamW phase. See
    # docs/MUZERO_AUDIT.md for the full chain of evidence.
    comptime OptType = Adam[LR=Self.LR, WEIGHT_DECAY=1e-4]

    comptime batch_size: Int = 128
    comptime buffer_capacity: Int = 100000
    comptime unroll_steps: Int = 5
    comptime td_steps: Int = 10

    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = 64

    # EfficientZero strategy choices:
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = LambdaReturn[0.95]       # Lambda returns for sample efficiency
    comptime Players = SinglePlayer

    comptime USE_REANALYZE: Bool = True         # Always use Reanalyze


# ═══════════════════════════════════════════════════════════════════════════
# MuZero TicTacToe Config (27D obs, 9 actions, self-play)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroTicTacToeConfig[
    LATENT: Int = 128,
    HIDDEN: Int = 128,
    BINS: Int = 51,
    LR: Float64 = 1e-3,
    CAP: Int = 50000,
    BS: Int = 64,
    K: Int = 5,
    N: Int = 10,
    SIMS: Int = 100,
    NODES: Int = 128,
    C_PUCT: Float64 = 1.0,
](MuZeroConfig):
    """MuZero for TicTacToe via learned dynamics + self-play.

    Unlike AlphaZero which uses true game rules in MCTS, MuZero learns
    the dynamics model g(s, a) → (r, s') and plans in latent space.
    Uses Monte Carlo returns (no bootstrapping) since board games
    have clear terminal outcomes.
    """

    comptime NAME: String = "MuZero-TicTacToe"

    # Dimensions
    comptime obs_dim: Int = 27    # 3 planes × 3×3
    comptime action_dim: Int = 9
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + 9
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = 9 + Self.BINS

    # Networks. MinMaxNorm at end of rep + on dynamics hidden split — see
    # MuZeroMLPConfig (configs.mojo:150-183) for the rationale; matches
    # muzero-general/models.py:147-170 latent normalization.
    comptime RepModel = Sequential[
        LinearMish[27, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.LATENT],
        MinMaxNorm[Self.LATENT],
    ]

    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Sequential[
                Linear[Self.HIDDEN, Self.LATENT],
                MinMaxNorm[Self.LATENT],
            ],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, 9],       # Policy head
            Linear[Self.HIDDEN, Self.BINS],  # Value head (distributional)
        ],
    ]

    # AdamW with weight_decay=1e-4 — matches muzero-general's
    # `torch.optim.Adam(weight_decay=1e-4)` (cartpole.py:86, trainer.py:48).
    # Our `Adam` struct does not support weight_decay (only `AdamW` does);
    # using bare `Adam` previously meant the agent's `weight_decay` field
    # was a dead knob and rep weights drifted unbounded under the K-step
    # unroll, causing pre-min-max activations to reach 10⁶ magnitude and
    # post-scale hidden state to saturate uniform across obs (state-blind).
    # See docs/MUZERO_AUDIT.md Phase G post-mortem 2026-05-04.
    # Adam with PyTorch-style L2-in-gradient weight decay (WEIGHT_DECAY=1e-4
    # to match muzero-general/games/cartpole.py:86 exactly). Switched from
    # AdamW (decoupled decay) on 2026-05-05: AdamW's `param *= (1 - LR*W)`
    # continues shrinking weights at rate LR·W·param indefinitely, which
    # over-decays small late-training gradients. PyTorch's L2-in-gradient
    # adds `W·param` to grad before m/v update — when grad → 0 in late
    # training, v_hat → (W·param)² and the per-step update caps at
    # LR·sign(param), avoiding the "bleed-to-zero" weight collapse we
    # observed for MuZero CartPole through the AdamW phase. See
    # docs/MUZERO_AUDIT.md for the full chain of evidence.
    # WEIGHT_DECAY=0 (Phase H17, 2026-05-07). Reference uses 1e-4 with
    # PyTorch's Adam L2-in-grad, but our batch-then-train pattern leaves
    # stretches where the true gradient is small; then the L2 term
    # `WD·param` dominates Adam's m/v and per-step update collapses to
    # `-LR·sign(param)`, bleeding weights toward zero linearly. The
    # NVIDIA TTT run (epochs=2) showed pred_norm shrinking 232 → 37 over
    # iters 39→79 — model still hit perfect play but tipped past iter 85
    # when MCTS priors became uninformative. Removing decay eliminates
    # the bleed. The earlier audit comment claiming Adam L2-in-grad
    # "avoids bleed-to-zero" was incorrect — confirmed by the smooth
    # `pred_norm` decline in the dashboard.
    comptime OptType = Adam[LR=Self.LR, WEIGHT_DECAY=0.0]

    # Training
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K
    comptime td_steps: Int = Self.N

    # MCTS
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    # Strategy types — board game self-play
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Backup = MonteCarloReturn
    comptime Players = SelfPlay

    comptime USE_REANALYZE: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# MuZero ConnectFour Config (126D obs, 7 actions, self-play)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroConnectFourConfig[
    LATENT: Int = 256,
    HIDDEN: Int = 256,
    BINS: Int = 51,
    LR: Float64 = 1e-3,
    WD: Float64 = 1e-4,
    CAP: Int = 200000,
    BS: Int = 128,
    K: Int = 5,
    N: Int = 10,
    SIMS: Int = 100,
    NODES: Int = 256,
    C_PUCT: Float64 = 2.0,
](MuZeroConfig):
    """MuZero for Connect Four via learned dynamics + self-play.

    Larger networks than TicTacToe due to higher board complexity
    (6×7 board, 3 channels). Uses ResNet-style residual blocks in
    the representation and dynamics networks for deeper feature
    extraction. AdamW with weight decay for regularization.
    """

    comptime NAME: String = "MuZero-ConnectFour"

    # Dimensions
    comptime obs_dim: Int = 126   # 3 planes × 6×7
    comptime action_dim: Int = 7
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + 7
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = 7 + Self.BINS

    # Representation with ResBlock + MinMaxNorm. See MuZeroMLPConfig
    # (configs.mojo:150-183) for the latent-norm rationale.
    comptime RepModel = Sequential[
        LinearMish[126, Self.HIDDEN],
        Residual[Sequential[
            LinearMish[Self.HIDDEN, Self.HIDDEN],
            Linear[Self.HIDDEN, Self.HIDDEN],
        ]],
        Linear[Self.HIDDEN, Self.LATENT],
        MinMaxNorm[Self.LATENT],
    ]

    # Dynamics with ResBlock; MinMaxNorm only on hidden split, reward bins raw.
    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, Self.HIDDEN],
        Residual[Sequential[
            LinearMish[Self.HIDDEN, Self.HIDDEN],
            Linear[Self.HIDDEN, Self.HIDDEN],
        ]],
        Parallel[
            Sequential[
                Linear[Self.HIDDEN, Self.LATENT],
                MinMaxNorm[Self.LATENT],
            ],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    # Prediction with deeper heads
    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, 7],       # Policy head
            Linear[Self.HIDDEN, Self.BINS],  # Value head (distributional)
        ],
    ]

    comptime OptType = AdamW[LR=Self.LR, WEIGHT_DECAY=Self.WD]

    # Training
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K
    comptime td_steps: Int = Self.N

    # MCTS
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    # Strategy types — board game self-play
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 1.0]  # alpha=1.0 for C4 (fewer actions)
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Backup = MonteCarloReturn
    comptime Players = SelfPlay

    comptime USE_REANALYZE: Bool = False
