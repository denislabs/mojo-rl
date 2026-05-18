"""Phase 3 agent rewiring smoke test: MuZero with USE_NEW_MCTS=True.

Same config as ``test_muzero_gpu.mojo`` but with a sibling config that
sets ``USE_NEW_MCTS = True``, so the GPU MCTS pipeline routes through
``GenericGPUMCTS.search_gpu`` + ``extract_actions_temp`` instead of the
inline kernel block.

Asserts: compiles + runs to completion + at least one training step ran.
Convergence parity against the legacy path is a longer-running validation
(separate from this smoke test).

Usage:
    pixi run -e apple mojo run -I . tests/deep_agents/test_muzero_gpu_new_mcts.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.muzero import GenericMuZeroAgent
from mojo_rl.deep_agents.muzero.configs import (
    MuZeroConfig,
    LearnedDynamics,
    CategoricalEncoding,
    MinMaxScale,
    DirichletNoise,
    MuZeroPUCT,
    NStepBootstrap,
    SinglePlayer,
)
from mojo_rl.nn.model import Linear, LinearMish, Sequential, Parallel
from mojo_rl.nn.model import MinMaxNorm
from mojo_rl.nn.optimizer import Adam
from mojo_rl.envs.cartpole import CartPoleEnv


# A direct copy of MuZeroMLPConfig with USE_NEW_MCTS flipped. Inheriting
# would be cleaner but ``MuZeroMLPConfig``'s comptime fields aren't easily
# overrideable; defining a sibling config is the smallest-change path.
struct DevMuZeroNewMCTSConfig[
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
    """Dev clone of MuZeroMLPConfig with the planner rewiring on."""

    comptime NAME: String = "MuZero-MLP-NewMCTS"

    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

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
    comptime OptType = Adam[LR=Self.LR, WEIGHT_DECAY=1e-4]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K
    comptime td_steps: Int = Self.N

    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = NStepBootstrap
    comptime Players = SinglePlayer

    comptime USE_REANALYZE: Bool = False
    comptime USE_NEW_MCTS: Bool = True   # ← the actual change


def main() raises:
    print("=== MuZero GPU Test — USE_NEW_MCTS=True ===")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    comptime Config = DevMuZeroNewMCTSConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=32,
        HIDDEN=32,
        BINS=21,
        SIMS=8,           # divisible by BATCH_SIMS=8 — production-shape
        K=3,
        N=5,
        BS=16,
        CAP=10000,
    ]

    var agent = GenericMuZeroAgent[Config, 16](
        gamma=0.99,
        temperature_decay_steps=5000,
    )
    print("Agent created:", Config.NAME, "USE_NEW_MCTS=", Config.USE_NEW_MCTS)

    print("Training with n_envs=16 GPU environments...")
    var metrics = agent.train_gpu[CartPoleGPU](
        ctx,
        num_steps=3200,
        warmup_steps=320,
        print_every=1600,
    )

    print("\n=== Results ===")
    print("GPU train steps:", agent.train_step_count)

    if agent.train_step_count > 0:
        print("PASS: GPU training completed via GenericGPUMCTS")
    else:
        print("FAIL: no training steps")

    _ = metrics
    print("=== Done ===")
