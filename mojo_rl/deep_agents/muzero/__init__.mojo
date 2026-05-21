# MuZero: Model-Based RL with Learned Model and MCTS Planning
# Learns representation, dynamics, and prediction networks. Uses Monte Carlo
# Tree Search (MCTS) with the learned model for action selection. Trains via
# K-step unrolled forward/backward through all three networks.
#
# Reference: Schrittwieser et al., 2020 — Mastering Atari, Go, Chess and
# Shogi by Planning with a Learned Model (Nature)

from .configs import (
    MuZeroConfig,
    MuZeroMLPConfig,
    MuZeroCNNConfig,
    MuZeroResNetConfig,
    MuZeroLargeConfig,
    EfficientZeroConfig,
    MuZeroTicTacToeConfig,
    MuZeroTicTacToeCNNConfig,
    MuZeroConnectFourConfig,
)
# Strategy + value-encoding traits live in planners. Re-exported here so
# existing `from mojo_rl.deep_agents.muzero import SelfPlay, ...` style
# imports still work for downstream consumers.
from mojo_rl.planners.tree_search.strategies import (
    SearchMode, LearnedDynamics, TrueGameRules,
    HiddenScaling, MinMaxScale, NoScale,
    ExplorationNoise, DirichletNoise, EpsilonNoise, NoNoise,
    PUCTFormula, MuZeroPUCT, AlphaGoPUCT, UCB1Formula,
    BackupMode, NStepBootstrap, MonteCarloReturn, LambdaReturn,
    PlayerMode, SinglePlayer, SelfPlay,
)
from mojo_rl.planners.common.value_encoding import (
    ValueEncoding, CategoricalEncoding, ScalarEncoding, SymlogEncoding,
)
from .state import MuZeroCPUState, MuZeroGPUState
from .muzero import GenericMuZeroAgent
# Legacy CPU MCTS (.mcts) removed 2026-05-21: all CPU MCTS now routes
# through ``planners.tree_search.GenericCPUMCTS`` via the agent's
# ``_mcts_search_visits_cpu`` helper. GPU MCTS state still re-exported
# via the planner module directly (the legacy ``.gpu_mcts`` shim was
# also retired in the same pass).
from mojo_rl.planners.tree_search.mcts_gpu import GPUMCTSState
from .utils import scalar_transform, inverse_scalar_transform, MinMaxStats
from .evaluators import (
    Evaluator, GPUEvaluator,
    RandomOpponent, MinimaxTicTacToe, GPUMinimaxTicTacToe, GPUMinimaxConnectFour,
)
