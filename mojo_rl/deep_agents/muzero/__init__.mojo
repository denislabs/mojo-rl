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
)
from .strategies import (
    SearchMode, LearnedDynamics, TrueGameRules,
    ValueEncoding, CategoricalEncoding, ScalarEncoding, SymlogEncoding,
    HiddenScaling, MinMaxScale, NoScale,
    ExplorationNoise, DirichletNoise, EpsilonNoise, NoNoise,
    PUCTFormula, MuZeroPUCT, AlphaGoPUCT, UCB1Formula,
    BackupMode, NStepBootstrap, MonteCarloReturn, LambdaReturn,
    PlayerMode, SinglePlayer, SelfPlay,
)
from .state import MuZeroCPUState, MuZeroGPUState
from .muzero import GenericMuZeroAgent
from .mcts import MCTS, MCTSNode
from .gpu_mcts import GPUMCTSState
from .utils import scalar_transform, inverse_scalar_transform, MinMaxStats
from .evaluators import (
    Evaluator, GPUEvaluator,
    RandomOpponent, MinimaxTicTacToe, GPUMinimaxTicTacToe, GPUMinimaxConnectFour,
)
