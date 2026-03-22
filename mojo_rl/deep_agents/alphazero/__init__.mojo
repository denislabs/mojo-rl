# AlphaZero: Self-Play RL with True Game Rules and MCTS
#
# Simple supervised training on (obs, mcts_policy, game_outcome) tuples.
# One prediction network: f(obs) → (policy, value).
# MCTS uses true game rules for expansion (env.step).
#
# Reference: Silver et al., 2017

from .configs import (
    AlphaZeroConfig,
    AlphaZeroTicTacToeConfig,
    AlphaZeroConnectFourConfig,
    AlphaZeroChessConfig,
)
from .state import AlphaZeroCPUState, AlphaZeroGPUState
from .alphazero import GenericAlphaZeroAgent
