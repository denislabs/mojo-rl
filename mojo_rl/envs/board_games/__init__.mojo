"""Board game environments for two-player self-play RL training.

Each game implements TwoPlayerDiscreteEnv + GPUTwoPlayerDiscreteEnv,
supporting both single-agent (opponent-in-env) and self-play training modes.

Usage:
    from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv
    from mojo_rl.envs.board_games.connect_four import ConnectFourEnv
"""

from .tic_tac_toe import TicTacToeEnv
from .connect_four import ConnectFourEnv
from .go import GoEnv
from .chess import ChessEnv
