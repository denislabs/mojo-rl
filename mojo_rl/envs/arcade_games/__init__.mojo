"""Native Arcade game engines for GPU-batched RL training.

Each game is a self-contained struct implementing BoxDiscreteActionEnv + GPUDiscreteEnv
+ RenderableEnv, following the same pattern as CartPole/Pendulum.

Usage:
    from mojo_rl.envs.arcade_games.pong import PongEnv

    # CPU path
    var env = PongEnv[DType.float64]()
    var obs = env.reset_obs_list()
    var result = env.step_obs(1)  # UP

    # GPU path (via GPUDiscreteEnv trait)
    PongEnv.step_kernel_gpu[BATCH, STATE, OBS](ctx, states, ...)
"""

from .pong import PongEnv, PongPixelEnv
from .breakout import BreakoutEnv
from .space_invaders import SpaceInvadersEnv
from .core.gpu_env import ArcadeGameState, ArcadeGameAction
