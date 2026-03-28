"""Render any physics3d environment with random actions.

Uncomment the environment you want to visualize.

Run with:
    pixi run -e apple mojo run -I . examples/render_random_demo.mojo
"""

from std.random import seed
from mojo_rl.envs.render_random import render_random

# ---- Uncomment ONE environment ----
# from mojo_rl.envs.half_cheetah import HalfCheetah
# from mojo_rl.envs.ant import Ant
from mojo_rl.envs.hopper import Hopper

# from mojo_rl.envs.walker2d import Walker2d
# from mojo_rl.envs.swimmer import Swimmer

# from mojo_rl.envs.humanoid import Humanoid


def main() raises:
    seed(42)

    # ---- Uncomment matching env ----
    # var env = HalfCheetah()
    # var env = Ant()
    var env = Hopper()
    # var env = Walker2d()
    # var env = Swimmer()
    # var env = Humanoid()

    render_random(env, num_steps=3000, frame_delay_ms=100)
