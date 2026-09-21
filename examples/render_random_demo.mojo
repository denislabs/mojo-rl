"""Render any physics3d environment with random actions.

Uncomment the environment you want to visualize.

Run with:
    pixi run -e apple mojo run -I . examples/render_random_demo.mojo
"""

from std.random import seed
from noeira.envs.render_random import render_random

# ---- Uncomment ONE environment ----
from noeira.envs.half_cheetah import HalfCheetah
from noeira.envs.ant import Ant
from noeira.envs.hopper import Hopper
from noeira.envs.walker2d import Walker2d
from noeira.envs.swimmer import Swimmer
from noeira.envs.humanoid import Humanoid
from noeira.envs.inverted_pendulum import InvertedPendulum
from noeira.envs.inverted_double_pendulum import InvertedDoublePendulum
from noeira.envs.reacher import Reacher
from noeira.envs.pusher import Pusher


def main() raises:
    seed(1)  # seed 1 produces self-collision contacts

    # ---- Uncomment matching env ----
    # var env = HalfCheetah()
    var env = Ant()
    # var env = Hopper[TERMINATE_ON_UNHEALTHY=False]()
    # var env = Walker2d()
    # var env = Swimmer()
    # var env = Humanoid()
    # var env = InvertedPendulum()
    # var env = InvertedDoublePendulum()
    # var env = Reacher()
    # var env = Pusher()

    # Use show_velocity=False for non-locomotion envs (Reacher, Pusher)
    render_random(env, num_steps=3000, frame_delay_ms=32, show_velocity=False)
