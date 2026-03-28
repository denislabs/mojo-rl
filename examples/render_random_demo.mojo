"""Render any physics3d environment with random actions.

Uncomment the environment you want to visualize.

Run with:
    pixi run -e apple mojo run -I . examples/render_random_demo.mojo
"""

from std.random import seed
from mojo_rl.envs.render_random import render_random

# ---- Uncomment ONE environment ----
from mojo_rl.envs.half_cheetah import HalfCheetah
from mojo_rl.envs.ant import Ant
from mojo_rl.envs.hopper import Hopper
from mojo_rl.envs.walker2d import Walker2d
from mojo_rl.envs.swimmer import Swimmer
from mojo_rl.envs.humanoid import Humanoid
from mojo_rl.envs.inverted_pendulum import InvertedPendulum
from mojo_rl.envs.inverted_double_pendulum import InvertedDoublePendulum
from mojo_rl.envs.reacher import Reacher
from mojo_rl.envs.pusher import Pusher


def main() raises:
    seed(1)  # seed 1 produces self-collision contacts

    # ---- Uncomment matching env ----
    # var env = HalfCheetah()
    # var env = Ant()
    # var env = Hopper[TERMINATE_ON_UNHEALTHY=False]()
    # var env = Walker2d()
    # var env = Swimmer()
    # var env = Humanoid()
    # var env = InvertedPendulum()
    # var env = InvertedDoublePendulum()
    var env = Reacher()
    # var env = Pusher()

    # Use show_velocity=False for non-locomotion envs (Reacher, Pusher)
    render_random(env, num_steps=3000, frame_delay_ms=100, show_velocity=False)
