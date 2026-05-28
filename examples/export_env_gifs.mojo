"""Export GIF recordings of all Physics3D environments with random actions.

Each environment runs for a fixed number of steps with random actions,
producing a GIF in the gifs/ directory. Useful for slides and demos.

Run with:
    pixi run -e apple mojo run -I . examples/export_env_gifs.mojo
"""

from std.random import seed
from mojo_rl.envs.render_random import render_random

from mojo_rl.envs.ant import Ant
from mojo_rl.envs.half_cheetah import HalfCheetah
from mojo_rl.envs.hopper import Hopper
from mojo_rl.envs.walker2d import Walker2d
from mojo_rl.envs.swimmer import Swimmer
from mojo_rl.envs.humanoid import Humanoid
from mojo_rl.envs.inverted_pendulum import InvertedPendulum
from mojo_rl.envs.inverted_double_pendulum import InvertedDoublePendulum
from mojo_rl.envs.reacher import Reacher
from mojo_rl.envs.pusher import Pusher
from mojo_rl.envs.metaworld import SawyerReach

from mojo_rl.core import BoxContinuousActionEnv, RenderableEnv


def record_env[
    E: BoxContinuousActionEnv & RenderableEnv,
](
    mut env: E, name: String, steps: Int = 500, fps: Int = 30, skip: Int = 1
) raises:
    print("=" * 60)
    print("Recording:", name)
    print("=" * 60)
    var path = "gifs/" + name + ".gif"
    render_random(
        env,
        num_steps=steps,
        frame_delay_ms=0,
        verbose=False,
        show_velocity=False,
        record_path=path,
        record_fps=fps,
        record_skip=skip,
    )
    print("  Saved:", path)
    print()


def main() raises:
    seed(1)

    # --- Uncomment the env(s) you want to export ---
    # Locomotion envs (show_velocity=False for cleaner GIFs)
    # var ant = Ant()
    # record_env(ant, "ant", steps=500, skip=3)

    # var cheetah = HalfCheetah()
    # record_env(cheetah, "half_cheetah", steps=500)

    # var hopper = Hopper[TERMINATE_ON_UNHEALTHY=False]()
    # record_env(hopper, "hopper", steps=500, skip=3)

    # var walker = Walker2d()
    # record_env(walker, "walker2d", steps=500, skip=3)

    # var swimmer = Swimmer()
    # record_env(swimmer, "swimmer", steps=500, skip=3)

    # var humanoid = Humanoid()
    # record_env(humanoid, "humanoid", steps=500, skip=3)

    # Balance envs
    # var ip = InvertedPendulum()
    # record_env(ip, "inverted_pendulum", steps=300)

    # var idp = InvertedDoublePendulum()
    # record_env(idp, "inverted_double_pendulum", steps=300)

    # Manipulation envs
    # var reacher = Reacher()
    # record_env(reacher, "reacher", steps=300)

    # var pusher = Pusher()
    # record_env(pusher, "pusher", steps=300)

    var sawyer_reach = SawyerReach()
    record_env(sawyer_reach, "sawyer_reach", steps=500, skip=5)

    print("Done! GIFs saved to gifs/")
