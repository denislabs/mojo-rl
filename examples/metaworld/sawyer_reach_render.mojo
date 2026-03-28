"""Render Sawyer Reach-v3 with random actions using render_random.

Run with:
    pixi run -e apple mojo run -I . examples/metaworld/sawyer_reach_render.mojo
"""

from std.random import seed
from mojo_rl.envs.render_random import render_random
from mojo_rl.envs.metaworld import SawyerReach


def main() raises:
    seed(42)
    var env = SawyerReach()
    render_random(env, num_steps=3000, frame_delay_ms=100)
