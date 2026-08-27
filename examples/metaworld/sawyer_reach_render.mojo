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
    # ⚠ `frame_delay_ms` IS A SLEEP, NOT A BUDGET — 200 CAPPED THIS AT 5 FPS.
    # This example read as "the Sawyer env is very slow"; it is not. Measured
    # on M1 Pro: physics 0.19 ms/env step, the whole render loop 15 ms/frame
    # (65 FPS with the delay at 0). The 200 ms sleep was 93% of the frame.
    # 32 matches `examples/render_random_demo.mojo`; Sawyer's env step is
    # 12.5 ms of sim time (0.0025 x FRAME_SKIP 5), so this runs ~2.6x slow.
    render_random(env, num_steps=3000, frame_delay_ms=32)
