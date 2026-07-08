"""Evaluate the scripted CarRacing driver (data-generation policy for offline
Dreamer 4). Runs a few episodes and reports return + tiles visited.

Run: pixi run mojo run -I . examples/car_racing/scripted_driver_eval.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB
from mojo_rl.envs.car_racing.scripted_driver import scripted_car_racing_action


def main() raises:
    comptime Env = CarRacingMB[DT, False]         # clean obs, CPU
    comptime MAX_STEPS = 1000
    var env = Env()
    print("scripted driver eval (", MAX_STEPS, "steps/ep):")
    for ep in range(4):
        _ = env.reset()
        var ret: Float64 = 0.0
        for t in range(MAX_STEPS):
            var a = scripted_car_racing_action(env, t)
            var r = env.step_action(a)
            ret += r[1]
            if r[2]:
                break
        print("  ep", ep, " return=", ret, " tiles=", env.tiles_visited,
              "/", env.track.track_length)
