"""Playable CarRacing (multi-body, CPU) — drive the car yourself.

Manual driving on the SAME faithful `CarRacingMB` env the hybrid agent trains
and evals on, so you can reproduce by hand what the policy experiences — in
particular the "stuck in the grass" behavior (wheels turning but the car not
moving).

Controls (hold keys; combos allowed, e.g. UP+LEFT to power through a turn):
    UP     gas       (full throttle, 1.0)
    DOWN   brake     (0.8)
    LEFT   steer left  (-1.0)
    RIGHT  steer right (+1.0)
    R      reset the track
    ESC / close window  quit

A throttled HUD line prints ~3x/sec: forward speed, ON GRASS vs ON TRACK,
tiles visited, and the raw controls being applied — so you can tell whether
"not moving on grass" is the physics (gas held, speed stays ~0) or just that
no gas was applied.

NOTE: the trained discrete agent only ever applies gas=0.2 (action 3) and
cannot combine steering with gas (each action is one of noop/left/right/
gas/brake). Here you get full continuous control, so if the car moves for you
on grass but the agent gets stuck, the issue is the policy (out-of-distribution
grass states), not the simulator.

Run with:
    pixi run mojo run -I . examples/car_racing/play_car_racing.mojo
"""

from std.memory import alloc

from mojo_rl.nn.constants import DT
from mojo_rl.envs.car_racing import CarRacingMB
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode
from mojo_rl.core.fmt import fit


comptime MAX_STEPS = 1_000_000  # effectively no truncation while exploring
comptime FRAME_DELAY_MS = 20  # ~50 FPS (CarRacing runs at 50 FPS)
comptime HUD_EVERY = 15  # print a status line ~3x/sec


def main() raises:
    print("=" * 70)
    print("Playable CarRacing (multi-body, CPU)")
    print("=" * 70)
    print("  UP=gas  DOWN=brake  LEFT/RIGHT=steer  R=reset  ESC=quit")
    print("  Drive onto the grass (green) and try to power back with UP.")
    print()

    var env = CarRacingMB[DT](max_steps=MAX_STEPS)
    _ = env.init_renderer()
    _ = env.reset_obs_list()

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    var frame = 0
    var prev_r = False  # R edge-detect so one press = one reset

    while env.is_renderer_open():
        # Render first — begin_frame() pumps SDL events, updating key state.
        env.render_frame()
        var keys = get_keyboard_state(numkeys_ptr)

        var steer = 0.0
        if keys[Int(Scancode.SCANCODE_LEFT)]:
            steer = 1.0
        elif keys[Int(Scancode.SCANCODE_RIGHT)]:
            steer = -1.0

        var gas = 1.0 if keys[Int(Scancode.SCANCODE_UP)] else 0.0
        var brake = 0.8 if keys[Int(Scancode.SCANCODE_DOWN)] else 0.0

        # R = manual reset (edge-triggered).
        var r_now = Bool(keys[Int(Scancode.SCANCODE_R)])
        if r_now and not prev_r:
            _ = env.reset_obs_list()
        prev_r = r_now

        var result = env.step(steer, gas, brake)
        var reward = result[1]
        var done = result[2]

        frame += 1
        if frame % HUD_EVERY == 0:
            var surface = "GRASS" if env.on_grass() else "track"
            print(
                "speed:", fit(String(env.hull_speed()), 5),
                " on:", surface,
                " tiles:", env.tiles_visited, "/", env.track_length(),
                "  [steer", steer, "gas", gas, "brake", brake, "]",
            )

        if done:
            print(
                "--- episode end  reward:", reward,
                " tiles:", env.tiles_visited, "/", env.track_length(),
                " (resetting) ---",
            )
            _ = env.reset_obs_list()
            frame = 0

        env.renderer_delay(FRAME_DELAY_MS)
        if env.check_renderer_quit():
            break

    numkeys_ptr.free()
    env.close_renderer()
    print("=" * 70)
    print("Done.")
    print("=" * 70)
