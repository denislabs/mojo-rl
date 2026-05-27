"""Playable Craftax-Classic with GIF recording.

Close the window to stop recording and save the GIF.

Run:
  pixi run mojo run -I . examples/craftax_classic/play_classic_gif.mojo

Controls:
  Arrows / WASD    Move (left/right/up/down)
  Space            DO (mine/attack/eat — depends on facing tile)
  Z                SLEEP
  Q                Place stone
  E                Place crafting table
  R                Place furnace
  T                Place plant (sapling)
  1                Make wood pickaxe
  2                Make stone pickaxe
  3                Make iron pickaxe
  4                Make wood sword
  5                Make stone sword
  6                Make iron sword
  Esc              Quit & save GIF
"""

from std.memory import alloc
from mojo_rl.envs.craftax_classic import CraftaxClassicEnv
from mojo_rl.envs.craftax_classic.constants import (
    ACTION_NOOP,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_DO,
    ACTION_SLEEP,
    ACTION_PLACE_STONE,
    ACTION_PLACE_TABLE,
    ACTION_PLACE_FURNACE,
    ACTION_PLACE_PLANT,
    ACTION_MAKE_WOOD_PICKAXE,
    ACTION_MAKE_STONE_PICKAXE,
    ACTION_MAKE_IRON_PICKAXE,
    ACTION_MAKE_WOOD_SWORD,
    ACTION_MAKE_STONE_SWORD,
    ACTION_MAKE_IRON_SWORD,
    NUM_ACHIEVEMENTS,
)
from mojo_rl.envs.craftax_classic.state import S_ACHIEVEMENTS_BASE
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode


@always_inline
def _read_action(keys: UnsafePointer[Bool, ImmutAnyOrigin]) -> Int:
    if keys[Int(Scancode.SCANCODE_LEFT)] or keys[Int(Scancode.SCANCODE_A)]:
        return ACTION_LEFT
    if keys[Int(Scancode.SCANCODE_RIGHT)] or keys[Int(Scancode.SCANCODE_D)]:
        return ACTION_RIGHT
    if keys[Int(Scancode.SCANCODE_UP)] or keys[Int(Scancode.SCANCODE_W)]:
        return ACTION_UP
    if keys[Int(Scancode.SCANCODE_DOWN)] or keys[Int(Scancode.SCANCODE_S)]:
        return ACTION_DOWN
    if keys[Int(Scancode.SCANCODE_SPACE)]:
        return ACTION_DO
    if keys[Int(Scancode.SCANCODE_Z)]:
        return ACTION_SLEEP
    if keys[Int(Scancode.SCANCODE_Q)]:
        return ACTION_PLACE_STONE
    if keys[Int(Scancode.SCANCODE_E)]:
        return ACTION_PLACE_TABLE
    if keys[Int(Scancode.SCANCODE_R)]:
        return ACTION_PLACE_FURNACE
    if keys[Int(Scancode.SCANCODE_T)]:
        return ACTION_PLACE_PLANT
    if keys[Int(Scancode.SCANCODE_1)]:
        return ACTION_MAKE_WOOD_PICKAXE
    if keys[Int(Scancode.SCANCODE_2)]:
        return ACTION_MAKE_STONE_PICKAXE
    if keys[Int(Scancode.SCANCODE_3)]:
        return ACTION_MAKE_IRON_PICKAXE
    if keys[Int(Scancode.SCANCODE_4)]:
        return ACTION_MAKE_WOOD_SWORD
    if keys[Int(Scancode.SCANCODE_5)]:
        return ACTION_MAKE_STONE_SWORD
    if keys[Int(Scancode.SCANCODE_6)]:
        return ACTION_MAKE_IRON_SWORD
    return ACTION_NOOP


def main() raises:
    print("=== Playable Craftax-Classic — Recording to GIF ===")
    print("Arrows/WASD = move    Space = do    Z = sleep    Esc = quit & save")
    print("Q/E/R/T     = place stone / table / furnace / plant")
    print("1..6        = craft pickaxes (1/2/3) / swords (4/5/6)")

    var env = CraftaxClassicEnv[DType.float32]()
    _ = env.init_renderer()
    _ = env.reset_obs_list()

    env.start_recording("gifs/craftax_classic_playable.gif", fps=15, skip=2)

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    var prev_action: Int = ACTION_NOOP
    var step_cooldown: Int = 0

    var step_count: Int = 0
    var episode: Int = 1

    while env.is_renderer_open():
        env.render_frame()

        var keys = get_keyboard_state(numkeys_ptr)
        var action = _read_action(keys)

        var fire = False
        if action != ACTION_NOOP:
            if action != prev_action:
                fire = True
                step_cooldown = 6
            elif step_cooldown == 0:
                fire = True
                step_cooldown = 6
        if step_cooldown > 0:
            step_cooldown -= 1
        prev_action = action

        if fire:
            var result = env.step_obs(action)
            var done = result[2]
            step_count += 1
            if done:
                var n = 0
                for k in range(NUM_ACHIEVEMENTS):
                    if env.state[S_ACHIEVEMENTS_BASE + k] > Float32(0.5):
                        n += 1
                print(
                    "Episode",
                    episode,
                    "done.  step=",
                    step_count,
                    " achievements=",
                    n,
                    "/",
                    NUM_ACHIEVEMENTS,
                )
                episode += 1
                step_count = 0
                _ = env.reset_obs_list()

        env.renderer_delay(33)

    env.stop_recording()
    numkeys_ptr.free()
    env.close_renderer()
    print("Saved: gifs/craftax_classic_playable.gif")
    print("=== Done ===")
