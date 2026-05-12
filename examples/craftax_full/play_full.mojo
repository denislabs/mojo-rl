"""Playable Craftax-Full — keyboard-controlled human demo.

Run:
  pixi run mojo run -I . examples/craftax_full/play_full.mojo

Controls (mirrors the reference `play_craftax.py`):
  WASD             Move (left/right/up/down)
  SPACE            DO (mine/attack/eat — depends on facing tile)
  TAB              SLEEP
  E                REST
  ,/.              ASCEND / DESCEND (ladder under you)
  T                Place crafting table
  R                Place stone
  F                Place furnace
  P                Place plant (sapling)
  J                Place torch (on your tile)
  1..4             Make wood/stone/iron/diamond pickaxe
  5..8             Make wood/stone/iron/diamond sword
  Y                Make iron armour (next free slot)
  U                Make diamond armour
  O                Make arrow      (LEFT_BRACKET)  Make torch
  I                Shoot arrow
  G/H              Cast fireball / iceball
  M                Read book (random spell unlock)
  K                Enchant sword           L     Enchant armour
  SEMICOLON        Enchant bow
  RIGHT_BRACKET    Level-up dexterity
  MINUS            Level-up strength
  EQUALS           Level-up intelligence
  Z/X/C/V/B/N      Drink potion red/green/blue/pink/cyan/yellow
  Esc / close      Quit
"""

from std.memory import alloc
from mojo_rl.envs.craftax_full import CraftaxFullEnv
from mojo_rl.envs.craftax_full.constants import (
    NUM_ACHIEVEMENTS,
    ACTION_NOOP,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_DO,
    ACTION_SLEEP,
    ACTION_REST,
    ACTION_DESCEND,
    ACTION_ASCEND,
    ACTION_PLACE_STONE,
    ACTION_PLACE_TABLE,
    ACTION_PLACE_FURNACE,
    ACTION_PLACE_PLANT,
    ACTION_PLACE_TORCH,
    ACTION_MAKE_WOOD_PICKAXE,
    ACTION_MAKE_STONE_PICKAXE,
    ACTION_MAKE_IRON_PICKAXE,
    ACTION_MAKE_DIAMOND_PICKAXE,
    ACTION_MAKE_WOOD_SWORD,
    ACTION_MAKE_STONE_SWORD,
    ACTION_MAKE_IRON_SWORD,
    ACTION_MAKE_DIAMOND_SWORD,
    ACTION_MAKE_IRON_ARMOUR,
    ACTION_MAKE_DIAMOND_ARMOUR,
    ACTION_MAKE_ARROW,
    ACTION_MAKE_TORCH,
    ACTION_SHOOT_ARROW,
    ACTION_CAST_FIREBALL,
    ACTION_CAST_ICEBALL,
    ACTION_READ_BOOK,
    ACTION_ENCHANT_SWORD,
    ACTION_ENCHANT_ARMOUR,
    ACTION_ENCHANT_BOW,
    ACTION_LEVEL_UP_DEXTERITY,
    ACTION_LEVEL_UP_STRENGTH,
    ACTION_LEVEL_UP_INTELLIGENCE,
    ACTION_DRINK_POTION_RED,
    ACTION_DRINK_POTION_GREEN,
    ACTION_DRINK_POTION_BLUE,
    ACTION_DRINK_POTION_PINK,
    ACTION_DRINK_POTION_CYAN,
    ACTION_DRINK_POTION_YELLOW,
)
from mojo_rl.envs.craftax_full.state import S_ACHIEVEMENTS_BASE
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode


@always_inline
def _read_action(keys: UnsafePointer[Bool, ImmutAnyOrigin]) -> Int:
    """Map current keyboard state to a Craftax-Full action. NOOP if no
    relevant key is held. First match wins — movement keys take priority
    over crafting / spell keys."""
    # Movement.
    if keys[Int(Scancode.SCANCODE_A)]:
        return ACTION_LEFT
    if keys[Int(Scancode.SCANCODE_D)]:
        return ACTION_RIGHT
    if keys[Int(Scancode.SCANCODE_W)]:
        return ACTION_UP
    if keys[Int(Scancode.SCANCODE_S)]:
        return ACTION_DOWN
    # Interact / sleep / rest / ladders.
    if keys[Int(Scancode.SCANCODE_SPACE)]:
        return ACTION_DO
    if keys[Int(Scancode.SCANCODE_TAB)]:
        return ACTION_SLEEP
    if keys[Int(Scancode.SCANCODE_E)]:
        return ACTION_REST
    if keys[Int(Scancode.SCANCODE_COMMA)]:
        return ACTION_ASCEND
    if keys[Int(Scancode.SCANCODE_PERIOD)]:
        return ACTION_DESCEND
    # Placement.
    if keys[Int(Scancode.SCANCODE_T)]:
        return ACTION_PLACE_TABLE
    if keys[Int(Scancode.SCANCODE_R)]:
        return ACTION_PLACE_STONE
    if keys[Int(Scancode.SCANCODE_F)]:
        return ACTION_PLACE_FURNACE
    if keys[Int(Scancode.SCANCODE_P)]:
        return ACTION_PLACE_PLANT
    if keys[Int(Scancode.SCANCODE_J)]:
        return ACTION_PLACE_TORCH
    # Crafting: pickaxes 1..4, swords 5..8.
    if keys[Int(Scancode.SCANCODE_1)]:
        return ACTION_MAKE_WOOD_PICKAXE
    if keys[Int(Scancode.SCANCODE_2)]:
        return ACTION_MAKE_STONE_PICKAXE
    if keys[Int(Scancode.SCANCODE_3)]:
        return ACTION_MAKE_IRON_PICKAXE
    if keys[Int(Scancode.SCANCODE_4)]:
        return ACTION_MAKE_DIAMOND_PICKAXE
    if keys[Int(Scancode.SCANCODE_5)]:
        return ACTION_MAKE_WOOD_SWORD
    if keys[Int(Scancode.SCANCODE_6)]:
        return ACTION_MAKE_STONE_SWORD
    if keys[Int(Scancode.SCANCODE_7)]:
        return ACTION_MAKE_IRON_SWORD
    if keys[Int(Scancode.SCANCODE_8)]:
        return ACTION_MAKE_DIAMOND_SWORD
    # Armour / consumables.
    if keys[Int(Scancode.SCANCODE_Y)]:
        return ACTION_MAKE_IRON_ARMOUR
    if keys[Int(Scancode.SCANCODE_U)]:
        return ACTION_MAKE_DIAMOND_ARMOUR
    if keys[Int(Scancode.SCANCODE_O)]:
        return ACTION_MAKE_ARROW
    if keys[Int(Scancode.SCANCODE_LEFTBRACKET)]:
        return ACTION_MAKE_TORCH
    if keys[Int(Scancode.SCANCODE_I)]:
        return ACTION_SHOOT_ARROW
    # Spells.
    if keys[Int(Scancode.SCANCODE_G)]:
        return ACTION_CAST_FIREBALL
    if keys[Int(Scancode.SCANCODE_H)]:
        return ACTION_CAST_ICEBALL
    if keys[Int(Scancode.SCANCODE_M)]:
        return ACTION_READ_BOOK
    # Enchant.
    if keys[Int(Scancode.SCANCODE_K)]:
        return ACTION_ENCHANT_SWORD
    if keys[Int(Scancode.SCANCODE_L)]:
        return ACTION_ENCHANT_ARMOUR
    if keys[Int(Scancode.SCANCODE_SEMICOLON)]:
        return ACTION_ENCHANT_BOW
    # Level-up.
    if keys[Int(Scancode.SCANCODE_RIGHTBRACKET)]:
        return ACTION_LEVEL_UP_DEXTERITY
    if keys[Int(Scancode.SCANCODE_MINUS)]:
        return ACTION_LEVEL_UP_STRENGTH
    if keys[Int(Scancode.SCANCODE_EQUALS)]:
        return ACTION_LEVEL_UP_INTELLIGENCE
    # Potions (Z X C V B N → red/green/blue/pink/cyan/yellow).
    if keys[Int(Scancode.SCANCODE_Z)]:
        return ACTION_DRINK_POTION_RED
    if keys[Int(Scancode.SCANCODE_X)]:
        return ACTION_DRINK_POTION_GREEN
    if keys[Int(Scancode.SCANCODE_C)]:
        return ACTION_DRINK_POTION_BLUE
    if keys[Int(Scancode.SCANCODE_V)]:
        return ACTION_DRINK_POTION_PINK
    if keys[Int(Scancode.SCANCODE_B)]:
        return ACTION_DRINK_POTION_CYAN
    if keys[Int(Scancode.SCANCODE_N)]:
        return ACTION_DRINK_POTION_YELLOW
    return ACTION_NOOP


def main() raises:
    print("=== Playable Craftax-Full ===")
    print("WASD = move    SPACE = do    TAB = sleep    E = rest")
    print(",/. = ascend / descend ladder")
    print("T/R/F/P/J = place table / stone / furnace / plant / torch")
    print("1..4 = pickaxes wood/stone/iron/diamond")
    print("5..8 = swords   wood/stone/iron/diamond")
    print("Y/U  = make iron / diamond armour")
    print("O/[  = make arrow / torch")
    print("I    = shoot arrow")
    print("G/H/M = cast fireball / iceball / read book")
    print("K/L/; = enchant sword / armour / bow")
    print("] - = = level-up dexterity / strength / intelligence")
    print("Z/X/C/V/B/N = drink potion red/green/blue/pink/cyan/yellow")
    print()

    var env = CraftaxFullEnv[DType.float32]()
    _ = env.init_renderer()
    _ = env.reset_obs_list()

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    # Per-key edge detection: only fire each key once per press.
    var prev_action: Int = ACTION_NOOP
    var step_cooldown: Int = 0  # frames until next action accepted

    var step_count: Int = 0
    var episode: Int = 1
    var total_ach: Int = 0

    while env.is_renderer_open():
        env.render_frame()

        var keys = get_keyboard_state(numkeys_ptr)
        var action = _read_action(keys)

        # Throttle: accept one action every 6 frames OR on key change.
        # Without this, holding a key spams steps.
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
                total_ach += n
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

        env.renderer_delay(33)  # ~30 fps

    numkeys_ptr.free()
    env.close_renderer()
    print("=== Done ===  Total achievements across episodes:", total_ach)
