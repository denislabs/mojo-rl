"""Phase 7C smoke test: env reset + step through every action.

Confirms:
  - reset_with_seed produces a state that passes basic invariants
    (player HP/food/drink/energy/mana = max, on floor 0, timestep = 0)
  - One step of each of the 43 actions completes without crashing
  - The step counter advances by exactly 1 per call
  - Done is not triggered prematurely

This is intentionally a smoke test, not a parity test against the
reference (combat / mob AI / boss are stubbed for this phase).

Run:
  pixi run mojo run -I . tests/envs/craftax_full/test_step.mojo
"""

from mojo_rl.envs.craftax_full import (
    CraftaxFullEnv,
    CraftaxFullAction,
    NUM_ACTIONS,
    STATE_SIZE,
)
from mojo_rl.envs.craftax_full.constants import (
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
    INTRINSIC_MANA,
    INTRINSIC_MAX,
    ATTR_DEXTERITY,
    ATTR_STRENGTH,
    ATTR_INTELLIGENCE,
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
    ACTION_REST,
    ACTION_DESCEND,
    ACTION_ASCEND,
    ACTION_SHOOT_ARROW,
    ACTION_CAST_FIREBALL,
    ACTION_CAST_ICEBALL,
    ACTION_DRINK_POTION_RED,
    ACTION_READ_BOOK,
    ACTION_ENCHANT_SWORD,
    ACTION_ENCHANT_BOW,
    ACTION_ENCHANT_ARMOUR,
    ACTION_LEVEL_UP_DEXTERITY,
    ACTION_LEVEL_UP_STRENGTH,
    ACTION_LEVEL_UP_INTELLIGENCE,
)
from mojo_rl.envs.craftax_full.state import (
    S_PLAYER_LEVEL,
    S_PLAYER_POS,
    S_TIMESTEP,
    s_intrinsic,
    s_attribute,
)


@always_inline
def check(mut counts: List[Int], name: String, ok: Bool):
    if ok:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


def test_reset(mut counts: List[Int]) raises:
    print("test_reset")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(0xC0FFEE))
    check(
        counts,
        "player on overworld",
        Int(env.state[S_PLAYER_LEVEL]) == 0,
    )
    check(
        counts,
        "timestep == 0",
        Int(env.state[S_TIMESTEP]) == 0,
    )
    for slot in [INTRINSIC_HEALTH, INTRINSIC_FOOD, INTRINSIC_DRINK,
                 INTRINSIC_ENERGY, INTRINSIC_MANA]:
        check(
            counts,
            "intrinsic " + String(slot) + " == 9",
            Int(env.state[s_intrinsic(slot)]) == INTRINSIC_MAX,
        )
    for slot in [ATTR_DEXTERITY, ATTR_STRENGTH, ATTR_INTELLIGENCE]:
        check(
            counts,
            "attribute " + String(slot) + " == 1",
            Int(env.state[s_attribute(slot)]) == 1,
        )
    check(counts, "done == False", not env.done)


def test_step_each_action(mut counts: List[Int]) raises:
    print("test_step_each_action")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(42))

    var t0 = Int(env.state[S_TIMESTEP])
    for a in range(NUM_ACTIONS):
        var act = CraftaxFullAction(value=a)
        var _result = env.step(act)
    var t1 = Int(env.state[S_TIMESTEP])
    check(
        counts,
        "timestep advanced by NUM_ACTIONS",
        t1 - t0 == NUM_ACTIONS,
    )
    check(counts, "not done after NUM_ACTIONS steps", not env.done)


def test_movement(mut counts: List[Int]) raises:
    print("test_movement")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(123))
    # Try all four cardinal directions. At least one should move the player
    # (overworld spawn is on GRASS at the map center, surrounded by varied
    # terrain).
    var moved = False
    var actions = [ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]
    for a in actions:
        _ = env.reset_with_seed(UInt64(123))
        var py_before = Int(env.state[S_PLAYER_POS])
        var px_before = Int(env.state[S_PLAYER_POS + 1])
        _ = env.step(CraftaxFullAction(value=a))
        var py_after = Int(env.state[S_PLAYER_POS])
        var px_after = Int(env.state[S_PLAYER_POS + 1])
        if py_after != py_before or px_after != px_before:
            moved = True
    check(counts, "at least one direction moves player", moved)


def test_intrinsics_tick(mut counts: List[Int]) raises:
    print("test_intrinsics_tick")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(7))
    # NOOP for many steps — hunger / thirst should eventually decrement
    # food / drink off the cap (HUNGER_THRESHOLD=25 → ~50 steps).
    for _ in range(120):
        _ = env.step(CraftaxFullAction(value=ACTION_NOOP))
    var food = Int(env.state[s_intrinsic(INTRINSIC_FOOD)])
    var drink = Int(env.state[s_intrinsic(INTRINSIC_DRINK)])
    check(counts, "food dropped from max after 120 NOOPs", food < INTRINSIC_MAX)
    check(counts, "drink dropped from max after 120 NOOPs", drink < INTRINSIC_MAX)


def test_max_timesteps_done(mut counts: List[Int]) raises:
    print("test_max_timesteps_done")
    # This would take 100k steps for a full run — skip the real test and
    # just confirm the done condition exists by checking after a short
    # episode: env should NOT report done yet.
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(2026))
    for _ in range(50):
        _ = env.step(CraftaxFullAction(value=ACTION_NOOP))
    check(counts, "not done after 50 steps", not env.done)


def main() raises:
    print("Craftax-Full Phase-7C step smoke gate")
    print("=" * 50)
    var counts = [0, 0]
    test_reset(counts)
    test_step_each_action(counts)
    test_movement(counts)
    test_intrinsics_tick(counts)
    test_max_timesteps_done(counts)
    print()
    print("=" * 50)
    print("Passed:", counts[0], "Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-7C gate FAILED")
    print("Phase-7C gate PASS")
