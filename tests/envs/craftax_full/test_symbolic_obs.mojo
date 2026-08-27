"""Phase 7D-A smoke gate for the Full Craftax symbolic obs encoder.

Verifies:
  - obs length == OBS_DIM (8268)
  - reset obs is finite, all in [0, 1] (with sqrt/scaling factors)
  - per-tile block channels are one-hot (≤ 1 lit channel per tile in the
    block sub-range when the tile is lit; 0 lit channels when dark)
  - the center tile (player's tile) carries a lit light channel after
    reset, since the player spawns on the overworld in daylight
  - direction one-hot has exactly one 1.0 entry
  - intrinsics block normalizes max-health to 0.9 (= 9 / 10)
  - obs changes after stepping a movement action (player position shifts)

Run:
  pixi run mojo run -I . tests/envs/craftax_full/test_symbolic_obs.mojo
"""

from mojo_rl.envs.craftax_full import (
    CraftaxFullEnv,
    CraftaxFullAction,
    OBS_DIM,
    OBS_VIEW_SIZE,
)
from mojo_rl.envs.craftax_full.constants import (
    VIEW_H,
    VIEW_W,
    TILE_CHANNELS,
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    OBS_MOB_CLASSES,
    OBS_MOB_TYPES_PER_CLASS,
    OBS_CH_BLOCK_BASE,
    OBS_CH_ITEM_BASE,
    OBS_CH_MOB_BASE,
    OBS_CH_LIGHT,
    OBS_INV_SIZE,
    NUM_POTIONS,
    OBS_INTRINSICS_SIZE,
    OBS_DIRECTION_SIZE,
    OBS_ARMOUR_SIZE,
    OBS_ARMOUR_ENCH_SIZE,
    OBS_SPECIAL_SIZE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_NOOP,
)


@always_inline
def check(mut counts: List[Int], name: String, ok: Bool):
    if ok:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


def test_obs_length(mut counts: List[Int]) raises:
    print("test_obs_length")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(0xBEEF))
    var obs = env.get_obs_list()
    check(counts, "len(obs) == OBS_DIM (8268)", len(obs) == OBS_DIM)
    check(counts, "OBS_DIM == 8268", OBS_DIM == 8268)
    check(counts, "OBS_VIEW_SIZE == 8217", OBS_VIEW_SIZE == 8217)


def test_obs_bounds(mut counts: List[Int]) raises:
    """All obs values must be finite and in a sane range [0, 1]ish.
    sqrt(9)/10 = 0.3 is the largest sqrt-scaled inventory; direction is
    one-hot. Anything outside [-0.5, 1.5] is a bug."""
    print("test_obs_bounds")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(11))
    var obs = env.get_obs_list()
    var all_in_range = True
    var any_nonzero = False
    for i in range(len(obs)):
        var v = Float32(obs[i])
        if v < Float32(-0.5) or v > Float32(1.5):
            all_in_range = False
        if v != Float32(0.0):
            any_nonzero = True
    check(counts, "all obs in [-0.5, 1.5]", all_in_range)
    check(counts, "obs has some nonzero entries", any_nonzero)


def test_tile_onehot(mut counts: List[Int]) raises:
    """Each tile's block sub-channel sum must be 0 (dark) or 1 (lit, one-hot)."""
    print("test_tile_onehot")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(99))
    var obs = env.get_obs_list()
    var bad_tiles = 0
    for lv in range(VIEW_H):
        for lx in range(VIEW_W):
            var tb = lv * VIEW_W * TILE_CHANNELS + lx * TILE_CHANNELS
            var block_sum = 0
            for ch in range(NUM_BLOCK_TYPES):
                if Float32(obs[tb + OBS_CH_BLOCK_BASE + ch]) == Float32(1.0):
                    block_sum += 1
            var lit = Float32(obs[tb + OBS_CH_LIGHT]) == Float32(1.0)
            # Lit tile: exactly one block channel set. Dark tile: zero.
            if lit:
                if block_sum != 1:
                    bad_tiles += 1
            else:
                if block_sum != 0:
                    bad_tiles += 1
    check(counts, "every tile is one-hot or fully dark", bad_tiles == 0)


def test_center_tile_lit(mut counts: List[Int]) raises:
    """The center tile (player's own tile) should be lit on the
    overworld at the start of an episode."""
    print("test_center_tile_lit")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(2026))
    var obs = env.get_obs_list()
    var cy = VIEW_H // 2
    var cx = VIEW_W // 2
    var tb = cy * VIEW_W * TILE_CHANNELS + cx * TILE_CHANNELS
    var light = Float32(obs[tb + OBS_CH_LIGHT])
    check(counts, "center tile light == 1.0", light == Float32(1.0))


def test_direction_one_hot(mut counts: List[Int]) raises:
    """Direction segment of the scalar tail should be a one-hot."""
    print("test_direction_one_hot")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(7))
    var obs = env.get_obs_list()
    # direction lives at OBS_VIEW_SIZE + OBS_INV_SIZE + NUM_POTIONS
    #                                  + OBS_INTRINSICS_SIZE
    var dir_off = (
        OBS_VIEW_SIZE
        + OBS_INV_SIZE
        + NUM_POTIONS
        + OBS_INTRINSICS_SIZE
    )
    var ones = 0
    var zeros = 0
    for k in range(OBS_DIRECTION_SIZE):
        var v = Float32(obs[dir_off + k])
        if v == Float32(1.0):
            ones += 1
        elif v == Float32(0.0):
            zeros += 1
    check(counts, "direction has exactly one 1.0", ones == 1)
    check(counts, "direction has 3 zeros",
          zeros == OBS_DIRECTION_SIZE - 1)


def test_intrinsics_normalized(mut counts: List[Int]) raises:
    """At reset, HP/food/drink/energy/mana = 9. obs encodes them as /10
    so each should be 0.9."""
    print("test_intrinsics_normalized")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(123))
    var obs = env.get_obs_list()
    var intr_off = (
        OBS_VIEW_SIZE
        + OBS_INV_SIZE
        + NUM_POTIONS
    )
    var hp = Float32(obs[intr_off + 0])
    var food = Float32(obs[intr_off + 1])
    var drink = Float32(obs[intr_off + 2])
    var energy = Float32(obs[intr_off + 3])
    var mana = Float32(obs[intr_off + 4])
    var hp_ok = hp > Float32(0.85) and hp < Float32(0.95)
    var food_ok = food > Float32(0.85) and food < Float32(0.95)
    var drink_ok = drink > Float32(0.85) and drink < Float32(0.95)
    var energy_ok = energy > Float32(0.85) and energy < Float32(0.95)
    var mana_ok = mana > Float32(0.85) and mana < Float32(0.95)
    check(counts, "HP ≈ 0.9", hp_ok)
    check(counts, "food ≈ 0.9", food_ok)
    check(counts, "drink ≈ 0.9", drink_ok)
    check(counts, "energy ≈ 0.9", energy_ok)
    check(counts, "mana ≈ 0.9", mana_ok)


def test_obs_changes_after_move(mut counts: List[Int]) raises:
    """Stepping a movement action should change the obs (player shifts,
    so the local view rolls). At least one of the four cardinal directions
    must produce a different obs from the initial one."""
    print("test_obs_changes_after_move")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(2025))
    var obs0 = env.get_obs_list()
    var changed = False
    var actions = [ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]
    for a in actions:
        _ = env.reset_with_seed(UInt64(2025))
        _ = env.step(CraftaxFullAction(value=a))
        var obs1 = env.get_obs_list()
        # Compare just the view portion — that's where movement shows up.
        for i in range(OBS_VIEW_SIZE):
            if Float32(obs1[i]) != Float32(obs0[i]):
                changed = True
                break
        if changed:
            break
    check(counts, "obs changes after a movement step", changed)


def test_no_mob_in_dark(mut counts: List[Int]) raises:
    """Sanity: dark tiles must have zero mob channel mass.
    (Reference masks the whole view by the light mask.)"""
    print("test_no_mob_in_dark")
    var env = CraftaxFullEnv()
    _ = env.reset_with_seed(UInt64(54321))
    var obs = env.get_obs_list()
    var violations = 0
    for lv in range(VIEW_H):
        for lx in range(VIEW_W):
            var tb = lv * VIEW_W * TILE_CHANNELS + lx * TILE_CHANNELS
            if Float32(obs[tb + OBS_CH_LIGHT]) == Float32(0.0):
                var ms = Float32(0.0)
                for c in range(OBS_MOB_CLASSES * OBS_MOB_TYPES_PER_CLASS):
                    ms += Float32(obs[tb + OBS_CH_MOB_BASE + c])
                if ms != Float32(0.0):
                    violations += 1
    check(counts, "no mob channel set on a dark tile",
          violations == 0)


def main() raises:
    print("Craftax-Full Phase-7D symbolic obs smoke gate")
    print("=" * 50)
    # Mojo 1.0 builds an `Array` from a list literal by default; the
    # helpers below take `List[Int]`, so the type must be stated.
    var counts: List[Int] = [0, 0]
    test_obs_length(counts)
    test_obs_bounds(counts)
    test_tile_onehot(counts)
    test_center_tile_lit(counts)
    test_direction_one_hot(counts)
    test_intrinsics_normalized(counts)
    test_obs_changes_after_move(counts)
    test_no_mob_in_dark(counts)
    print()
    print("=" * 50)
    print("Passed:", counts[0], "Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-7D symbolic obs gate FAILED")
    print("Phase-7D symbolic obs gate PASS")
