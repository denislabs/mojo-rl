"""Phase-4 gate: structural correctness of the 1345-D symbolic observation.

Verifies:
  - obs length == OBS_DIM == 1345
  - Center tile of view = player's tile (block one-hot)
  - Off-map tiles render as BLOCK_OUT_OF_BOUNDS
  - Mob channels (17..20) light up exactly where mobs are in view
  - Inventory and intrinsics sections normalize correctly
  - Direction one-hot matches player_direction
  - Light/sleep scalars round-trip

Run:
  pixi run mojo run -I . tests/envs/craftax_classic/test_obs_layout.mojo
"""

from mojo_rl.envs.craftax_classic import CraftaxClassicEnv
from mojo_rl.envs.craftax_classic.constants import (
    MAP_W,
    BLOCK_GRASS,
    BLOCK_WATER,
    BLOCK_STONE,
    BLOCK_OUT_OF_BOUNDS,
    VIEW_H,
    VIEW_W,
    TILE_CHANNELS,
    NUM_BLOCK_TYPES,
    NUM_INTRINSICS,
    NUM_INVENTORY,
    NUM_DIRECTIONS,
    OBS_VIEW_SIZE,
    DIR_UP,
    DIR_DOWN,
    INV_WOOD,
    INV_DIAMOND,
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    MOB_FY,
    MOB_FX,
    MOB_HP,
    MOB_FIELDS,
)
from mojo_rl.envs.craftax_classic.state import (
    S_MAP_BASE,
    S_PLAYER_POS,
    S_INV_BASE,
    S_INTRINSICS_BASE,
    S_ZOMBIES_BASE,
    S_COWS_BASE,
    S_LIGHT_LEVEL,
    S_IS_SLEEPING,
)
from mojo_rl.nn.constants import DT as dtype


@always_inline
def check(mut counts: List[Int], name: String, condition: Bool):
    if condition:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


@always_inline
def approx_equal(a: Float64, b: Float64, eps: Float64 = 0.001) -> Bool:
    var d = a - b
    if d < Float64(0.0):
        d = -d
    return d < eps


@always_inline
def tile_channel(
    obs: List[Float32], lv: Int, lx: Int, ch: Int
) -> Float32:
    return obs[lv * VIEW_W * TILE_CHANNELS + lx * TILE_CHANNELS + ch]


def test_obs_dim(mut counts: List[Int]) raises:
    print("test_obs_dim")
    var env = CraftaxClassicEnv[dtype]()
    var obs = env.reset_obs_list()
    check(counts, "obs_dim == 1345", len(obs) == 1345)


def test_center_tile_is_grass(mut counts: List[Int]) raises:
    """After reset, the player tile is GRASS — center of the 7×9 view at
    (lv=3, lx=4) should be one-hot BLOCK_GRASS = 2."""
    print("test_center_tile_is_grass")
    var env = CraftaxClassicEnv[dtype]()
    _ = env.reset_with_seed(42, False)
    var obs_dyn = env.get_obs_list()
    var obs = List[Float32](capacity=len(obs_dyn))
    for i in range(len(obs_dyn)):
        obs.append(Float32(obs_dyn[i]))
    var center_v = VIEW_H // 2
    var center_x = VIEW_W // 2
    check(
        counts,
        "center tile = GRASS one-hot",
        approx_equal(
            Float64(tile_channel(obs, center_v, center_x, BLOCK_GRASS)),
            1.0,
        ),
    )
    # And no other block channel at the center.
    var any_other = False
    for ch in range(NUM_BLOCK_TYPES):
        if ch == BLOCK_GRASS:
            continue
        if tile_channel(obs, center_v, center_x, ch) > Float32(0.5):
            any_other = True
            break
    check(counts, "no other block at center", not any_other)


def test_out_of_bounds_padding(mut counts: List[Int]) raises:
    """Move player to a corner so part of the view is off-map and
    confirm those tiles render as BLOCK_OUT_OF_BOUNDS."""
    print("test_out_of_bounds_padding")
    var env = CraftaxClassicEnv[dtype]()
    _ = env.reset_with_seed(42, False)
    # Force player to (0, 0). The top-left half of the view (lv<3, lx<4)
    # will be off the map.
    env.state[S_PLAYER_POS] = Float32(0)
    env.state[S_PLAYER_POS + 1] = Float32(0)
    var obs_dyn = env.get_obs_list()
    var obs = List[Float32](capacity=len(obs_dyn))
    for i in range(len(obs_dyn)):
        obs.append(Float32(obs_dyn[i]))
    # Tile (0, 0) is at world (-3, -4) → out of bounds.
    check(
        counts,
        "tile (0,0) is OUT_OF_BOUNDS",
        approx_equal(
            Float64(tile_channel(obs, 0, 0, BLOCK_OUT_OF_BOUNDS)), 1.0
        ),
    )


def test_inventory_normalization(mut counts: List[Int]) raises:
    print("test_inventory_normalization")
    var env = CraftaxClassicEnv[dtype]()
    _ = env.reset_with_seed(42, False)
    env.state[S_INV_BASE + INV_WOOD] = Float32(5)
    env.state[S_INV_BASE + INV_DIAMOND] = Float32(3)
    var obs = env.get_obs_list()
    check(
        counts,
        "inv_wood / 10 = 0.5",
        approx_equal(Float64(obs[OBS_VIEW_SIZE + INV_WOOD]), 0.5),
    )
    check(
        counts,
        "inv_diamond / 10 = 0.3",
        approx_equal(Float64(obs[OBS_VIEW_SIZE + INV_DIAMOND]), 0.3),
    )


def test_intrinsics_normalization(mut counts: List[Int]) raises:
    print("test_intrinsics_normalization")
    var env = CraftaxClassicEnv[dtype]()
    _ = env.reset_with_seed(42, False)
    # After reset all intrinsics start at 9 → obs / 10 = 0.9.
    var obs = env.get_obs_list()
    var off = OBS_VIEW_SIZE + NUM_INVENTORY
    var all_correct = True
    for k in range(NUM_INTRINSICS):
        if not approx_equal(Float64(obs[off + k]), 0.9):
            all_correct = False
            break
    check(counts, "intrinsics / 10 = 0.9 after reset", all_correct)


def test_direction_one_hot(mut counts: List[Int]) raises:
    print("test_direction_one_hot")
    var env = CraftaxClassicEnv[dtype]()
    _ = env.reset_with_seed(42, False)
    # Reset sets DIR_UP = 2.
    var obs = env.get_obs_list()
    var off = OBS_VIEW_SIZE + NUM_INVENTORY + NUM_INTRINSICS
    var correct = True
    for k in range(NUM_DIRECTIONS):
        var expected: Float32 = 1.0 if k == DIR_UP else 0.0
        if not approx_equal(Float64(obs[off + k]), Float64(expected)):
            correct = False
            break
    check(counts, "direction one-hot at DIR_UP", correct)


def test_light_and_sleep_scalars(mut counts: List[Int]) raises:
    print("test_light_and_sleep_scalars")
    var env = CraftaxClassicEnv[dtype]()
    _ = env.reset_with_seed(42, False)
    var off = OBS_VIEW_SIZE + NUM_INVENTORY + NUM_INTRINSICS + NUM_DIRECTIONS
    var obs = env.get_obs_list()
    check(
        counts,
        "light_level matches state",
        approx_equal(
            Float64(obs[off + 0]), Float64(env.state[S_LIGHT_LEVEL])
        ),
    )
    check(
        counts,
        "is_sleeping = 0 after reset",
        approx_equal(Float64(obs[off + 1]), 0.0),
    )
    env.state[S_IS_SLEEPING] = Float32(1.0)
    var obs2 = env.get_obs_list()
    check(
        counts,
        "is_sleeping = 1 after manual set",
        approx_equal(Float64(obs2[off + 1]), 1.0),
    )


def test_zombie_in_view(mut counts: List[Int]) raises:
    """Place a zombie 2 tiles below the player; it should appear at local
    (5, 4) in the 7×9 view (center 3,4 + dy=2)."""
    print("test_zombie_in_view")
    var env = CraftaxClassicEnv[dtype]()
    _ = env.reset_with_seed(42, False)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    env.state[S_ZOMBIES_BASE + MOB_FY] = Float32(py + 2)
    env.state[S_ZOMBIES_BASE + MOB_FX] = Float32(px)
    env.state[S_ZOMBIES_BASE + MOB_HP] = Float32(5)
    var obs_dyn = env.get_obs_list()
    var obs = List[Float32](capacity=len(obs_dyn))
    for i in range(len(obs_dyn)):
        obs.append(Float32(obs_dyn[i]))
    var center_v = VIEW_H // 2
    var center_x = VIEW_W // 2
    check(
        counts,
        "zombie channel set at (center+2, center)",
        approx_equal(
            Float64(
                tile_channel(
                    obs, center_v + 2, center_x, NUM_BLOCK_TYPES + 0
                )
            ),
            1.0,
        ),
    )
    # And it should NOT be set on cow/skeleton/arrow channels for that tile.
    check(
        counts,
        "cow channel NOT set",
        tile_channel(obs, center_v + 2, center_x, NUM_BLOCK_TYPES + 1)
        < Float32(0.5),
    )


def test_cow_outside_view_not_shown(mut counts: List[Int]) raises:
    """A cow 10 tiles away from the player must not appear anywhere in
    the 7×9 view."""
    print("test_cow_outside_view_not_shown")
    var env = CraftaxClassicEnv[dtype]()
    _ = env.reset_with_seed(42, False)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    env.state[S_COWS_BASE + MOB_FY] = Float32(py + 10)
    env.state[S_COWS_BASE + MOB_FX] = Float32(px + 10)
    env.state[S_COWS_BASE + MOB_HP] = Float32(3)
    var obs_dyn = env.get_obs_list()
    var any_cow_channel_set = False
    for lv in range(VIEW_H):
        for lx in range(VIEW_W):
            var v = Float32(
                obs_dyn[
                    lv * VIEW_W * TILE_CHANNELS
                    + lx * TILE_CHANNELS
                    + NUM_BLOCK_TYPES
                    + 1
                ]
            )
            if v > Float32(0.5):
                any_cow_channel_set = True
                break
    check(counts, "no cow channel anywhere in view", not any_cow_channel_set)


def main() raises:
    print("Craftax-Classic Phase-4 observation gate")
    print("=" * 50)
    # Mojo 1.0 builds an `Array` from a list literal by default; the
    # helpers below take `List[Int]`, so the type must be stated.
    var counts: List[Int] = [0, 0]
    test_obs_dim(counts)
    test_center_tile_is_grass(counts)
    test_out_of_bounds_padding(counts)
    test_inventory_normalization(counts)
    test_intrinsics_normalization(counts)
    test_direction_one_hot(counts)
    test_light_and_sleep_scalars(counts)
    test_zombie_in_view(counts)
    test_cow_outside_view_not_shown(counts)

    print()
    print("=" * 50)
    print("Passed:", counts[0])
    print("Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-4 gate FAILED")
    print("Phase-4 gate PASS")
