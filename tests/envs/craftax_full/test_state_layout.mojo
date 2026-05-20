"""Phase 7A gate: structural sanity of Full Craftax constants + state layout.

Compile-time-only checks: enum sizes match the reference, section offsets
don't overlap, and section sizes sum to STATE_SIZE.

Run:
  pixi run mojo run -I . tests/envs/craftax_full/test_state_layout.mojo
"""

from mojo_rl.envs.craftax_full import (
    MAP_H,
    MAP_W,
    MAP_SIZE_PER_FLOOR,
    MAP_TOTAL_SIZE,
    NUM_FLOORS,
    VIEW_H,
    VIEW_W,
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    NUM_ACTIONS,
    NUM_MOB_CATEGORIES,
    NUM_PROJECTILE_TYPES,
    NUM_INVENTORY,
    NUM_ACHIEVEMENTS,
    OBS_VIEW_SIZE,
    OBS_SCALAR_SIZE,
    OBS_DIM,
    TILE_CHANNELS,
    STATE_SIZE,
)
from mojo_rl.envs.craftax_full.constants import (
    achievement_reward_weight,
    floor_mob_health,
    melee_damage,
    projectile_damage,
    floor_mob_spawn_chance,
    required_pickaxe_tier,
    BLOCK_DIAMOND,
    BLOCK_IRON,
    BLOCK_STONE,
    BLOCK_GRASS,
    ACH_COLLECT_WOOD,
    ACH_DEFEAT_NECROMANCER,
    ACH_COLLECT_RUBY,
    PROJ_FIREBALL,
    PROJ_ICEBALL,
)
from mojo_rl.envs.craftax_full.state import (
    S_MAP_BASE,
    S_ITEM_MAP_BASE,
    S_MOB_MAP_BASE,
    S_LIGHT_MAP_BASE,
    S_DOWN_LADDERS_BASE,
    S_INV_BASE,
    S_ACHIEVEMENTS_BASE,
    S_TIMESTEP,
    S_RNG_BASE,
    S_RNG_WORDS,
    s_map,
    s_inv,
    s_achievement,
)


@always_inline
def check(mut counts: List[Int], name: String, ok: Bool):
    if ok:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


def test_enum_sizes(mut counts: List[Int]) raises:
    print("test_enum_sizes")
    check(counts, "NUM_BLOCK_TYPES == 37", NUM_BLOCK_TYPES == 37)
    check(counts, "NUM_ITEM_TYPES == 5", NUM_ITEM_TYPES == 5)
    check(counts, "NUM_ACTIONS == 43", NUM_ACTIONS == 43)
    check(counts, "NUM_MOB_CATEGORIES == 4", NUM_MOB_CATEGORIES == 4)
    check(counts, "NUM_PROJECTILE_TYPES == 8", NUM_PROJECTILE_TYPES == 8)
    check(counts, "NUM_INVENTORY == 24", NUM_INVENTORY == 24)
    check(counts, "NUM_ACHIEVEMENTS == 67", NUM_ACHIEVEMENTS == 67)
    check(counts, "NUM_FLOORS == 9", NUM_FLOORS == 9)
    check(counts, "MAP_H == 48 and MAP_W == 48",
          MAP_H == 48 and MAP_W == 48)
    check(counts, "VIEW_H == 9 and VIEW_W == 11",
          VIEW_H == 9 and VIEW_W == 11)


def test_layout_monotonic(mut counts: List[Int]) raises:
    """Section base offsets must be strictly increasing."""
    print("test_layout_monotonic")
    check(counts, "S_MAP_BASE < S_ITEM_MAP_BASE",
          S_MAP_BASE < S_ITEM_MAP_BASE)
    check(counts, "S_ITEM_MAP_BASE < S_MOB_MAP_BASE",
          S_ITEM_MAP_BASE < S_MOB_MAP_BASE)
    check(counts, "S_MOB_MAP_BASE < S_LIGHT_MAP_BASE",
          S_MOB_MAP_BASE < S_LIGHT_MAP_BASE)
    check(counts, "S_LIGHT_MAP_BASE < S_DOWN_LADDERS_BASE",
          S_LIGHT_MAP_BASE < S_DOWN_LADDERS_BASE)
    check(counts, "S_INV_BASE < S_ACHIEVEMENTS_BASE",
          S_INV_BASE < S_ACHIEVEMENTS_BASE)
    check(counts, "S_ACHIEVEMENTS_BASE < S_TIMESTEP",
          S_ACHIEVEMENTS_BASE < S_TIMESTEP)
    check(counts, "S_TIMESTEP < S_RNG_BASE",
          S_TIMESTEP < S_RNG_BASE)
    check(counts, "S_RNG_BASE + 4 == STATE_SIZE",
          S_RNG_BASE + S_RNG_WORDS == STATE_SIZE)


def test_map_indexer_range(mut counts: List[Int]) raises:
    """s_map(floor, y, x) must land inside [S_MAP_BASE, S_MAP_BASE+MAP_TOTAL_SIZE)."""
    print("test_map_indexer_range")
    var first = s_map(0, 0, 0)
    var last = s_map(NUM_FLOORS - 1, MAP_H - 1, MAP_W - 1)
    check(counts, "s_map(0,0,0) == S_MAP_BASE", first == S_MAP_BASE)
    check(counts, "s_map(last) at MAP end",
          last == S_MAP_BASE + MAP_TOTAL_SIZE - 1)
    # Floor 1 starts MAP_SIZE_PER_FLOOR after floor 0.
    check(counts, "floor 1 stride correct",
          s_map(1, 0, 0) == S_MAP_BASE + MAP_SIZE_PER_FLOOR)


def test_inventory_and_achievements_in_range(mut counts: List[Int]) raises:
    """Inventory + achievement helpers must land in their declared ranges."""
    print("test_inventory_and_achievements_in_range")
    var inv0 = s_inv(0)
    var inv_last = s_inv(NUM_INVENTORY - 1)
    var ach0 = s_achievement(0)
    var ach_last = s_achievement(NUM_ACHIEVEMENTS - 1)
    check(counts, "s_inv(0) == S_INV_BASE", inv0 == S_INV_BASE)
    check(counts, "s_inv(N-1) == S_INV_BASE + N-1",
          inv_last == S_INV_BASE + NUM_INVENTORY - 1)
    check(counts, "s_achievement(0) == S_ACHIEVEMENTS_BASE",
          ach0 == S_ACHIEVEMENTS_BASE)
    check(counts, "s_achievement(N-1) == base + N-1",
          ach_last == S_ACHIEVEMENTS_BASE + NUM_ACHIEVEMENTS - 1)


def test_obs_shape(mut counts: List[Int]) raises:
    """Symbolic obs dimensions match Craftax reference (8268-D)."""
    print("test_obs_shape")
    # Reference tile encoding: 37 block + 5 item + 5*8 mob + 1 light = 83.
    check(counts, "TILE_CHANNELS == 83", TILE_CHANNELS == 83)
    check(counts, "OBS_VIEW_SIZE == 99 * 83 = 8217",
          OBS_VIEW_SIZE == 8217)
    # Reference scalar tail = 16 + 6 + 9 + 4 + 4 + 4 + 8 = 51.
    check(counts, "OBS_SCALAR_SIZE == 51", OBS_SCALAR_SIZE == 51)
    check(counts, "OBS_DIM == 8268", OBS_DIM == 8268)
    check(counts, "OBS_DIM == view + scalar",
          OBS_DIM == OBS_VIEW_SIZE + OBS_SCALAR_SIZE)


def test_combat_tables(mut counts: List[Int]) raises:
    """Spot-check the per-floor / per-species tables."""
    print("test_combat_tables")
    # Floor 0: zombie HP = 5, cow HP = 3, skeleton HP = 3.
    check(counts, "floor 0 passive HP = 3", floor_mob_health(0, 0) == 3)
    check(counts, "floor 0 melee HP = 5", floor_mob_health(0, 1) == 5)
    check(counts, "floor 0 ranged HP = 3", floor_mob_health(0, 2) == 3)
    # Floor 7 ice troll melee HP = 24.
    check(counts, "floor 7 melee HP = 24", floor_mob_health(7, 1) == 24)
    # Zombie melee damage = (2, 0, 0).
    var zd = melee_damage(0)
    check(counts, "zombie damage = (2,0,0)",
          zd[0] == 2 and zd[1] == 0 and zd[2] == 0)
    # Fireball projectile damage = (0, 3, 0).
    var fb = projectile_damage(PROJ_FIREBALL)
    check(counts, "fireball damage = (0,3,0)",
          fb[0] == 0 and fb[1] == 3 and fb[2] == 0)
    var ib = projectile_damage(PROJ_ICEBALL)
    check(counts, "iceball damage = (0,0,3)",
          ib[0] == 0 and ib[1] == 0 and ib[2] == 3)
    # Diamond needs iron pickaxe (tier 3). Stone needs tier 1.
    check(counts, "diamond requires tier 3",
          required_pickaxe_tier(BLOCK_DIAMOND) == 3)
    check(counts, "stone requires tier 1",
          required_pickaxe_tier(BLOCK_STONE) == 1)
    check(counts, "grass requires tier 0",
          required_pickaxe_tier(BLOCK_GRASS) == 0)
    # Spawn chances: floor 0 melee day = 0.02, dungeons = 0.06.
    check(counts, "floor 0 melee-day spawn = 0.02",
          floor_mob_spawn_chance(0, 1) == Float32(0.02))
    check(counts, "floor 2 melee-day spawn = 0.06",
          floor_mob_spawn_chance(2, 1) == Float32(0.06))


def test_achievement_reward_tiers(mut counts: List[Int]) raises:
    """Achievement tier weights match the reference's
    `achievement_mapping`: 1 / 3 / 5 / 8 buckets."""
    print("test_achievement_reward_tiers")
    # COLLECT_WOOD (id=0) → tier 1.
    check(counts, "COLLECT_WOOD weight == 1.0",
          achievement_reward_weight(ACH_COLLECT_WOOD) == Float32(1.0))
    # COLLECT_RUBY (id=59) → intermediate (3).
    check(counts, "COLLECT_RUBY weight == 3.0",
          achievement_reward_weight(ACH_COLLECT_RUBY) == Float32(3.0))
    # DEFEAT_NECROMANCER (id=49) → very advanced (8).
    check(counts, "DEFEAT_NECROMANCER weight == 8.0",
          achievement_reward_weight(ACH_DEFEAT_NECROMANCER) == Float32(8.0))


def main() raises:
    print("Craftax-Full Phase-7A state layout gate")
    print("=" * 50)
    var counts = [0, 0]
    test_enum_sizes(counts)
    test_layout_monotonic(counts)
    test_map_indexer_range(counts)
    test_inventory_and_achievements_in_range(counts)
    test_obs_shape(counts)
    test_combat_tables(counts)
    test_achievement_reward_tiers(counts)
    print()
    print("=" * 50)
    print("Passed:", counts[0], "Failed:", counts[1])
    print("STATE_SIZE =", STATE_SIZE,
          " floats per env (~",
          (STATE_SIZE * 4) // 1024, " KB)")
    if counts[1] > 0:
        raise Error("Phase-7A gate FAILED")
    print("Phase-7A gate PASS")
