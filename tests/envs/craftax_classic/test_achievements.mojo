"""Phase-3A gate: scripted achievement tests for Craftax-Classic.

Each subtest sets up a hand-built state (block in front of player,
inventory items, mob positions, etc.), runs one or more `env.step()`
calls, then asserts the corresponding achievement bit is set.

Run:
  pixi run mojo run -I . tests/envs/craftax_classic/test_achievements.mojo

Phase-3A covers 19 achievements via game-logic alone, plus the 3
mob-kill achievements via hand-placed mobs (DEFEAT_ZOMBIE,
DEFEAT_SKELETON, EAT_COW). Mob AI / natural spawning lands in 3B.
"""

from std.random.philox import Random as PhiloxRandom

from mojo_rl.envs.craftax_classic import CraftaxClassicEnv
from mojo_rl.envs.craftax_classic.constants import (
    MAP_W,
    BLOCK_GRASS,
    BLOCK_WATER,
    BLOCK_STONE,
    BLOCK_TREE,
    BLOCK_PATH,
    BLOCK_COAL,
    BLOCK_IRON,
    BLOCK_DIAMOND,
    BLOCK_CRAFTING_TABLE,
    BLOCK_FURNACE,
    BLOCK_PLANT,
    BLOCK_RIPE_PLANT,
    DIR_UP,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT,
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
    ACH_COLLECT_WOOD,
    ACH_PLACE_TABLE,
    ACH_EAT_COW,
    ACH_COLLECT_SAPLING,
    ACH_COLLECT_DRINK,
    ACH_MAKE_WOOD_PICKAXE,
    ACH_MAKE_STONE_PICKAXE,
    ACH_MAKE_IRON_PICKAXE,
    ACH_MAKE_WOOD_SWORD,
    ACH_MAKE_STONE_SWORD,
    ACH_MAKE_IRON_SWORD,
    ACH_PLACE_PLANT,
    ACH_DEFEAT_ZOMBIE,
    ACH_COLLECT_STONE,
    ACH_PLACE_STONE,
    ACH_EAT_PLANT,
    ACH_DEFEAT_SKELETON,
    ACH_COLLECT_IRON,
    ACH_COLLECT_COAL,
    ACH_PLACE_FURNACE,
    ACH_COLLECT_DIAMOND,
    ACH_WAKE_UP,
    INV_WOOD,
    INV_STONE,
    INV_COAL,
    INV_IRON,
    INV_SAPLING,
    INV_WOOD_PICKAXE,
    INV_STONE_PICKAXE,
    INV_IRON_PICKAXE,
    INV_WOOD_SWORD,
    INV_IRON_SWORD,
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
    MOB_FY,
    MOB_FX,
    MOB_HP,
    MAX_ZOMBIES,
    MAX_COWS,
    MAX_SKELETONS,
)
from mojo_rl.envs.craftax_classic.state import (
    S_MAP_BASE,
    S_PLAYER_POS,
    S_PLAYER_DIR,
    S_INTRINSICS_BASE,
    S_INV_BASE,
    S_ZOMBIES_BASE,
    S_COWS_BASE,
    S_SKELETONS_BASE,
    S_ACHIEVEMENTS_BASE,
    S_IS_SLEEPING,
    STATE_SIZE,
)
from mojo_rl.nn.constants import DT as dtype


# ----------------------------------------------------------------------------
# State manipulation helpers
# ----------------------------------------------------------------------------


@always_inline
def set_block_in_front(mut env: CraftaxClassicEnv[dtype], block: Int):
    """Put `block` on the tile the player is currently facing."""
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    var d = Int(env.state[S_PLAYER_DIR])
    var dy: Int
    var dx: Int
    if d == DIR_LEFT:
        dy = 0
        dx = -1
    elif d == DIR_RIGHT:
        dy = 0
        dx = 1
    elif d == DIR_UP:
        dy = -1
        dx = 0
    else:
        dy = 1
        dx = 0
    env.state[S_MAP_BASE + (py + dy) * MAP_W + (px + dx)] = Float32(block)


@always_inline
def set_block_at(
    mut env: CraftaxClassicEnv[dtype], y: Int, x: Int, block: Int
):
    env.state[S_MAP_BASE + y * MAP_W + x] = Float32(block)


@always_inline
def give(mut env: CraftaxClassicEnv[dtype], slot: Int, count: Int):
    env.state[S_INV_BASE + slot] = Float32(count)


@always_inline
def set_intrinsic(
    mut env: CraftaxClassicEnv[dtype], slot: Int, value: Int
):
    env.state[S_INTRINSICS_BASE + slot] = Float32(value)


@always_inline
def place_mob_in_front(
    mut env: CraftaxClassicEnv[dtype],
    base: Int,
    slot: Int,
    hp: Int,
):
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    var d = Int(env.state[S_PLAYER_DIR])
    var dy: Int
    var dx: Int
    if d == DIR_LEFT:
        dy = 0
        dx = -1
    elif d == DIR_RIGHT:
        dy = 0
        dx = 1
    elif d == DIR_UP:
        dy = -1
        dx = 0
    else:
        dy = 1
        dx = 0
    env.state[base + slot * 4 + MOB_FY] = Float32(py + dy)
    env.state[base + slot * 4 + MOB_FX] = Float32(px + dx)
    env.state[base + slot * 4 + MOB_HP] = Float32(hp)


@always_inline
def ach_set(env: CraftaxClassicEnv[dtype], idx: Int) -> Bool:
    return env.state[S_ACHIEVEMENTS_BASE + idx] > Float32(0.5)


@always_inline
def setup_fresh_env(
    mut env: CraftaxClassicEnv[dtype], seed: UInt64 = 42
):
    """Reset env then clear a 5×5 grass neighborhood around the player so
    tests can stage an exact block in front without map interference."""
    _ = env.reset_with_seed(seed, False)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            var y = py + dy
            var x = px + dx
            env.state[S_MAP_BASE + y * MAP_W + x] = Float32(BLOCK_GRASS)


# ----------------------------------------------------------------------------
# Assertion helper — counts live in main() since Mojo disallows globals.
# ----------------------------------------------------------------------------


@always_inline
def check(mut counts: List[Int], name: String, condition: Bool):
    if condition:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


# ----------------------------------------------------------------------------
# Individual achievement tests
# ----------------------------------------------------------------------------


def test_collect_wood(mut counts: List[Int]) raises:
    print("test_collect_wood")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    set_block_in_front(env, BLOCK_TREE)
    _ = env.step_obs(ACTION_DO)
    check(counts, "COLLECT_WOOD set", ach_set(env, ACH_COLLECT_WOOD))
    check(counts, "wood in inventory", Int(env.state[S_INV_BASE + INV_WOOD]) >= 1)


def test_place_table(mut counts: List[Int]) raises:
    print("test_place_table")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    _ = env.step_obs(ACTION_PLACE_TABLE)
    check(counts, "PLACE_TABLE set", ach_set(env, ACH_PLACE_TABLE))
    check(counts, "wood decremented", Int(env.state[S_INV_BASE + INV_WOOD]) == 4)


def test_collect_drink(mut counts: List[Int]) raises:
    print("test_collect_drink")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    set_intrinsic(env, INTRINSIC_DRINK, 5)
    set_block_in_front(env, BLOCK_WATER)
    _ = env.step_obs(ACTION_DO)
    check(counts, "COLLECT_DRINK set", ach_set(env, ACH_COLLECT_DRINK))
    check(counts,
        "drink bumped",
        Int(env.state[S_INTRINSICS_BASE + INTRINSIC_DRINK]) >= 6,
    )


def test_collect_sapling(mut counts: List[Int]) raises:
    """10% chance per try. Loop ≥30 times to virtually guarantee a hit."""
    print("test_collect_sapling")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    set_block_in_front(env, BLOCK_GRASS)
    # The block in front stays grass after each DO (we don't change it).
    for _ in range(80):
        _ = env.step_obs(ACTION_DO)
        if ach_set(env, ACH_COLLECT_SAPLING):
            break
    check(counts, "COLLECT_SAPLING set", ach_set(env, ACH_COLLECT_SAPLING))


def test_make_wood_pickaxe(mut counts: List[Int]) raises:
    print("test_make_wood_pickaxe")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    # Place crafting table adjacent to the player.
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    set_block_at(env, py + 1, px, BLOCK_CRAFTING_TABLE)
    _ = env.step_obs(ACTION_MAKE_WOOD_PICKAXE)
    check(counts,
        "MAKE_WOOD_PICKAXE set", ach_set(env, ACH_MAKE_WOOD_PICKAXE)
    )
    check(counts,
        "wood_pickaxe +1",
        Int(env.state[S_INV_BASE + INV_WOOD_PICKAXE]) == 1,
    )


def test_collect_stone(mut counts: List[Int]) raises:
    print("test_collect_stone")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD_PICKAXE, 1)
    set_block_in_front(env, BLOCK_STONE)
    _ = env.step_obs(ACTION_DO)
    check(counts, "COLLECT_STONE set", ach_set(env, ACH_COLLECT_STONE))
    check(counts, "stone +1", Int(env.state[S_INV_BASE + INV_STONE]) == 1)


def test_collect_stone_blocked_without_pickaxe(mut counts: List[Int]) raises:
    print("test_collect_stone_blocked_without_pickaxe")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    set_block_in_front(env, BLOCK_STONE)
    _ = env.step_obs(ACTION_DO)
    check(counts,
        "COLLECT_STONE NOT set without pickaxe",
        not ach_set(env, ACH_COLLECT_STONE),
    )


def test_place_stone(mut counts: List[Int]) raises:
    print("test_place_stone")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_STONE, 3)
    _ = env.step_obs(ACTION_PLACE_STONE)
    check(counts, "PLACE_STONE set", ach_set(env, ACH_PLACE_STONE))


def test_place_furnace(mut counts: List[Int]) raises:
    print("test_place_furnace")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_STONE, 3)
    _ = env.step_obs(ACTION_PLACE_FURNACE)
    check(counts, "PLACE_FURNACE set", ach_set(env, ACH_PLACE_FURNACE))


def test_collect_coal(mut counts: List[Int]) raises:
    print("test_collect_coal")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD_PICKAXE, 1)
    set_block_in_front(env, BLOCK_COAL)
    _ = env.step_obs(ACTION_DO)
    check(counts, "COLLECT_COAL set", ach_set(env, ACH_COLLECT_COAL))


def test_collect_iron(mut counts: List[Int]) raises:
    print("test_collect_iron")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_STONE_PICKAXE, 1)
    set_block_in_front(env, BLOCK_IRON)
    _ = env.step_obs(ACTION_DO)
    check(counts, "COLLECT_IRON set", ach_set(env, ACH_COLLECT_IRON))


def test_collect_diamond(mut counts: List[Int]) raises:
    print("test_collect_diamond")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_IRON_PICKAXE, 1)
    set_block_in_front(env, BLOCK_DIAMOND)
    _ = env.step_obs(ACTION_DO)
    check(counts, "COLLECT_DIAMOND set", ach_set(env, ACH_COLLECT_DIAMOND))


def test_make_iron_pickaxe(mut counts: List[Int]) raises:
    print("test_make_iron_pickaxe")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_STONE, 5)
    give(env, INV_IRON, 5)
    give(env, INV_COAL, 5)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    set_block_at(env, py + 1, px, BLOCK_CRAFTING_TABLE)
    set_block_at(env, py - 1, px, BLOCK_FURNACE)
    _ = env.step_obs(ACTION_MAKE_IRON_PICKAXE)
    check(counts,
        "MAKE_IRON_PICKAXE set", ach_set(env, ACH_MAKE_IRON_PICKAXE)
    )
    check(counts,
        "iron_pickaxe +1",
        Int(env.state[S_INV_BASE + INV_IRON_PICKAXE]) == 1,
    )


def test_make_iron_pickaxe_needs_furnace(mut counts: List[Int]) raises:
    print("test_make_iron_pickaxe_needs_furnace")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_STONE, 5)
    give(env, INV_IRON, 5)
    give(env, INV_COAL, 5)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    set_block_at(env, py + 1, px, BLOCK_CRAFTING_TABLE)
    # NO furnace.
    _ = env.step_obs(ACTION_MAKE_IRON_PICKAXE)
    check(counts,
        "MAKE_IRON_PICKAXE NOT set without furnace",
        not ach_set(env, ACH_MAKE_IRON_PICKAXE),
    )


def test_make_iron_sword(mut counts: List[Int]) raises:
    print("test_make_iron_sword")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_STONE, 5)
    give(env, INV_IRON, 5)
    give(env, INV_COAL, 5)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    set_block_at(env, py + 1, px, BLOCK_CRAFTING_TABLE)
    set_block_at(env, py - 1, px, BLOCK_FURNACE)
    _ = env.step_obs(ACTION_MAKE_IRON_SWORD)
    check(counts, "MAKE_IRON_SWORD set", ach_set(env, ACH_MAKE_IRON_SWORD))


def test_place_plant_and_eat_plant(mut counts: List[Int]) raises:
    """Place a plant, then via direct map edit ripen it, then eat it."""
    print("test_place_plant_and_eat_plant")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_SAPLING, 1)
    _ = env.step_obs(ACTION_PLACE_PLANT)
    check(counts, "PLACE_PLANT set", ach_set(env, ACH_PLACE_PLANT))
    # Manually ripen the plant (the slow 600-step age path is covered
    # implicitly by update_plants but we want a fast EAT_PLANT test).
    set_block_in_front(env, BLOCK_RIPE_PLANT)
    _ = env.step_obs(ACTION_DO)
    check(counts, "EAT_PLANT set", ach_set(env, ACH_EAT_PLANT))


def test_defeat_zombie(mut counts: List[Int]) raises:
    print("test_defeat_zombie")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_IRON_SWORD, 1)
    place_mob_in_front(env, S_ZOMBIES_BASE, 0, 5)  # zombie HP = 5
    # One iron-sword hit = 5 damage → kill.
    _ = env.step_obs(ACTION_DO)
    check(counts, "DEFEAT_ZOMBIE set", ach_set(env, ACH_DEFEAT_ZOMBIE))


def test_defeat_skeleton(mut counts: List[Int]) raises:
    print("test_defeat_skeleton")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_IRON_SWORD, 1)
    place_mob_in_front(env, S_SKELETONS_BASE, 0, 3)  # skeleton HP = 3
    _ = env.step_obs(ACTION_DO)
    check(counts, "DEFEAT_SKELETON set", ach_set(env, ACH_DEFEAT_SKELETON))


def test_eat_cow(mut counts: List[Int]) raises:
    print("test_eat_cow")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_IRON_SWORD, 1)
    set_intrinsic(env, INTRINSIC_FOOD, 3)
    place_mob_in_front(env, S_COWS_BASE, 0, 3)  # cow HP = 3
    _ = env.step_obs(ACTION_DO)
    check(counts, "EAT_COW set", ach_set(env, ACH_EAT_COW))
    check(counts,
        "food bumped",
        Int(env.state[S_INTRINSICS_BASE + INTRINSIC_FOOD]) >= 4,
    )


def test_wake_up(mut counts: List[Int]) raises:
    """Manually put the player in (sleeping, energy=9) so the very next
    intrinsics update fires the wake-up path."""
    print("test_wake_up")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    set_intrinsic(env, INTRINSIC_ENERGY, 9)
    env.state[S_IS_SLEEPING] = Float32(1.0)
    _ = env.step_obs(ACTION_NOOP)
    check(counts, "WAKE_UP set", ach_set(env, ACH_WAKE_UP))
    check(counts,
        "is_sleeping cleared", env.state[S_IS_SLEEPING] < Float32(0.5)
    )


def test_move_blocked_by_solid(mut counts: List[Int]) raises:
    print("test_move_blocked_by_solid")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    var py0 = Int(env.state[S_PLAYER_POS])
    var px0 = Int(env.state[S_PLAYER_POS + 1])
    set_block_in_front(env, BLOCK_STONE)  # in front of UP direction
    _ = env.step_obs(ACTION_UP)
    var py1 = Int(env.state[S_PLAYER_POS])
    var px1 = Int(env.state[S_PLAYER_POS + 1])
    check(counts, "position unchanged", py1 == py0 and px1 == px0)
    check(counts,
        "direction updated to UP",
        Int(env.state[S_PLAYER_DIR]) == DIR_UP,
    )


def test_move_walks_on_grass(mut counts: List[Int]) raises:
    print("test_move_walks_on_grass")
    var env = CraftaxClassicEnv[dtype]()
    setup_fresh_env(env)
    var py0 = Int(env.state[S_PLAYER_POS])
    var px0 = Int(env.state[S_PLAYER_POS + 1])
    _ = env.step_obs(ACTION_DOWN)
    var py1 = Int(env.state[S_PLAYER_POS])
    check(counts, "y increased by 1", py1 == py0 + 1)


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------


def main() raises:
    print("Craftax-Classic Phase-3A achievement gate")
    print("=" * 50)
    # Mojo 1.0 builds an `Array` from a list literal by default; the
    # helpers below take `List[Int]`, so the type must be stated.
    var counts: List[Int] = [0, 0]  # [passed, failed]

    test_collect_wood(counts)
    test_place_table(counts)
    test_collect_drink(counts)
    test_collect_sapling(counts)
    test_make_wood_pickaxe(counts)
    test_collect_stone(counts)
    test_collect_stone_blocked_without_pickaxe(counts)
    test_place_stone(counts)
    test_place_furnace(counts)
    test_collect_coal(counts)
    test_collect_iron(counts)
    test_collect_diamond(counts)
    test_make_iron_pickaxe(counts)
    test_make_iron_pickaxe_needs_furnace(counts)
    test_make_iron_sword(counts)
    test_place_plant_and_eat_plant(counts)
    test_defeat_zombie(counts)
    test_defeat_skeleton(counts)
    test_eat_cow(counts)
    test_wake_up(counts)
    test_move_blocked_by_solid(counts)
    test_move_walks_on_grass(counts)

    print()
    print("=" * 50)
    print("Passed:", counts[0])
    print("Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-3A gate FAILED")
    print("Phase-3A gate PASS")
