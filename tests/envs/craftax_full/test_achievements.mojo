"""Phase 7F gate: scripted achievement tests for Craftax-Full.

Each subtest sets up a hand-built state (block in front of player, inventory,
mobs, etc.), runs one or more `env.step()` calls, then asserts the matching
achievement bit (and any direct side effect) fires correctly.

Goal: verify that ≥30 of the 67 Craftax-Full achievements are unlockable from
the current `apply_step_inline` implementation. Mob-kill achievements that
don't yet have a `set_achievement` call in `game_logic.mojo` (DEFEAT_ZOMBIE,
DEFEAT_GNOME_WARRIOR, etc.) are NOT in scope here — they wait on mob-AI
phases.

Run:
  pixi run mojo run -I . tests/envs/craftax_full/test_achievements.mojo
"""

from mojo_rl.envs.craftax_full import (
    CraftaxFullEnv,
    CraftaxFullAction,
)
from mojo_rl.nn.constants import DT as dtype
from mojo_rl.envs.craftax_full.constants import (
    MAP_W,
    NUM_ACHIEVEMENTS,
    NUM_FLOORS,
    BLOCK_GRASS,
    BLOCK_WATER,
    BLOCK_STONE,
    BLOCK_TREE,
    BLOCK_PATH,
    BLOCK_COAL,
    BLOCK_IRON,
    BLOCK_DIAMOND,
    BLOCK_RUBY,
    BLOCK_SAPPHIRE,
    BLOCK_CRAFTING_TABLE,
    BLOCK_FURNACE,
    BLOCK_PLANT,
    BLOCK_RIPE_PLANT,
    BLOCK_CHEST,
    BLOCK_ENCHANTMENT_TABLE_FIRE,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP,
    DIR_DOWN,
    ITEM_LADDER_DOWN,
    INV_WOOD,
    INV_STONE,
    INV_COAL,
    INV_IRON,
    INV_DIAMOND,
    INV_SAPPHIRE,
    INV_RUBY,
    INV_SAPLING,
    INV_PICKAXE,
    INV_SWORD,
    INV_BOW,
    INV_ARROWS,
    INV_ARMOUR_HEAD,
    INV_TORCHES,
    INV_BOOKS,
    INV_POTIONS_BASE,
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
    INTRINSIC_MANA,
    INTRINSIC_IS_SLEEPING,
    INTRINSIC_MAX,
    NUM_INTRINSICS,
    MAX_MELEE_MOBS,
    MAX_PASSIVE_MOBS,
    MAX_RANGED_MOBS,
    MOB_FY,
    MOB_FX,
    MOB_HP,
    MOB_MASK,
    MOB_TYPE_ID,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    ACTION_DO,
    ACTION_NOOP,
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
    ACTION_DESCEND,
    ACTION_SHOOT_ARROW,
    ACTION_CAST_FIREBALL,
    ACTION_CAST_ICEBALL,
    ACTION_DRINK_POTION_RED,
    ACTION_READ_BOOK,
    ACTION_ENCHANT_SWORD,
    ACH_COLLECT_WOOD,
    ACH_COLLECT_STONE,
    ACH_COLLECT_COAL,
    ACH_COLLECT_IRON,
    ACH_COLLECT_DIAMOND,
    ACH_COLLECT_RUBY,
    ACH_COLLECT_SAPPHIRE,
    ACH_COLLECT_SAPLING,
    ACH_COLLECT_DRINK,
    ACH_PLACE_TABLE,
    ACH_PLACE_STONE,
    ACH_PLACE_FURNACE,
    ACH_PLACE_PLANT,
    ACH_PLACE_TORCH,
    ACH_MAKE_WOOD_PICKAXE,
    ACH_MAKE_STONE_PICKAXE,
    ACH_MAKE_IRON_PICKAXE,
    ACH_MAKE_DIAMOND_PICKAXE,
    ACH_MAKE_WOOD_SWORD,
    ACH_MAKE_STONE_SWORD,
    ACH_MAKE_IRON_SWORD,
    ACH_MAKE_DIAMOND_SWORD,
    ACH_MAKE_IRON_ARMOUR,
    ACH_MAKE_DIAMOND_ARMOUR,
    ACH_MAKE_ARROW,
    ACH_MAKE_TORCH,
    ACH_EAT_COW,
    ACH_EAT_PLANT,
    ACH_WAKE_UP,
    ACH_OPEN_CHEST,
    ACH_DRINK_POTION,
    ACH_LEARN_FIREBALL,
    ACH_LEARN_ICEBALL,
    ACH_CAST_FIREBALL,
    ACH_CAST_ICEBALL,
    ACH_FIRE_BOW,
    ACH_FIND_BOW,
    ACH_ENCHANT_SWORD,
    ACH_ENTER_DUNGEON,
    SPELL_FIREBALL,
    SPELL_ICEBALL,
)
from mojo_rl.envs.craftax_full.state import (
    STATE_SIZE,
    S_PLAYER_POS,
    S_PLAYER_LEVEL,
    S_PLAYER_DIR,
    S_ACHIEVEMENTS_BASE,
    s_map,
    s_item_map,
    s_mob_map,
    s_intrinsic,
    s_inv,
    s_monsters_killed,
    s_melee_mob,
    s_passive_mob,
    s_learned_spell,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


@always_inline
def _dir_offset(d: Int) -> Tuple[Int, Int]:
    if d == DIR_LEFT:
        return (0, -1)
    if d == DIR_RIGHT:
        return (0, 1)
    if d == DIR_UP:
        return (-1, 0)
    return (1, 0)


@always_inline
def set_block_in_front(mut env: CraftaxFullEnv[dtype], block: Int):
    var floor = Int(env.state[S_PLAYER_LEVEL])
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    var off = _dir_offset(Int(env.state[S_PLAYER_DIR]))
    env.state[s_map(floor, py + off[0], px + off[1])] = Float32(block)


@always_inline
def set_block_at(
    mut env: CraftaxFullEnv[dtype], floor: Int, y: Int, x: Int, block: Int
):
    env.state[s_map(floor, y, x)] = Float32(block)


@always_inline
def give(mut env: CraftaxFullEnv[dtype], slot: Int, count: Int):
    env.state[s_inv(slot)] = Float32(count)


@always_inline
def set_intr(mut env: CraftaxFullEnv[dtype], slot: Int, value: Int):
    env.state[s_intrinsic(slot)] = Float32(value)


@always_inline
def place_passive_mob_in_front(
    mut env: CraftaxFullEnv[dtype], slot: Int, hp: Int, type_id: Int = 0
):
    var floor = Int(env.state[S_PLAYER_LEVEL])
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    var off = _dir_offset(Int(env.state[S_PLAYER_DIR]))
    var fy = py + off[0]
    var fx = px + off[1]
    env.state[s_passive_mob(floor, slot, MOB_FY)] = Float32(fy)
    env.state[s_passive_mob(floor, slot, MOB_FX)] = Float32(fx)
    env.state[s_passive_mob(floor, slot, MOB_HP)] = Float32(hp)
    env.state[s_passive_mob(floor, slot, MOB_MASK)] = Float32(1.0)
    env.state[s_passive_mob(floor, slot, MOB_TYPE_ID)] = Float32(type_id)
    env.state[s_mob_map(floor, fy, fx)] = Float32(1.0)  # mark occupancy


@always_inline
def ach_set(env: CraftaxFullEnv[dtype], idx: Int) -> Bool:
    return env.state[S_ACHIEVEMENTS_BASE + idx] > Float32(0.5)


def setup_fresh_env(mut env: CraftaxFullEnv[dtype], seed: UInt64 = 42):
    """Reset env then carve a 5×5 grass neighborhood around the player so
    tests can stage a known block in front without interference from
    procgen lava/stone walls."""
    _ = env.reset_with_seed(seed)
    var floor = Int(env.state[S_PLAYER_LEVEL])
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            env.state[s_map(floor, py + dy, px + dx)] = Float32(BLOCK_GRASS)


@always_inline
def check(mut counts: List[Int], name: String, condition: Bool):
    if condition:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


# ----------------------------------------------------------------------------
# Block-collection achievements (DO on a block in front)
# ----------------------------------------------------------------------------


def test_collect_wood(mut counts: List[Int]) raises:
    print("test_collect_wood")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    set_block_in_front(env, BLOCK_TREE)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "COLLECT_WOOD set", ach_set(env, ACH_COLLECT_WOOD))
    check(counts, "wood +1",
          Int(env.state[s_inv(INV_WOOD)]) >= 1)


def test_collect_stone(mut counts: List[Int]) raises:
    print("test_collect_stone")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_PICKAXE, 1)  # wood pickaxe tier 1
    set_block_in_front(env, BLOCK_STONE)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "COLLECT_STONE set", ach_set(env, ACH_COLLECT_STONE))


def test_collect_coal(mut counts: List[Int]) raises:
    print("test_collect_coal")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_PICKAXE, 1)
    set_block_in_front(env, BLOCK_COAL)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "COLLECT_COAL set", ach_set(env, ACH_COLLECT_COAL))


def test_collect_iron(mut counts: List[Int]) raises:
    print("test_collect_iron")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_PICKAXE, 2)  # stone pickaxe
    set_block_in_front(env, BLOCK_IRON)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "COLLECT_IRON set", ach_set(env, ACH_COLLECT_IRON))


def test_collect_diamond(mut counts: List[Int]) raises:
    print("test_collect_diamond")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_PICKAXE, 3)  # iron pickaxe
    set_block_in_front(env, BLOCK_DIAMOND)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "COLLECT_DIAMOND set", ach_set(env, ACH_COLLECT_DIAMOND))


def test_collect_ruby(mut counts: List[Int]) raises:
    print("test_collect_ruby")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_PICKAXE, 3)
    set_block_in_front(env, BLOCK_RUBY)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "COLLECT_RUBY set", ach_set(env, ACH_COLLECT_RUBY))


def test_collect_sapphire(mut counts: List[Int]) raises:
    print("test_collect_sapphire")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_PICKAXE, 3)
    set_block_in_front(env, BLOCK_SAPPHIRE)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "COLLECT_SAPPHIRE set", ach_set(env, ACH_COLLECT_SAPPHIRE))


def test_collect_drink(mut counts: List[Int]) raises:
    print("test_collect_drink")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    set_intr(env, INTRINSIC_DRINK, 5)
    set_block_in_front(env, BLOCK_WATER)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "COLLECT_DRINK set", ach_set(env, ACH_COLLECT_DRINK))


def test_collect_sapling(mut counts: List[Int]) raises:
    """10% chance per try. Loop ≥80 times to virtually guarantee a hit."""
    print("test_collect_sapling")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    set_block_in_front(env, BLOCK_GRASS)
    for _ in range(80):
        _ = env.step(CraftaxFullAction(value=ACTION_DO))
        if ach_set(env, ACH_COLLECT_SAPLING):
            break
    check(counts, "COLLECT_SAPLING set",
          ach_set(env, ACH_COLLECT_SAPLING))


# ----------------------------------------------------------------------------
# Placement achievements
# ----------------------------------------------------------------------------


def test_place_table(mut counts: List[Int]) raises:
    print("test_place_table")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    _ = env.step(CraftaxFullAction(value=ACTION_PLACE_TABLE))
    check(counts, "PLACE_TABLE set", ach_set(env, ACH_PLACE_TABLE))
    check(counts, "wood decremented",
          Int(env.state[s_inv(INV_WOOD)]) == 4)


def test_place_stone(mut counts: List[Int]) raises:
    print("test_place_stone")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_STONE, 3)
    _ = env.step(CraftaxFullAction(value=ACTION_PLACE_STONE))
    check(counts, "PLACE_STONE set", ach_set(env, ACH_PLACE_STONE))


def test_place_furnace(mut counts: List[Int]) raises:
    print("test_place_furnace")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_STONE, 3)
    _ = env.step(CraftaxFullAction(value=ACTION_PLACE_FURNACE))
    check(counts, "PLACE_FURNACE set", ach_set(env, ACH_PLACE_FURNACE))


def test_place_plant(mut counts: List[Int]) raises:
    print("test_place_plant")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_SAPLING, 1)
    _ = env.step(CraftaxFullAction(value=ACTION_PLACE_PLANT))
    check(counts, "PLACE_PLANT set", ach_set(env, ACH_PLACE_PLANT))


def test_place_torch(mut counts: List[Int]) raises:
    print("test_place_torch")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_TORCHES, 1)
    _ = env.step(CraftaxFullAction(value=ACTION_PLACE_TORCH))
    check(counts, "PLACE_TORCH set", ach_set(env, ACH_PLACE_TORCH))


# ----------------------------------------------------------------------------
# Crafting achievements (need adjacent table / furnace)
# ----------------------------------------------------------------------------


@always_inline
def _stage_workstation(
    mut env: CraftaxFullEnv[dtype], with_furnace: Bool = False
):
    """Drop a crafting table (and optionally a furnace) adjacent to the
    player. Uses tiles distinct from the one in front so DO/place tests
    can still set their own block."""
    var floor = Int(env.state[S_PLAYER_LEVEL])
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    set_block_at(env, floor, py - 1, px, BLOCK_CRAFTING_TABLE)
    if with_furnace:
        set_block_at(env, floor, py, px - 1, BLOCK_FURNACE)


def test_make_wood_pickaxe(mut counts: List[Int]) raises:
    print("test_make_wood_pickaxe")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    _stage_workstation(env)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_WOOD_PICKAXE))
    check(counts, "MAKE_WOOD_PICKAXE set",
          ach_set(env, ACH_MAKE_WOOD_PICKAXE))
    check(counts, "pickaxe tier 1",
          Int(env.state[s_inv(INV_PICKAXE)]) == 1)


def test_make_stone_pickaxe(mut counts: List[Int]) raises:
    print("test_make_stone_pickaxe")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_STONE, 5)
    give(env, INV_PICKAXE, 1)  # need pickaxe_tier < 2
    _stage_workstation(env)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_STONE_PICKAXE))
    check(counts, "MAKE_STONE_PICKAXE set",
          ach_set(env, ACH_MAKE_STONE_PICKAXE))


def test_make_iron_pickaxe(mut counts: List[Int]) raises:
    print("test_make_iron_pickaxe")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_STONE, 5)
    give(env, INV_IRON, 5)
    give(env, INV_COAL, 5)
    give(env, INV_PICKAXE, 2)
    _stage_workstation(env, with_furnace=True)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_IRON_PICKAXE))
    check(counts, "MAKE_IRON_PICKAXE set",
          ach_set(env, ACH_MAKE_IRON_PICKAXE))


def test_make_diamond_pickaxe(mut counts: List[Int]) raises:
    print("test_make_diamond_pickaxe")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_DIAMOND, 1)
    give(env, INV_PICKAXE, 3)
    _stage_workstation(env)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_DIAMOND_PICKAXE))
    check(counts, "MAKE_DIAMOND_PICKAXE set",
          ach_set(env, ACH_MAKE_DIAMOND_PICKAXE))


def test_make_wood_sword(mut counts: List[Int]) raises:
    print("test_make_wood_sword")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    _stage_workstation(env)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_WOOD_SWORD))
    check(counts, "MAKE_WOOD_SWORD set",
          ach_set(env, ACH_MAKE_WOOD_SWORD))


def test_make_stone_sword(mut counts: List[Int]) raises:
    print("test_make_stone_sword")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_STONE, 5)
    give(env, INV_SWORD, 1)
    _stage_workstation(env)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_STONE_SWORD))
    check(counts, "MAKE_STONE_SWORD set",
          ach_set(env, ACH_MAKE_STONE_SWORD))


def test_make_iron_sword(mut counts: List[Int]) raises:
    print("test_make_iron_sword")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_IRON, 5)
    give(env, INV_COAL, 5)
    give(env, INV_SWORD, 2)
    _stage_workstation(env, with_furnace=True)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_IRON_SWORD))
    check(counts, "MAKE_IRON_SWORD set",
          ach_set(env, ACH_MAKE_IRON_SWORD))


def test_make_diamond_sword(mut counts: List[Int]) raises:
    print("test_make_diamond_sword")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_DIAMOND, 1)
    give(env, INV_SWORD, 3)
    _stage_workstation(env)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_DIAMOND_SWORD))
    check(counts, "MAKE_DIAMOND_SWORD set",
          ach_set(env, ACH_MAKE_DIAMOND_SWORD))


def test_make_iron_armour(mut counts: List[Int]) raises:
    print("test_make_iron_armour")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_IRON, 5)
    give(env, INV_COAL, 5)
    _stage_workstation(env, with_furnace=True)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_IRON_ARMOUR))
    check(counts, "MAKE_IRON_ARMOUR set",
          ach_set(env, ACH_MAKE_IRON_ARMOUR))
    check(counts, "head armour tier 1",
          Int(env.state[s_inv(INV_ARMOUR_HEAD)]) == 1)


def test_make_diamond_armour(mut counts: List[Int]) raises:
    print("test_make_diamond_armour")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_DIAMOND, 4)
    _stage_workstation(env)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_DIAMOND_ARMOUR))
    check(counts, "MAKE_DIAMOND_ARMOUR set",
          ach_set(env, ACH_MAKE_DIAMOND_ARMOUR))


def test_make_arrow(mut counts: List[Int]) raises:
    print("test_make_arrow")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_STONE, 5)
    _stage_workstation(env)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_ARROW))
    check(counts, "MAKE_ARROW set", ach_set(env, ACH_MAKE_ARROW))


def test_make_torch(mut counts: List[Int]) raises:
    print("test_make_torch")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_WOOD, 5)
    give(env, INV_COAL, 5)
    _stage_workstation(env)
    _ = env.step(CraftaxFullAction(value=ACTION_MAKE_TORCH))
    check(counts, "MAKE_TORCH set", ach_set(env, ACH_MAKE_TORCH))


# ----------------------------------------------------------------------------
# Mob-kill + eat achievements
# ----------------------------------------------------------------------------


def test_eat_plant(mut counts: List[Int]) raises:
    print("test_eat_plant")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    set_block_in_front(env, BLOCK_RIPE_PLANT)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "EAT_PLANT set", ach_set(env, ACH_EAT_PLANT))


def test_eat_cow(mut counts: List[Int]) raises:
    """Place a passive cow (HP 1) in front, give a wood sword (damage 2 > 1),
    DO → kill → EAT_COW fires via _try_attack_mob."""
    print("test_eat_cow")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_SWORD, 1)  # wood sword, damage 2
    set_intr(env, INTRINSIC_FOOD, 3)
    place_passive_mob_in_front(env, 0, 1)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "EAT_COW set", ach_set(env, ACH_EAT_COW))


# ----------------------------------------------------------------------------
# Misc achievements
# ----------------------------------------------------------------------------


def test_wake_up(mut counts: List[Int]) raises:
    """Manually put the player asleep with energy = 9. The very next
    intrinsics tick fires the wake-up branch (still in NOOP because the
    sleeping check zeroes the user action)."""
    print("test_wake_up")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    set_intr(env, INTRINSIC_ENERGY, INTRINSIC_MAX)
    set_intr(env, INTRINSIC_IS_SLEEPING, 1)
    _ = env.step(CraftaxFullAction(value=ACTION_NOOP))
    check(counts, "WAKE_UP set", ach_set(env, ACH_WAKE_UP))
    check(counts, "is_sleeping cleared",
          Int(env.state[s_intrinsic(INTRINSIC_IS_SLEEPING)]) == 0)


def test_open_chest(mut counts: List[Int]) raises:
    print("test_open_chest")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    set_block_in_front(env, BLOCK_CHEST)
    _ = env.step(CraftaxFullAction(value=ACTION_DO))
    check(counts, "OPEN_CHEST set", ach_set(env, ACH_OPEN_CHEST))


def test_drink_potion(mut counts: List[Int]) raises:
    print("test_drink_potion")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    env.state[s_inv(INV_POTIONS_BASE + 0)] = Float32(1)  # 1 red potion
    _ = env.step(CraftaxFullAction(value=ACTION_DRINK_POTION_RED))
    check(counts, "DRINK_POTION set", ach_set(env, ACH_DRINK_POTION))


def test_learn_spell(mut counts: List[Int]) raises:
    """READ_BOOK randomly teaches either fireball or iceball. Loop until
    we observe at least one of the two learn achievements."""
    print("test_learn_spell")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    var got_fb = False
    var got_ib = False
    for _ in range(40):
        give(env, INV_BOOKS, 1)
        _ = env.step(CraftaxFullAction(value=ACTION_READ_BOOK))
        if ach_set(env, ACH_LEARN_FIREBALL):
            got_fb = True
        if ach_set(env, ACH_LEARN_ICEBALL):
            got_ib = True
        if got_fb and got_ib:
            break
    check(counts, "LEARN_FIREBALL or LEARN_ICEBALL set",
          got_fb or got_ib)


def test_cast_fireball(mut counts: List[Int]) raises:
    print("test_cast_fireball")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    env.state[s_learned_spell(SPELL_FIREBALL)] = Float32(1.0)
    set_intr(env, INTRINSIC_MANA, INTRINSIC_MAX)
    _ = env.step(CraftaxFullAction(value=ACTION_CAST_FIREBALL))
    check(counts, "CAST_FIREBALL set", ach_set(env, ACH_CAST_FIREBALL))


def test_cast_iceball(mut counts: List[Int]) raises:
    print("test_cast_iceball")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    env.state[s_learned_spell(SPELL_ICEBALL)] = Float32(1.0)
    set_intr(env, INTRINSIC_MANA, INTRINSIC_MAX)
    _ = env.step(CraftaxFullAction(value=ACTION_CAST_ICEBALL))
    check(counts, "CAST_ICEBALL set", ach_set(env, ACH_CAST_ICEBALL))


def test_fire_bow(mut counts: List[Int]) raises:
    print("test_fire_bow")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_BOW, 1)
    give(env, INV_ARROWS, 3)
    _ = env.step(CraftaxFullAction(value=ACTION_SHOOT_ARROW))
    check(counts, "FIRE_BOW set", ach_set(env, ACH_FIRE_BOW))


def test_find_bow(mut counts: List[Int]) raises:
    """Calculate_inventory_achievements runs every step. Putting a bow in
    the inventory then taking any action should mark FIND_BOW."""
    print("test_find_bow")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_BOW, 1)
    _ = env.step(CraftaxFullAction(value=ACTION_NOOP))
    check(counts, "FIND_BOW set", ach_set(env, ACH_FIND_BOW))


def test_enchant_sword(mut counts: List[Int]) raises:
    print("test_enchant_sword")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    give(env, INV_SWORD, 1)
    give(env, INV_RUBY, 1)
    # Fire enchant table adjacent to the player (above).
    var floor = Int(env.state[S_PLAYER_LEVEL])
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    set_block_at(env, floor, py - 1, px, BLOCK_ENCHANTMENT_TABLE_FIRE)
    _ = env.step(CraftaxFullAction(value=ACTION_ENCHANT_SWORD))
    check(counts, "ENCHANT_SWORD set", ach_set(env, ACH_ENCHANT_SWORD))


def test_enter_dungeon(mut counts: List[Int]) raises:
    """Set the kill quota for floor 0, drop a LADDER_DOWN under the player,
    and DESCEND → ACH_ENTER_DUNGEON."""
    print("test_enter_dungeon")
    var env = CraftaxFullEnv[dtype]()
    setup_fresh_env(env)
    env.state[s_monsters_killed(0)] = Float32(MONSTERS_KILLED_TO_CLEAR_LEVEL)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    env.state[s_item_map(0, py, px)] = Float32(ITEM_LADDER_DOWN)
    _ = env.step(CraftaxFullAction(value=ACTION_DESCEND))
    check(counts, "ENTER_DUNGEON set", ach_set(env, ACH_ENTER_DUNGEON))
    check(counts, "player level == 1",
          Int(env.state[S_PLAYER_LEVEL]) == 1)


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------


def main() raises:
    print("Craftax-Full Phase-7F achievement gate")
    print("=" * 50)
    # Mojo 1.0 builds an `Array` from a list literal by default; the
    # helpers below take `List[Int]`, so the type must be stated.
    var counts: List[Int] = [0, 0]
    test_collect_wood(counts)
    test_collect_stone(counts)
    test_collect_coal(counts)
    test_collect_iron(counts)
    test_collect_diamond(counts)
    test_collect_ruby(counts)
    test_collect_sapphire(counts)
    test_collect_drink(counts)
    test_collect_sapling(counts)
    test_place_table(counts)
    test_place_stone(counts)
    test_place_furnace(counts)
    test_place_plant(counts)
    test_place_torch(counts)
    test_make_wood_pickaxe(counts)
    test_make_stone_pickaxe(counts)
    test_make_iron_pickaxe(counts)
    test_make_diamond_pickaxe(counts)
    test_make_wood_sword(counts)
    test_make_stone_sword(counts)
    test_make_iron_sword(counts)
    test_make_diamond_sword(counts)
    test_make_iron_armour(counts)
    test_make_diamond_armour(counts)
    test_make_arrow(counts)
    test_make_torch(counts)
    test_eat_plant(counts)
    test_eat_cow(counts)
    test_wake_up(counts)
    test_open_chest(counts)
    test_drink_potion(counts)
    test_learn_spell(counts)
    test_cast_fireball(counts)
    test_cast_iceball(counts)
    test_fire_bow(counts)
    test_find_bow(counts)
    test_enchant_sword(counts)
    test_enter_dungeon(counts)

    # Count unique achievement bits unlocked across the entire test run
    # (a single env instance can only set one or two, but the suite as a
    # whole proves ≥30 are reachable). We don't share state across tests,
    # so instead we count the number of distinct achievements asserted.
    var ach_ids = [
        ACH_COLLECT_WOOD, ACH_COLLECT_STONE, ACH_COLLECT_COAL,
        ACH_COLLECT_IRON, ACH_COLLECT_DIAMOND, ACH_COLLECT_RUBY,
        ACH_COLLECT_SAPPHIRE, ACH_COLLECT_DRINK, ACH_COLLECT_SAPLING,
        ACH_PLACE_TABLE, ACH_PLACE_STONE, ACH_PLACE_FURNACE,
        ACH_PLACE_PLANT, ACH_PLACE_TORCH,
        ACH_MAKE_WOOD_PICKAXE, ACH_MAKE_STONE_PICKAXE,
        ACH_MAKE_IRON_PICKAXE, ACH_MAKE_DIAMOND_PICKAXE,
        ACH_MAKE_WOOD_SWORD, ACH_MAKE_STONE_SWORD,
        ACH_MAKE_IRON_SWORD, ACH_MAKE_DIAMOND_SWORD,
        ACH_MAKE_IRON_ARMOUR, ACH_MAKE_DIAMOND_ARMOUR,
        ACH_MAKE_ARROW, ACH_MAKE_TORCH,
        ACH_EAT_PLANT, ACH_EAT_COW,
        ACH_WAKE_UP, ACH_OPEN_CHEST, ACH_DRINK_POTION,
        ACH_CAST_FIREBALL, ACH_CAST_ICEBALL, ACH_FIRE_BOW,
        ACH_FIND_BOW, ACH_ENCHANT_SWORD, ACH_ENTER_DUNGEON,
    ]
    var unique_ach = len(ach_ids)

    print()
    print("=" * 50)
    print("Passed:", counts[0], " Failed:", counts[1])
    print("Distinct achievements exercised:", unique_ach,
          "/", NUM_ACHIEVEMENTS)
    if counts[1] > 0:
        raise Error("Phase-7F achievement gate FAILED")
    if unique_ach < 30:
        raise Error(
            "Phase-7F coverage too low (need ≥30 achievements, got "
            + String(unique_ach) + ")"
        )
    print("Phase-7F achievement gate PASS")
