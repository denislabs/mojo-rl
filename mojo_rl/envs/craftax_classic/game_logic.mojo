"""Core game logic for Craftax-Classic.

Ports `references/Craftax-main/craftax/craftax_classic/game_logic.py`
subroutine-by-subroutine. All functions operate on a single env's flat
state slice (`UnsafePointer[Float32, MutAnyOrigin]`) and are
`@always_inline` so they work in both the CPU `step_obs` path and the
GPU `step_kernel_gpu` per-thread body.

This file covers Phase 3A:
  - Helpers (direction lookup, bounds, solid-block predicate, achievement
    bits, inventory accessors, find_mob_at, manhattan distance).
  - _move_player
  - _place_block
  - _do_crafting
  - _update_intrinsics
  - _update_plants
  - _do_action (full version, including cow/zombie/skeleton attacks)
  - _cap_inventory
  - apply_step_inline (top-level step orchestrator; _update_mobs and
    _spawn_mobs are stubbed for Phase 3B)
"""

from std.math import cos as math_cos
from std.random.philox import Random as PhiloxRandom

from .constants import (
    MAP_H,
    MAP_W,
    MAP_SIZE,
    VIEW_H,
    VIEW_W,
    NUM_BLOCK_TYPES,
    NUM_MOB_CHANNELS,
    TILE_CHANNELS,
    OBS_VIEW_SIZE,
    OBS_DIM,
    BLOCK_INVALID,
    BLOCK_OUT_OF_BOUNDS,
    NUM_DIRECTIONS,
    NUM_INTRINSICS,
    NUM_INVENTORY,
    NUM_ACHIEVEMENTS,
    MAX_ZOMBIES,
    MAX_COWS,
    MAX_SKELETONS,
    MAX_ARROWS,
    MAX_PLANTS,
    ARROW_FIELDS,
    ARROW_FDIR,
    MOB_DESPAWN_DISTANCE,
    INTRINSIC_MAX,
    DAY_LENGTH,
    MAX_TIMESTEPS,
    # Block IDs
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
    BLOCK_LAVA,
    BLOCK_PLANT,
    BLOCK_RIPE_PLANT,
    # Directions
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP,
    DIR_DOWN,
    # Actions
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
    # Inventory slot indices
    INV_WOOD,
    INV_STONE,
    INV_COAL,
    INV_IRON,
    INV_DIAMOND,
    INV_SAPLING,
    INV_WOOD_PICKAXE,
    INV_STONE_PICKAXE,
    INV_IRON_PICKAXE,
    INV_WOOD_SWORD,
    INV_STONE_SWORD,
    INV_IRON_SWORD,
    INV_MAX_PER_SLOT,
    # Intrinsic indices
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
    INTRINSIC_F_RECOVER,
    INTRINSIC_F_HUNGER,
    INTRINSIC_F_THIRST,
    INTRINSIC_F_FATIGUE,
    # Achievement bits
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
    # Mob field layout
    MOB_FIELDS,
    MOB_FY,
    MOB_FX,
    MOB_HP,
    MOB_CD,
    # Plant field layout
    PLANT_FIELDS,
    PLANT_FY,
    PLANT_FX,
    PLANT_FAGE,
    PLANT_RIPEN_AGE,
    # Damage tables
    DAMAGE_FIST,
    DAMAGE_WOOD_SWORD,
    DAMAGE_STONE_SWORD,
    DAMAGE_IRON_SWORD,
    # Mob health + attack params
    ZOMBIE_HEALTH,
    COW_HEALTH,
    SKELETON_HEALTH,
    ZOMBIE_ATTACK_DAMAGE,
    ZOMBIE_ATTACK_DAMAGE_SLEEP,
    ZOMBIE_ATTACK_COOLDOWN,
    SKELETON_ATTACK_COOLDOWN,
    ARROW_DAMAGE,
    # Mob AI tuning
    ZOMBIE_CHASE_PROB,
    ZOMBIE_CHASE_RANGE,
    SKELETON_RANDOM_OVERRIDE,
    SKELETON_FLEE_RANGE,
    SKELETON_RANGE_MIN,
    SKELETON_FIRE_MIN,
    SKELETON_FIRE_MAX,
    # Spawn params
    SPAWN_COW_CHANCE,
    SPAWN_ZOMBIE_BASE_CHANCE,
    SPAWN_ZOMBIE_NIGHT_CHANCE,
    SPAWN_SKELETON_CHANCE,
    # Intrinsic thresholds
    HUNGER_THRESHOLD,
    THIRST_THRESHOLD,
    FATIGUE_HIGH_THRESHOLD,
    FATIGUE_LOW_THRESHOLD,
    RECOVER_HIGH_THRESHOLD,
    RECOVER_LOW_THRESHOLD,
    # Eat / drink
    COW_EAT_BOOST,
    PLANT_EAT_BOOST,
    WATER_DRINK_BOOST,
    SAPLING_DROP_CHANCE,
)
from .state import (
    S_MAP_BASE,
    S_PLAYER_POS,
    S_PLAYER_DIR,
    S_INTRINSICS_BASE,
    S_INTRINSICS_F_BASE,
    S_INV_BASE,
    S_ZOMBIES_BASE,
    S_COWS_BASE,
    S_SKELETONS_BASE,
    S_ARROWS_BASE,
    S_PLANTS_BASE,
    S_PLANT_MASK_BASE,
    S_ACHIEVEMENTS_BASE,
    S_LIGHT_LEVEL,
    S_IS_SLEEPING,
    S_TIMESTEP,
    STATE_SIZE,
)


# ============================================================================
# Type alias
# ============================================================================

comptime State = UnsafePointer[Float32, MutAnyOrigin]


# ============================================================================
# Direction table — DIRECTIONS[action] = (dy, dx)
# ============================================================================

@always_inline
def dir_offset(d: Int) -> Tuple[Int, Int]:
    """Map a DIR_* code (0..3) to (dy, dx). Returns (0, 0) otherwise.

    Note: direction codes (0..3) and action codes (1..4) overlap
    numerically — pass a direction code only, never a raw action.
    """
    if d == DIR_LEFT:
        return (0, -1)
    if d == DIR_RIGHT:
        return (0, 1)
    if d == DIR_UP:
        return (-1, 0)
    if d == DIR_DOWN:
        return (1, 0)
    return (0, 0)


# ============================================================================
# Bounds / block predicates
# ============================================================================

@always_inline
def in_bounds(y: Int, x: Int) -> Bool:
    return 0 <= y and y < MAP_H and 0 <= x and x < MAP_W


@always_inline
def is_solid(block: Int) -> Bool:
    """Reference SOLID_BLOCKS: WATER, STONE, TREE, COAL, IRON, DIAMOND,
    CRAFTING_TABLE, FURNACE, PLANT, RIPE_PLANT.
    """
    return (
        block == BLOCK_WATER
        or block == BLOCK_STONE
        or block == BLOCK_TREE
        or block == BLOCK_COAL
        or block == BLOCK_IRON
        or block == BLOCK_DIAMOND
        or block == BLOCK_CRAFTING_TABLE
        or block == BLOCK_FURNACE
        or block == BLOCK_PLANT
        or block == BLOCK_RIPE_PLANT
    )


# ============================================================================
# State accessors
# ============================================================================

@always_inline
def get_map(s: State, y: Int, x: Int) -> Int:
    return Int(s[S_MAP_BASE + y * MAP_W + x])


@always_inline
def set_map(s: State, y: Int, x: Int, block: Int):
    s[S_MAP_BASE + y * MAP_W + x] = Float32(block)


@always_inline
def get_inv(s: State, slot: Int) -> Int:
    return Int(s[S_INV_BASE + slot])


@always_inline
def add_inv(s: State, slot: Int, delta: Int):
    var v = Int(s[S_INV_BASE + slot]) + delta
    if v < 0:
        v = 0
    if v > INV_MAX_PER_SLOT:
        v = INV_MAX_PER_SLOT
    s[S_INV_BASE + slot] = Float32(v)


@always_inline
def get_intr(s: State, slot: Int) -> Int:
    return Int(s[S_INTRINSICS_BASE + slot])


@always_inline
def set_intr(s: State, slot: Int, v: Int):
    var v2 = v
    if v2 < 0:
        v2 = 0
    if v2 > INTRINSIC_MAX:
        v2 = INTRINSIC_MAX
    s[S_INTRINSICS_BASE + slot] = Float32(v2)


@always_inline
def add_intr(s: State, slot: Int, delta: Int):
    set_intr(s, slot, get_intr(s, slot) + delta)


@always_inline
def get_intr_f(s: State, slot: Int) -> Float32:
    return s[S_INTRINSICS_F_BASE + slot]


@always_inline
def set_intr_f(s: State, slot: Int, v: Float32):
    s[S_INTRINSICS_F_BASE + slot] = v


@always_inline
def player_pos(s: State) -> Tuple[Int, Int]:
    return (Int(s[S_PLAYER_POS]), Int(s[S_PLAYER_POS + 1]))


@always_inline
def player_dir(s: State) -> Int:
    return Int(s[S_PLAYER_DIR])


@always_inline
def is_sleeping(s: State) -> Bool:
    return s[S_IS_SLEEPING] > Float32(0.5)


@always_inline
def set_sleeping(s: State, v: Bool):
    s[S_IS_SLEEPING] = Float32(1.0) if v else Float32(0.0)


@always_inline
def set_achievement(s: State, idx: Int):
    s[S_ACHIEVEMENTS_BASE + idx] = Float32(1.0)


@always_inline
def get_ach(s: State, idx: Int) -> Bool:
    return s[S_ACHIEVEMENTS_BASE + idx] > Float32(0.5)


@always_inline
def sum_achievements(s: State) -> Int:
    var n = 0
    for i in range(NUM_ACHIEVEMENTS):
        if get_ach(s, i):
            n += 1
    return n


# ----------------------------------------------------------------------------
# Mob field accessors (kind = base offset, n = MAX of that kind).
# ----------------------------------------------------------------------------

@always_inline
def mob_hp(s: State, base: Int, i: Int) -> Int:
    return Int(s[base + i * MOB_FIELDS + MOB_HP])


@always_inline
def mob_set_hp(s: State, base: Int, i: Int, hp: Int):
    s[base + i * MOB_FIELDS + MOB_HP] = Float32(hp)


@always_inline
def mob_pos(s: State, base: Int, i: Int) -> Tuple[Int, Int]:
    return (
        Int(s[base + i * MOB_FIELDS + MOB_FY]),
        Int(s[base + i * MOB_FIELDS + MOB_FX]),
    )


@always_inline
def mob_set_pos(s: State, base: Int, i: Int, y: Int, x: Int):
    s[base + i * MOB_FIELDS + MOB_FY] = Float32(y)
    s[base + i * MOB_FIELDS + MOB_FX] = Float32(x)


@always_inline
def mob_cd(s: State, base: Int, i: Int) -> Int:
    return Int(s[base + i * MOB_FIELDS + MOB_CD])


@always_inline
def mob_set_cd(s: State, base: Int, i: Int, cd: Int):
    s[base + i * MOB_FIELDS + MOB_CD] = Float32(cd)


# Returns (kind_base, slot_idx) of the first alive mob at (y, x), or (-1, -1)
# if none. Kinds in scan order: zombies, cows, skeletons.
@always_inline
def find_mob_at(s: State, y: Int, x: Int) -> Tuple[Int, Int]:
    for i in range(MAX_ZOMBIES):
        if mob_hp(s, S_ZOMBIES_BASE, i) > 0:
            var p = mob_pos(s, S_ZOMBIES_BASE, i)
            if p[0] == y and p[1] == x:
                return (S_ZOMBIES_BASE, i)
    for i in range(MAX_COWS):
        if mob_hp(s, S_COWS_BASE, i) > 0:
            var p = mob_pos(s, S_COWS_BASE, i)
            if p[0] == y and p[1] == x:
                return (S_COWS_BASE, i)
    for i in range(MAX_SKELETONS):
        if mob_hp(s, S_SKELETONS_BASE, i) > 0:
            var p = mob_pos(s, S_SKELETONS_BASE, i)
            if p[0] == y and p[1] == x:
                return (S_SKELETONS_BASE, i)
    return (-1, -1)


@always_inline
def is_in_mob(s: State, y: Int, x: Int) -> Bool:
    var hit = find_mob_at(s, y, x)
    return hit[0] >= 0


# ----------------------------------------------------------------------------
# Plant accessors
# ----------------------------------------------------------------------------

@always_inline
def plant_mask(s: State, i: Int) -> Bool:
    return s[S_PLANT_MASK_BASE + i] > Float32(0.5)


@always_inline
def plant_set_mask(s: State, i: Int, v: Bool):
    s[S_PLANT_MASK_BASE + i] = Float32(1.0) if v else Float32(0.0)


@always_inline
def plant_pos(s: State, i: Int) -> Tuple[Int, Int]:
    return (
        Int(s[S_PLANTS_BASE + i * PLANT_FIELDS + PLANT_FY]),
        Int(s[S_PLANTS_BASE + i * PLANT_FIELDS + PLANT_FX]),
    )


@always_inline
def plant_set_pos(s: State, i: Int, y: Int, x: Int):
    s[S_PLANTS_BASE + i * PLANT_FIELDS + PLANT_FY] = Float32(y)
    s[S_PLANTS_BASE + i * PLANT_FIELDS + PLANT_FX] = Float32(x)


@always_inline
def plant_age(s: State, i: Int) -> Int:
    return Int(s[S_PLANTS_BASE + i * PLANT_FIELDS + PLANT_FAGE])


@always_inline
def plant_set_age(s: State, i: Int, a: Int):
    s[S_PLANTS_BASE + i * PLANT_FIELDS + PLANT_FAGE] = Float32(a)


# ============================================================================
# _cap_inventory
# ============================================================================

@always_inline
def cap_inventory(s: State):
    for i in range(NUM_INVENTORY):
        var v = Int(s[S_INV_BASE + i])
        if v < 0:
            v = 0
        elif v > INV_MAX_PER_SLOT:
            v = INV_MAX_PER_SLOT
        s[S_INV_BASE + i] = Float32(v)


# ============================================================================
# _move_player
# ============================================================================

@always_inline
def move_player(s: State, action: Int):
    """Reference move_player (lines 1374-1397).

    For cardinal actions, attempt to step in that direction. The move is
    valid iff the target tile is in bounds AND (not solid) AND no mob.
    Lava is walkable (death happens via intrinsics). Direction always
    updates if the action is cardinal (even when blocked).
    """
    if action < ACTION_LEFT or action > ACTION_DOWN:
        return
    var d = action - ACTION_LEFT  # action 1..4 → direction 0..3
    var off = dir_offset(d)
    var pp = player_pos(s)
    var ny = pp[0] + off[0]
    var nx = pp[1] + off[1]

    # Direction updates regardless of whether move succeeds.
    s[S_PLAYER_DIR] = Float32(d)

    if not in_bounds(ny, nx):
        return
    var blk = get_map(s, ny, nx)
    if is_solid(blk):
        return
    if is_in_mob(s, ny, nx):
        return

    s[S_PLAYER_POS] = Float32(ny)
    s[S_PLAYER_POS + 1] = Float32(nx)


# ============================================================================
# _place_block
# ============================================================================

@always_inline
def add_growing_plant(s: State, y: Int, x: Int) -> Bool:
    """Find first empty plant slot and register a plant at (y, x). Returns
    True on success."""
    for i in range(MAX_PLANTS):
        if not plant_mask(s, i):
            plant_set_pos(s, i, y, x)
            plant_set_age(s, i, 0)
            plant_set_mask(s, i, True)
            return True
    return False


@always_inline
def place_block(s: State, action: Int):
    """Reference place_block (lines 586-745). The four placement actions
    drop a block in front of the player.

    Guards:
      - Target tile must be in bounds and have no mob.
      - PLACE_STONE allowed on water (for bridge building) or on any
        non-solid block; the others require non-solid.
      - PLACE_PLANT requires target == GRASS and registers in the plant
        array (if there's an empty slot — reference behavior).
    """
    if action < ACTION_PLACE_STONE or action > ACTION_PLACE_PLANT:
        return
    var pp = player_pos(s)
    var d = player_dir(s)
    var off = dir_offset(d)
    var ty = pp[0] + off[0]
    var tx = pp[1] + off[1]
    if not in_bounds(ty, tx) or is_in_mob(s, ty, tx):
        return
    var target = get_map(s, ty, tx)

    if action == ACTION_PLACE_TABLE:
        if get_inv(s, INV_WOOD) >= 1 and not is_solid(target):
            add_inv(s, INV_WOOD, -1)
            set_map(s, ty, tx, BLOCK_CRAFTING_TABLE)
            set_achievement(s, ACH_PLACE_TABLE)
    elif action == ACTION_PLACE_FURNACE:
        if get_inv(s, INV_STONE) >= 1 and not is_solid(target):
            add_inv(s, INV_STONE, -1)
            set_map(s, ty, tx, BLOCK_FURNACE)
            set_achievement(s, ACH_PLACE_FURNACE)
    elif action == ACTION_PLACE_STONE:
        # Stone allowed on water (bridge), or any non-solid tile.
        var ok = target == BLOCK_WATER or not is_solid(target)
        if get_inv(s, INV_STONE) >= 1 and ok:
            add_inv(s, INV_STONE, -1)
            set_map(s, ty, tx, BLOCK_STONE)
            set_achievement(s, ACH_PLACE_STONE)
    elif action == ACTION_PLACE_PLANT:
        if get_inv(s, INV_SAPLING) >= 1 and target == BLOCK_GRASS:
            add_inv(s, INV_SAPLING, -1)
            set_map(s, ty, tx, BLOCK_PLANT)
            _ = add_growing_plant(s, ty, tx)
            set_achievement(s, ACH_PLACE_PLANT)


# ============================================================================
# _do_crafting
# ============================================================================

@always_inline
def is_near(s: State, block_type: Int) -> Bool:
    """True iff any of the 8 cells around the player contains `block_type`.
    Mirrors reference is_near_block."""
    var pp = player_pos(s)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            var y = pp[0] + dy
            var x = pp[1] + dx
            if in_bounds(y, x) and get_map(s, y, x) == block_type:
                return True
    return False


@always_inline
def do_crafting(s: State, action: Int):
    """Reference do_crafting (lines 397-546). Recipes are checked in order
    so later ones see updated inventory if multiple actions were
    triggered — but only one action is given per step, so effectively
    one recipe at most."""
    if action < ACTION_MAKE_WOOD_PICKAXE or action > ACTION_MAKE_IRON_SWORD:
        return
    var at_table = is_near(s, BLOCK_CRAFTING_TABLE)
    var at_furnace = is_near(s, BLOCK_FURNACE)

    if action == ACTION_MAKE_WOOD_PICKAXE:
        if at_table and get_inv(s, INV_WOOD) >= 1:
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_WOOD_PICKAXE, 1)
            set_achievement(s, ACH_MAKE_WOOD_PICKAXE)
    elif action == ACTION_MAKE_STONE_PICKAXE:
        if (
            at_table
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_STONE) >= 1
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_STONE, -1)
            add_inv(s, INV_STONE_PICKAXE, 1)
            set_achievement(s, ACH_MAKE_STONE_PICKAXE)
    elif action == ACTION_MAKE_IRON_PICKAXE:
        if (
            at_table
            and at_furnace
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_STONE) >= 1
            and get_inv(s, INV_IRON) >= 1
            and get_inv(s, INV_COAL) >= 1
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_STONE, -1)
            add_inv(s, INV_IRON, -1)
            add_inv(s, INV_COAL, -1)
            add_inv(s, INV_IRON_PICKAXE, 1)
            set_achievement(s, ACH_MAKE_IRON_PICKAXE)
    elif action == ACTION_MAKE_WOOD_SWORD:
        if at_table and get_inv(s, INV_WOOD) >= 1:
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_WOOD_SWORD, 1)
            set_achievement(s, ACH_MAKE_WOOD_SWORD)
    elif action == ACTION_MAKE_STONE_SWORD:
        if (
            at_table
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_STONE) >= 1
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_STONE, -1)
            add_inv(s, INV_STONE_SWORD, 1)
            set_achievement(s, ACH_MAKE_STONE_SWORD)
    elif action == ACTION_MAKE_IRON_SWORD:
        if (
            at_table
            and at_furnace
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_STONE) >= 1
            and get_inv(s, INV_IRON) >= 1
            and get_inv(s, INV_COAL) >= 1
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_STONE, -1)
            add_inv(s, INV_IRON, -1)
            add_inv(s, INV_COAL, -1)
            add_inv(s, INV_IRON_SWORD, 1)
            set_achievement(s, ACH_MAKE_IRON_SWORD)


# ============================================================================
# _do_action — mine, attack, eat, drink, sleep
# ============================================================================

@always_inline
def best_sword_damage(s: State) -> Int:
    if get_inv(s, INV_IRON_SWORD) > 0:
        return DAMAGE_IRON_SWORD
    if get_inv(s, INV_STONE_SWORD) > 0:
        return DAMAGE_STONE_SWORD
    if get_inv(s, INV_WOOD_SWORD) > 0:
        return DAMAGE_WOOD_SWORD
    return DAMAGE_FIST


@always_inline
def update_plant_age_on_eat(s: State, y: Int, x: Int):
    for i in range(MAX_PLANTS):
        if plant_mask(s, i):
            var p = plant_pos(s, i)
            if p[0] == y and p[1] == x:
                plant_set_age(s, i, 0)
                return


@always_inline
def do_action(s: State, action: Int, mut rng: PhiloxRandom):
    """Reference do_action (lines 69-382). Either attack a mob OR mine /
    eat / drink the tile in front of the player. Attack takes priority:
    if the front tile holds a mob, we hit it and do not also mine.
    """
    if action != ACTION_DO:
        return
    var pp = player_pos(s)
    var d = player_dir(s)
    var off = dir_offset(d)
    var ty = pp[0] + off[0]
    var tx = pp[1] + off[1]
    if not in_bounds(ty, tx):
        return

    # ------------------------------------------------------------------
    # Attack any mob standing on the target tile.
    # ------------------------------------------------------------------
    var hit = find_mob_at(s, ty, tx)
    if hit[0] >= 0:
        var base = hit[0]
        var idx = hit[1]
        var dmg = best_sword_damage(s)
        var new_hp = mob_hp(s, base, idx) - dmg
        if new_hp < 0:
            new_hp = 0
        mob_set_hp(s, base, idx, new_hp)
        if new_hp == 0:
            if base == S_ZOMBIES_BASE:
                set_achievement(s, ACH_DEFEAT_ZOMBIE)
            elif base == S_COWS_BASE:
                set_achievement(s, ACH_EAT_COW)
                add_intr(s, INTRINSIC_FOOD, COW_EAT_BOOST)
                set_intr_f(s, INTRINSIC_F_HUNGER, Float32(0.0))
            elif base == S_SKELETONS_BASE:
                set_achievement(s, ACH_DEFEAT_SKELETON)
        return

    # ------------------------------------------------------------------
    # Mine / interact with the tile in front of the player.
    # ------------------------------------------------------------------
    var blk = get_map(s, ty, tx)

    if blk == BLOCK_TREE:
        set_map(s, ty, tx, BLOCK_GRASS)
        add_inv(s, INV_WOOD, 1)
        set_achievement(s, ACH_COLLECT_WOOD)
    elif blk == BLOCK_STONE:
        if get_inv(s, INV_WOOD_PICKAXE) > 0:
            set_map(s, ty, tx, BLOCK_PATH)
            add_inv(s, INV_STONE, 1)
            set_achievement(s, ACH_COLLECT_STONE)
    elif blk == BLOCK_COAL:
        if get_inv(s, INV_WOOD_PICKAXE) > 0:
            set_map(s, ty, tx, BLOCK_PATH)
            add_inv(s, INV_COAL, 1)
            set_achievement(s, ACH_COLLECT_COAL)
    elif blk == BLOCK_IRON:
        if get_inv(s, INV_STONE_PICKAXE) > 0:
            set_map(s, ty, tx, BLOCK_PATH)
            add_inv(s, INV_IRON, 1)
            set_achievement(s, ACH_COLLECT_IRON)
    elif blk == BLOCK_DIAMOND:
        if get_inv(s, INV_IRON_PICKAXE) > 0:
            set_map(s, ty, tx, BLOCK_PATH)
            add_inv(s, INV_DIAMOND, 1)
            set_achievement(s, ACH_COLLECT_DIAMOND)
    elif blk == BLOCK_GRASS:
        # 10% chance of dropping a sapling when "mining" grass.
        var u = rng.step_uniform()
        if Float32(u[0]) < SAPLING_DROP_CHANCE:
            add_inv(s, INV_SAPLING, 1)
            set_achievement(s, ACH_COLLECT_SAPLING)
    elif blk == BLOCK_WATER:
        add_intr(s, INTRINSIC_DRINK, WATER_DRINK_BOOST)
        set_intr_f(s, INTRINSIC_F_THIRST, Float32(0.0))
        set_achievement(s, ACH_COLLECT_DRINK)
    elif blk == BLOCK_RIPE_PLANT:
        set_map(s, ty, tx, BLOCK_PLANT)
        add_intr(s, INTRINSIC_FOOD, PLANT_EAT_BOOST)
        set_intr_f(s, INTRINSIC_F_HUNGER, Float32(0.0))
        set_achievement(s, ACH_EAT_PLANT)
        update_plant_age_on_eat(s, ty, tx)


# ============================================================================
# _update_plants
# ============================================================================

@always_inline
def update_plants(s: State):
    """Reference update_plants (lines 1335-1371). Age each alive plant; at
    PLANT_RIPEN_AGE, the map tile becomes RIPE_PLANT. The map is also
    refreshed to PLANT every step (in case it was overwritten), matching
    reference behavior.
    """
    for i in range(MAX_PLANTS):
        if not plant_mask(s, i):
            continue
        var age = plant_age(s, i) + 1
        plant_set_age(s, i, age)
        var p = plant_pos(s, i)
        if age >= PLANT_RIPEN_AGE:
            set_map(s, p[0], p[1], BLOCK_RIPE_PLANT)
        else:
            set_map(s, p[0], p[1], BLOCK_PLANT)


# ============================================================================
# _update_intrinsics
# ============================================================================

@always_inline
def update_intrinsics(s: State, action: Int):
    """Reference update_player_intrinsics (lines 1237-1332).

    Order matters: sleep transition → hunger → thirst → fatigue →
    recovery → health. WAKE_UP is set when sleep ends naturally.
    """
    # ---- sleep start ----
    var energy = get_intr(s, INTRINSIC_ENERGY)
    var sleeping = is_sleeping(s)
    if action == ACTION_SLEEP and energy < INTRINSIC_MAX:
        sleeping = True
    set_sleeping(s, sleeping)

    # ---- sleep end ----
    if sleeping and energy >= INTRINSIC_MAX:
        set_sleeping(s, False)
        set_achievement(s, ACH_WAKE_UP)
        sleeping = False

    # ---- hunger ----
    var hunger_add = Float32(0.5) if sleeping else Float32(1.0)
    var hunger = get_intr_f(s, INTRINSIC_F_HUNGER) + hunger_add
    if hunger > HUNGER_THRESHOLD:
        add_intr(s, INTRINSIC_FOOD, -1)
        hunger = Float32(0.0)
    set_intr_f(s, INTRINSIC_F_HUNGER, hunger)

    # ---- thirst ----
    var thirst_add = Float32(0.5) if sleeping else Float32(1.0)
    var thirst = get_intr_f(s, INTRINSIC_F_THIRST) + thirst_add
    if thirst > THIRST_THRESHOLD:
        add_intr(s, INTRINSIC_DRINK, -1)
        thirst = Float32(0.0)
    set_intr_f(s, INTRINSIC_F_THIRST, thirst)

    # ---- fatigue ----
    var fatigue = get_intr_f(s, INTRINSIC_F_FATIGUE)
    if sleeping:
        fatigue -= Float32(1.0)
    else:
        fatigue += Float32(1.0)
    if fatigue > FATIGUE_HIGH_THRESHOLD:
        add_intr(s, INTRINSIC_ENERGY, -1)
        fatigue = Float32(0.0)
    elif fatigue < FATIGUE_LOW_THRESHOLD:
        add_intr(s, INTRINSIC_ENERGY, 1)
        fatigue = Float32(0.0)
    set_intr_f(s, INTRINSIC_F_FATIGUE, fatigue)

    # ---- recovery / health ----
    var food = get_intr(s, INTRINSIC_FOOD)
    var drink = get_intr(s, INTRINSIC_DRINK)
    energy = get_intr(s, INTRINSIC_ENERGY)
    var necessities_ok = food > 0 and drink > 0 and (energy > 0 or sleeping)
    var recover_add: Float32
    if necessities_ok:
        recover_add = Float32(2.0) if sleeping else Float32(1.0)
    else:
        recover_add = Float32(-0.5) if sleeping else Float32(-1.0)
    var recover = get_intr_f(s, INTRINSIC_F_RECOVER) + recover_add
    if recover > RECOVER_HIGH_THRESHOLD:
        add_intr(s, INTRINSIC_HEALTH, 1)
        recover = Float32(0.0)
    elif recover < RECOVER_LOW_THRESHOLD:
        add_intr(s, INTRINSIC_HEALTH, -1)
        recover = Float32(0.0)
    set_intr_f(s, INTRINSIC_F_RECOVER, recover)


# ============================================================================
# Light level (matches calculate_light_level in reference game_logic)
# ============================================================================

@always_inline
def compute_light_level(timestep: Int) -> Float32:
    var t_in_day = timestep - (timestep // DAY_LENGTH) * DAY_LENGTH
    var progress = Float32(t_in_day) / Float32(DAY_LENGTH) + Float32(0.3)
    var c = math_cos(Float32(3.14159265) * progress)
    var abs_c = c if c >= Float32(0.0) else -c
    return Float32(1.0) - abs_c * abs_c * abs_c


# ============================================================================
# Mob AI helpers (Phase 3B)
# ============================================================================

@always_inline
def manhattan(ay: Int, ax: Int, by: Int, bx: Int) -> Int:
    var dy = ay - by
    var dx = ax - bx
    if dy < 0:
        dy = -dy
    if dx < 0:
        dx = -dx
    return dy + dx


@always_inline
def random_cardinal_dir(mut rng: PhiloxRandom) -> Int:
    """Uniform pick from {DIR_LEFT, DIR_RIGHT, DIR_UP, DIR_DOWN}."""
    var u = rng.step_uniform()
    var r = Int(Float32(u[0]) * Float32(4.0))
    if r >= 4:
        r = 3
    return r


@always_inline
def random_8way_offset(mut rng: PhiloxRandom) -> Tuple[Int, Int]:
    """Reference cows use DIRECTIONS[1:9], which is 4 cardinals + 4 zero
    padding entries, giving a 50% chance the mob stays in place."""
    var u = rng.step_uniform()
    var r = Int(Float32(u[0]) * Float32(8.0))
    if r >= 8:
        r = 7
    if r == 0:
        return (0, -1)
    if r == 1:
        return (0, 1)
    if r == 2:
        return (-1, 0)
    if r == 3:
        return (1, 0)
    return (0, 0)


@always_inline
def chase_direction(
    my_y: Int, my_x: Int, py: Int, px: Int, mut rng: PhiloxRandom
) -> Int:
    """Reference: pick the axis with largest |Δ| (random tiebreak), then
    move ±1 along that axis toward the player. Returns a DIR_* code."""
    var dy = py - my_y
    var dx = px - my_x
    var ady = -dy if dy < 0 else dy
    var adx = -dx if dx < 0 else dx
    var pick_y: Bool
    if ady > adx:
        pick_y = True
    elif adx > ady:
        pick_y = False
    else:
        var u = rng.step_uniform()
        pick_y = Float32(u[0]) < Float32(0.5)
    if pick_y:
        return DIR_DOWN if dy > 0 else DIR_UP
    else:
        return DIR_RIGHT if dx > 0 else DIR_LEFT


@always_inline
def is_mob_walkable(s: State, y: Int, x: Int) -> Bool:
    """Match reference is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava.
    The "not_in_mob" check here uses find_mob_at, which scans alive slots;
    a mob moving to (y, x) won't show up if it currently sits at a
    different tile, so self-exclusion is automatic."""
    if not in_bounds(y, x):
        return False
    var b = get_map(s, y, x)
    if is_solid(b) or b == BLOCK_LAVA:
        return False
    if is_in_mob(s, y, x):
        return False
    return True


# ============================================================================
# _update_mobs — zombies, cows, skeletons, arrows
# ============================================================================

@always_inline
def update_zombies(s: State, mut rng: PhiloxRandom):
    var pp = player_pos(s)
    var sleep = is_sleeping(s)
    for i in range(MAX_ZOMBIES):
        if mob_hp(s, S_ZOMBIES_BASE, i) <= 0:
            continue
        var mp = mob_pos(s, S_ZOMBIES_BASE, i)
        var dist = manhattan(mp[0], mp[1], pp[0], pp[1])

        var rd = random_cardinal_dir(rng)
        var rd_off = dir_offset(rd)
        var rand_y = mp[0] + rd_off[0]
        var rand_x = mp[1] + rd_off[1]

        var cd = chase_direction(mp[0], mp[1], pp[0], pp[1], rng)
        var cd_off = dir_offset(cd)
        var chase_y = mp[0] + cd_off[0]
        var chase_x = mp[1] + cd_off[1]

        var u = rng.step_uniform()
        var use_chase = dist < ZOMBIE_CHASE_RANGE and Float32(
            u[0]
        ) < ZOMBIE_CHASE_PROB
        var prop_y = chase_y if use_chase else rand_y
        var prop_x = chase_x if use_chase else rand_x

        var cur_cd = mob_cd(s, S_ZOMBIES_BASE, i)
        var attacking = dist == 1 and cur_cd <= 0
        if attacking:
            prop_y = mp[0]
            prop_x = mp[1]
            var dmg = ZOMBIE_ATTACK_DAMAGE_SLEEP if sleep else (
                ZOMBIE_ATTACK_DAMAGE
            )
            add_intr(s, INTRINSIC_HEALTH, -dmg)
            if sleep:
                set_sleeping(s, False)
                set_achievement(s, ACH_WAKE_UP)
                sleep = False
            mob_set_cd(s, S_ZOMBIES_BASE, i, ZOMBIE_ATTACK_COOLDOWN)
        else:
            mob_set_cd(s, S_ZOMBIES_BASE, i, cur_cd - 1)

        # Despawn first (so we don't write a stale position).
        if dist >= MOB_DESPAWN_DISTANCE:
            mob_set_hp(s, S_ZOMBIES_BASE, i, 0)
            continue

        if is_mob_walkable(s, prop_y, prop_x):
            mob_set_pos(s, S_ZOMBIES_BASE, i, prop_y, prop_x)


@always_inline
def update_cows(s: State, mut rng: PhiloxRandom):
    var pp = player_pos(s)
    for i in range(MAX_COWS):
        if mob_hp(s, S_COWS_BASE, i) <= 0:
            continue
        var mp = mob_pos(s, S_COWS_BASE, i)
        var dist = manhattan(mp[0], mp[1], pp[0], pp[1])

        if dist >= MOB_DESPAWN_DISTANCE:
            mob_set_hp(s, S_COWS_BASE, i, 0)
            continue

        var off = random_8way_offset(rng)
        var prop_y = mp[0] + off[0]
        var prop_x = mp[1] + off[1]
        if is_mob_walkable(s, prop_y, prop_x):
            mob_set_pos(s, S_COWS_BASE, i, prop_y, prop_x)


@always_inline
def try_spawn_arrow(s: State, sy: Int, sx: Int, fire_dir: Int) -> Bool:
    """Find the first dead arrow slot and spawn an arrow heading `fire_dir`
    (a DIR_* code). Returns True on success."""
    for i in range(MAX_ARROWS):
        var base = S_ARROWS_BASE + i * ARROW_FIELDS
        if Int(s[base + MOB_HP]) <= 0:
            s[base + MOB_FY] = Float32(sy)
            s[base + MOB_FX] = Float32(sx)
            s[base + MOB_HP] = Float32(1)
            s[base + MOB_CD] = Float32(0)
            s[base + ARROW_FDIR] = Float32(fire_dir)
            return True
    return False


@always_inline
def update_skeletons(s: State, mut rng: PhiloxRandom):
    var pp = player_pos(s)
    for i in range(MAX_SKELETONS):
        if mob_hp(s, S_SKELETONS_BASE, i) <= 0:
            continue
        var mp = mob_pos(s, S_SKELETONS_BASE, i)
        var dist = manhattan(mp[0], mp[1], pp[0], pp[1])

        var rd = random_cardinal_dir(rng)
        var rd_off = dir_offset(rd)
        var rand_y = mp[0] + rd_off[0]
        var rand_x = mp[1] + rd_off[1]

        var cd = chase_direction(mp[0], mp[1], pp[0], pp[1], rng)
        var cd_off = dir_offset(cd)

        # Movement: chase if far, flee if too close, else random.
        var prop_y: Int
        var prop_x: Int
        var far = dist >= SKELETON_RANGE_MIN
        var too_close = dist <= SKELETON_FLEE_RANGE
        if too_close:
            prop_y = mp[0] - cd_off[0]
            prop_x = mp[1] - cd_off[1]
        elif far:
            prop_y = mp[0] + cd_off[0]
            prop_x = mp[1] + cd_off[1]
        else:
            prop_y = rand_y
            prop_x = rand_x

        # 15% chance to override with random (matches reference uniform > 0.85).
        var u = rng.step_uniform()
        if Float32(u[0]) > SKELETON_RANDOM_OVERRIDE:
            prop_y = rand_y
            prop_x = rand_x

        # Attack: in the firing band, OR fleeing-but-blocked.
        var cur_cd = mob_cd(s, S_SKELETONS_BASE, i)
        var in_fire_band = (
            dist >= SKELETON_FIRE_MIN and dist <= SKELETON_FIRE_MAX
        )
        var blocked_while_fleeing = too_close and not is_mob_walkable(
            s, prop_y, prop_x
        )
        var want_attack = (in_fire_band or blocked_while_fleeing) and cur_cd <= 0

        if want_attack:
            _ = try_spawn_arrow(s, mp[0], mp[1], cd)
            prop_y = mp[0]
            prop_x = mp[1]
            mob_set_cd(
                s, S_SKELETONS_BASE, i, SKELETON_ATTACK_COOLDOWN
            )
        else:
            mob_set_cd(s, S_SKELETONS_BASE, i, cur_cd - 1)

        if dist >= MOB_DESPAWN_DISTANCE:
            mob_set_hp(s, S_SKELETONS_BASE, i, 0)
            continue

        if is_mob_walkable(s, prop_y, prop_x):
            mob_set_pos(s, S_SKELETONS_BASE, i, prop_y, prop_x)


@always_inline
def update_arrows(s: State):
    """Arrows move 1 tile per step in their stored direction. Hitting the
    player damages them (and wakes if sleeping). Hitting a wall/mob
    destroys the arrow; hitting a crafting table or furnace also
    destroys those blocks (→ PATH). Water is passable."""
    var pp = player_pos(s)
    for i in range(MAX_ARROWS):
        var base = S_ARROWS_BASE + i * ARROW_FIELDS
        if Int(s[base + MOB_HP]) <= 0:
            continue

        var ay = Int(s[base + MOB_FY])
        var ax = Int(s[base + MOB_FX])
        var d = Int(s[base + ARROW_FDIR])
        var off = dir_offset(d)
        var ny = ay + off[0]
        var nx = ax + off[1]

        if not in_bounds(ny, nx):
            s[base + MOB_HP] = Float32(0)
            continue

        # Hit player.
        if ny == pp[0] and nx == pp[1]:
            add_intr(s, INTRINSIC_HEALTH, -ARROW_DAMAGE)
            if is_sleeping(s):
                set_sleeping(s, False)
                set_achievement(s, ACH_WAKE_UP)
            s[base + MOB_HP] = Float32(0)
            continue

        var blk = get_map(s, ny, nx)
        # Arrows pass over water; everything else solid stops them.
        if is_solid(blk) and blk != BLOCK_WATER:
            if blk == BLOCK_FURNACE or blk == BLOCK_CRAFTING_TABLE:
                set_map(s, ny, nx, BLOCK_PATH)
            s[base + MOB_HP] = Float32(0)
            continue

        if is_in_mob(s, ny, nx):
            s[base + MOB_HP] = Float32(0)
            continue

        s[base + MOB_FY] = Float32(ny)
        s[base + MOB_FX] = Float32(nx)


@always_inline
def update_mobs_all(s: State, mut rng: PhiloxRandom):
    update_zombies(s, rng)
    update_cows(s, rng)
    update_skeletons(s, rng)
    update_arrows(s)


# ============================================================================
# _spawn_mobs — probabilistic, biome / distance / day-night gated
# ============================================================================

@always_inline
def count_alive(s: State, base: Int, n: Int) -> Int:
    var c = 0
    for i in range(n):
        if mob_hp(s, base, i) > 0:
            c += 1
    return c


@always_inline
def first_dead_slot(s: State, base: Int, n: Int) -> Int:
    for i in range(n):
        if mob_hp(s, base, i) <= 0:
            return i
    return -1


@always_inline
def is_cow_spawn_tile(s: State, y: Int, x: Int, dist: Int) -> Bool:
    if dist <= 3 or dist >= MOB_DESPAWN_DISTANCE:
        return False
    if get_map(s, y, x) != BLOCK_GRASS:
        return False
    if is_in_mob(s, y, x):
        return False
    return True


@always_inline
def is_zombie_spawn_tile(s: State, y: Int, x: Int, dist: Int) -> Bool:
    if dist <= 9 or dist >= MOB_DESPAWN_DISTANCE:
        return False
    var b = get_map(s, y, x)
    if b != BLOCK_GRASS and b != BLOCK_PATH:
        return False
    if is_in_mob(s, y, x):
        return False
    return True


@always_inline
def is_skeleton_spawn_tile(s: State, y: Int, x: Int, dist: Int) -> Bool:
    if dist <= 9 or dist >= MOB_DESPAWN_DISTANCE:
        return False
    if get_map(s, y, x) != BLOCK_PATH:
        return False
    if is_in_mob(s, y, x):
        return False
    return True


@always_inline
def spawn_one_mob(
    s: State,
    base: Int,
    max_n: Int,
    hp: Int,
    chance: Float32,
    kind: Int,  # 0=cow, 1=zombie, 2=skeleton
    mut rng: PhiloxRandom,
):
    if count_alive(s, base, max_n) >= max_n:
        return
    var u = rng.step_uniform()
    if Float32(u[0]) >= chance:
        return

    var pp = player_pos(s)

    # Count valid tiles.
    var total = 0
    for y in range(MAP_H):
        for x in range(MAP_W):
            var d = manhattan(y, x, pp[0], pp[1])
            var ok = (
                is_cow_spawn_tile(s, y, x, d)
                if kind == 0
                else (
                    is_zombie_spawn_tile(s, y, x, d)
                    if kind == 1
                    else is_skeleton_spawn_tile(s, y, x, d)
                )
            )
            if ok:
                total += 1
    if total == 0:
        return

    var u2 = rng.step_uniform()
    var pick = Int(Float32(u2[0]) * Float32(total))
    if pick >= total:
        pick = total - 1

    var seen = 0
    for y in range(MAP_H):
        for x in range(MAP_W):
            var d = manhattan(y, x, pp[0], pp[1])
            var ok = (
                is_cow_spawn_tile(s, y, x, d)
                if kind == 0
                else (
                    is_zombie_spawn_tile(s, y, x, d)
                    if kind == 1
                    else is_skeleton_spawn_tile(s, y, x, d)
                )
            )
            if ok:
                if seen == pick:
                    var slot = first_dead_slot(s, base, max_n)
                    if slot >= 0:
                        mob_set_pos(s, base, slot, y, x)
                        mob_set_hp(s, base, slot, hp)
                        mob_set_cd(s, base, slot, 0)
                    return
                seen += 1


@always_inline
def spawn_mobs(s: State, mut rng: PhiloxRandom):
    # Cows.
    spawn_one_mob(
        s,
        S_COWS_BASE,
        MAX_COWS,
        COW_HEALTH,
        SPAWN_COW_CHANCE,
        0,
        rng,
    )
    # Zombies — night-gated.
    var light = s[S_LIGHT_LEVEL]
    var darkness = Float32(1.0) - light
    var z_chance = (
        SPAWN_ZOMBIE_BASE_CHANCE
        + SPAWN_ZOMBIE_NIGHT_CHANCE * darkness * darkness
    )
    spawn_one_mob(
        s, S_ZOMBIES_BASE, MAX_ZOMBIES, ZOMBIE_HEALTH, z_chance, 1, rng
    )
    # Skeletons.
    spawn_one_mob(
        s,
        S_SKELETONS_BASE,
        MAX_SKELETONS,
        SKELETON_HEALTH,
        SPAWN_SKELETON_CHANCE,
        2,
        rng,
    )


# ============================================================================
# Top-level step orchestrator
# ============================================================================

@always_inline
def apply_step_inline(
    s: State, action: Int, mut rng: PhiloxRandom
) -> Tuple[Float32, Bool]:
    """One env step. Returns (reward, done).

    Order mirrors `craftax_step` in the reference:
      crafting → action → place → move → mobs → spawn → plants →
      intrinsics → cap_inventory → reward → timestep + light.
    """
    var action_eff = ACTION_NOOP if is_sleeping(s) else action

    var init_ach = sum_achievements(s)
    var init_health = get_intr(s, INTRINSIC_HEALTH)

    do_crafting(s, action_eff)
    do_action(s, action_eff, rng)
    place_block(s, action_eff)
    move_player(s, action_eff)
    update_mobs_all(s, rng)
    spawn_mobs(s, rng)
    update_plants(s)
    update_intrinsics(s, action_eff)
    cap_inventory(s)

    # Reward and done.
    var new_ach = sum_achievements(s)
    var new_health = get_intr(s, INTRINSIC_HEALTH)
    var reward = Float32(new_ach - init_ach) + Float32(0.1) * Float32(
        new_health - init_health
    )

    # Bookkeeping: timestep + light level.
    var t = Int(s[S_TIMESTEP]) + 1
    s[S_TIMESTEP] = Float32(t)
    s[S_LIGHT_LEVEL] = compute_light_level(t)

    var dead = new_health <= 0
    var truncated = t >= MAX_TIMESTEPS
    var done = dead or truncated
    return (reward, done)


# ============================================================================
# Symbolic observation extraction (Phase 4)
# ============================================================================

@always_inline
def _splat_mob_channel(
    s: State,
    obs: UnsafePointer[Float32, MutAnyOrigin],
    base: Int,
    max_n: Int,
    py: Int,
    px: Int,
    half_h: Int,
    half_w: Int,
    channel: Int,
):
    """For each alive mob of kind `base`, if its tile lies inside the
    player's view window, set the corresponding mob channel."""
    for i in range(max_n):
        if mob_hp(s, base, i) <= 0:
            continue
        var mp = mob_pos(s, base, i)
        var lv = mp[0] - py + half_h
        var lx = mp[1] - px + half_w
        if 0 <= lv and lv < VIEW_H and 0 <= lx and lx < VIEW_W:
            var tile_base = lv * VIEW_W * TILE_CHANNELS + lx * TILE_CHANNELS
            obs[tile_base + NUM_BLOCK_TYPES + channel] = Float32(1.0)


@always_inline
def extract_obs_inline(s: State, obs: UnsafePointer[Float32, MutAnyOrigin]):
    """Build the 1345-D symbolic observation vector for one env.

    Layout (matches `references/.../renderer.py:render_craftax_symbolic`):
      [0 : 1323) 7×9×21 local view (per-tile one-hot block + 4 mob channels)
      [1323 : 1335) inventory / 10
      [1335 : 1339) health, food, drink, energy (each / 10)
      [1339 : 1343) direction one-hot (LEFT, RIGHT, UP, DOWN)
      [1343]        light_level
      [1344]        is_sleeping (0 or 1)
    """
    var pp = player_pos(s)
    var half_h = VIEW_H // 2
    var half_w = VIEW_W // 2
    var py = pp[0]
    var px = pp[1]

    # --- 1. View tiles: block one-hot + mob channels (zero first) ---
    for lv in range(VIEW_H):
        var wy = py - half_h + lv
        for lx in range(VIEW_W):
            var wx = px - half_w + lx
            var tile_base = lv * VIEW_W * TILE_CHANNELS + lx * TILE_CHANNELS
            for ch in range(TILE_CHANNELS):
                obs[tile_base + ch] = Float32(0.0)
            var blk = (
                get_map(s, wy, wx) if in_bounds(wy, wx)
                else BLOCK_OUT_OF_BOUNDS
            )
            obs[tile_base + blk] = Float32(1.0)

    # --- 2. Mob channels: zombie=17, cow=18, skeleton=19, arrow=20 ---
    _splat_mob_channel(
        s, obs, S_ZOMBIES_BASE, MAX_ZOMBIES, py, px, half_h, half_w, 0
    )
    _splat_mob_channel(
        s, obs, S_COWS_BASE, MAX_COWS, py, px, half_h, half_w, 1
    )
    _splat_mob_channel(
        s,
        obs,
        S_SKELETONS_BASE,
        MAX_SKELETONS,
        py,
        px,
        half_h,
        half_w,
        2,
    )
    # Arrows are 5-field; mob_pos works because it only reads fields 0,1.
    for i in range(MAX_ARROWS):
        var arr_base = S_ARROWS_BASE + i * ARROW_FIELDS
        if Int(s[arr_base + MOB_HP]) <= 0:
            continue
        var ay = Int(s[arr_base + MOB_FY])
        var ax = Int(s[arr_base + MOB_FX])
        var lv = ay - py + half_h
        var lx = ax - px + half_w
        if 0 <= lv and lv < VIEW_H and 0 <= lx and lx < VIEW_W:
            var tile_base = lv * VIEW_W * TILE_CHANNELS + lx * TILE_CHANNELS
            obs[tile_base + NUM_BLOCK_TYPES + 3] = Float32(1.0)

    # --- 3. Inventory / 10 ---
    var inv_off = OBS_VIEW_SIZE
    for i in range(NUM_INVENTORY):
        obs[inv_off + i] = Float32(get_inv(s, i)) * Float32(0.1)

    # --- 4. Intrinsics / 10 ---
    var intr_off = inv_off + NUM_INVENTORY
    for k in range(NUM_INTRINSICS):
        obs[intr_off + k] = Float32(get_intr(s, k)) * Float32(0.1)

    # --- 5. Direction one-hot (DIR_LEFT..DIR_DOWN = 0..3) ---
    var dir_off = intr_off + NUM_INTRINSICS
    var d = player_dir(s)
    for k in range(NUM_DIRECTIONS):
        obs[dir_off + k] = Float32(1.0) if k == d else Float32(0.0)

    # --- 6. Light + sleep ---
    var scalar_off = dir_off + NUM_DIRECTIONS
    obs[scalar_off + 0] = s[S_LIGHT_LEVEL]
    obs[scalar_off + 1] = s[S_IS_SLEEPING]
