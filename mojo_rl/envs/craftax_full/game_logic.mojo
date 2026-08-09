"""Step logic for Full Craftax.

Phase 7C/35 deliverable: full scaffold + the deterministic action paths
(movement, DO, descend/ascend, all crafting + placement, sleep/rest,
intrinsics tick, plant tick, level-up attributes, achievements, reward,
done). Combat is wired but mob spawn / movement / attack / projectiles /
boss logic are deliberately stubbed — they're real branches in the step
function that currently no-op so the rest of the pipeline stays
correct. The next session fills them in.

All functions operate on a single env's flat state slice
(`Pointer[Float32, MutAnyOrigin]`) so the same code can later be
called from a GPU kernel.

Mirrors `references/Craftax-main/craftax/craftax/game_logic.py`
function-by-function.
"""

from std.math import cos as math_cos
from std.random.philox import Random as PhiloxRandom

from .constants import (
    MAP_H,
    MAP_W,
    NUM_FLOORS,
    FLOOR_OVERWORLD,
    FLOOR_GRAVEYARD,
    NUM_ACTIONS,
    NUM_INTRINSICS,
    NUM_INTRINSICS_F,
    NUM_INVENTORY,
    NUM_ARMOUR_ENCHANTS,
    NUM_ACHIEVEMENTS,
    NUM_DIRECTIONS,
    NUM_SPELLS,
    NUM_POTIONS,
    NUM_ATTRIBUTES,
    INV_MAX_PER_SLOT,
    INTRINSIC_MAX,
    PLAYER_MAX_HEALTH,
    DAY_LENGTH,
    MAX_TIMESTEPS,
    MAX_MELEE_MOBS,
    MAX_PASSIVE_MOBS,
    MAX_RANGED_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PLAYER_PROJECTILES,
    MAX_GROWING_PLANTS,
    MOB_FIELDS,
    MOB_FY,
    MOB_FX,
    MOB_HP,
    MOB_MASK,
    MOB_CD,
    MOB_TYPE_ID,
    PROJ_FIELDS,
    PROJ_FDIR_Y,
    PROJ_FDIR_X,
    PLANT_FIELDS,
    PLANT_FY,
    PLANT_FX,
    PLANT_FAGE,
    PLANT_RIPEN_AGE,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    BOSS_FIGHT_SPAWN_TURNS,
    # Blocks
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
    BLOCK_SAND,
    BLOCK_LAVA,
    BLOCK_PLANT,
    BLOCK_RIPE_PLANT,
    BLOCK_WALL,
    BLOCK_DARKNESS,
    BLOCK_WALL_MOSS,
    BLOCK_STALAGMITE,
    BLOCK_SAPPHIRE,
    BLOCK_RUBY,
    BLOCK_CHEST,
    BLOCK_FOUNTAIN,
    BLOCK_FIRE_GRASS,
    BLOCK_ICE_GRASS,
    BLOCK_GRAVEL,
    BLOCK_FIRE_TREE,
    BLOCK_ICE_SHRUB,
    BLOCK_ENCHANTMENT_TABLE_FIRE,
    BLOCK_ENCHANTMENT_TABLE_ICE,
    BLOCK_NECROMANCER,
    BLOCK_GRAVE,
    BLOCK_GRAVE2,
    BLOCK_GRAVE3,
    BLOCK_NECROMANCER_VULNERABLE,
    # Items
    ITEM_NONE,
    ITEM_TORCH,
    ITEM_LADDER_DOWN,
    ITEM_LADDER_UP,
    ITEM_LADDER_DOWN_BLOCKED,
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
    ACTION_DRINK_POTION_GREEN,
    ACTION_DRINK_POTION_BLUE,
    ACTION_DRINK_POTION_PINK,
    ACTION_DRINK_POTION_CYAN,
    ACTION_DRINK_POTION_YELLOW,
    ACTION_READ_BOOK,
    ACTION_ENCHANT_SWORD,
    ACTION_ENCHANT_BOW,
    ACTION_ENCHANT_ARMOUR,
    ACTION_LEVEL_UP_DEXTERITY,
    ACTION_LEVEL_UP_STRENGTH,
    ACTION_LEVEL_UP_INTELLIGENCE,
    # Inventory
    INV_WOOD,
    INV_STONE,
    INV_COAL,
    INV_IRON,
    INV_DIAMOND,
    INV_SAPLING,
    INV_PICKAXE,
    INV_SWORD,
    INV_BOW,
    INV_ARROWS,
    INV_ARMOUR_HEAD,
    INV_ARMOUR_BODY,
    INV_ARMOUR_LEGS,
    INV_ARMOUR_FEET,
    INV_TORCHES,
    INV_RUBY,
    INV_SAPPHIRE,
    INV_POTIONS_BASE,
    INV_BOOKS,
    # Intrinsics
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
    INTRINSIC_MANA,
    INTRINSIC_IS_SLEEPING,
    INTRINSIC_IS_RESTING,
    INTRINSIC_F_RECOVER,
    INTRINSIC_F_HUNGER,
    INTRINSIC_F_THIRST,
    INTRINSIC_F_FATIGUE,
    INTRINSIC_F_RECOVER_MANA,
    # Attributes
    ATTR_XP,
    ATTR_DEXTERITY,
    ATTR_STRENGTH,
    ATTR_INTELLIGENCE,
    MAX_ATTRIBUTE,
    # Damage tables / consumption
    DAMAGE_FIST,
    DAMAGE_WOOD_SWORD,
    DAMAGE_STONE_SWORD,
    DAMAGE_IRON_SWORD,
    DAMAGE_DIAMOND_SWORD,
    DAMAGE_BOW,
    HUNGER_THRESHOLD,
    THIRST_THRESHOLD,
    FATIGUE_HIGH_THRESHOLD,
    FATIGUE_LOW_THRESHOLD,
    RECOVER_HIGH_THRESHOLD,
    RECOVER_LOW_THRESHOLD,
    RECOVER_MANA_THRESHOLD,
    COW_EAT_BOOST,
    BAT_EAT_BOOST,
    SNAIL_EAT_BOOST,
    PLANT_EAT_BOOST,
    WATER_DRINK_BOOST,
    FOUNTAIN_DRINK_BOOST,
    POTION_HEAL,
    SAPLING_DROP_CHANCE,
    MANA_COST_FIREBALL,
    MANA_COST_ICEBALL,
    SPELL_FIREBALL,
    SPELL_ICEBALL,
    ENCHANT_NONE,
    ENCHANT_FIRE,
    ENCHANT_ICE,
    # Achievements
    ACH_COLLECT_WOOD,
    ACH_PLACE_TABLE,
    ACH_EAT_COW,
    ACH_COLLECT_SAPLING,
    ACH_COLLECT_DRINK,
    ACH_MAKE_WOOD_PICKAXE,
    ACH_MAKE_WOOD_SWORD,
    ACH_PLACE_PLANT,
    ACH_DEFEAT_ZOMBIE,
    ACH_COLLECT_STONE,
    ACH_PLACE_STONE,
    ACH_EAT_PLANT,
    ACH_DEFEAT_SKELETON,
    ACH_MAKE_STONE_PICKAXE,
    ACH_MAKE_STONE_SWORD,
    ACH_WAKE_UP,
    ACH_PLACE_FURNACE,
    ACH_COLLECT_COAL,
    ACH_COLLECT_IRON,
    ACH_COLLECT_DIAMOND,
    ACH_MAKE_IRON_PICKAXE,
    ACH_MAKE_IRON_SWORD,
    ACH_MAKE_ARROW,
    ACH_MAKE_TORCH,
    ACH_PLACE_TORCH,
    ACH_MAKE_DIAMOND_SWORD,
    ACH_MAKE_DIAMOND_PICKAXE,
    ACH_MAKE_IRON_ARMOUR,
    ACH_MAKE_DIAMOND_ARMOUR,
    ACH_ENTER_GNOMISH_MINES,
    ACH_ENTER_DUNGEON,
    ACH_ENTER_SEWERS,
    ACH_ENTER_VAULT,
    ACH_ENTER_TROLL_MINES,
    ACH_ENTER_FIRE_REALM,
    ACH_ENTER_ICE_REALM,
    ACH_ENTER_GRAVEYARD,
    ACH_FIND_BOW,
    ACH_FIRE_BOW,
    ACH_COLLECT_SAPPHIRE,
    ACH_LEARN_FIREBALL,
    ACH_CAST_FIREBALL,
    ACH_LEARN_ICEBALL,
    ACH_CAST_ICEBALL,
    ACH_COLLECT_RUBY,
    ACH_OPEN_CHEST,
    ACH_DRINK_POTION,
    ACH_ENCHANT_SWORD,
    ACH_ENCHANT_ARMOUR,
    ACH_DEFEAT_NECROMANCER,
    ACH_DAMAGE_NECROMANCER,
    achievement_reward_weight,
    required_pickaxe_tier,
)
from .state import (
    s_map,
    s_item_map,
    s_mob_map,
    s_light_map,
    s_down_ladder,
    s_up_ladder,
    s_chest_opened,
    s_monsters_killed,
    s_inv,
    s_intrinsic,
    s_intrinsic_f,
    s_attribute,
    s_achievement,
    s_melee_mob,
    s_passive_mob,
    s_ranged_mob,
    s_mob_projectile,
    s_player_projectile,
    s_plant,
    s_plant_mask,
    s_potion_mapping,
    s_learned_spell,
    s_armour_enchant,
    s_rng,
    S_PLAYER_POS,
    S_PLAYER_LEVEL,
    S_PLAYER_DIR,
    S_SWORD_ENCHANT,
    S_BOW_ENCHANT,
    S_BOSS_PROGRESS,
    S_BOSS_TIMESTEPS,
    S_LIGHT_LEVEL,
    S_TIMESTEP,
    STATE_SIZE,
)


# ============================================================================
# Type alias
# ============================================================================

comptime State = Pointer[Float32, MutAnyOrigin]


# ============================================================================
# Direction
# ============================================================================

@always_inline
def dir_offset(d: Int) -> Tuple[Int, Int]:
    if d == DIR_LEFT:
        return (0, -1)
    if d == DIR_RIGHT:
        return (0, 1)
    if d == DIR_UP:
        return (-1, 0)
    if d == DIR_DOWN:
        return (1, 0)
    return (0, 0)


@always_inline
def in_bounds(y: Int, x: Int) -> Bool:
    return 0 <= y and y < MAP_H and 0 <= x and x < MAP_W


# ============================================================================
# Block predicates
# ============================================================================

@always_inline
def is_solid(block: Int) -> Bool:
    """Blocks the player (a LAND_CREATURE) cannot walk through.

    Walkable: GRASS, PATH, SAND, LAVA, FIRE_GRASS, ICE_GRASS, GRAVEL,
              NECROMANCER_VULNERABLE.
    Everything else: solid.
    """
    if block == BLOCK_GRASS:
        return False
    if block == BLOCK_PATH:
        return False
    if block == BLOCK_SAND:
        return False
    if block == BLOCK_LAVA:
        return False  # walkable, kills via intrinsics
    if block == BLOCK_FIRE_GRASS:
        return False
    if block == BLOCK_ICE_GRASS:
        return False
    if block == BLOCK_GRAVEL:
        return False
    if block == BLOCK_NECROMANCER_VULNERABLE:
        return False
    return True


# ============================================================================
# Map / item / mob-map / light accessors
# ============================================================================

@always_inline
def floor_idx(s: State) -> Int:
    return Int(s[unsafe_offset=S_PLAYER_LEVEL])


@always_inline
def get_map(s: State, floor: Int, y: Int, x: Int) -> Int:
    return Int(s[unsafe_offset=s_map(floor, y, x)])


@always_inline
def set_map(s: State, floor: Int, y: Int, x: Int, block: Int):
    s[unsafe_offset=s_map(floor, y, x)] = Float32(block)


@always_inline
def get_item(s: State, floor: Int, y: Int, x: Int) -> Int:
    return Int(s[unsafe_offset=s_item_map(floor, y, x)])


@always_inline
def set_item(s: State, floor: Int, y: Int, x: Int, item: Int):
    s[unsafe_offset=s_item_map(floor, y, x)] = Float32(item)


@always_inline
def get_mob_map(s: State, floor: Int, y: Int, x: Int) -> Int:
    return Int(s[unsafe_offset=s_mob_map(floor, y, x)])


@always_inline
def set_mob_map(s: State, floor: Int, y: Int, x: Int, v: Int):
    s[unsafe_offset=s_mob_map(floor, y, x)] = Float32(v)


@always_inline
def is_in_mob(s: State, floor: Int, y: Int, x: Int) -> Bool:
    return get_mob_map(s, floor, y, x) != 0


# ============================================================================
# Player accessors
# ============================================================================

@always_inline
def player_pos(s: State) -> Tuple[Int, Int]:
    return (Int(s[unsafe_offset=S_PLAYER_POS]), Int(s[unsafe_offset=S_PLAYER_POS + 1]))


@always_inline
def set_player_pos(s: State, y: Int, x: Int):
    s[unsafe_offset=S_PLAYER_POS] = Float32(y)
    s[unsafe_offset=S_PLAYER_POS + 1] = Float32(x)


@always_inline
def player_dir(s: State) -> Int:
    return Int(s[unsafe_offset=S_PLAYER_DIR])


@always_inline
def set_player_dir(s: State, d: Int):
    s[unsafe_offset=S_PLAYER_DIR] = Float32(d)


# ============================================================================
# Inventory / intrinsic accessors with clamping
# ============================================================================

@always_inline
def get_inv(s: State, slot: Int) -> Int:
    return Int(s[unsafe_offset=s_inv(slot)])


@always_inline
def set_inv(s: State, slot: Int, v: Int):
    var x = v
    if x < 0:
        x = 0
    if slot < INV_PICKAXE and x > INV_MAX_PER_SLOT:
        # Tier slots (pickaxe/sword/bow/armour) are not counts — uncapped
        # via the INV_MAX_PER_SLOT rule; raw materials are capped.
        x = INV_MAX_PER_SLOT
    s[unsafe_offset=s_inv(slot)] = Float32(x)


@always_inline
def add_inv(s: State, slot: Int, delta: Int):
    set_inv(s, slot, get_inv(s, slot) + delta)


@always_inline
def cap_inventory(s: State):
    for i in range(NUM_INVENTORY):
        var v = Int(s[unsafe_offset=s_inv(i)])
        if v < 0:
            v = 0
        elif v > INV_MAX_PER_SLOT and i != INV_PICKAXE and i != INV_SWORD:
            v = INV_MAX_PER_SLOT
        s[unsafe_offset=s_inv(i)] = Float32(v)


@always_inline
def get_intr(s: State, slot: Int) -> Int:
    return Int(s[unsafe_offset=s_intrinsic(slot)])


@always_inline
def set_intr(s: State, slot: Int, v: Int):
    var x = v
    if x < 0:
        x = 0
    if x > INTRINSIC_MAX:
        x = INTRINSIC_MAX
    s[unsafe_offset=s_intrinsic(slot)] = Float32(x)


@always_inline
def add_intr(s: State, slot: Int, delta: Int):
    set_intr(s, slot, get_intr(s, slot) + delta)


@always_inline
def get_intr_f(s: State, slot: Int) -> Float32:
    return s[unsafe_offset=s_intrinsic_f(slot)]


@always_inline
def set_intr_f(s: State, slot: Int, v: Float32):
    s[unsafe_offset=s_intrinsic_f(slot)] = v


@always_inline
def player_hp(s: State) -> Int:
    return get_intr(s, INTRINSIC_HEALTH)


@always_inline
def set_player_hp(s: State, hp: Int):
    set_intr(s, INTRINSIC_HEALTH, hp)


@always_inline
def is_sleeping(s: State) -> Bool:
    return get_intr(s, INTRINSIC_IS_SLEEPING) > 0


@always_inline
def set_sleeping(s: State, v: Bool):
    set_intr(s, INTRINSIC_IS_SLEEPING, 1 if v else 0)


@always_inline
def is_resting(s: State) -> Bool:
    return get_intr(s, INTRINSIC_IS_RESTING) > 0


@always_inline
def set_resting(s: State, v: Bool):
    set_intr(s, INTRINSIC_IS_RESTING, 1 if v else 0)


@always_inline
def get_attr(s: State, slot: Int) -> Int:
    return Int(s[unsafe_offset=s_attribute(slot)])


@always_inline
def set_attr(s: State, slot: Int, v: Int):
    s[unsafe_offset=s_attribute(slot)] = Float32(v)


@always_inline
def add_attr(s: State, slot: Int, delta: Int):
    set_attr(s, slot, get_attr(s, slot) + delta)


# ============================================================================
# Achievements
# ============================================================================

@always_inline
def set_achievement(s: State, idx: Int):
    s[unsafe_offset=s_achievement(idx)] = Float32(1.0)


@always_inline
def get_ach(s: State, idx: Int) -> Bool:
    return s[unsafe_offset=s_achievement(idx)] > Float32(0.5)


@always_inline
def sum_achievement_weights(s: State) -> Float32:
    var acc = Float32(0.0)
    for i in range(NUM_ACHIEVEMENTS):
        if get_ach(s, i):
            acc += achievement_reward_weight(i)
    return acc


# ============================================================================
# Helpers
# ============================================================================

@always_inline
def is_near(s: State, floor: Int, block_type: Int) -> Bool:
    """Any of the 8 neighbour cells equals `block_type`."""
    var pp = player_pos(s)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            var y = pp[0] + dy
            var x = pp[1] + dx
            if in_bounds(y, x) and get_map(s, floor, y, x) == block_type:
                return True
    return False


@always_inline
def pickaxe_tier(s: State) -> Int:
    return get_inv(s, INV_PICKAXE)


@always_inline
def sword_tier(s: State) -> Int:
    return get_inv(s, INV_SWORD)


@always_inline
def best_sword_damage(s: State) -> Int:
    var t = sword_tier(s)
    if t >= 4:
        return DAMAGE_DIAMOND_SWORD
    if t == 3:
        return DAMAGE_IRON_SWORD
    if t == 2:
        return DAMAGE_STONE_SWORD
    if t == 1:
        return DAMAGE_WOOD_SWORD
    return DAMAGE_FIST


# ============================================================================
# Plant helpers
# ============================================================================

@always_inline
def plant_mask_get(s: State, i: Int) -> Bool:
    return s[unsafe_offset=s_plant_mask(i)] > 0


@always_inline
def plant_mask_set(s: State, i: Int, v: Bool):
    s[unsafe_offset=s_plant_mask(i)] = Float32(1.0) if v else Float32(0.0)


@always_inline
def plant_get(s: State, i: Int, f: Int) -> Float32:
    return s[unsafe_offset=s_plant(i, f)]


@always_inline
def plant_set(s: State, i: Int, f: Int, v: Float32):
    s[unsafe_offset=s_plant(i, f)] = v


@always_inline
def add_growing_plant(s: State, y: Int, x: Int) -> Bool:
    for i in range(MAX_GROWING_PLANTS):
        if not plant_mask_get(s, i):
            plant_set(s, i, PLANT_FY, Float32(y))
            plant_set(s, i, PLANT_FX, Float32(x))
            plant_set(s, i, PLANT_FAGE, Float32(0))
            plant_mask_set(s, i, True)
            return True
    return False


@always_inline
def update_plants(s: State):
    """Per-step plant aging. PLANT becomes RIPE_PLANT once age >= ripen.
    Only ages plants on the player's current floor — the reference
    actually advances them every step regardless of floor; we do the same
    by ignoring `floor` (plants is a shared pool with floor not tracked
    in our layout). When a plant ripens we update its tile if the tile
    still says PLANT (otherwise the plant was destroyed).
    """
    var floor = floor_idx(s)
    for i in range(MAX_GROWING_PLANTS):
        if not plant_mask_get(s, i):
            continue
        var age = Int(plant_get(s, i, PLANT_FAGE)) + 1
        plant_set(s, i, PLANT_FAGE, Float32(age))
        if age >= PLANT_RIPEN_AGE:
            var py = Int(plant_get(s, i, PLANT_FY))
            var px = Int(plant_get(s, i, PLANT_FX))
            if in_bounds(py, px):
                if get_map(s, floor, py, px) == BLOCK_PLANT:
                    set_map(s, floor, py, px, BLOCK_RIPE_PLANT)


# ============================================================================
# Movement
# ============================================================================

@always_inline
def move_player(s: State, action: Int):
    """Cardinal step with bounds + solid + mob-occupancy gates.
    Direction always updates on a cardinal action."""
    if action < ACTION_LEFT or action > ACTION_DOWN:
        return
    var d = action - ACTION_LEFT
    set_player_dir(s, d)
    var off = dir_offset(d)
    var pp = player_pos(s)
    var ny = pp[0] + off[0]
    var nx = pp[1] + off[1]
    if not in_bounds(ny, nx):
        return
    var floor = floor_idx(s)
    if is_solid(get_map(s, floor, ny, nx)):
        return
    if is_in_mob(s, floor, ny, nx):
        return
    set_player_pos(s, ny, nx)


# ============================================================================
# Change floor (DESCEND / ASCEND)
# ============================================================================

@always_inline
def change_floor(s: State, action: Int):
    """DESCEND requires standing on a LADDER_DOWN item AND the current
    floor's kill quota met (monsters_killed >= 8). ASCEND requires
    standing on a LADDER_UP item.
    """
    if action != ACTION_DESCEND and action != ACTION_ASCEND:
        return
    var floor = floor_idx(s)
    var pp = player_pos(s)
    var item_here = get_item(s, floor, pp[0], pp[1])

    if action == ACTION_DESCEND:
        if item_here != ITEM_LADDER_DOWN:
            return
        if floor >= NUM_FLOORS - 1:
            return
        if Int(s[unsafe_offset=s_monsters_killed(floor)]) < MONSTERS_KILLED_TO_CLEAR_LEVEL:
            return
        var nf = floor + 1
        s[unsafe_offset=S_PLAYER_LEVEL] = Float32(nf)
        # Spawn at the destination floor's ladder_up coords (so we land
        # next to a ladder going back).
        var ny = Int(s[unsafe_offset=s_up_ladder(nf, 0)])
        var nx = Int(s[unsafe_offset=s_up_ladder(nf, 1)])
        if in_bounds(ny, nx):
            set_player_pos(s, ny, nx)
        # Floor-entry achievements.
        if nf == 1:
            set_achievement(s, ACH_ENTER_DUNGEON)
        elif nf == 2:
            set_achievement(s, ACH_ENTER_GNOMISH_MINES)
        elif nf == 3:
            set_achievement(s, ACH_ENTER_SEWERS)
        elif nf == 4:
            set_achievement(s, ACH_ENTER_VAULT)
        elif nf == 5:
            set_achievement(s, ACH_ENTER_TROLL_MINES)
        elif nf == 6:
            set_achievement(s, ACH_ENTER_FIRE_REALM)
        elif nf == 7:
            set_achievement(s, ACH_ENTER_ICE_REALM)
        elif nf == 8:
            set_achievement(s, ACH_ENTER_GRAVEYARD)
    else:  # ASCEND
        if item_here != ITEM_LADDER_UP:
            return
        if floor <= 0:
            return
        var nf = floor - 1
        s[unsafe_offset=S_PLAYER_LEVEL] = Float32(nf)
        var ny = Int(s[unsafe_offset=s_down_ladder(nf, 0)])
        var nx = Int(s[unsafe_offset=s_down_ladder(nf, 1)])
        if in_bounds(ny, nx):
            set_player_pos(s, ny, nx)


# ============================================================================
# Placement
# ============================================================================

@always_inline
def place_block(s: State, action: Int):
    if action != ACTION_PLACE_STONE and action != ACTION_PLACE_TABLE \
       and action != ACTION_PLACE_FURNACE and action != ACTION_PLACE_PLANT \
       and action != ACTION_PLACE_TORCH:
        return
    var floor = floor_idx(s)
    var pp = player_pos(s)
    var off = dir_offset(player_dir(s))
    var ty = pp[0] + off[0]
    var tx = pp[1] + off[1]
    if not in_bounds(ty, tx) or is_in_mob(s, floor, ty, tx):
        return
    var target = get_map(s, floor, ty, tx)

    if action == ACTION_PLACE_TABLE:
        if get_inv(s, INV_WOOD) >= 1 and not is_solid(target):
            add_inv(s, INV_WOOD, -1)
            set_map(s, floor, ty, tx, BLOCK_CRAFTING_TABLE)
            set_achievement(s, ACH_PLACE_TABLE)
    elif action == ACTION_PLACE_FURNACE:
        if get_inv(s, INV_STONE) >= 1 and not is_solid(target):
            add_inv(s, INV_STONE, -1)
            set_map(s, floor, ty, tx, BLOCK_FURNACE)
            set_achievement(s, ACH_PLACE_FURNACE)
    elif action == ACTION_PLACE_STONE:
        var ok = target == BLOCK_WATER or not is_solid(target)
        if get_inv(s, INV_STONE) >= 1 and ok:
            add_inv(s, INV_STONE, -1)
            set_map(s, floor, ty, tx, BLOCK_STONE)
            set_achievement(s, ACH_PLACE_STONE)
    elif action == ACTION_PLACE_PLANT:
        if get_inv(s, INV_SAPLING) >= 1 and target == BLOCK_GRASS:
            add_inv(s, INV_SAPLING, -1)
            set_map(s, floor, ty, tx, BLOCK_PLANT)
            _ = add_growing_plant(s, ty, tx)
            set_achievement(s, ACH_PLACE_PLANT)
    elif action == ACTION_PLACE_TORCH:
        # Drop a torch on the player's own tile (in front works too).
        # Reference: at player position, light_map updated by torch radius.
        if get_inv(s, INV_TORCHES) >= 1:
            add_inv(s, INV_TORCHES, -1)
            set_item(s, floor, pp[0], pp[1], ITEM_TORCH)
            set_achievement(s, ACH_PLACE_TORCH)


# ============================================================================
# Crafting
# ============================================================================

@always_inline
def do_crafting(s: State, action: Int):
    var floor = floor_idx(s)
    var at_table = is_near(s, floor, BLOCK_CRAFTING_TABLE)
    var at_furnace = is_near(s, floor, BLOCK_FURNACE)

    if action == ACTION_MAKE_WOOD_PICKAXE:
        if at_table and get_inv(s, INV_WOOD) >= 1 and pickaxe_tier(s) < 1:
            add_inv(s, INV_WOOD, -1)
            set_inv(s, INV_PICKAXE, 1)
            set_achievement(s, ACH_MAKE_WOOD_PICKAXE)
    elif action == ACTION_MAKE_STONE_PICKAXE:
        if (
            at_table
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_STONE) >= 1
            and pickaxe_tier(s) < 2
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_STONE, -1)
            set_inv(s, INV_PICKAXE, 2)
            set_achievement(s, ACH_MAKE_STONE_PICKAXE)
    elif action == ACTION_MAKE_IRON_PICKAXE:
        if (
            at_table
            and at_furnace
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_STONE) >= 1
            and get_inv(s, INV_IRON) >= 1
            and get_inv(s, INV_COAL) >= 1
            and pickaxe_tier(s) < 3
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_STONE, -1)
            add_inv(s, INV_IRON, -1)
            add_inv(s, INV_COAL, -1)
            set_inv(s, INV_PICKAXE, 3)
            set_achievement(s, ACH_MAKE_IRON_PICKAXE)
    elif action == ACTION_MAKE_DIAMOND_PICKAXE:
        if (
            at_table
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_DIAMOND) >= 1
            and pickaxe_tier(s) < 4
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_DIAMOND, -1)
            set_inv(s, INV_PICKAXE, 4)
            set_achievement(s, ACH_MAKE_DIAMOND_PICKAXE)
    elif action == ACTION_MAKE_WOOD_SWORD:
        if at_table and get_inv(s, INV_WOOD) >= 1 and sword_tier(s) < 1:
            add_inv(s, INV_WOOD, -1)
            set_inv(s, INV_SWORD, 1)
            set_achievement(s, ACH_MAKE_WOOD_SWORD)
    elif action == ACTION_MAKE_STONE_SWORD:
        if (
            at_table
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_STONE) >= 1
            and sword_tier(s) < 2
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_STONE, -1)
            set_inv(s, INV_SWORD, 2)
            set_achievement(s, ACH_MAKE_STONE_SWORD)
    elif action == ACTION_MAKE_IRON_SWORD:
        if (
            at_table
            and at_furnace
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_COAL) >= 1
            and get_inv(s, INV_IRON) >= 1
            and sword_tier(s) < 3
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_COAL, -1)
            add_inv(s, INV_IRON, -1)
            set_inv(s, INV_SWORD, 3)
            set_achievement(s, ACH_MAKE_IRON_SWORD)
    elif action == ACTION_MAKE_DIAMOND_SWORD:
        if (
            at_table
            and get_inv(s, INV_WOOD) >= 1
            and get_inv(s, INV_DIAMOND) >= 1
            and sword_tier(s) < 4
        ):
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_DIAMOND, -1)
            set_inv(s, INV_SWORD, 4)
            set_achievement(s, ACH_MAKE_DIAMOND_SWORD)
    elif action == ACTION_MAKE_IRON_ARMOUR:
        # One piece per call; we use head slot if free, else body, etc.
        if (
            at_table
            and at_furnace
            and get_inv(s, INV_IRON) >= 1
            and get_inv(s, INV_COAL) >= 1
        ):
            var placed = False
            for slot in [INV_ARMOUR_HEAD, INV_ARMOUR_BODY, INV_ARMOUR_LEGS, INV_ARMOUR_FEET]:
                if get_inv(s, slot) < 1:
                    set_inv(s, slot, 1)
                    placed = True
                    break
            if placed:
                add_inv(s, INV_IRON, -1)
                add_inv(s, INV_COAL, -1)
                set_achievement(s, ACH_MAKE_IRON_ARMOUR)
    elif action == ACTION_MAKE_DIAMOND_ARMOUR:
        if at_table and get_inv(s, INV_DIAMOND) >= 1:
            var placed = False
            for slot in [INV_ARMOUR_HEAD, INV_ARMOUR_BODY, INV_ARMOUR_LEGS, INV_ARMOUR_FEET]:
                if get_inv(s, slot) < 2:
                    set_inv(s, slot, 2)
                    placed = True
                    break
            if placed:
                add_inv(s, INV_DIAMOND, -1)
                set_achievement(s, ACH_MAKE_DIAMOND_ARMOUR)
    elif action == ACTION_MAKE_ARROW:
        if at_table and get_inv(s, INV_WOOD) >= 1 and get_inv(s, INV_STONE) >= 1:
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_STONE, -1)
            add_inv(s, INV_ARROWS, 1)
            set_achievement(s, ACH_MAKE_ARROW)
    elif action == ACTION_MAKE_TORCH:
        if at_table and get_inv(s, INV_WOOD) >= 1 and get_inv(s, INV_COAL) >= 1:
            add_inv(s, INV_WOOD, -1)
            add_inv(s, INV_COAL, -1)
            add_inv(s, INV_TORCHES, 1)
            set_achievement(s, ACH_MAKE_TORCH)


# ============================================================================
# do_action — mine / attack / eat / drink / sleep
# ============================================================================

@always_inline
def update_plant_age_on_eat(s: State, y: Int, x: Int):
    for i in range(MAX_GROWING_PLANTS):
        if plant_mask_get(s, i):
            var py = Int(plant_get(s, i, PLANT_FY))
            var px = Int(plant_get(s, i, PLANT_FX))
            if py == y and px == x:
                plant_set(s, i, PLANT_FAGE, Float32(0))
                return


@always_inline
def do_action(s: State, action: Int, mut rng: PhiloxRandom):
    if action != ACTION_DO:
        return
    var floor = floor_idx(s)
    var pp = player_pos(s)
    var off = dir_offset(player_dir(s))
    var ty = pp[0] + off[0]
    var tx = pp[1] + off[1]
    if not in_bounds(ty, tx):
        return

    # Mob attack takes priority — combat path is stubbed for this phase
    # (no mobs spawn yet), but we still call into a unified scan so the
    # plumbing works once mobs come online.
    if is_in_mob(s, floor, ty, tx):
        # See _try_attack_mob in the combat stub block.
        _try_attack_mob(s, floor, ty, tx)
        return

    var blk = get_map(s, floor, ty, tx)
    var pt = pickaxe_tier(s)

    if blk == BLOCK_TREE or blk == BLOCK_FIRE_TREE or blk == BLOCK_ICE_SHRUB \
       or blk == BLOCK_STALAGMITE:
        set_map(s, floor, ty, tx, BLOCK_GRASS if floor == 0 else BLOCK_PATH)
        add_inv(s, INV_WOOD, 1)
        set_achievement(s, ACH_COLLECT_WOOD)
    elif blk == BLOCK_STONE:
        if pt >= required_pickaxe_tier(blk):
            set_map(s, floor, ty, tx, BLOCK_PATH)
            add_inv(s, INV_STONE, 1)
            set_achievement(s, ACH_COLLECT_STONE)
    elif blk == BLOCK_COAL:
        if pt >= required_pickaxe_tier(blk):
            set_map(s, floor, ty, tx, BLOCK_PATH)
            add_inv(s, INV_COAL, 1)
            set_achievement(s, ACH_COLLECT_COAL)
    elif blk == BLOCK_IRON:
        if pt >= required_pickaxe_tier(blk):
            set_map(s, floor, ty, tx, BLOCK_PATH)
            add_inv(s, INV_IRON, 1)
            set_achievement(s, ACH_COLLECT_IRON)
    elif blk == BLOCK_DIAMOND:
        if pt >= required_pickaxe_tier(blk):
            set_map(s, floor, ty, tx, BLOCK_PATH)
            add_inv(s, INV_DIAMOND, 1)
            set_achievement(s, ACH_COLLECT_DIAMOND)
    elif blk == BLOCK_RUBY:
        if pt >= required_pickaxe_tier(blk):
            set_map(s, floor, ty, tx, BLOCK_PATH)
            add_inv(s, INV_RUBY, 1)
            set_achievement(s, ACH_COLLECT_RUBY)
    elif blk == BLOCK_SAPPHIRE:
        if pt >= required_pickaxe_tier(blk):
            set_map(s, floor, ty, tx, BLOCK_PATH)
            add_inv(s, INV_SAPPHIRE, 1)
            set_achievement(s, ACH_COLLECT_SAPPHIRE)
    elif blk == BLOCK_WALL or blk == BLOCK_WALL_MOSS:
        if pt >= required_pickaxe_tier(blk):
            set_map(s, floor, ty, tx, BLOCK_PATH)
    elif blk == BLOCK_GRASS or blk == BLOCK_FIRE_GRASS or blk == BLOCK_ICE_GRASS:
        var u = rng.step_uniform()
        if Float32(u[0]) < SAPLING_DROP_CHANCE:
            add_inv(s, INV_SAPLING, 1)
            set_achievement(s, ACH_COLLECT_SAPLING)
    elif blk == BLOCK_PLANT:
        # Reset growth.
        update_plant_age_on_eat(s, ty, tx)
    elif blk == BLOCK_RIPE_PLANT:
        set_map(s, floor, ty, tx, BLOCK_PLANT)
        update_plant_age_on_eat(s, ty, tx)
        add_intr(s, INTRINSIC_FOOD, PLANT_EAT_BOOST)
        set_intr_f(s, INTRINSIC_F_HUNGER, Float32(0.0))
        set_achievement(s, ACH_EAT_PLANT)
    elif blk == BLOCK_WATER:
        add_intr(s, INTRINSIC_DRINK, WATER_DRINK_BOOST)
        set_intr_f(s, INTRINSIC_F_THIRST, Float32(0.0))
        set_achievement(s, ACH_COLLECT_DRINK)
    elif blk == BLOCK_FOUNTAIN:
        add_intr(s, INTRINSIC_DRINK, FOUNTAIN_DRINK_BOOST)
        set_intr_f(s, INTRINSIC_F_THIRST, Float32(0.0))
        set_achievement(s, ACH_COLLECT_DRINK)
    elif blk == BLOCK_CHEST:
        # Mark opened + give a small bundle (real loot table lands in
        # the next session). For now: 1 wood, 1 stone — and the
        # achievement.
        set_map(s, floor, ty, tx, BLOCK_PATH)
        s[unsafe_offset=s_chest_opened(floor)] = Float32(1.0)
        add_inv(s, INV_WOOD, 1)
        add_inv(s, INV_STONE, 1)
        set_achievement(s, ACH_OPEN_CHEST)
    elif blk == BLOCK_ENCHANTMENT_TABLE_FIRE or blk == BLOCK_ENCHANTMENT_TABLE_ICE:
        # Block itself is interactive via the dedicated ENCHANT_* actions;
        # DO on it does nothing.
        pass


# ============================================================================
# Mob combat stub (real impl in the next session)
# ============================================================================

@always_inline
def _try_attack_mob(s: State, floor: Int, y: Int, x: Int):
    """Apply best-sword damage to the first alive mob at (y, x) on the
    given floor. Awards the kill achievement on death.

    Stub: mob arrays are zeroed at reset and no spawn pass populates
    them yet, so this is currently a no-op in practice. Wiring is in
    place so combat starts working as soon as spawn is implemented.
    """
    var dmg = best_sword_damage(s)
    # Scan melee mobs.
    for i in range(MAX_MELEE_MOBS):
        if s[unsafe_offset=s_melee_mob(floor, i, MOB_MASK)] > 0:
            var my = Int(s[unsafe_offset=s_melee_mob(floor, i, MOB_FY)])
            var mx = Int(s[unsafe_offset=s_melee_mob(floor, i, MOB_FX)])
            if my == y and mx == x:
                var new_hp = Int(
                    s[unsafe_offset=s_melee_mob(floor, i, MOB_HP)]
                ) - dmg
                if new_hp < 0:
                    new_hp = 0
                s[unsafe_offset=s_melee_mob(floor, i, MOB_HP)] = Float32(new_hp)
                if new_hp == 0:
                    s[unsafe_offset=s_melee_mob(floor, i, MOB_MASK)] = Float32(0.0)
                    set_mob_map(s, floor, my, mx, 0)
                    s[unsafe_offset=s_monsters_killed(floor)] += Float32(1.0)
                    # XP reward.
                    add_attr(s, ATTR_XP, 1)
                return
    for i in range(MAX_PASSIVE_MOBS):
        if s[unsafe_offset=s_passive_mob(floor, i, MOB_MASK)] > 0:
            var my = Int(s[unsafe_offset=s_passive_mob(floor, i, MOB_FY)])
            var mx = Int(s[unsafe_offset=s_passive_mob(floor, i, MOB_FX)])
            if my == y and mx == x:
                var new_hp = Int(
                    s[unsafe_offset=s_passive_mob(floor, i, MOB_HP)]
                ) - dmg
                if new_hp < 0:
                    new_hp = 0
                s[unsafe_offset=s_passive_mob(floor, i, MOB_HP)] = Float32(new_hp)
                if new_hp == 0:
                    s[unsafe_offset=s_passive_mob(floor, i, MOB_MASK)] = Float32(0.0)
                    set_mob_map(s, floor, my, mx, 0)
                    # Eat the corpse (cow/bat/snail boost).
                    add_intr(s, INTRINSIC_FOOD, COW_EAT_BOOST)
                    set_intr_f(s, INTRINSIC_F_HUNGER, Float32(0.0))
                    set_achievement(s, ACH_EAT_COW)
                return
    for i in range(MAX_RANGED_MOBS):
        if s[unsafe_offset=s_ranged_mob(floor, i, MOB_MASK)] > 0:
            var my = Int(s[unsafe_offset=s_ranged_mob(floor, i, MOB_FY)])
            var mx = Int(s[unsafe_offset=s_ranged_mob(floor, i, MOB_FX)])
            if my == y and mx == x:
                var new_hp = Int(
                    s[unsafe_offset=s_ranged_mob(floor, i, MOB_HP)]
                ) - dmg
                if new_hp < 0:
                    new_hp = 0
                s[unsafe_offset=s_ranged_mob(floor, i, MOB_HP)] = Float32(new_hp)
                if new_hp == 0:
                    s[unsafe_offset=s_ranged_mob(floor, i, MOB_MASK)] = Float32(0.0)
                    set_mob_map(s, floor, my, mx, 0)
                    s[unsafe_offset=s_monsters_killed(floor)] += Float32(1.0)
                    add_attr(s, ATTR_XP, 1)
                return


# ============================================================================
# Sleep / rest
# ============================================================================

@always_inline
def do_sleep(s: State, action: Int):
    if action == ACTION_SLEEP:
        set_sleeping(s, True)
    elif action == ACTION_REST:
        set_resting(s, True)


@always_inline
def wake_up_if_recovered(s: State):
    """Match reference: stop sleeping when energy is full; stop resting
    when health is full."""
    if is_sleeping(s) and get_intr(s, INTRINSIC_ENERGY) >= INTRINSIC_MAX:
        set_sleeping(s, False)
        set_achievement(s, ACH_WAKE_UP)
    if is_resting(s) and player_hp(s) >= PLAYER_MAX_HEALTH:
        set_resting(s, False)


# ============================================================================
# Intrinsics tick
# ============================================================================

@always_inline
def update_player_intrinsics(s: State, action: Int):
    """Once per step: accumulate hunger/thirst/fatigue/recover and apply
    threshold-triggered changes. Mirrors the reference (which uses
    floats and tier thresholds). Mana recovers up to 9.
    """
    var floor = floor_idx(s)

    # Hunger
    var hunger = get_intr_f(s, INTRINSIC_F_HUNGER) + Float32(0.5)
    if hunger >= HUNGER_THRESHOLD:
        hunger = Float32(0.0)
        add_intr(s, INTRINSIC_FOOD, -1)
    set_intr_f(s, INTRINSIC_F_HUNGER, hunger)

    # Thirst
    var thirst = get_intr_f(s, INTRINSIC_F_THIRST) + Float32(0.5)
    if thirst >= THIRST_THRESHOLD:
        thirst = Float32(0.0)
        add_intr(s, INTRINSIC_DRINK, -1)
    set_intr_f(s, INTRINSIC_F_THIRST, thirst)

    # Fatigue (sleep replenishes energy; not sleeping drains it).
    var fatigue = get_intr_f(s, INTRINSIC_F_FATIGUE)
    if is_sleeping(s):
        fatigue -= Float32(1.0)
    else:
        fatigue += Float32(0.5)
    if fatigue >= FATIGUE_HIGH_THRESHOLD:
        fatigue = Float32(0.0)
        add_intr(s, INTRINSIC_ENERGY, -1)
    elif fatigue <= FATIGUE_LOW_THRESHOLD:
        fatigue = Float32(0.0)
        add_intr(s, INTRINSIC_ENERGY, 1)
    set_intr_f(s, INTRINSIC_F_FATIGUE, fatigue)

    # Recover (heals when food + drink + energy all > 0; drains otherwise).
    var recov = get_intr_f(s, INTRINSIC_F_RECOVER)
    var well_fed = (
        get_intr(s, INTRINSIC_FOOD) > 0
        and get_intr(s, INTRINSIC_DRINK) > 0
        and get_intr(s, INTRINSIC_ENERGY) > 0
    )
    if well_fed:
        recov += Float32(1.0)
    else:
        recov -= Float32(0.5)
    if recov >= RECOVER_HIGH_THRESHOLD:
        recov = Float32(0.0)
        add_intr(s, INTRINSIC_HEALTH, 1)
    elif recov <= RECOVER_LOW_THRESHOLD:
        recov = Float32(0.0)
        add_intr(s, INTRINSIC_HEALTH, -1)
    set_intr_f(s, INTRINSIC_F_RECOVER, recov)

    # Mana recovery.
    var rm = get_intr_f(s, INTRINSIC_F_RECOVER_MANA) + Float32(0.5)
    if rm >= RECOVER_MANA_THRESHOLD:
        rm = Float32(0.0)
        add_intr(s, INTRINSIC_MANA, 1)
    set_intr_f(s, INTRINSIC_F_RECOVER_MANA, rm)

    # Standing on LAVA deals 1 hp / step.
    var pp = player_pos(s)
    if get_map(s, floor, pp[0], pp[1]) == BLOCK_LAVA:
        add_intr(s, INTRINSIC_HEALTH, -1)

    wake_up_if_recovered(s)


# ============================================================================
# Day/night light
# ============================================================================

@always_inline
def update_light_level(s: State, timestep_after: Int):
    var progress = (Float32(timestep_after) / Float32(DAY_LENGTH)) - Float32(
        Int(timestep_after // DAY_LENGTH)
    ) + Float32(0.3)
    var c = math_cos(Float32(3.14159265) * progress)
    var ac = c if c >= Float32(0.0) else -c
    s[unsafe_offset=S_LIGHT_LEVEL] = Float32(1.0) - ac * ac * ac


# ============================================================================
# Spells / arrows / potions / books / enchants — stubbed (resource cost +
# achievement only).
# ============================================================================

@always_inline
def shoot_projectile(s: State, action: Int):
    if action != ACTION_SHOOT_ARROW:
        return
    if get_inv(s, INV_BOW) >= 1 and get_inv(s, INV_ARROWS) >= 1:
        add_inv(s, INV_ARROWS, -1)
        set_achievement(s, ACH_FIRE_BOW)


@always_inline
def cast_spell(s: State, action: Int):
    if action == ACTION_CAST_FIREBALL:
        if (
            s[unsafe_offset=s_learned_spell(SPELL_FIREBALL)] > 0
            and get_intr(s, INTRINSIC_MANA) >= MANA_COST_FIREBALL
        ):
            add_intr(s, INTRINSIC_MANA, -MANA_COST_FIREBALL)
            set_achievement(s, ACH_CAST_FIREBALL)
    elif action == ACTION_CAST_ICEBALL:
        if (
            s[unsafe_offset=s_learned_spell(SPELL_ICEBALL)] > 0
            and get_intr(s, INTRINSIC_MANA) >= MANA_COST_ICEBALL
        ):
            add_intr(s, INTRINSIC_MANA, -MANA_COST_ICEBALL)
            set_achievement(s, ACH_CAST_ICEBALL)


@always_inline
def drink_potion(s: State, action: Int):
    if action < ACTION_DRINK_POTION_RED or action > ACTION_DRINK_POTION_YELLOW:
        return
    var color = action - ACTION_DRINK_POTION_RED  # 0..5
    if get_inv(s, INV_POTIONS_BASE + color) < 1:
        return
    add_inv(s, INV_POTIONS_BASE + color, -1)
    set_achievement(s, ACH_DRINK_POTION)
    # Reference maps each color to one of {heal, mana, energy, water,
    # food, damage} via the random potion_mapping table. Until we
    # implement that table, treat every potion as a small heal.
    add_intr(s, INTRINSIC_HEALTH, POTION_HEAL)


@always_inline
def read_book(s: State, mut rng: PhiloxRandom, action: Int):
    if action != ACTION_READ_BOOK:
        return
    if get_inv(s, INV_BOOKS) < 1:
        return
    add_inv(s, INV_BOOKS, -1)
    var u = rng.step_uniform()
    if Float32(u[0]) < Float32(0.5):
        s[unsafe_offset=s_learned_spell(SPELL_FIREBALL)] = Float32(1.0)
        set_achievement(s, ACH_LEARN_FIREBALL)
    else:
        s[unsafe_offset=s_learned_spell(SPELL_ICEBALL)] = Float32(1.0)
        set_achievement(s, ACH_LEARN_ICEBALL)


@always_inline
def enchant(s: State, mut rng: PhiloxRandom, action: Int):
    if action != ACTION_ENCHANT_SWORD and action != ACTION_ENCHANT_BOW \
       and action != ACTION_ENCHANT_ARMOUR:
        return
    var floor = floor_idx(s)
    var has_fire = is_near(s, floor, BLOCK_ENCHANTMENT_TABLE_FIRE)
    var has_ice = is_near(s, floor, BLOCK_ENCHANTMENT_TABLE_ICE)
    if not (has_fire or has_ice):
        return
    var enchant_kind = ENCHANT_FIRE if has_fire else ENCHANT_ICE
    # Costs: a gem matching the table.
    if has_fire and get_inv(s, INV_RUBY) < 1:
        return
    if has_ice and get_inv(s, INV_SAPPHIRE) < 1:
        return
    if has_fire:
        add_inv(s, INV_RUBY, -1)
    else:
        add_inv(s, INV_SAPPHIRE, -1)

    if action == ACTION_ENCHANT_SWORD:
        if sword_tier(s) >= 1:
            s[unsafe_offset=S_SWORD_ENCHANT] = Float32(enchant_kind)
            set_achievement(s, ACH_ENCHANT_SWORD)
    elif action == ACTION_ENCHANT_BOW:
        if get_inv(s, INV_BOW) >= 1:
            s[unsafe_offset=S_BOW_ENCHANT] = Float32(enchant_kind)
    elif action == ACTION_ENCHANT_ARMOUR:
        for slot in [INV_ARMOUR_HEAD, INV_ARMOUR_BODY, INV_ARMOUR_LEGS, INV_ARMOUR_FEET]:
            if get_inv(s, slot) >= 1:
                var idx = slot - INV_ARMOUR_HEAD
                s[unsafe_offset=s_armour_enchant(idx)] = Float32(enchant_kind)
                set_achievement(s, ACH_ENCHANT_ARMOUR)
                break


# ============================================================================
# Level-up attributes (cost: XP)
# ============================================================================

@always_inline
def level_up_attributes(s: State, action: Int):
    if action != ACTION_LEVEL_UP_DEXTERITY and action != ACTION_LEVEL_UP_STRENGTH \
       and action != ACTION_LEVEL_UP_INTELLIGENCE:
        return
    var slot = ATTR_DEXTERITY
    if action == ACTION_LEVEL_UP_STRENGTH:
        slot = ATTR_STRENGTH
    elif action == ACTION_LEVEL_UP_INTELLIGENCE:
        slot = ATTR_INTELLIGENCE
    if get_attr(s, slot) >= MAX_ATTRIBUTE:
        return
    var cur = get_attr(s, slot)
    var cost = cur + 1  # 1 XP for first level, 2 for second, etc.
    if get_attr(s, ATTR_XP) < cost:
        return
    add_attr(s, ATTR_XP, -cost)
    add_attr(s, slot, 1)


# ============================================================================
# Boss logic (stub) + spawn / mob update (stubs)
# ============================================================================

@always_inline
def boss_logic(s: State):
    """Stub. Real logic: alternating necromancer-vulnerable/invulnerable
    phases, projectile flurries, etc. lands in the next session."""
    var floor = floor_idx(s)
    if floor != FLOOR_GRAVEYARD:
        return
    # No-op for now.


@always_inline
def update_mobs(s: State, mut rng: PhiloxRandom):
    """Stub — mobs are not spawned yet."""
    pass


@always_inline
def spawn_mobs(s: State, mut rng: PhiloxRandom):
    """Stub — mobs are not spawned yet."""
    pass


# ============================================================================
# Inventory achievements (BOW pickup, etc.)
# ============================================================================

@always_inline
def calculate_inventory_achievements(s: State):
    if get_inv(s, INV_BOW) > 0:
        set_achievement(s, ACH_FIND_BOW)


# ============================================================================
# Reward + done
# ============================================================================

@always_inline
def is_game_over(s: State, timestep_after: Int) -> Bool:
    if player_hp(s) <= 0:
        return True
    if timestep_after >= MAX_TIMESTEPS:
        return True
    if get_ach(s, ACH_DEFEAT_NECROMANCER):
        return True
    return False


# ============================================================================
# Top-level step orchestration
# ============================================================================

def apply_step_inline(s: State, action: Int, mut rng: PhiloxRandom) -> Tuple[Float32, Bool]:
    """Apply one step. Returns (reward, done).

    Phase ordering mirrors `craftax_step` in the reference: change_floor,
    crafting, do_action, place, shoot, cast, potion, read_book, enchant,
    boss, level_up, move, mob update, spawn, plants, intrinsics, cap,
    inventory_achievements, reward.
    """
    var init_ach_weight = sum_achievement_weights(s)
    var init_hp = player_hp(s)

    # Interrupt action if asleep / resting.
    var eff_action = action
    if is_sleeping(s) or is_resting(s):
        eff_action = ACTION_NOOP

    change_floor(s, eff_action)
    do_crafting(s, eff_action)
    do_action(s, eff_action, rng)
    place_block(s, eff_action)
    shoot_projectile(s, eff_action)
    cast_spell(s, eff_action)
    drink_potion(s, eff_action)
    read_book(s, rng, eff_action)
    enchant(s, rng, eff_action)
    boss_logic(s)
    level_up_attributes(s, eff_action)
    move_player(s, eff_action)
    update_mobs(s, rng)
    spawn_mobs(s, rng)
    update_plants(s)
    do_sleep(s, eff_action)
    update_player_intrinsics(s, eff_action)
    cap_inventory(s)
    calculate_inventory_achievements(s)

    # Reward
    var final_ach_weight = sum_achievement_weights(s)
    var ach_reward = final_ach_weight - init_ach_weight
    var hp_reward = Float32(player_hp(s) - init_hp) * Float32(0.1)
    var reward = ach_reward + hp_reward

    # Advance timestep + recompute light.
    var t1 = Int(s[unsafe_offset=S_TIMESTEP]) + 1
    s[unsafe_offset=S_TIMESTEP] = Float32(t1)
    update_light_level(s, t1)

    return (reward, is_game_over(s, t1))
