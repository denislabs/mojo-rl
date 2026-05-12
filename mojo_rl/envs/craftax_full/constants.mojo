"""Compile-time constants for Full Craftax (9-floor dungeon variant).

Mirrors `references/Craftax-main/craftax/craftax/constants.py`. Numeric IDs
for blocks/actions/mobs/achievements match the original Python enums so we
can reuse the same sprite assets and (eventually) compare obs vectors with
the reference implementation.
"""

# ============================================================================
# Map / floor geometry
# ============================================================================

comptime MAP_H: Int = 48
comptime MAP_W: Int = 48
comptime MAP_SIZE_PER_FLOOR: Int = MAP_H * MAP_W  # 2304
comptime NUM_FLOORS: Int = 9
comptime MAP_TOTAL_SIZE: Int = NUM_FLOORS * MAP_SIZE_PER_FLOOR  # 20736

# Floor order matches reference `generate_world` splice
# (smooth[0], dung[0], smooth[1], dung[1], dung[2], smooth[2], smooth[3..5]).
comptime FLOOR_OVERWORLD: Int = 0
comptime FLOOR_DUNGEON: Int = 1
comptime FLOOR_GNOMISH_MINES: Int = 2
comptime FLOOR_SEWERS: Int = 3
comptime FLOOR_VAULTS: Int = 4
comptime FLOOR_TROLL_MINES: Int = 5
comptime FLOOR_FIRE: Int = 6
comptime FLOOR_ICE: Int = 7
comptime FLOOR_GRAVEYARD: Int = 8

# Agent's local view window — Full Craftax uses (9, 11) (odd × odd).
comptime VIEW_H: Int = 9
comptime VIEW_W: Int = 11
comptime VIEW_SIZE: Int = VIEW_H * VIEW_W  # 99


# ============================================================================
# Block types — match BlockType enum in reference (37 entries)
# ============================================================================

comptime BLOCK_INVALID: Int = 0
comptime BLOCK_OUT_OF_BOUNDS: Int = 1
comptime BLOCK_GRASS: Int = 2
comptime BLOCK_WATER: Int = 3
comptime BLOCK_STONE: Int = 4
comptime BLOCK_TREE: Int = 5
comptime BLOCK_WOOD: Int = 6
comptime BLOCK_PATH: Int = 7
comptime BLOCK_COAL: Int = 8
comptime BLOCK_IRON: Int = 9
comptime BLOCK_DIAMOND: Int = 10
comptime BLOCK_CRAFTING_TABLE: Int = 11
comptime BLOCK_FURNACE: Int = 12
comptime BLOCK_SAND: Int = 13
comptime BLOCK_LAVA: Int = 14
comptime BLOCK_PLANT: Int = 15
comptime BLOCK_RIPE_PLANT: Int = 16
comptime BLOCK_WALL: Int = 17
comptime BLOCK_DARKNESS: Int = 18
comptime BLOCK_WALL_MOSS: Int = 19
comptime BLOCK_STALAGMITE: Int = 20
comptime BLOCK_SAPPHIRE: Int = 21
comptime BLOCK_RUBY: Int = 22
comptime BLOCK_CHEST: Int = 23
comptime BLOCK_FOUNTAIN: Int = 24
comptime BLOCK_FIRE_GRASS: Int = 25
comptime BLOCK_ICE_GRASS: Int = 26
comptime BLOCK_GRAVEL: Int = 27
comptime BLOCK_FIRE_TREE: Int = 28
comptime BLOCK_ICE_SHRUB: Int = 29
comptime BLOCK_ENCHANTMENT_TABLE_FIRE: Int = 30
comptime BLOCK_ENCHANTMENT_TABLE_ICE: Int = 31
comptime BLOCK_NECROMANCER: Int = 32
comptime BLOCK_GRAVE: Int = 33
comptime BLOCK_GRAVE2: Int = 34
comptime BLOCK_GRAVE3: Int = 35
comptime BLOCK_NECROMANCER_VULNERABLE: Int = 36

comptime NUM_BLOCK_TYPES: Int = 37


# ============================================================================
# Item types — placeable / pickable items occupying the item_map layer
# ============================================================================

comptime ITEM_NONE: Int = 0
comptime ITEM_TORCH: Int = 1
comptime ITEM_LADDER_DOWN: Int = 2
comptime ITEM_LADDER_UP: Int = 3
comptime ITEM_LADDER_DOWN_BLOCKED: Int = 4

comptime NUM_ITEM_TYPES: Int = 5


# ============================================================================
# Actions — match Action enum in reference (43 entries)
# ============================================================================

comptime ACTION_NOOP: Int = 0
comptime ACTION_LEFT: Int = 1
comptime ACTION_RIGHT: Int = 2
comptime ACTION_UP: Int = 3
comptime ACTION_DOWN: Int = 4
comptime ACTION_DO: Int = 5
comptime ACTION_SLEEP: Int = 6
comptime ACTION_PLACE_STONE: Int = 7
comptime ACTION_PLACE_TABLE: Int = 8
comptime ACTION_PLACE_FURNACE: Int = 9
comptime ACTION_PLACE_PLANT: Int = 10
comptime ACTION_MAKE_WOOD_PICKAXE: Int = 11
comptime ACTION_MAKE_STONE_PICKAXE: Int = 12
comptime ACTION_MAKE_IRON_PICKAXE: Int = 13
comptime ACTION_MAKE_WOOD_SWORD: Int = 14
comptime ACTION_MAKE_STONE_SWORD: Int = 15
comptime ACTION_MAKE_IRON_SWORD: Int = 16
comptime ACTION_REST: Int = 17
comptime ACTION_DESCEND: Int = 18
comptime ACTION_ASCEND: Int = 19
comptime ACTION_MAKE_DIAMOND_PICKAXE: Int = 20
comptime ACTION_MAKE_DIAMOND_SWORD: Int = 21
comptime ACTION_MAKE_IRON_ARMOUR: Int = 22
comptime ACTION_MAKE_DIAMOND_ARMOUR: Int = 23
comptime ACTION_SHOOT_ARROW: Int = 24
comptime ACTION_MAKE_ARROW: Int = 25
comptime ACTION_CAST_FIREBALL: Int = 26
comptime ACTION_CAST_ICEBALL: Int = 27
comptime ACTION_PLACE_TORCH: Int = 28
comptime ACTION_DRINK_POTION_RED: Int = 29
comptime ACTION_DRINK_POTION_GREEN: Int = 30
comptime ACTION_DRINK_POTION_BLUE: Int = 31
comptime ACTION_DRINK_POTION_PINK: Int = 32
comptime ACTION_DRINK_POTION_CYAN: Int = 33
comptime ACTION_DRINK_POTION_YELLOW: Int = 34
comptime ACTION_READ_BOOK: Int = 35
comptime ACTION_ENCHANT_SWORD: Int = 36
comptime ACTION_ENCHANT_ARMOUR: Int = 37
comptime ACTION_MAKE_TORCH: Int = 38
comptime ACTION_LEVEL_UP_DEXTERITY: Int = 39
comptime ACTION_LEVEL_UP_STRENGTH: Int = 40
comptime ACTION_LEVEL_UP_INTELLIGENCE: Int = 41
comptime ACTION_ENCHANT_BOW: Int = 42

comptime NUM_ACTIONS: Int = 43


# ============================================================================
# Mobs — 4 categories × 8 species per category (per-floor mapping)
# ============================================================================

# Mob category indices (used by mob_map encoding + state mob arrays).
comptime MOB_CAT_PASSIVE: Int = 0
comptime MOB_CAT_MELEE: Int = 1
comptime MOB_CAT_RANGED: Int = 2
comptime MOB_CAT_PROJECTILE: Int = 3
comptime NUM_MOB_CATEGORIES: Int = 4

# Per-floor species ID for each (mob_cat, floor) — the same `type_id` slot
# used in the state arrays. Reproduced from `FLOOR_MOB_MAPPING` in the
# reference. Each floor has one passive / melee / ranged species index, and
# the same index drives sprite + damage / health table lookups.
# Layout: floor-major, then category. `FLOOR_MOB_SPECIES[floor*3+cat] = id`.
# Passive species IDs (8 in total): 0=cow, 1=bat, 2=snail.
# Melee species IDs: 0=zombie, 1=gnome warrior, 2=orc soldier, 3=lizard,
#   4=knight, 5=troll, 6=pigman, 7=frost troll.
# Ranged species IDs: 0=skeleton, 1=gnome archer, 2=orc mage, 3=kobold,
#   4=knight archer, 5=deep thing, 6=fire elemental, 7=ice elemental.
# Projectile species IDs: 0=arrow, 1=dagger, 2=fireball, 3=iceball,
#   4=arrow2, 5=slimeball, 6=fireball2, 7=iceball2.

# Max simultaneous mobs per env per floor (matches StaticEnvParams).
comptime MAX_MELEE_MOBS: Int = 3
comptime MAX_PASSIVE_MOBS: Int = 3
comptime MAX_RANGED_MOBS: Int = 2
comptime MAX_MOB_PROJECTILES: Int = 3
comptime MAX_PLAYER_PROJECTILES: Int = 3
comptime MAX_GROWING_PLANTS: Int = 10  # plant pool is shared across floors

# Per-mob fields. hp=0 ⇒ dead.
comptime MOB_FIELDS: Int = 6  # y, x, hp, mask, cd, type_id
comptime MOB_FY: Int = 0
comptime MOB_FX: Int = 1
comptime MOB_HP: Int = 2
comptime MOB_MASK: Int = 3
comptime MOB_CD: Int = 4
comptime MOB_TYPE_ID: Int = 5

# Projectiles also carry a 2-vector direction.
comptime PROJ_FIELDS: Int = 8  # MOB_FIELDS + direction_y, direction_x
comptime PROJ_FDIR_Y: Int = 6
comptime PROJ_FDIR_X: Int = 7

# Mob category index per floor — see FLOOR_MOB_MAPPING in the reference.
# Used to look up the species ID at runtime via the helper table below.
# Reproduced as flat row-major: (floor, cat) → species_id.
comptime _FMM_ROWS: Int = NUM_FLOORS
comptime _FMM_COLS: Int = 3
comptime FLOOR_MOB_SPECIES_SIZE: Int = _FMM_ROWS * _FMM_COLS  # 27

@always_inline
def floor_mob_species(floor: Int, cat: Int) -> Int:
    """Species index for (floor, mob category in {passive, melee, ranged}).

    Identical to FLOOR_MOB_MAPPING in `craftax/constants.py`.
    """
    # cat: 0=passive, 1=melee, 2=ranged. Mirrors FLOOR_MOB_MAPPING.
    if floor == 0:
        return 0  # overworld: cow / zombie / skeleton
    elif floor == 1:
        # dungeon: snail / orc soldier / orc mage
        return 2
    elif floor == 2:
        # gnomish mines: bat / gnome warrior / gnome archer
        return 1
    elif floor == 3:
        # sewers: snail / lizard / kobold
        if cat == 0:
            return 2
        return 3
    elif floor == 4:
        # vaults: snail / knight / knight archer
        if cat == 0:
            return 2
        return 4
    elif floor == 5:
        # troll mines: bat / troll / deep thing
        if cat == 0:
            return 1
        return 5
    elif floor == 6:
        # fire: bat / pigman / fire elemental
        if cat == 0:
            return 1
        return 6
    elif floor == 7:
        # ice: bat / ice troll / ice elemental
        if cat == 0:
            return 1
        return 7
    else:
        return 0  # boss (no regular spawns)


# ============================================================================
# Projectile types
# ============================================================================

comptime PROJ_ARROW: Int = 0
comptime PROJ_DAGGER: Int = 1
comptime PROJ_FIREBALL: Int = 2
comptime PROJ_ICEBALL: Int = 3
comptime PROJ_ARROW2: Int = 4
comptime PROJ_SLIMEBALL: Int = 5
comptime PROJ_FIREBALL2: Int = 6
comptime PROJ_ICEBALL2: Int = 7
comptime NUM_PROJECTILE_TYPES: Int = 8


# ============================================================================
# Plants (shared pool — not per-floor in the reference)
# ============================================================================

comptime PLANT_FIELDS: Int = 3
comptime PLANT_FY: Int = 0
comptime PLANT_FX: Int = 1
comptime PLANT_FAGE: Int = 2


# ============================================================================
# Inventory (Full Craftax extends Classic — 21 distinct slots)
# ============================================================================

comptime INV_WOOD: Int = 0
comptime INV_STONE: Int = 1
comptime INV_COAL: Int = 2
comptime INV_IRON: Int = 3
comptime INV_DIAMOND: Int = 4
comptime INV_SAPLING: Int = 5
comptime INV_PICKAXE: Int = 6     # tier-encoded (0..4: none, wood, stone, iron, diamond)
comptime INV_SWORD: Int = 7
comptime INV_BOW: Int = 8
comptime INV_ARROWS: Int = 9
comptime INV_ARMOUR_HEAD: Int = 10   # 4 armour slots — tier-encoded each
comptime INV_ARMOUR_BODY: Int = 11
comptime INV_ARMOUR_LEGS: Int = 12
comptime INV_ARMOUR_FEET: Int = 13
comptime INV_TORCHES: Int = 14
comptime INV_RUBY: Int = 15
comptime INV_SAPPHIRE: Int = 16
comptime INV_POTIONS_BASE: Int = 17   # 6 contiguous slots: potion counts by color
# index 23 onward: future / reserved
comptime INV_BOOKS: Int = 23

comptime NUM_POTIONS: Int = 6  # red, green, blue, pink, cyan, yellow
comptime NUM_INVENTORY: Int = 24

comptime INV_MAX_PER_SLOT: Int = 9


# ============================================================================
# Player intrinsics
# ============================================================================

# Integer intrinsics: 5 stats (HP, food, drink, energy, mana) + 2 booleans
# (is_sleeping, is_resting). All fit in [0..9] except booleans in {0, 1}.
comptime INTRINSIC_HEALTH: Int = 0
comptime INTRINSIC_FOOD: Int = 1
comptime INTRINSIC_DRINK: Int = 2
comptime INTRINSIC_ENERGY: Int = 3
comptime INTRINSIC_MANA: Int = 4
comptime INTRINSIC_IS_SLEEPING: Int = 5
comptime INTRINSIC_IS_RESTING: Int = 6
comptime NUM_INTRINSICS: Int = 7

# Float accumulators (recover, hunger, thirst, fatigue, recover_mana).
comptime INTRINSIC_F_RECOVER: Int = 0
comptime INTRINSIC_F_HUNGER: Int = 1
comptime INTRINSIC_F_THIRST: Int = 2
comptime INTRINSIC_F_FATIGUE: Int = 3
comptime INTRINSIC_F_RECOVER_MANA: Int = 4
comptime NUM_INTRINSICS_F: Int = 5

comptime INTRINSIC_MAX: Int = 9
comptime PLAYER_MAX_HEALTH: Int = 9


# ============================================================================
# Player attributes (XP-driven RPG stats)
# ============================================================================

comptime ATTR_XP: Int = 0
comptime ATTR_DEXTERITY: Int = 1
comptime ATTR_STRENGTH: Int = 2
comptime ATTR_INTELLIGENCE: Int = 3
comptime NUM_ATTRIBUTES: Int = 4

comptime MAX_ATTRIBUTE: Int = 5  # cap on dex/str/intel


# ============================================================================
# Direction
# ============================================================================

comptime DIR_LEFT: Int = 0
comptime DIR_RIGHT: Int = 1
comptime DIR_UP: Int = 2
comptime DIR_DOWN: Int = 3
comptime NUM_DIRECTIONS: Int = 4


# ============================================================================
# Enchantments
# ============================================================================

# Encoded as integer: 0=none, 1=fire, 2=ice.
comptime ENCHANT_NONE: Int = 0
comptime ENCHANT_FIRE: Int = 1
comptime ENCHANT_ICE: Int = 2

# 4 armour slots (head, body, legs, feet) — each holds an enchantment id.
comptime NUM_ARMOUR_ENCHANTS: Int = 4


# ============================================================================
# Achievements — match Achievement enum in reference (67 unique values)
# ============================================================================
#
# NOTE: the reference enum has non-contiguous IDs (54, 59, 60, 65, 66 are
# inserted between the regular sequence). We keep the same IDs so achievement
# bits in the state vector map 1:1 with the reference. ACH_COUNT = 67.

comptime ACH_COLLECT_WOOD: Int = 0
comptime ACH_PLACE_TABLE: Int = 1
comptime ACH_EAT_COW: Int = 2
comptime ACH_COLLECT_SAPLING: Int = 3
comptime ACH_COLLECT_DRINK: Int = 4
comptime ACH_MAKE_WOOD_PICKAXE: Int = 5
comptime ACH_MAKE_WOOD_SWORD: Int = 6
comptime ACH_PLACE_PLANT: Int = 7
comptime ACH_DEFEAT_ZOMBIE: Int = 8
comptime ACH_COLLECT_STONE: Int = 9
comptime ACH_PLACE_STONE: Int = 10
comptime ACH_EAT_PLANT: Int = 11
comptime ACH_DEFEAT_SKELETON: Int = 12
comptime ACH_MAKE_STONE_PICKAXE: Int = 13
comptime ACH_MAKE_STONE_SWORD: Int = 14
comptime ACH_WAKE_UP: Int = 15
comptime ACH_PLACE_FURNACE: Int = 16
comptime ACH_COLLECT_COAL: Int = 17
comptime ACH_COLLECT_IRON: Int = 18
comptime ACH_COLLECT_DIAMOND: Int = 19
comptime ACH_MAKE_IRON_PICKAXE: Int = 20
comptime ACH_MAKE_IRON_SWORD: Int = 21
comptime ACH_MAKE_ARROW: Int = 22
comptime ACH_MAKE_TORCH: Int = 23
comptime ACH_PLACE_TORCH: Int = 24
comptime ACH_MAKE_DIAMOND_SWORD: Int = 25
comptime ACH_MAKE_IRON_ARMOUR: Int = 26
comptime ACH_MAKE_DIAMOND_ARMOUR: Int = 27
comptime ACH_ENTER_GNOMISH_MINES: Int = 28
comptime ACH_ENTER_DUNGEON: Int = 29
comptime ACH_ENTER_SEWERS: Int = 30
comptime ACH_ENTER_VAULT: Int = 31
comptime ACH_ENTER_TROLL_MINES: Int = 32
comptime ACH_ENTER_FIRE_REALM: Int = 33
comptime ACH_ENTER_ICE_REALM: Int = 34
comptime ACH_ENTER_GRAVEYARD: Int = 35
comptime ACH_DEFEAT_GNOME_WARRIOR: Int = 36
comptime ACH_DEFEAT_GNOME_ARCHER: Int = 37
comptime ACH_DEFEAT_ORC_SOLDIER: Int = 38
comptime ACH_DEFEAT_ORC_MAGE: Int = 39
comptime ACH_DEFEAT_LIZARD: Int = 40
comptime ACH_DEFEAT_KOBOLD: Int = 41
comptime ACH_DEFEAT_TROLL: Int = 42
comptime ACH_DEFEAT_DEEP_THING: Int = 43
comptime ACH_DEFEAT_PIGMAN: Int = 44
comptime ACH_DEFEAT_FIRE_ELEMENTAL: Int = 45
comptime ACH_DEFEAT_FROST_TROLL: Int = 46
comptime ACH_DEFEAT_ICE_ELEMENTAL: Int = 47
comptime ACH_DAMAGE_NECROMANCER: Int = 48
comptime ACH_DEFEAT_NECROMANCER: Int = 49
comptime ACH_EAT_BAT: Int = 50
comptime ACH_EAT_SNAIL: Int = 51
comptime ACH_FIND_BOW: Int = 52
comptime ACH_FIRE_BOW: Int = 53
comptime ACH_COLLECT_SAPPHIRE: Int = 54
comptime ACH_LEARN_FIREBALL: Int = 55
comptime ACH_CAST_FIREBALL: Int = 56
comptime ACH_LEARN_ICEBALL: Int = 57
comptime ACH_CAST_ICEBALL: Int = 58
comptime ACH_COLLECT_RUBY: Int = 59
comptime ACH_MAKE_DIAMOND_PICKAXE: Int = 60
comptime ACH_OPEN_CHEST: Int = 61
comptime ACH_DRINK_POTION: Int = 62
comptime ACH_ENCHANT_SWORD: Int = 63
comptime ACH_ENCHANT_ARMOUR: Int = 64
comptime ACH_DEFEAT_KNIGHT: Int = 65
comptime ACH_DEFEAT_ARCHER: Int = 66

comptime NUM_ACHIEVEMENTS: Int = 67

# Reward weight per achievement tier (from `achievement_mapping`):
#   tier 1 (≤ 24)            → +1
#   intermediate / advanced  → +3
#   very advanced            → +8
#   regular                  → +5
# We store this as a Mojo helper so reward computation can lookup at runtime.

@always_inline
def achievement_reward_weight(ach_id: Int) -> Float32:
    # Tier 1: introductory progression (≤ 24).
    if ach_id <= 24:
        return Float32(1.0)
    # Very advanced: late-game / boss progression.
    if (
        ach_id == ACH_ENTER_FIRE_REALM
        or ach_id == ACH_ENTER_ICE_REALM
        or ach_id == ACH_ENTER_GRAVEYARD
        or ach_id == ACH_DEFEAT_PIGMAN
        or ach_id == ACH_DEFEAT_FIRE_ELEMENTAL
        or ach_id == ACH_DEFEAT_FROST_TROLL
        or ach_id == ACH_DEFEAT_ICE_ELEMENTAL
        or ach_id == ACH_DAMAGE_NECROMANCER
        or ach_id == ACH_DEFEAT_NECROMANCER
    ):
        return Float32(8.0)
    # Intermediate: mid-tier crafting & dungeon entry.
    if (
        ach_id == ACH_COLLECT_SAPPHIRE
        or ach_id == ACH_COLLECT_RUBY
        or ach_id == ACH_MAKE_DIAMOND_PICKAXE
        or ach_id == ACH_MAKE_DIAMOND_SWORD
        or ach_id == ACH_MAKE_IRON_ARMOUR
        or ach_id == ACH_MAKE_DIAMOND_ARMOUR
        or ach_id == ACH_ENTER_GNOMISH_MINES
        or ach_id == ACH_ENTER_DUNGEON
        or ach_id == ACH_DEFEAT_GNOME_WARRIOR
        or ach_id == ACH_DEFEAT_GNOME_ARCHER
        or ach_id == ACH_DEFEAT_ORC_SOLDIER
        or ach_id == ACH_DEFEAT_ORC_MAGE
        or ach_id == ACH_EAT_BAT
        or ach_id == ACH_EAT_SNAIL
        or ach_id == ACH_FIND_BOW
        or ach_id == ACH_FIRE_BOW
        or ach_id == ACH_OPEN_CHEST
        or ach_id == ACH_DRINK_POTION
    ):
        return Float32(3.0)
    # Default: regular advanced.
    return Float32(5.0)


# ============================================================================
# Episode / world dynamics
# ============================================================================

comptime MAX_TIMESTEPS: Int = 100_000
comptime DAY_LENGTH: Int = 300
comptime MOB_DESPAWN_DISTANCE: Int = 14
comptime PLANT_RIPEN_AGE: Int = 600

# Reward weights / mob progression.
comptime HEALTH_REWARD_WEIGHT: Float64 = 0.1
comptime MONSTERS_KILLED_TO_CLEAR_LEVEL: Int = 8
comptime BOSS_FIGHT_SPAWN_TURNS: Int = 7
comptime BOSS_FIGHT_EXTRA_DAMAGE: Float64 = 0.5

# Base spawn probabilities per step on the overworld; per-floor multipliers
# live in `floor_mob_spawn_chance()` below (matches FLOOR_MOB_SPAWN_CHANCE).
comptime SPAWN_PASSIVE_BASE: Float32 = 0.1
comptime SPAWN_MELEE_DAY_BASE: Float32 = 0.02
comptime SPAWN_RANGED_BASE: Float32 = 0.05
comptime SPAWN_MELEE_NIGHT_BASE: Float32 = 0.1


# ============================================================================
# Per-floor spawn / health / damage / collision tables (compact accessors)
# ============================================================================
#
# Reference encodes these as JAX arrays indexed by (floor, mob_cat). We expose
# the same semantics via @always_inline helpers — the call sites become
# branches at compile time when the floor is known and a small lookup
# otherwise.

@always_inline
def floor_mob_spawn_chance(floor: Int, cat: Int) -> Float32:
    """Spawn probability per step. cat ∈ {0=passive, 1=melee, 2=ranged, 3=melee-night}."""
    if cat == 0:
        # Passive: 0 on the ice realm, 0.1 everywhere else.
        if floor == FLOOR_ICE:
            return Float32(0.0)
        return SPAWN_PASSIVE_BASE
    elif cat == 1:
        # Melee (day): overworld uses 0.02, dungeons use 0.06.
        if floor == FLOOR_OVERWORLD:
            return SPAWN_MELEE_DAY_BASE
        return Float32(0.06)
    elif cat == 2:
        return SPAWN_RANGED_BASE
    else:  # cat == 3: melee-night override only on overworld
        if floor == FLOOR_OVERWORLD:
            return SPAWN_MELEE_NIGHT_BASE
        return Float32(0.0)


@always_inline
def floor_mob_health(floor: Int, cat: Int) -> Int:
    """Mob HP per (floor, cat). Mirrors MOB_TYPE_HEALTH_MAPPING."""
    # Hand-baked from reference table.
    if floor == 0:
        if cat == 0: return 3   # cow
        if cat == 1: return 5   # zombie
        if cat == 2: return 3   # skeleton
        return 0
    if floor == 1:
        if cat == 0: return 4
        if cat == 1: return 7
        if cat == 2: return 5
        return 0
    if floor == 2:
        if cat == 0: return 6
        if cat == 1: return 9
        if cat == 2: return 6
        return 0
    if floor == 3:
        if cat == 0: return 8
        if cat == 1: return 11
        if cat == 2: return 8
        return 0
    if floor == 4:
        if cat == 0: return 0
        if cat == 1: return 12
        if cat == 2: return 12
        return 0
    if floor == 5:
        if cat == 0: return 0
        if cat == 1: return 20
        if cat == 2: return 4
        return 0
    if floor == 6:
        if cat == 0: return 0
        if cat == 1: return 20
        if cat == 2: return 14
        return 0
    if floor == 7:
        if cat == 0: return 0
        if cat == 1: return 24
        if cat == 2: return 16
        return 0
    # Floor 8: boss — no regular mob spawns.
    return 0


# Damage table — what a hit from this (cat, species) deals to (HP, food/drink, energy/mana).
# Index by species_id and channel. Returns (hp, food, drink) damage triplet.
# Channels follow the reference: index 0 = HP, 1 = food, 2 = drink/mana.
# Mirrors MOB_TYPE_DAMAGE_MAPPING.

@always_inline
def melee_damage(species_id: Int) -> Tuple[Int, Int, Int]:
    if species_id == 0: return (2, 0, 0)  # zombie
    if species_id == 1: return (4, 0, 0)  # gnome warrior
    if species_id == 2: return (3, 0, 0)  # orc soldier
    if species_id == 3: return (5, 0, 0)  # lizard
    if species_id == 4: return (6, 0, 0)  # knight
    if species_id == 5: return (6, 1, 1)  # troll
    if species_id == 6: return (3, 5, 0)  # pigman
    if species_id == 7: return (4, 0, 5)  # ice troll
    return (0, 0, 0)


@always_inline
def projectile_damage(species_id: Int) -> Tuple[Int, Int, Int]:
    if species_id == PROJ_ARROW:     return (2, 0, 0)
    if species_id == PROJ_DAGGER:    return (4, 0, 0)
    if species_id == PROJ_FIREBALL:  return (0, 3, 0)
    if species_id == PROJ_ICEBALL:   return (0, 0, 3)
    if species_id == PROJ_ARROW2:    return (5, 0, 0)
    if species_id == PROJ_SLIMEBALL: return (4, 3, 3)
    if species_id == PROJ_FIREBALL2: return (3, 5, 0)
    if species_id == PROJ_ICEBALL2:  return (4, 0, 5)
    return (0, 0, 0)


# Defense (fractional resistance per channel). Returns (hp_def, food_def, drink_def).
# Mirrors MOB_TYPE_DEFENSE_MAPPING for melee/ranged categories.
@always_inline
def floor_mob_defense(
    floor: Int, cat: Int
) -> Tuple[Float32, Float32, Float32]:
    if floor == 4 and (cat == 1 or cat == 2):
        return (Float32(0.5), Float32(0.0), Float32(0.0))
    if floor == 5 and cat == 1:
        return (Float32(0.2), Float32(0.0), Float32(0.0))
    if floor == 6 and (cat == 1 or cat == 2):
        return (Float32(0.9), Float32(1.0), Float32(0.0))
    if floor == 7 and (cat == 1 or cat == 2):
        return (Float32(0.9), Float32(0.0), Float32(1.0))
    return (Float32(0.0), Float32(0.0), Float32(0.0))


# ============================================================================
# Damage tables for player tools
# ============================================================================
#
# DAMAGE_FIST / DAMAGE_<TIER>_<TOOL> tables mirror Classic but extended.

comptime DAMAGE_FIST: Int = 1
comptime DAMAGE_WOOD_SWORD: Int = 2
comptime DAMAGE_STONE_SWORD: Int = 3
comptime DAMAGE_IRON_SWORD: Int = 5
comptime DAMAGE_DIAMOND_SWORD: Int = 8
comptime DAMAGE_BOW: Int = 3

# Pickaxe tier required to mine specific block types.
@always_inline
def required_pickaxe_tier(block_id: Int) -> Int:
    """0=fist, 1=wood, 2=stone, 3=iron, 4=diamond."""
    if block_id == BLOCK_STONE or block_id == BLOCK_COAL:
        return 1
    if block_id == BLOCK_IRON or block_id == BLOCK_WALL or block_id == BLOCK_WALL_MOSS:
        return 2
    if block_id == BLOCK_DIAMOND or block_id == BLOCK_SAPPHIRE or block_id == BLOCK_RUBY:
        return 3
    return 0


# ============================================================================
# Mob attack cooldowns
# ============================================================================

comptime MELEE_ATTACK_COOLDOWN: Int = 5
comptime RANGED_ATTACK_COOLDOWN: Int = 4


# ============================================================================
# Intrinsic accumulator thresholds
# ============================================================================

comptime HUNGER_THRESHOLD: Float32 = 25.0
comptime THIRST_THRESHOLD: Float32 = 20.0
comptime FATIGUE_HIGH_THRESHOLD: Float32 = 30.0
comptime FATIGUE_LOW_THRESHOLD: Float32 = -10.0
comptime RECOVER_HIGH_THRESHOLD: Float32 = 25.0
comptime RECOVER_LOW_THRESHOLD: Float32 = -15.0
comptime RECOVER_MANA_THRESHOLD: Float32 = 25.0

comptime COW_EAT_BOOST: Int = 6
comptime BAT_EAT_BOOST: Int = 6
comptime SNAIL_EAT_BOOST: Int = 6
comptime PLANT_EAT_BOOST: Int = 4
comptime WATER_DRINK_BOOST: Int = 1
comptime FOUNTAIN_DRINK_BOOST: Int = 5
comptime POTION_HEAL: Int = 4

comptime SAPLING_DROP_CHANCE: Float32 = 0.1


# ============================================================================
# Spells / mana
# ============================================================================

comptime SPELL_FIREBALL: Int = 0
comptime SPELL_ICEBALL: Int = 1
comptime NUM_SPELLS: Int = 2

comptime MANA_COST_FIREBALL: Int = 1
comptime MANA_COST_ICEBALL: Int = 1


# ============================================================================
# Observation shape (derived)
# ============================================================================

# Tile encoding for symbolic obs: block one-hot + item one-hot + 4 mob-cat
# binary presence channels.
comptime TILE_CHANNELS: Int = (
    NUM_BLOCK_TYPES + NUM_ITEM_TYPES + NUM_MOB_CATEGORIES
)  # 37 + 5 + 4 = 46
comptime OBS_VIEW_SIZE: Int = VIEW_SIZE * TILE_CHANNELS  # 99 * 46 = 4554

# Scalar tail of the symbolic obs (matches the order used by the renderer/test
# fixtures): inventory (24) + intrinsics int (7) + intrinsics_f (5) +
# attributes (4) + direction one-hot (4) + light_level + sleep flag +
# floor_level (one-hot of NUM_FLOORS) + boss_progress + boss_timestep +
# learned_spells (NUM_SPELLS) + sword_enchant + bow_enchant +
# armour_enchants (4) + monsters_killed (NUM_FLOORS) + chests_opened (NUM_FLOORS).
comptime OBS_SCALAR_SIZE: Int = (
    NUM_INVENTORY      # 24
    + NUM_INTRINSICS   # 7
    + NUM_INTRINSICS_F # 5
    + NUM_ATTRIBUTES   # 4
    + NUM_DIRECTIONS   # 4
    + 2                # light_level, is_sleeping
    + NUM_FLOORS       # floor one-hot
    + 2                # boss_progress, boss_timesteps_to_spawn
    + NUM_SPELLS       # learned spells (2)
    + 1 + 1            # sword_enchant, bow_enchant
    + NUM_ARMOUR_ENCHANTS  # 4
    + NUM_FLOORS       # monsters_killed
    + NUM_FLOORS       # chests_opened
)

comptime OBS_DIM: Int = OBS_VIEW_SIZE + OBS_SCALAR_SIZE
