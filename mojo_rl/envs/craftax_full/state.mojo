"""Flat state layout for Full Craftax.

State is one big `InlineArray[Scalar[dtype], STATE_SIZE]` per env, with the
sections laid out in the order below. Offsets are pure compile-time arithmetic
so kernels specialize on them.

Sections (in order):
  - map               9 × 48 × 48                 block IDs (float-encoded)
  - item_map          9 × 48 × 48                 items per tile (torch/ladder)
  - mob_map           9 × 48 × 48                 packed mob-cat IDs per tile
  - light_map         9 × 48 × 48                 per-tile lighting [0..1]
  - down_ladders      9 × 2                       ladder-down (y, x) per floor
  - up_ladders        9 × 2                       ladder-up (y, x) per floor
  - chests_opened     9                           bool per floor
  - monsters_killed   9                           per-floor counter
  - player_pos        2                           (y, x)
  - player_level      1                           current floor index
  - player_dir        1
  - intrinsics_i      NUM_INTRINSICS              HP/food/drink/energy/mana
                                                  + is_sleeping/is_resting
  - intrinsics_f      NUM_INTRINSICS_F            float accumulators
  - attributes        NUM_ATTRIBUTES              xp + dex/str/intel
  - inventory         NUM_INVENTORY               24 slots (see constants)
  - melee_mobs        9 × MAX_MELEE × MOB_FIELDS
  - passive_mobs      9 × MAX_PASSIVE × MOB_FIELDS
  - ranged_mobs       9 × MAX_RANGED × MOB_FIELDS
  - mob_projectiles   9 × MAX_MOB_PROJ × PROJ_FIELDS
  - player_proj       9 × MAX_PLAYER_PROJ × PROJ_FIELDS
  - plants            MAX_GROWING_PLANTS × PLANT_FIELDS
  - plant_mask        MAX_GROWING_PLANTS
  - potion_mapping    NUM_POTIONS                 color → effect (per-run perm)
  - learned_spells    NUM_SPELLS                  bool per spell
  - sword_enchant     1                           {0,1,2}
  - bow_enchant       1
  - armour_enchants   NUM_ARMOUR_ENCHANTS = 4
  - boss_progress     1
  - boss_timesteps    1
  - light_level       1
  - achievements      NUM_ACHIEVEMENTS = 67
  - timestep          1
  - rng               4                            Philox counter words

Total: STATE_SIZE compile-time constant — see bottom of file.
"""

from .constants import (
    MAP_W,
    MAP_SIZE_PER_FLOOR,
    NUM_FLOORS,
    MAP_TOTAL_SIZE,
    NUM_INTRINSICS,
    NUM_INTRINSICS_F,
    NUM_ATTRIBUTES,
    NUM_INVENTORY,
    MAX_MELEE_MOBS,
    MAX_PASSIVE_MOBS,
    MAX_RANGED_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PLAYER_PROJECTILES,
    MAX_GROWING_PLANTS,
    MOB_FIELDS,
    PROJ_FIELDS,
    PLANT_FIELDS,
    NUM_POTIONS,
    NUM_SPELLS,
    NUM_ARMOUR_ENCHANTS,
    NUM_ACHIEVEMENTS,
)


# ============================================================================
# Section base offsets — all comptime
# ============================================================================

comptime S_MAP_BASE: Int = 0
comptime S_ITEM_MAP_BASE: Int = S_MAP_BASE + MAP_TOTAL_SIZE
comptime S_MOB_MAP_BASE: Int = S_ITEM_MAP_BASE + MAP_TOTAL_SIZE
comptime S_LIGHT_MAP_BASE: Int = S_MOB_MAP_BASE + MAP_TOTAL_SIZE

comptime S_DOWN_LADDERS_BASE: Int = S_LIGHT_MAP_BASE + MAP_TOTAL_SIZE
comptime S_UP_LADDERS_BASE: Int = S_DOWN_LADDERS_BASE + NUM_FLOORS * 2
comptime S_CHESTS_OPENED_BASE: Int = S_UP_LADDERS_BASE + NUM_FLOORS * 2
comptime S_MONSTERS_KILLED_BASE: Int = S_CHESTS_OPENED_BASE + NUM_FLOORS

comptime S_PLAYER_POS: Int = S_MONSTERS_KILLED_BASE + NUM_FLOORS
comptime S_PLAYER_LEVEL: Int = S_PLAYER_POS + 2
comptime S_PLAYER_DIR: Int = S_PLAYER_LEVEL + 1

comptime S_INTRINSICS_BASE: Int = S_PLAYER_DIR + 1
comptime S_INTRINSICS_F_BASE: Int = S_INTRINSICS_BASE + NUM_INTRINSICS
comptime S_ATTRIBUTES_BASE: Int = S_INTRINSICS_F_BASE + NUM_INTRINSICS_F

comptime S_INV_BASE: Int = S_ATTRIBUTES_BASE + NUM_ATTRIBUTES

comptime S_MELEE_MOBS_BASE: Int = S_INV_BASE + NUM_INVENTORY
comptime S_PASSIVE_MOBS_BASE: Int = (
    S_MELEE_MOBS_BASE + NUM_FLOORS * MAX_MELEE_MOBS * MOB_FIELDS
)
comptime S_RANGED_MOBS_BASE: Int = (
    S_PASSIVE_MOBS_BASE + NUM_FLOORS * MAX_PASSIVE_MOBS * MOB_FIELDS
)
comptime S_MOB_PROJECTILES_BASE: Int = (
    S_RANGED_MOBS_BASE + NUM_FLOORS * MAX_RANGED_MOBS * MOB_FIELDS
)
comptime S_PLAYER_PROJECTILES_BASE: Int = (
    S_MOB_PROJECTILES_BASE + NUM_FLOORS * MAX_MOB_PROJECTILES * PROJ_FIELDS
)

comptime S_PLANTS_BASE: Int = (
    S_PLAYER_PROJECTILES_BASE + NUM_FLOORS * MAX_PLAYER_PROJECTILES * PROJ_FIELDS
)
comptime S_PLANT_MASK_BASE: Int = (
    S_PLANTS_BASE + MAX_GROWING_PLANTS * PLANT_FIELDS
)

comptime S_POTION_MAPPING_BASE: Int = S_PLANT_MASK_BASE + MAX_GROWING_PLANTS
comptime S_LEARNED_SPELLS_BASE: Int = S_POTION_MAPPING_BASE + NUM_POTIONS

comptime S_SWORD_ENCHANT: Int = S_LEARNED_SPELLS_BASE + NUM_SPELLS
comptime S_BOW_ENCHANT: Int = S_SWORD_ENCHANT + 1
comptime S_ARMOUR_ENCHANTS_BASE: Int = S_BOW_ENCHANT + 1

comptime S_BOSS_PROGRESS: Int = S_ARMOUR_ENCHANTS_BASE + NUM_ARMOUR_ENCHANTS
comptime S_BOSS_TIMESTEPS: Int = S_BOSS_PROGRESS + 1

comptime S_LIGHT_LEVEL: Int = S_BOSS_TIMESTEPS + 1
comptime S_ACHIEVEMENTS_BASE: Int = S_LIGHT_LEVEL + 1
comptime S_TIMESTEP: Int = S_ACHIEVEMENTS_BASE + NUM_ACHIEVEMENTS

comptime S_RNG_BASE: Int = S_TIMESTEP + 1
comptime S_RNG_WORDS: Int = 4

comptime STATE_SIZE: Int = S_RNG_BASE + S_RNG_WORDS


# ============================================================================
# Compile-time field-level accessors (pure offset arithmetic)
# ============================================================================
#
# Map indexing is (floor, y, x); we always do `floor * MAP_SIZE_PER_FLOOR +
# y * MAP_W + x` so floor 0 is contiguous, then floor 1, etc.
# Same convention for item_map / mob_map / light_map.

@always_inline
def s_map(floor: Int, y: Int, x: Int) -> Int:
    return S_MAP_BASE + floor * MAP_SIZE_PER_FLOOR + y * MAP_W + x


@always_inline
def s_item_map(floor: Int, y: Int, x: Int) -> Int:
    return S_ITEM_MAP_BASE + floor * MAP_SIZE_PER_FLOOR + y * MAP_W + x


@always_inline
def s_mob_map(floor: Int, y: Int, x: Int) -> Int:
    return S_MOB_MAP_BASE + floor * MAP_SIZE_PER_FLOOR + y * MAP_W + x


@always_inline
def s_light_map(floor: Int, y: Int, x: Int) -> Int:
    return S_LIGHT_MAP_BASE + floor * MAP_SIZE_PER_FLOOR + y * MAP_W + x


@always_inline
def s_down_ladder(floor: Int, axis: Int) -> Int:
    """axis: 0 = y, 1 = x."""
    return S_DOWN_LADDERS_BASE + floor * 2 + axis


@always_inline
def s_up_ladder(floor: Int, axis: Int) -> Int:
    return S_UP_LADDERS_BASE + floor * 2 + axis


@always_inline
def s_chest_opened(floor: Int) -> Int:
    return S_CHESTS_OPENED_BASE + floor


@always_inline
def s_monsters_killed(floor: Int) -> Int:
    return S_MONSTERS_KILLED_BASE + floor


@always_inline
def s_intrinsic(slot: Int) -> Int:
    return S_INTRINSICS_BASE + slot


@always_inline
def s_intrinsic_f(slot: Int) -> Int:
    return S_INTRINSICS_F_BASE + slot


@always_inline
def s_attribute(slot: Int) -> Int:
    return S_ATTRIBUTES_BASE + slot


@always_inline
def s_inv(slot: Int) -> Int:
    return S_INV_BASE + slot


@always_inline
def s_melee_mob(floor: Int, i: Int, field: Int) -> Int:
    return (
        S_MELEE_MOBS_BASE
        + floor * MAX_MELEE_MOBS * MOB_FIELDS
        + i * MOB_FIELDS
        + field
    )


@always_inline
def s_passive_mob(floor: Int, i: Int, field: Int) -> Int:
    return (
        S_PASSIVE_MOBS_BASE
        + floor * MAX_PASSIVE_MOBS * MOB_FIELDS
        + i * MOB_FIELDS
        + field
    )


@always_inline
def s_ranged_mob(floor: Int, i: Int, field: Int) -> Int:
    return (
        S_RANGED_MOBS_BASE
        + floor * MAX_RANGED_MOBS * MOB_FIELDS
        + i * MOB_FIELDS
        + field
    )


@always_inline
def s_mob_projectile(floor: Int, i: Int, field: Int) -> Int:
    return (
        S_MOB_PROJECTILES_BASE
        + floor * MAX_MOB_PROJECTILES * PROJ_FIELDS
        + i * PROJ_FIELDS
        + field
    )


@always_inline
def s_player_projectile(floor: Int, i: Int, field: Int) -> Int:
    return (
        S_PLAYER_PROJECTILES_BASE
        + floor * MAX_PLAYER_PROJECTILES * PROJ_FIELDS
        + i * PROJ_FIELDS
        + field
    )


@always_inline
def s_plant(i: Int, field: Int) -> Int:
    return S_PLANTS_BASE + i * PLANT_FIELDS + field


@always_inline
def s_plant_mask(i: Int) -> Int:
    return S_PLANT_MASK_BASE + i


@always_inline
def s_potion_mapping(slot: Int) -> Int:
    return S_POTION_MAPPING_BASE + slot


@always_inline
def s_learned_spell(slot: Int) -> Int:
    return S_LEARNED_SPELLS_BASE + slot


@always_inline
def s_armour_enchant(slot: Int) -> Int:
    return S_ARMOUR_ENCHANTS_BASE + slot


@always_inline
def s_achievement(i: Int) -> Int:
    return S_ACHIEVEMENTS_BASE + i


@always_inline
def s_rng(word: Int) -> Int:
    return S_RNG_BASE + word
