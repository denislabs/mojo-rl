"""State layout for Craftax-Classic.

State is a flat array of Scalar[dtype] per env, indexed by the offsets below.
The whole layout is compile-time-constant so kernels can specialize.

Sections, in order:
  - map           MAP_SIZE                 block IDs (float-encoded)
  - player_pos    2                        px, py
  - player_dir    1                        0..3
  - intrinsics_i  NUM_INTRINSICS           health, food, drink, energy (0..9)
  - intrinsics_f  NUM_INTRINSICS_F         recover, hunger, thirst, fatigue
  - inventory     NUM_INVENTORY            see constants.mojo
  - zombies       MAX_ZOMBIES * MOB_FIELDS
  - cows          MAX_COWS * MOB_FIELDS
  - skeletons     MAX_SKELETONS * MOB_FIELDS
  - arrows        MAX_ARROWS * ARROW_FIELDS
  - plants        MAX_PLANTS * PLANT_FIELDS
  - plant_mask    MAX_PLANTS
  - achievements  NUM_ACHIEVEMENTS
  - misc          3                        light_level, is_sleeping, timestep
  - rng           4                        Philox counter (32-bit words)

The total compile-time size lives in STATE_SIZE.
"""

from .constants import (
    MAP_W,
    MAP_SIZE,
    NUM_INTRINSICS,
    NUM_INTRINSICS_F,
    NUM_INVENTORY,
    MAX_ZOMBIES,
    MAX_COWS,
    MAX_SKELETONS,
    MAX_ARROWS,
    MAX_PLANTS,
    MOB_FIELDS,
    ARROW_FIELDS,
    PLANT_FIELDS,
    NUM_ACHIEVEMENTS,
)


# ============================================================================
# Section base offsets
# ============================================================================

comptime S_MAP_BASE: Int = 0

comptime S_PLAYER_POS: Int = S_MAP_BASE + MAP_SIZE  # 4096
comptime S_PLAYER_DIR: Int = S_PLAYER_POS + 2

comptime S_INTRINSICS_BASE: Int = S_PLAYER_DIR + 1
comptime S_INTRINSICS_F_BASE: Int = S_INTRINSICS_BASE + NUM_INTRINSICS

comptime S_INV_BASE: Int = S_INTRINSICS_F_BASE + NUM_INTRINSICS_F

comptime S_ZOMBIES_BASE: Int = S_INV_BASE + NUM_INVENTORY
comptime S_COWS_BASE: Int = S_ZOMBIES_BASE + MAX_ZOMBIES * MOB_FIELDS
comptime S_SKELETONS_BASE: Int = S_COWS_BASE + MAX_COWS * MOB_FIELDS
comptime S_ARROWS_BASE: Int = S_SKELETONS_BASE + MAX_SKELETONS * MOB_FIELDS

comptime S_PLANTS_BASE: Int = S_ARROWS_BASE + MAX_ARROWS * ARROW_FIELDS
comptime S_PLANT_MASK_BASE: Int = S_PLANTS_BASE + MAX_PLANTS * PLANT_FIELDS

comptime S_ACHIEVEMENTS_BASE: Int = S_PLANT_MASK_BASE + MAX_PLANTS

comptime S_LIGHT_LEVEL: Int = S_ACHIEVEMENTS_BASE + NUM_ACHIEVEMENTS
comptime S_IS_SLEEPING: Int = S_LIGHT_LEVEL + 1
comptime S_TIMESTEP: Int = S_IS_SLEEPING + 1

comptime S_RNG_BASE: Int = S_TIMESTEP + 1
comptime S_RNG_WORDS: Int = 4

comptime STATE_SIZE: Int = S_RNG_BASE + S_RNG_WORDS


# ============================================================================
# Field-level accessors (compile-time helpers)
# ============================================================================
#
# These are pure `comptime` arithmetic helpers — call sites resolve to
# constants. Use them everywhere instead of hand-computed offsets so the
# layout stays in one place.

@always_inline
def s_map(y: Int, x: Int) -> Int:
    """Index into the map for tile (y, x). Row-major: y * MAP_W + x."""
    return S_MAP_BASE + y * MAP_W + x


@always_inline
def s_inv(slot: Int) -> Int:
    """Index of inventory slot `slot` (0..NUM_INVENTORY-1)."""
    return S_INV_BASE + slot


@always_inline
def s_intrinsic(slot: Int) -> Int:
    """Index of integer intrinsic `slot` (0..NUM_INTRINSICS-1)."""
    return S_INTRINSICS_BASE + slot


@always_inline
def s_intrinsic_f(slot: Int) -> Int:
    """Index of float intrinsic `slot` (0..NUM_INTRINSICS_F-1)."""
    return S_INTRINSICS_F_BASE + slot


@always_inline
def s_zombie(i: Int, field: Int) -> Int:
    """Index of field `field` of zombie `i`."""
    return S_ZOMBIES_BASE + i * MOB_FIELDS + field


@always_inline
def s_cow(i: Int, field: Int) -> Int:
    return S_COWS_BASE + i * MOB_FIELDS + field


@always_inline
def s_skeleton(i: Int, field: Int) -> Int:
    return S_SKELETONS_BASE + i * MOB_FIELDS + field


@always_inline
def s_arrow(i: Int, field: Int) -> Int:
    return S_ARROWS_BASE + i * ARROW_FIELDS + field


@always_inline
def s_plant(i: Int, field: Int) -> Int:
    return S_PLANTS_BASE + i * PLANT_FIELDS + field


@always_inline
def s_plant_mask(i: Int) -> Int:
    return S_PLANT_MASK_BASE + i


@always_inline
def s_achievement(i: Int) -> Int:
    return S_ACHIEVEMENTS_BASE + i


@always_inline
def s_rng(word: Int) -> Int:
    return S_RNG_BASE + word
