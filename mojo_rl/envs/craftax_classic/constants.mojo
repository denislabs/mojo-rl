"""Compile-time constants for Craftax-Classic.

Mirrors `references/Craftax-main/craftax/craftax_classic/constants.py`.
See `docs/CRAFTAX_PORT.md` for the overall design.
"""

# ============================================================================
# Map
# ============================================================================

comptime MAP_H: Int = 64
comptime MAP_W: Int = 64
comptime MAP_SIZE: Int = MAP_H * MAP_W  # 4096 tiles

# Agent's local view window (centered on player)
comptime VIEW_H: Int = 7
comptime VIEW_W: Int = 9
comptime VIEW_SIZE: Int = VIEW_H * VIEW_W  # 63 tiles


# ============================================================================
# Block types — match BlockType enum in reference constants.py
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

comptime NUM_BLOCK_TYPES: Int = 17


# ============================================================================
# Mobs
# ============================================================================

# Channel order used in observation tile encoding.
comptime MOB_CH_ZOMBIE: Int = 0
comptime MOB_CH_COW: Int = 1
comptime MOB_CH_SKELETON: Int = 2
comptime MOB_CH_ARROW: Int = 3
comptime NUM_MOB_CHANNELS: Int = 4

# Max simultaneous entities per env (matches StaticEnvParams in reference).
comptime MAX_ZOMBIES: Int = 3
comptime MAX_COWS: Int = 3
comptime MAX_SKELETONS: Int = 2
comptime MAX_ARROWS: Int = 3

# Per-mob fields. hp=0 ⇒ dead (no separate mask).
comptime MOB_FIELDS: Int = 4
comptime MOB_FY: Int = 0  # y (row index in map)
comptime MOB_FX: Int = 1  # x (col index)
comptime MOB_HP: Int = 2
comptime MOB_CD: Int = 3  # attack cooldown

# Arrows carry an extra direction field.
comptime ARROW_FIELDS: Int = 5
comptime ARROW_FDIR: Int = 4  # direction code (0..4)


# ============================================================================
# Plants
# ============================================================================

comptime MAX_PLANTS: Int = 10
comptime PLANT_FIELDS: Int = 3
comptime PLANT_FY: Int = 0
comptime PLANT_FX: Int = 1
comptime PLANT_FAGE: Int = 2


# ============================================================================
# Inventory
# ============================================================================

# Order matches the symbolic obs layout in obs_description.md (Classic subset).
comptime INV_WOOD: Int = 0
comptime INV_STONE: Int = 1
comptime INV_COAL: Int = 2
comptime INV_IRON: Int = 3
comptime INV_DIAMOND: Int = 4
comptime INV_SAPLING: Int = 5
comptime INV_WOOD_PICKAXE: Int = 6
comptime INV_STONE_PICKAXE: Int = 7
comptime INV_IRON_PICKAXE: Int = 8
comptime INV_WOOD_SWORD: Int = 9
comptime INV_STONE_SWORD: Int = 10
comptime INV_IRON_SWORD: Int = 11
comptime NUM_INVENTORY: Int = 12

comptime INV_MAX_PER_SLOT: Int = 9  # reference caps each item slot at 9


# ============================================================================
# Player intrinsics
# ============================================================================

# Integer intrinsics in [0..9]: health, food, drink, energy.
comptime INTRINSIC_HEALTH: Int = 0
comptime INTRINSIC_FOOD: Int = 1
comptime INTRINSIC_DRINK: Int = 2
comptime INTRINSIC_ENERGY: Int = 3
comptime NUM_INTRINSICS: Int = 4

# Float accumulator intrinsics: recover, hunger, thirst, fatigue.
comptime INTRINSIC_F_RECOVER: Int = 0
comptime INTRINSIC_F_HUNGER: Int = 1
comptime INTRINSIC_F_THIRST: Int = 2
comptime INTRINSIC_F_FATIGUE: Int = 3
comptime NUM_INTRINSICS_F: Int = 4

comptime INTRINSIC_MAX: Int = 9
comptime PLAYER_MAX_HEALTH: Int = 9


# ============================================================================
# Direction
# ============================================================================

comptime DIR_LEFT: Int = 0
comptime DIR_RIGHT: Int = 1
comptime DIR_UP: Int = 2
comptime DIR_DOWN: Int = 3
comptime NUM_DIRECTIONS: Int = 4


# ============================================================================
# Actions — match Action enum in reference constants.py
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

comptime NUM_ACTIONS: Int = 17


# ============================================================================
# Achievements (22 total) — match Achievement enum in reference
# ============================================================================

comptime ACH_COLLECT_WOOD: Int = 0
comptime ACH_PLACE_TABLE: Int = 1
comptime ACH_EAT_COW: Int = 2
comptime ACH_COLLECT_SAPLING: Int = 3
comptime ACH_COLLECT_DRINK: Int = 4
comptime ACH_MAKE_WOOD_PICKAXE: Int = 5
comptime ACH_MAKE_STONE_PICKAXE: Int = 6
comptime ACH_MAKE_IRON_PICKAXE: Int = 7
comptime ACH_MAKE_WOOD_SWORD: Int = 8
comptime ACH_MAKE_STONE_SWORD: Int = 9
comptime ACH_MAKE_IRON_SWORD: Int = 10
comptime ACH_PLACE_PLANT: Int = 11
comptime ACH_DEFEAT_ZOMBIE: Int = 12
comptime ACH_COLLECT_STONE: Int = 13
comptime ACH_PLACE_STONE: Int = 14
comptime ACH_EAT_PLANT: Int = 15
comptime ACH_DEFEAT_SKELETON: Int = 16
comptime ACH_COLLECT_IRON: Int = 17
comptime ACH_COLLECT_COAL: Int = 18
comptime ACH_PLACE_FURNACE: Int = 19
comptime ACH_COLLECT_DIAMOND: Int = 20
comptime ACH_WAKE_UP: Int = 21

comptime NUM_ACHIEVEMENTS: Int = 22


# ============================================================================
# Episode / world dynamics
# ============================================================================

comptime MAX_TIMESTEPS: Int = 10_000
comptime DAY_LENGTH: Int = 300
comptime MOB_DESPAWN_DISTANCE: Int = 14
comptime PLANT_RIPEN_AGE: Int = 600

# Reward weights
comptime HEALTH_REWARD_WEIGHT: Float64 = 0.1

# Mob hit points (match reference EnvParams).
comptime ZOMBIE_HEALTH: Int = 5
comptime COW_HEALTH: Int = 3
comptime SKELETON_HEALTH: Int = 3

# Spawn probabilities per step (match reference EnvParams).
comptime SPAWN_COW_CHANCE: Float32 = 0.1
comptime SPAWN_ZOMBIE_BASE_CHANCE: Float32 = 0.02
comptime SPAWN_ZOMBIE_NIGHT_CHANCE: Float32 = 0.1
comptime SPAWN_SKELETON_CHANCE: Float32 = 0.05

# Damage tables (match reference do_action).
comptime DAMAGE_FIST: Int = 1
comptime DAMAGE_WOOD_SWORD: Int = 2
comptime DAMAGE_STONE_SWORD: Int = 3
comptime DAMAGE_IRON_SWORD: Int = 5

# Mob attack cooldowns (match reference update_mobs).
comptime ZOMBIE_ATTACK_DAMAGE: Int = 2
comptime ZOMBIE_ATTACK_DAMAGE_SLEEP: Int = 7
comptime ZOMBIE_ATTACK_COOLDOWN: Int = 5
comptime SKELETON_ATTACK_COOLDOWN: Int = 4
comptime ARROW_DAMAGE: Int = 2

# Intrinsic accumulator thresholds (match reference update_player_intrinsics).
comptime HUNGER_THRESHOLD: Float32 = 25.0
comptime THIRST_THRESHOLD: Float32 = 20.0
comptime FATIGUE_HIGH_THRESHOLD: Float32 = 30.0
comptime FATIGUE_LOW_THRESHOLD: Float32 = -10.0
comptime RECOVER_HIGH_THRESHOLD: Float32 = 25.0
comptime RECOVER_LOW_THRESHOLD: Float32 = -15.0

# Eat / drink boosts.
comptime COW_EAT_BOOST: Int = 6
comptime PLANT_EAT_BOOST: Int = 4
comptime WATER_DRINK_BOOST: Int = 1

# Misc gameplay.
comptime SAPLING_DROP_CHANCE: Float32 = 0.1
comptime ZOMBIE_CHASE_PROB: Float32 = 0.75
comptime ZOMBIE_CHASE_RANGE: Int = 10
comptime SKELETON_RANDOM_OVERRIDE: Float32 = 0.85  # uniform > this → random
comptime SKELETON_FLEE_RANGE: Int = 3
comptime SKELETON_RANGE_MIN: Int = 10
comptime SKELETON_FIRE_MIN: Int = 4
comptime SKELETON_FIRE_MAX: Int = 5


# ============================================================================
# Observation shape (derived)
# ============================================================================

# Tile encoding: NUM_BLOCK_TYPES one-hot + NUM_MOB_CHANNELS binary presence.
comptime TILE_CHANNELS: Int = NUM_BLOCK_TYPES + NUM_MOB_CHANNELS  # 21
comptime OBS_VIEW_SIZE: Int = VIEW_SIZE * TILE_CHANNELS  # 63 * 21 = 1323

# Symbolic obs layout:
#   [0                          : OBS_VIEW_SIZE)        local view (one-hot blocks + mobs)
#   [OBS_VIEW_SIZE              : +NUM_INVENTORY)       inventory (normalized)
#   [+NUM_INVENTORY             : +NUM_INTRINSICS)      intrinsics / 9
#   [+NUM_INTRINSICS            : +NUM_DIRECTIONS)      direction one-hot
#   [+NUM_DIRECTIONS            : +2)                   light_level, is_sleeping
comptime OBS_DIM: Int = (
    OBS_VIEW_SIZE + NUM_INVENTORY + NUM_INTRINSICS + NUM_DIRECTIONS + 2
)  # 1345
