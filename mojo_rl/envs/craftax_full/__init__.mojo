"""Full Craftax — 9-floor dungeon variant (37 blocks / 43 actions / 67 achievements).

Mojo port of `craftax/craftax/` from the Craftax JAX benchmark
(Matthews et al. 2024, ICML). See `docs/CRAFTAX_PORT.md` §5bis.

Phase 7A: constants + flat state layout only. World gen / game logic / obs
come in 7B–7D.
"""

from .constants import (
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
    NUM_INTRINSICS,
    NUM_INTRINSICS_F,
    NUM_ATTRIBUTES,
    NUM_DIRECTIONS,
    NUM_SPELLS,
    NUM_ACHIEVEMENTS,
    NUM_POTIONS,
    NUM_ARMOUR_ENCHANTS,
    OBS_VIEW_SIZE,
    OBS_SCALAR_SIZE,
    OBS_DIM,
    TILE_CHANNELS,
)
from .state import STATE_SIZE
from .world_gen_configs import (
    SmoothGenConfig,
    DungeonConfig,
    overworld_config,
    gnomish_mines_config,
    troll_mines_config,
    fire_level_config,
    ice_level_config,
    boss_level_config,
    dungeon_config,
    sewer_config,
    vaults_config,
)
from .world_gen import (
    generate_full_world,
    calculate_light_level,
)
