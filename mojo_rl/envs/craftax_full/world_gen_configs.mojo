"""World-generation configs for Full Craftax.

Ports `references/Craftax-main/craftax/craftax/world_gen/world_gen_configs.py`.

Six SmoothGen configs (overworld, gnomish mines, troll mines, fire, ice,
boss) and three Dungeon configs (dungeon, sewer, vaults). The reference
splices them in the order:
    floor 0: SMOOTH[0] = OVERWORLD
    floor 1: DUNG[0]   = DUNGEON
    floor 2: SMOOTH[1] = GNOMISH_MINES
    floor 3: DUNG[1]   = SEWER
    floor 4: DUNG[2]   = VAULTS
    floor 5: SMOOTH[2] = TROLL_MINES
    floor 6: SMOOTH[3] = FIRE
    floor 7: SMOOTH[4] = ICE
    floor 8: SMOOTH[5] = BOSS
"""

from .constants import (
    BLOCK_GRASS,
    BLOCK_WATER,
    BLOCK_STONE,
    BLOCK_TREE,
    BLOCK_PATH,
    BLOCK_COAL,
    BLOCK_IRON,
    BLOCK_DIAMOND,
    BLOCK_SAND,
    BLOCK_LAVA,
    BLOCK_SAPPHIRE,
    BLOCK_RUBY,
    BLOCK_STALAGMITE,
    BLOCK_FIRE_GRASS,
    BLOCK_FIRE_TREE,
    BLOCK_ICE_GRASS,
    BLOCK_ICE_SHRUB,
    BLOCK_WALL,
    BLOCK_WALL_MOSS,
    BLOCK_GRAVE,
    BLOCK_GRAVE2,
    BLOCK_GRAVE3,
    BLOCK_NECROMANCER,
    BLOCK_FOUNTAIN,
    BLOCK_OUT_OF_BOUNDS,
    BLOCK_ENCHANTMENT_TABLE_FIRE,
    BLOCK_ENCHANTMENT_TABLE_ICE,
)


# ============================================================================
# Smooth-gen config: drives the per-floor noise → block-id rules.
# ============================================================================
#
# Fields match the reference's SmoothGenConfig 1-1. Five ore slots are baked
# in (matches the reference's `jnp.array([5])` ore slot count); unused slots
# point at BLOCK_OUT_OF_BOUNDS with chance 0.

@fieldwise_init
struct SmoothGenConfig(Copyable, Movable):
    var default_block: Int
    var sea_block: Int
    var coast_block: Int
    var mountain_block: Int
    var path_block: Int
    var inner_mountain_block: Int
    # 5 ore slots (match reference) — each (req_block, ore_block, chance).
    var ore_req_block_0: Int
    var ore_req_block_1: Int
    var ore_req_block_2: Int
    var ore_req_block_3: Int
    var ore_req_block_4: Int
    var ore_block_0: Int
    var ore_block_1: Int
    var ore_block_2: Int
    var ore_block_3: Int
    var ore_block_4: Int
    var ore_chance_0: Float32
    var ore_chance_1: Float32
    var ore_chance_2: Float32
    var ore_chance_3: Float32
    var ore_chance_4: Float32
    var tree_req_block: Int
    var tree_block: Int
    var lava_block: Int
    var player_spawn_block: Int
    var valid_ladder_block: Int
    var ladder_up: Bool
    var ladder_down: Bool
    var water_strength: Float32
    var water_max: Float32
    var mountain_strength: Float32
    var mountain_max: Float32
    var default_light: Float32
    var water_threshold: Float32
    var sand_threshold: Float32
    var tree_threshold_uniform: Float32
    var tree_threshold_perlin: Float32


@fieldwise_init
struct DungeonConfig(Copyable, Movable):
    var special_block: Int
    var fountain_block: Int
    var rare_path_replacement_block: Int


# ============================================================================
# Six smooth configs (ordered for `SMOOTHGENS[i]` to match the reference)
# ============================================================================

def overworld_config() -> SmoothGenConfig:
    return SmoothGenConfig(
        default_block=BLOCK_GRASS,
        sea_block=BLOCK_WATER,
        coast_block=BLOCK_SAND,
        mountain_block=BLOCK_STONE,
        path_block=BLOCK_PATH,
        inner_mountain_block=BLOCK_PATH,
        ore_req_block_0=BLOCK_STONE,
        ore_req_block_1=BLOCK_STONE,
        ore_req_block_2=BLOCK_STONE,
        ore_req_block_3=BLOCK_STONE,
        ore_req_block_4=BLOCK_STONE,
        ore_block_0=BLOCK_COAL,
        ore_block_1=BLOCK_IRON,
        ore_block_2=BLOCK_DIAMOND,
        ore_block_3=BLOCK_OUT_OF_BOUNDS,
        ore_block_4=BLOCK_OUT_OF_BOUNDS,
        ore_chance_0=Float32(0.03),
        ore_chance_1=Float32(0.02),
        ore_chance_2=Float32(0.001),
        ore_chance_3=Float32(0.0),
        ore_chance_4=Float32(0.0),
        tree_req_block=BLOCK_GRASS,
        tree_block=BLOCK_TREE,
        lava_block=BLOCK_LAVA,
        player_spawn_block=BLOCK_GRASS,
        valid_ladder_block=BLOCK_PATH,
        ladder_up=False,
        ladder_down=True,
        water_strength=Float32(5.0),
        water_max=Float32(1.0),
        mountain_strength=Float32(5.0),
        mountain_max=Float32(1.0),
        default_light=Float32(1.0),
        water_threshold=Float32(0.7),
        sand_threshold=Float32(0.6),
        tree_threshold_uniform=Float32(0.8),
        tree_threshold_perlin=Float32(0.5),
    )


def gnomish_mines_config() -> SmoothGenConfig:
    return SmoothGenConfig(
        default_block=BLOCK_PATH,
        sea_block=BLOCK_WATER,
        coast_block=BLOCK_PATH,
        mountain_block=BLOCK_STONE,
        path_block=BLOCK_STONE,
        inner_mountain_block=BLOCK_STONE,
        ore_req_block_0=BLOCK_STONE,
        ore_req_block_1=BLOCK_STONE,
        ore_req_block_2=BLOCK_STONE,
        ore_req_block_3=BLOCK_STONE,
        ore_req_block_4=BLOCK_STONE,
        ore_block_0=BLOCK_COAL,
        ore_block_1=BLOCK_IRON,
        ore_block_2=BLOCK_DIAMOND,
        ore_block_3=BLOCK_SAPPHIRE,
        ore_block_4=BLOCK_RUBY,
        ore_chance_0=Float32(0.04),
        ore_chance_1=Float32(0.02),
        ore_chance_2=Float32(0.005),
        ore_chance_3=Float32(0.0025),
        ore_chance_4=Float32(0.0025),
        tree_req_block=BLOCK_PATH,
        tree_block=BLOCK_STALAGMITE,
        lava_block=BLOCK_LAVA,
        player_spawn_block=BLOCK_PATH,
        valid_ladder_block=BLOCK_PATH,
        ladder_up=True,
        ladder_down=True,
        water_strength=Float32(5.0),
        water_max=Float32(1.0),
        mountain_strength=Float32(17.0),
        mountain_max=Float32(1.5),
        default_light=Float32(0.0),
        water_threshold=Float32(0.7),
        sand_threshold=Float32(0.6),
        tree_threshold_uniform=Float32(0.8),
        tree_threshold_perlin=Float32(0.5),
    )


def troll_mines_config() -> SmoothGenConfig:
    return SmoothGenConfig(
        default_block=BLOCK_PATH,
        sea_block=BLOCK_WATER,
        coast_block=BLOCK_PATH,
        mountain_block=BLOCK_STONE,
        path_block=BLOCK_STONE,
        inner_mountain_block=BLOCK_STONE,
        ore_req_block_0=BLOCK_STONE,
        ore_req_block_1=BLOCK_STONE,
        ore_req_block_2=BLOCK_STONE,
        ore_req_block_3=BLOCK_STONE,
        ore_req_block_4=BLOCK_STONE,
        ore_block_0=BLOCK_COAL,
        ore_block_1=BLOCK_IRON,
        ore_block_2=BLOCK_DIAMOND,
        ore_block_3=BLOCK_SAPPHIRE,
        ore_block_4=BLOCK_RUBY,
        ore_chance_0=Float32(0.04),
        ore_chance_1=Float32(0.03),
        ore_chance_2=Float32(0.01),
        ore_chance_3=Float32(0.01),
        ore_chance_4=Float32(0.01),
        tree_req_block=BLOCK_PATH,
        tree_block=BLOCK_STALAGMITE,
        lava_block=BLOCK_LAVA,
        player_spawn_block=BLOCK_PATH,
        valid_ladder_block=BLOCK_PATH,
        ladder_up=True,
        ladder_down=True,
        water_strength=Float32(5.0),
        water_max=Float32(1.0),
        mountain_strength=Float32(17.0),
        mountain_max=Float32(1.5),
        default_light=Float32(0.0),
        water_threshold=Float32(0.7),
        sand_threshold=Float32(0.6),
        tree_threshold_uniform=Float32(0.8),
        tree_threshold_perlin=Float32(0.5),
    )


def fire_level_config() -> SmoothGenConfig:
    return SmoothGenConfig(
        default_block=BLOCK_FIRE_GRASS,
        sea_block=BLOCK_LAVA,
        coast_block=BLOCK_SAND,
        mountain_block=BLOCK_STONE,
        path_block=BLOCK_STONE,
        inner_mountain_block=BLOCK_STONE,
        ore_req_block_0=BLOCK_STONE,
        ore_req_block_1=BLOCK_STONE,
        ore_req_block_2=BLOCK_STONE,
        ore_req_block_3=BLOCK_STONE,
        ore_req_block_4=BLOCK_STONE,
        ore_block_0=BLOCK_COAL,
        ore_block_1=BLOCK_IRON,
        ore_block_2=BLOCK_DIAMOND,
        ore_block_3=BLOCK_SAPPHIRE,
        ore_block_4=BLOCK_RUBY,
        ore_chance_0=Float32(0.05),
        ore_chance_1=Float32(0.0),
        ore_chance_2=Float32(0.0),
        ore_chance_3=Float32(0.0),
        ore_chance_4=Float32(0.025),
        tree_req_block=BLOCK_FIRE_GRASS,
        tree_block=BLOCK_FIRE_TREE,
        lava_block=BLOCK_LAVA,
        player_spawn_block=BLOCK_FIRE_GRASS,
        valid_ladder_block=BLOCK_FIRE_GRASS,
        ladder_up=True,
        ladder_down=True,
        water_strength=Float32(5.0),
        water_max=Float32(1.0),
        mountain_strength=Float32(5.0),
        mountain_max=Float32(1.0),
        default_light=Float32(1.0),
        water_threshold=Float32(0.5),
        sand_threshold=Float32(0.6),
        tree_threshold_uniform=Float32(0.8),
        tree_threshold_perlin=Float32(0.5),
    )


def ice_level_config() -> SmoothGenConfig:
    return SmoothGenConfig(
        default_block=BLOCK_ICE_GRASS,
        sea_block=BLOCK_WATER,
        coast_block=BLOCK_ICE_GRASS,
        mountain_block=BLOCK_STONE,
        path_block=BLOCK_STONE,
        inner_mountain_block=BLOCK_STONE,
        ore_req_block_0=BLOCK_STONE,
        ore_req_block_1=BLOCK_STONE,
        ore_req_block_2=BLOCK_STONE,
        ore_req_block_3=BLOCK_STONE,
        ore_req_block_4=BLOCK_STONE,
        ore_block_0=BLOCK_COAL,
        ore_block_1=BLOCK_IRON,
        ore_block_2=BLOCK_DIAMOND,
        ore_block_3=BLOCK_SAPPHIRE,
        ore_block_4=BLOCK_RUBY,
        ore_chance_0=Float32(0.0),
        ore_chance_1=Float32(0.0),
        ore_chance_2=Float32(0.005),
        ore_chance_3=Float32(0.02),
        ore_chance_4=Float32(0.0),
        tree_req_block=BLOCK_ICE_GRASS,
        tree_block=BLOCK_ICE_SHRUB,
        lava_block=BLOCK_WATER,
        player_spawn_block=BLOCK_ICE_GRASS,
        valid_ladder_block=BLOCK_ICE_GRASS,
        ladder_up=True,
        ladder_down=True,
        water_strength=Float32(5.0),
        water_max=Float32(1.0),
        mountain_strength=Float32(17.0),
        mountain_max=Float32(1.5),
        default_light=Float32(0.0),
        water_threshold=Float32(0.5),
        sand_threshold=Float32(0.6),
        tree_threshold_uniform=Float32(0.4),
        tree_threshold_perlin=Float32(0.5),
    )


def boss_level_config() -> SmoothGenConfig:
    return SmoothGenConfig(
        default_block=BLOCK_PATH,
        sea_block=BLOCK_PATH,
        coast_block=BLOCK_PATH,
        mountain_block=BLOCK_WALL,
        path_block=BLOCK_WALL,
        inner_mountain_block=BLOCK_WALL,
        ore_req_block_0=BLOCK_WALL,
        ore_req_block_1=BLOCK_GRAVE,
        ore_req_block_2=BLOCK_GRAVE,
        ore_req_block_3=BLOCK_WALL,
        ore_req_block_4=BLOCK_WALL,
        ore_block_0=BLOCK_WALL_MOSS,
        ore_block_1=BLOCK_GRAVE2,
        ore_block_2=BLOCK_GRAVE3,
        ore_block_3=BLOCK_SAPPHIRE,
        ore_block_4=BLOCK_RUBY,
        ore_chance_0=Float32(0.1),
        ore_chance_1=Float32(0.333),
        ore_chance_2=Float32(0.5),
        ore_chance_3=Float32(0.0),
        ore_chance_4=Float32(0.0),
        tree_req_block=BLOCK_PATH,
        tree_block=BLOCK_GRAVE,
        lava_block=BLOCK_WALL,
        player_spawn_block=BLOCK_NECROMANCER,
        valid_ladder_block=BLOCK_PATH,
        ladder_up=False,
        ladder_down=False,
        water_strength=Float32(5.0),
        water_max=Float32(1.0),
        mountain_strength=Float32(10.0),
        mountain_max=Float32(10.0),
        default_light=Float32(0.0),
        water_threshold=Float32(0.7),
        sand_threshold=Float32(0.6),
        tree_threshold_uniform=Float32(0.95),
        tree_threshold_perlin=Float32(-1.0),
    )


# ============================================================================
# Three dungeon configs
# ============================================================================

def dungeon_config() -> DungeonConfig:
    return DungeonConfig(
        special_block=BLOCK_PATH,
        fountain_block=BLOCK_FOUNTAIN,
        rare_path_replacement_block=BLOCK_PATH,
    )


def sewer_config() -> DungeonConfig:
    return DungeonConfig(
        special_block=BLOCK_ENCHANTMENT_TABLE_ICE,
        fountain_block=BLOCK_WATER,
        rare_path_replacement_block=BLOCK_WATER,
    )


def vaults_config() -> DungeonConfig:
    return DungeonConfig(
        special_block=BLOCK_ENCHANTMENT_TABLE_FIRE,
        fountain_block=BLOCK_FOUNTAIN,
        rare_path_replacement_block=BLOCK_PATH,
    )
