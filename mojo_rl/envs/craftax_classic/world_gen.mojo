"""World generation for Craftax-Classic.

Ports `references/Craftax-main/craftax/craftax_classic/world_gen.py`.

Generates a 64×64 BlockType map from a seed by combining four Perlin
noise fields (water, mountain, paths, trees), a player-proximity distance
map, and random ore/diamond placement.

The implementation here is CPU-only for now; the GPU port reuses the same
threshold structure but inlines the noise pass.
"""

from std.math import cos as math_cos
from std.random.philox import Random as PhiloxRandom

from .constants import (
    MAP_H,
    MAP_W,
    MAP_SIZE,
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
)
from .noise import generate_fractal_noise_2d_normalized


# ============================================================================
# Resolution presets — match reference world_gen.py exactly
# ============================================================================

comptime RES_WATER_H: Int = MAP_H // 16  # 4
comptime RES_WATER_W: Int = MAP_W // 16  # 4

comptime RES_MOUNTAIN_H: Int = MAP_H // 16  # 4
comptime RES_MOUNTAIN_W: Int = MAP_W // 16  # 4

comptime RES_PATH_H: Int = MAP_H // 8   # 8
comptime RES_PATH_W: Int = MAP_W // 2   # 32

comptime RES_TREE_H: Int = MAP_H // 4   # 16
comptime RES_TREE_W: Int = MAP_W // 4   # 16


# ============================================================================
# Light level
# ============================================================================

@always_inline
def calculate_light_level(timestep: Int, day_length: Int) -> Float32:
    """Match `calculate_light_level` in reference game_logic.py.

    light = 1 - |cos(π * progress)|^3  where  progress = (t / day_length) % 1 + 0.3
    """
    var progress = (Float32(timestep) / Float32(day_length)) - Float32(
        Int(timestep // day_length)
    ) + Float32(0.3)
    var c = math_cos(Float32(3.14159265) * progress)
    var abs_c = c if c >= Float32(0.0) else -c
    return Float32(1.0) - abs_c * abs_c * abs_c


# ============================================================================
# World generator (CPU)
# ============================================================================


@always_inline
def apply_world_gen_rules(
    water: UnsafePointer[Float32, MutAnyOrigin],
    mountain: UnsafePointer[Float32, MutAnyOrigin],
    path: UnsafePointer[Float32, MutAnyOrigin],
    tree: UnsafePointer[Float32, MutAnyOrigin],
    map_out: UnsafePointer[Float32, MutAnyOrigin],
    mut rng: PhiloxRandom,
    always_diamond: Bool,
) -> Tuple[Int, Int]:
    """Combine 4 normalized noise fields into a block-ID map.

    All 4 input fields are length MAP_SIZE, normalized to [0, 1].
    Writes block IDs into `map_out[y * MAP_W + x]`. Returns spawn (y, x).

    Shared by the CPU world generator and the GPU per-thread world gen,
    so callers only need to vary how they allocate the four scratch
    buffers (InlineArray on CPU, per-env DeviceBuffer slice on GPU).
    """
    var py = MAP_H // 2
    var px = MAP_W // 2

    # ------------------------------------------------------------------
    # Pass 1: thresholds → block IDs.
    # Player proximity = Manhattan dist / 5, clipped to 1 — inlined.
    # ------------------------------------------------------------------
    var mountain_threshold = Float32(0.7)

    for y in range(MAP_H):
        var dy = y - py
        var ady = dy if dy >= 0 else -dy
        for x in range(MAP_W):
            var dx = x - px
            var adx = dx if dx >= 0 else -dx
            var prox = Float32(ady + adx) * Float32(0.2)
            if prox > Float32(1.0):
                prox = Float32(1.0)

            var i = y * MAP_W + x
            var w_adj = water[i] + prox - Float32(1.0)
            var m_adj = mountain[i] + Float32(0.05) + prox - Float32(1.0)
            var path_x = path[i]
            var path_y = path[x * MAP_W + y]  # reference uses path.T
            var w_raw = water[i]
            var m_raw = mountain[i]

            var b = BLOCK_GRASS
            if w_adj > Float32(0.7):
                b = BLOCK_WATER
            elif w_adj < Float32(0.75) and w_adj > Float32(0.6):
                b = BLOCK_SAND

            if m_adj > mountain_threshold:
                b = BLOCK_STONE

            if m_adj > mountain_threshold and path_x > Float32(0.8):
                b = BLOCK_PATH
            if m_adj > mountain_threshold and path_y > Float32(0.8):
                b = BLOCK_PATH

            if m_raw > Float32(0.85) and w_raw > Float32(0.4):
                b = BLOCK_PATH

            map_out[i] = Float32(b)

    # ------------------------------------------------------------------
    # Pass 2: ores. Each stone tile draws independently — same RNG
    # ordering as reference (coal, iron, diamond).
    # ------------------------------------------------------------------
    for i in range(MAP_SIZE):
        if Int(map_out[i]) == BLOCK_STONE:
            var u = rng.step_uniform()
            if Float32(u[0]) < Float32(0.04):
                map_out[i] = Float32(BLOCK_COAL)

    for i in range(MAP_SIZE):
        if Int(map_out[i]) == BLOCK_STONE:
            var u = rng.step_uniform()
            if Float32(u[0]) < Float32(0.03):
                map_out[i] = Float32(BLOCK_IRON)

    for i in range(MAP_SIZE):
        if Int(map_out[i]) == BLOCK_STONE and mountain[i] > Float32(0.8):
            var u = rng.step_uniform()
            if Float32(u[0]) < Float32(0.005):
                map_out[i] = Float32(BLOCK_DIAMOND)

    # Trees on grass with sparsity gate.
    for i in range(MAP_SIZE):
        if Int(map_out[i]) == BLOCK_GRASS and tree[i] > Float32(0.5):
            var u = rng.step_uniform()
            if Float32(u[0]) > Float32(0.8):
                map_out[i] = Float32(BLOCK_TREE)

    # Lava (overwrites everything inside the lava zone).
    for i in range(MAP_SIZE):
        if mountain[i] > Float32(0.85) and tree[i] > Float32(0.7):
            map_out[i] = Float32(BLOCK_LAVA)

    # Player tile must be grass.
    map_out[py * MAP_W + px] = Float32(BLOCK_GRASS)

    # Optional always_diamond: pick a uniform stone tile and overwrite.
    if always_diamond:
        var stone_count = 0
        for i in range(MAP_SIZE):
            if Int(map_out[i]) == BLOCK_STONE:
                stone_count += 1
        if stone_count > 0:
            var u = rng.step_uniform()
            var k = Int(Float32(u[0]) * Float32(stone_count))
            if k >= stone_count:
                k = stone_count - 1
            var seen = 0
            for i in range(MAP_SIZE):
                if Int(map_out[i]) == BLOCK_STONE:
                    if seen == k:
                        map_out[i] = Float32(BLOCK_DIAMOND)
                        break
                    seen += 1

    return (py, px)


@always_inline
def generate_world_inline(
    rng_seed: UInt64,
    water: UnsafePointer[Float32, MutAnyOrigin],
    mountain: UnsafePointer[Float32, MutAnyOrigin],
    path: UnsafePointer[Float32, MutAnyOrigin],
    tree: UnsafePointer[Float32, MutAnyOrigin],
    map_out: UnsafePointer[Float32, MutAnyOrigin],
    always_diamond: Bool = False,
) -> Tuple[Int, Int]:
    """End-to-end world gen: noise + rule application.

    Caller provides four scratch buffers of length MAP_SIZE for the
    intermediate noise fields. CPU callers stack-allocate them via
    InlineArray; GPU callers slice them off a per-env workspace.
    """
    var rng = PhiloxRandom(seed=rng_seed, offset=0)
    generate_fractal_noise_2d_normalized[
        MAP_H, MAP_W, RES_WATER_H, RES_WATER_W
    ](rng, water)
    generate_fractal_noise_2d_normalized[
        MAP_H, MAP_W, RES_MOUNTAIN_H, RES_MOUNTAIN_W
    ](rng, mountain)
    generate_fractal_noise_2d_normalized[
        MAP_H, MAP_W, RES_PATH_H, RES_PATH_W
    ](rng, path)
    generate_fractal_noise_2d_normalized[
        MAP_H, MAP_W, RES_TREE_H, RES_TREE_W
    ](rng, tree)
    return apply_world_gen_rules(
        water, mountain, path, tree, map_out, rng, always_diamond
    )


def generate_world_cpu(
    rng_seed: UInt64,
    map_out: UnsafePointer[Float32, MutAnyOrigin],
    always_diamond: Bool = False,
) -> Tuple[Int, Int]:
    """CPU entry point: stack-allocates noise scratch + calls inline core."""
    var water = InlineArray[Float32, MAP_SIZE](fill=Float32(0.0))
    var mountain = InlineArray[Float32, MAP_SIZE](fill=Float32(0.0))
    var path = InlineArray[Float32, MAP_SIZE](fill=Float32(0.0))
    var tree = InlineArray[Float32, MAP_SIZE](fill=Float32(0.0))
    return generate_world_inline(
        rng_seed,
        water.unsafe_ptr().bitcast[Float32](),
        mountain.unsafe_ptr().bitcast[Float32](),
        path.unsafe_ptr().bitcast[Float32](),
        tree.unsafe_ptr().bitcast[Float32](),
        map_out,
        always_diamond,
    )
