"""World generation for Full Craftax (9-floor dungeon variant).

Ports `references/Craftax-main/craftax/craftax/world_gen/world_gen.py`.

Per-floor pipeline:
  - Floors 0, 2, 5, 6, 7, 8 are *smoothworlds*: 4 Perlin noise fields
    (water, mountain, path, tree) thresholded into block IDs, with
    config-driven block substitutions.
  - Floors 1, 3, 4 are *dungeons*: 8 random rooms placed in a 3×3 chunk
    grid, connected by L-paths, decorated with torches / chest / fountain,
    walls dilated and split into WALL / WALL_MOSS / DARKNESS bands.

Each floor writes into its slice of `map / item_map / light_map / ladders`
inside a `state` buffer. Mobs and projectiles are zeroed; the initial
inventory / intrinsics / achievements are filled at reset, not here.

CPU-only for Phase 7B. GPU port comes after game logic.
"""

from std.math import cos as math_cos
from std.random.philox import Random as PhiloxRandom

from mojo_rl.envs.craftax_classic.noise import (
    generate_fractal_noise_2d_normalized,
)
from .constants import (
    MAP_H,
    MAP_W,
    MAP_SIZE_PER_FLOOR,
    NUM_FLOORS,
    BLOCK_WALL,
    BLOCK_WALL_MOSS,
    BLOCK_DARKNESS,
    BLOCK_PATH,
    BLOCK_CHEST,
    BLOCK_LAVA,
    BLOCK_OUT_OF_BOUNDS,
    ITEM_NONE,
    ITEM_TORCH,
    ITEM_LADDER_DOWN,
    ITEM_LADDER_UP,
    DAY_LENGTH,
)
from .state import (
    s_map,
    s_item_map,
    s_light_map,
    s_down_ladder,
    s_up_ladder,
)
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


# ============================================================================
# Perlin noise resolutions — match the reference's per-axis ratios.
#   small_res = (H/16, W/16) — water, mountain
#   x_res     = (H/8,  W/2)  — path
#   larger_res = (H/4, W/4)  — trees
# For 48×48 → (3,3), (6,24), (12,12).
# ============================================================================

comptime RES_SMALL_H: Int = MAP_H // 16  # 3
comptime RES_SMALL_W: Int = MAP_W // 16  # 3
comptime RES_PATH_H: Int = MAP_H // 8  # 6
comptime RES_PATH_W: Int = MAP_W // 2  # 24
comptime RES_LARGER_H: Int = MAP_H // 4  # 12
comptime RES_LARGER_W: Int = MAP_W // 4  # 12

# Mountain threshold is a constant in the reference (not per-config).
comptime MOUNTAIN_THRESHOLD: Float32 = 0.7


# ============================================================================
# Day/night light level — shared with the reset path.
# Mirrors `calculate_light_level` from the reference's game_logic.py.
# ============================================================================


@always_inline
def calculate_light_level(timestep: Int) -> Float32:
    """Formula: light = 1 - |cos(π * progress)|^3 with progress = (t/day) + 0.3.
    """
    var progress = (
        (Float32(timestep) / Float32(DAY_LENGTH))
        - Float32(Int(timestep // DAY_LENGTH))
        + Float32(0.3)
    )
    var c = math_cos(Float32(3.14159265) * progress)
    var ac = c if c >= Float32(0.0) else -c
    return Float32(1.0) - ac * ac * ac


# ============================================================================
# Helpers
# ============================================================================


@always_inline
def _abs_int(v: Int) -> Int:
    return v if v >= 0 else -v


@always_inline
def _pick_random_block(
    map_buf: UnsafePointer[Float32, MutAnyOrigin],
    target_block: Int,
    mut rng: PhiloxRandom,
) -> Tuple[Int, Int]:
    """Uniformly sample one tile equal to `target_block`. Returns (y, x).

    If no such tile exists, returns (0, 0).
    """
    var count = 0
    for i in range(MAP_SIZE_PER_FLOOR):
        if Int(map_buf[i]) == target_block:
            count += 1
    if count == 0:
        return (0, 0)
    var u = rng.step_uniform()
    var k = Int(Float32(u[0]) * Float32(count))
    if k >= count:
        k = count - 1
    if k < 0:
        k = 0
    var seen = 0
    for i in range(MAP_SIZE_PER_FLOOR):
        if Int(map_buf[i]) == target_block:
            if seen == k:
                return (i // MAP_W, i % MAP_W)
            seen += 1
    return (0, 0)


# ============================================================================
# Smoothworld floor generator
# ============================================================================


def _generate_smoothworld_floor(
    mut rng: PhiloxRandom,
    config: SmoothGenConfig,
    player_y: Int,
    player_x: Int,
    map_out: UnsafePointer[Float32, MutAnyOrigin],
    item_map_out: UnsafePointer[Float32, MutAnyOrigin],
    light_map_out: UnsafePointer[Float32, MutAnyOrigin],
    water: UnsafePointer[Float32, MutAnyOrigin],
    mountain: UnsafePointer[Float32, MutAnyOrigin],
    path: UnsafePointer[Float32, MutAnyOrigin],
    tree: UnsafePointer[Float32, MutAnyOrigin],
) -> Tuple[Int, Int, Int, Int]:
    """Generate one smoothworld floor. Writes into the four maps.

    Returns (ladder_down_y, ladder_down_x, ladder_up_y, ladder_up_x).
    """
    # Pass 0: Perlin noise fields.
    generate_fractal_noise_2d_normalized[
        MAP_H, MAP_W, RES_SMALL_H, RES_SMALL_W
    ](rng, water)
    generate_fractal_noise_2d_normalized[
        MAP_H, MAP_W, RES_SMALL_H, RES_SMALL_W
    ](rng, mountain)
    generate_fractal_noise_2d_normalized[MAP_H, MAP_W, RES_PATH_H, RES_PATH_W](
        rng, path
    )
    generate_fractal_noise_2d_normalized[
        MAP_H, MAP_W, RES_LARGER_H, RES_LARGER_W
    ](rng, tree)

    # Pass 1: thresholds → block IDs.
    for y in range(MAP_H):
        var dy = y - player_y
        var ady = _abs_int(dy)
        for x in range(MAP_W):
            var dx = x - player_x
            var adx = _abs_int(dx)
            var dist = Float32(ady + adx)

            var prox_w = dist / config.water_strength
            if prox_w > config.water_max:
                prox_w = config.water_max
            var prox_m = dist / config.mountain_strength
            if prox_m > config.mountain_max:
                prox_m = config.mountain_max

            var i = y * MAP_W + x
            var w_raw = water[i]
            var m_raw = mountain[i]
            var w_adj = w_raw + prox_w - Float32(1.0)
            var m_adj = m_raw + Float32(0.05) + prox_m - Float32(1.0)
            var path_x = path[i]
            var path_y = path[x * MAP_W + y]  # transposed sample (matches ref)

            var b = config.default_block

            # Water → sand band → ...
            if w_adj > config.water_threshold:
                b = config.sea_block
            elif w_adj > config.sand_threshold:
                b = config.coast_block

            # Mountains
            if m_adj > MOUNTAIN_THRESHOLD:
                b = config.mountain_block

            # Paths cut through mountains in both directions.
            if m_adj > MOUNTAIN_THRESHOLD and path_x > Float32(0.8):
                b = config.path_block
            if m_adj > MOUNTAIN_THRESHOLD and path_y > Float32(0.8):
                b = config.path_block

            # Deep caves through mountains.
            if m_raw > Float32(0.85) and w_raw > Float32(0.4):
                b = config.inner_mountain_block

            map_out[i] = Float32(b)

    # Pass 2: trees (per-tile uniform draw, only on tree_req tiles).
    for i in range(MAP_SIZE_PER_FLOOR):
        if (
            Int(map_out[i]) == config.tree_req_block
            and tree[i] > config.tree_threshold_perlin
        ):
            var u = rng.step_uniform()
            if Float32(u[0]) > config.tree_threshold_uniform:
                map_out[i] = Float32(config.tree_block)

    # Pass 3: 5 ore slots — each draws uniform per matching tile.
    for slot in range(5):
        var req: Int
        var ore: Int
        var chance: Float32
        if slot == 0:
            req = config.ore_req_block_0
            ore = config.ore_block_0
            chance = config.ore_chance_0
        elif slot == 1:
            req = config.ore_req_block_1
            ore = config.ore_block_1
            chance = config.ore_chance_1
        elif slot == 2:
            req = config.ore_req_block_2
            ore = config.ore_block_2
            chance = config.ore_chance_2
        elif slot == 3:
            req = config.ore_req_block_3
            ore = config.ore_block_3
            chance = config.ore_chance_3
        else:
            req = config.ore_req_block_4
            ore = config.ore_block_4
            chance = config.ore_chance_4

        if chance <= Float32(0.0) or ore == BLOCK_OUT_OF_BOUNDS:
            continue
        for i in range(MAP_SIZE_PER_FLOOR):
            if Int(map_out[i]) == req:
                var u = rng.step_uniform()
                if Float32(u[0]) < chance:
                    map_out[i] = Float32(ore)

    # Pass 4: lava overrides (uses raw noise, not adjusted).
    for i in range(MAP_SIZE_PER_FLOOR):
        if mountain[i] > Float32(0.85) and tree[i] > Float32(0.7):
            map_out[i] = Float32(config.lava_block)

    # Pass 5: player spawn — fixed center tile.
    map_out[player_y * MAP_W + player_x] = Float32(config.player_spawn_block)

    # Pass 6: item_map / light_map defaults.
    for i in range(MAP_SIZE_PER_FLOOR):
        item_map_out[i] = Float32(ITEM_NONE)
        light_map_out[i] = config.default_light

    # Pass 7: ladders on valid_ladder_block tiles.
    var (ld_y, ld_x) = _pick_random_block(
        map_out, config.valid_ladder_block, rng
    )
    var (lu_y, lu_x) = _pick_random_block(
        map_out, config.valid_ladder_block, rng
    )

    if config.ladder_down:
        item_map_out[ld_y * MAP_W + ld_x] = Float32(ITEM_LADDER_DOWN)
    if config.ladder_up:
        item_map_out[lu_y * MAP_W + lu_x] = Float32(ITEM_LADDER_UP)

    # Pass 8: torch glow around ladder_up
    # (9×9 region, intensity = 1 - manhattan/5, blended with default_light).
    for dy in range(-4, 5):
        var ty = lu_y + dy
        if ty < 0 or ty >= MAP_H:
            continue
        for dx in range(-4, 5):
            var tx = lu_x + dx
            if tx < 0 or tx >= MAP_W:
                continue
            var manhattan = _abs_int(dy) + _abs_int(dx)
            var intensity = Float32(1.0) - Float32(manhattan) * Float32(0.2)
            if intensity < Float32(0.0):
                intensity = Float32(0.0)
            if intensity > Float32(1.0):
                intensity = Float32(1.0)
            var blended = (
                intensity * (Float32(1.0) - config.default_light)
                + config.default_light
            )
            var idx = ty * MAP_W + tx
            if blended > light_map_out[idx]:
                light_map_out[idx] = blended

    # Pass 9: lava light spread (only if lava is the volcanic kind).
    if config.lava_block == BLOCK_LAVA:
        # 3×3 weighted kernel: corners 0.2, edges 0.7, center 1.0.
        for y in range(MAP_H):
            for x in range(MAP_W):
                var i = y * MAP_W + x
                if Int(map_out[i]) != BLOCK_LAVA:
                    continue
                # Add the kernel centered on this lava tile.
                for ky in range(-1, 2):
                    var ny = y + ky
                    if ny < 0 or ny >= MAP_H:
                        continue
                    for kx in range(-1, 2):
                        var nx = x + kx
                        if nx < 0 or nx >= MAP_W:
                            continue
                        var w: Float32
                        if ky == 0 and kx == 0:
                            w = Float32(1.0)
                        elif ky == 0 or kx == 0:
                            w = Float32(0.7)
                        else:
                            w = Float32(0.2)
                        var ni = ny * MAP_W + nx
                        var v = light_map_out[ni] + w
                        if v > Float32(1.0):
                            v = Float32(1.0)
                        light_map_out[ni] = v

    return (ld_y, ld_x, lu_y, lu_x)


# ============================================================================
# Dungeon floor generator
# ============================================================================

comptime NUM_ROOMS: Int = 8
comptime MIN_ROOM_SIZE: Int = 5
comptime MAX_ROOM_SIZE: Int = 10
comptime CHUNK_SIZE: Int = 16
comptime CHUNKS_PER_AXIS: Int = MAP_H // CHUNK_SIZE  # 3
comptime TOTAL_CHUNKS: Int = CHUNKS_PER_AXIS * CHUNKS_PER_AXIS  # 9
comptime RARE_CHANCE: Float32 = 0.1


def _generate_dungeon_floor(
    mut rng: PhiloxRandom,
    config: DungeonConfig,
    map_out: UnsafePointer[Float32, MutAnyOrigin],
    item_map_out: UnsafePointer[Float32, MutAnyOrigin],
    light_map_out: UnsafePointer[Float32, MutAnyOrigin],
) -> Tuple[Int, Int, Int, Int]:
    """Room-based dungeon. Returns ladder positions like smoothworld."""
    # Pre-roll room sizes (uniform in [min_room_size, max_room_size)).
    var room_y = InlineArray[Int, NUM_ROOMS](fill=0)
    var room_x = InlineArray[Int, NUM_ROOMS](fill=0)
    var room_h = InlineArray[Int, NUM_ROOMS](fill=MIN_ROOM_SIZE)
    var room_w = InlineArray[Int, NUM_ROOMS](fill=MIN_ROOM_SIZE)
    for r in range(NUM_ROOMS):
        var u = rng.step_uniform()
        var rh = MIN_ROOM_SIZE + Int(
            Float32(u[0]) * Float32(MAX_ROOM_SIZE - MIN_ROOM_SIZE)
        )
        var rw = MIN_ROOM_SIZE + Int(
            Float32(u[1]) * Float32(MAX_ROOM_SIZE - MIN_ROOM_SIZE)
        )
        if rh >= MAX_ROOM_SIZE:
            rh = MAX_ROOM_SIZE - 1
        if rw >= MAX_ROOM_SIZE:
            rw = MAX_ROOM_SIZE - 1
        room_h[r] = rh
        room_w[r] = rw

    # Initialize: all WALL, no items.
    for i in range(MAP_SIZE_PER_FLOOR):
        map_out[i] = Float32(BLOCK_WALL)
        item_map_out[i] = Float32(ITEM_NONE)

    # Place rooms, each in a distinct chunk.
    var occupied = InlineArray[Bool, TOTAL_CHUNKS](fill=False)
    for r in range(NUM_ROOMS):
        # Sample a free chunk uniformly.
        var free = 0
        for c in range(TOTAL_CHUNKS):
            if not occupied[c]:
                free += 1
        if free <= 0:
            break  # 8 rooms in 9 chunks — should never happen

        var u_chunk = rng.step_uniform()
        var k = Int(Float32(u_chunk[0]) * Float32(free))
        if k >= free:
            k = free - 1
        if k < 0:
            k = 0
        var seen = 0
        var chunk_id = 0
        for c in range(TOTAL_CHUNKS):
            if not occupied[c]:
                if seen == k:
                    chunk_id = c
                    break
                seen += 1
        occupied[chunk_id] = True

        var cy = chunk_id // CHUNKS_PER_AXIS
        var cx = chunk_id % CHUNKS_PER_AXIS

        var u_off = rng.step_uniform()
        var off_y = Int(Float32(u_off[0]) * Float32(CHUNK_SIZE - MIN_ROOM_SIZE))
        var off_x = Int(Float32(u_off[1]) * Float32(CHUNK_SIZE - MIN_ROOM_SIZE))
        var py = cy * CHUNK_SIZE + off_y
        var px = cx * CHUNK_SIZE + off_x
        # Clamp to fit the room within bounds (no padding).
        if py + room_h[r] > MAP_H:
            py = MAP_H - room_h[r]
        if px + room_w[r] > MAP_W:
            px = MAP_W - room_w[r]
        if py < 0:
            py = 0
        if px < 0:
            px = 0
        room_y[r] = py
        room_x[r] = px

        # Fill room interior with PATH.
        for ry in range(room_h[r]):
            for rx in range(room_w[r]):
                map_out[(py + ry) * MAP_W + (px + rx)] = Float32(BLOCK_PATH)

        # Torches in 4 corners.
        item_map_out[py * MAP_W + px] = Float32(ITEM_TORCH)
        item_map_out[(py + room_h[r] - 1) * MAP_W + px] = Float32(ITEM_TORCH)
        item_map_out[py * MAP_W + (px + room_w[r] - 1)] = Float32(ITEM_TORCH)
        item_map_out[
            (py + room_h[r] - 1) * MAP_W + (px + room_w[r] - 1)
        ] = Float32(ITEM_TORCH)

        # Chest at a random interior cell.
        var u_chest = rng.step_uniform()
        var cy_in = 1 + Int(Float32(u_chest[0]) * Float32(room_h[r] - 2))
        var cx_in = 1 + Int(Float32(u_chest[1]) * Float32(room_w[r] - 2))
        if cy_in >= room_h[r] - 1:
            cy_in = room_h[r] - 2
        if cx_in >= room_w[r] - 1:
            cx_in = room_w[r] - 2
        map_out[(py + cy_in) * MAP_W + (px + cx_in)] = Float32(BLOCK_CHEST)

        # Fountain at a random interior cell (50% chance).
        var u_fn = rng.step_uniform()
        if Float32(u_fn[0]) > Float32(0.5):
            var fy_in = 1 + Int(Float32(u_fn[1]) * Float32(room_h[r] - 2))
            var fx_in = 1 + Int(Float32(u_fn[2]) * Float32(room_w[r] - 2))
            if fy_in >= room_h[r] - 1:
                fy_in = room_h[r] - 2
            if fx_in >= room_w[r] - 1:
                fx_in = room_w[r] - 2
            map_out[(py + fy_in) * MAP_W + (px + fx_in)] = Float32(
                config.fountain_block
            )

    # Connect rooms: room[i] → random of included; initially included = {last}.
    var included = InlineArray[Bool, NUM_ROOMS](fill=False)
    included[NUM_ROOMS - 1] = True
    for i in range(NUM_ROOMS):
        var inc = 0
        for r in range(NUM_ROOMS):
            if included[r]:
                inc += 1
        if inc == 0:
            included[i] = True
            continue
        var u = rng.step_uniform()
        var k = Int(Float32(u[0]) * Float32(inc))
        if k >= inc:
            k = inc - 1
        if k < 0:
            k = 0
        var seen2 = 0
        var sink = i
        for r in range(NUM_ROOMS):
            if included[r]:
                if seen2 == k:
                    sink = r
                    break
                seen2 += 1

        var sy = room_y[i]
        var sx = room_x[i]
        var ty = room_y[sink]
        var tx = room_x[sink]

        # Horizontal segment first.
        var step_x = 1
        if tx < sx:
            step_x = -1
        var cx_run = sx
        while cx_run != tx:
            var idx = sy * MAP_W + cx_run
            if Int(map_out[idx]) == BLOCK_WALL:
                map_out[idx] = Float32(BLOCK_PATH)
            cx_run += step_x

        # Then vertical from (sy, tx) to (ty, tx).
        var step_y = 1
        if ty < sy:
            step_y = -1
        var cy_run = sy
        while cy_run != ty:
            var idx = cy_run * MAP_W + tx
            if Int(map_out[idx]) == BLOCK_WALL:
                map_out[idx] = Float32(BLOCK_PATH)
            cy_run += step_y

        included[i] = True

    # Special block in room 0 at offset (2, 2).
    var sp_y = room_y[0] + 2
    var sp_x = room_x[0] + 2
    if sp_y < MAP_H and sp_x < MAP_W:
        map_out[sp_y * MAP_W + sp_x] = Float32(config.special_block)

    # Pre-compute c_path (non-wall) + adj_path (4-neighbor dilation).
    var c_path = InlineArray[Bool, MAP_SIZE_PER_FLOOR](fill=False)
    for i in range(MAP_SIZE_PER_FLOOR):
        c_path[i] = Int(map_out[i]) != BLOCK_WALL

    var adj_path = InlineArray[Bool, MAP_SIZE_PER_FLOOR](fill=False)
    for y in range(MAP_H):
        for x in range(MAP_W):
            var i = y * MAP_W + x
            var v = c_path[i]
            if y > 0:
                v = v or c_path[i - MAP_W]
            if y < MAP_H - 1:
                v = v or c_path[i + MAP_W]
            if x > 0:
                v = v or c_path[i - 1]
            if x < MAP_W - 1:
                v = v or c_path[i + 1]
            adj_path[i] = v

    # Apply post-processing: WALL → {WALL, WALL_MOSS, DARKNESS}, plus rare
    # path replacement on plain PATH tiles. Single per-tile uniform draw.
    for i in range(MAP_SIZE_PER_FLOOR):
        var u = rng.step_uniform()
        var is_rare = Float32(u[0]) < RARE_CHANCE
        var b = Int(map_out[i])
        var item = Int(item_map_out[i])
        if not adj_path[i]:
            map_out[i] = Float32(BLOCK_DARKNESS)
        elif b == BLOCK_WALL:
            if is_rare:
                map_out[i] = Float32(BLOCK_WALL_MOSS)
        elif b == BLOCK_PATH and item == ITEM_NONE:
            if is_rare:
                map_out[i] = Float32(config.rare_path_replacement_block)

    # Light map: always fully lit (per reference).
    for i in range(MAP_SIZE_PER_FLOOR):
        light_map_out[i] = Float32(1.0)

    # Ladders: random PATH tile (post-rare).
    var (ld_y, ld_x) = _pick_random_block(map_out, BLOCK_PATH, rng)
    item_map_out[ld_y * MAP_W + ld_x] = Float32(ITEM_LADDER_DOWN)
    var (lu_y, lu_x) = _pick_random_block(map_out, BLOCK_PATH, rng)
    item_map_out[lu_y * MAP_W + lu_x] = Float32(ITEM_LADDER_UP)

    return (ld_y, ld_x, lu_y, lu_x)


# ============================================================================
# Top-level: fill the full state buffer
# ============================================================================


@always_inline
def _floor_is_dungeon(floor: Int) -> Bool:
    return floor == 1 or floor == 3 or floor == 4


@always_inline
def generate_full_world_inline(
    seed: UInt64,
    state_ptr: UnsafePointer[Float32, MutAnyOrigin],
    water: UnsafePointer[Float32, MutAnyOrigin],
    mountain: UnsafePointer[Float32, MutAnyOrigin],
    path: UnsafePointer[Float32, MutAnyOrigin],
    tree: UnsafePointer[Float32, MutAnyOrigin],
) -> Tuple[Int, Int]:
    """GPU-safe core. Same world-gen as `generate_full_world` but takes
    pre-allocated scratch (`water/mountain/path/tree`, each
    `MAP_SIZE_PER_FLOOR` floats) so callers can stage workspaces in a
    DeviceBuffer for batched runs. Returns (player_y, player_x).

    Writes only the map / item_map / light_map slabs + down/up ladder
    coords; the caller zeroes everything else.
    """
    var rng = PhiloxRandom(seed=seed, offset=0)

    var player_y = MAP_H // 2
    var player_x = MAP_W // 2

    for floor in range(NUM_FLOORS):
        var map_off = s_map(floor, 0, 0)
        var item_off = s_item_map(floor, 0, 0)
        var light_off = s_light_map(floor, 0, 0)
        var map_ptr = state_ptr + map_off
        var item_ptr = state_ptr + item_off
        var light_ptr = state_ptr + light_off

        var ladders: Tuple[Int, Int, Int, Int]
        if _floor_is_dungeon(floor):
            var dcfg = dungeon_config() if floor == 1 else (
                sewer_config() if floor == 3 else vaults_config()
            )
            ladders = _generate_dungeon_floor(
                rng, dcfg, map_ptr, item_ptr, light_ptr
            )
        else:
            var scfg: SmoothGenConfig
            if floor == 0:
                scfg = overworld_config()
            elif floor == 2:
                scfg = gnomish_mines_config()
            elif floor == 5:
                scfg = troll_mines_config()
            elif floor == 6:
                scfg = fire_level_config()
            elif floor == 7:
                scfg = ice_level_config()
            else:
                scfg = boss_level_config()  # floor 8
            ladders = _generate_smoothworld_floor(
                rng,
                scfg,
                player_y,
                player_x,
                map_ptr,
                item_ptr,
                light_ptr,
                water,
                mountain,
                path,
                tree,
            )

        state_ptr[s_down_ladder(floor, 0)] = Float32(ladders[0])
        state_ptr[s_down_ladder(floor, 1)] = Float32(ladders[1])
        state_ptr[s_up_ladder(floor, 0)] = Float32(ladders[2])
        state_ptr[s_up_ladder(floor, 1)] = Float32(ladders[3])

    return (player_y, player_x)


def generate_full_world(
    seed: UInt64,
    state_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> Tuple[Int, Int]:
    """CPU entry — allocates scratch on the heap, delegates to the inline
    kernel, then frees.

    Writes:
      - 9 × (map / item_map / light_map) blocks (each 48×48 floats)
      - down/up ladder coords per floor
    Returns (player_y, player_x) — center tile of the overworld.

    The caller is responsible for zeroing all other state regions (mobs,
    inventory, etc.). World gen only touches the map slabs and ladder
    coords.
    """
    var water = alloc[Float32](MAP_SIZE_PER_FLOOR)
    var mountain = alloc[Float32](MAP_SIZE_PER_FLOOR)
    var path = alloc[Float32](MAP_SIZE_PER_FLOOR)
    var tree = alloc[Float32](MAP_SIZE_PER_FLOOR)
    var spawn = generate_full_world_inline(
        seed,
        state_ptr,
        water,
        mountain,
        path,
        tree,
    )
    water.free()
    mountain.free()
    path.free()
    tree.free()
    return spawn
