"""Full Craftax symbolic observation encoder.

Mirrors `render_craftax_symbolic` in
`references/Craftax-main/craftax/craftax/renderer.py`. Output is a flat
Float32 vector of length `OBS_DIM = 8268`:

    [0 : OBS_VIEW_SIZE)            9 × 11 × 83 view tiles, row-major
    [OBS_VIEW_SIZE : OBS_DIM)      51-element scalar tail

Per-tile channel layout (83 channels):
    [OBS_CH_BLOCK_BASE : +NUM_BLOCK_TYPES)        block one-hot (37)
    [OBS_CH_ITEM_BASE  : +NUM_ITEM_TYPES)         item one-hot  (5)
    [OBS_CH_MOB_BASE   : +OBS_MOB_CLASSES*8)      mob presence  (40)
                                                  class_idx*8 + type_id
                                                  classes:
                                                      0 = melee
                                                      1 = passive
                                                      2 = ranged
                                                      3 = mob projectile
                                                      4 = player projectile
    [OBS_CH_LIGHT]                                light visibility (1)

Darkness masking: tiles whose `light_map[player_level][y][x] <= 0.05` get
all channels zeroed; only the light channel carries the (binary) lit
flag. Mobs sitting on dark tiles likewise vanish (matches reference).

Scalar tail order (51 elements):
    inventory(16) | potions(6) | intrinsics(9) | direction(4) |
    armour(4)     | armour_enchants(4) | special(8)

CPU only for now — GPU encoder lands later if needed.
"""

from std.math import sqrt
from std.memory import Pointer

from .constants import (
    MAP_H,
    MAP_W,
    NUM_FLOORS,
    VIEW_H,
    VIEW_W,
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    NUM_INTRINSICS,
    NUM_ATTRIBUTES,
    NUM_DIRECTIONS,
    NUM_SPELLS,
    NUM_ARMOUR_ENCHANTS,
    NUM_POTIONS,
    OBS_MOB_CLASSES,
    OBS_MOB_TYPES_PER_CLASS,
    OBS_CH_BLOCK_BASE,
    OBS_CH_ITEM_BASE,
    OBS_CH_MOB_BASE,
    OBS_CH_LIGHT,
    OBS_INV_SIZE,
    OBS_INTRINSICS_SIZE,
    OBS_DIRECTION_SIZE,
    OBS_ARMOUR_SIZE,
    OBS_ARMOUR_ENCH_SIZE,
    OBS_SPECIAL_SIZE,
    OBS_VIEW_SIZE,
    OBS_SCALAR_SIZE,
    OBS_DIM,
    TILE_CHANNELS,
    LIGHT_VISIBILITY_THRESHOLD,
    BLOCK_OUT_OF_BOUNDS,
    ITEM_NONE,
    MAX_MELEE_MOBS,
    MAX_PASSIVE_MOBS,
    MAX_RANGED_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PLAYER_PROJECTILES,
    MOB_FIELDS,
    PROJ_FIELDS,
    MOB_FY,
    MOB_FX,
    MOB_MASK,
    MOB_TYPE_ID,
    OBS_MOB_CLASS_MELEE,
    OBS_MOB_CLASS_PASSIVE,
    OBS_MOB_CLASS_RANGED,
    OBS_MOB_CLASS_MOB_PROJ,
    OBS_MOB_CLASS_PLAYER_PROJ,
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
    INTRINSIC_MANA,
    INTRINSIC_IS_SLEEPING,
    INTRINSIC_IS_RESTING,
    ATTR_XP,
    ATTR_DEXTERITY,
    ATTR_STRENGTH,
    ATTR_INTELLIGENCE,
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
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
)
from .state import (
    S_PLAYER_POS,
    S_PLAYER_LEVEL,
    S_PLAYER_DIR,
    S_LIGHT_LEVEL,
    S_BOSS_TIMESTEPS,
    S_BOW_ENCHANT,
    S_SWORD_ENCHANT,
    S_MELEE_MOBS_BASE,
    S_PASSIVE_MOBS_BASE,
    S_RANGED_MOBS_BASE,
    S_MOB_PROJECTILES_BASE,
    S_PLAYER_PROJECTILES_BASE,
    s_map,
    s_item_map,
    s_light_map,
    s_intrinsic,
    s_attribute,
    s_inv,
    s_armour_enchant,
    s_learned_spell,
    s_monsters_killed,
    s_melee_mob,
    s_ranged_mob,
)


comptime State = Pointer[Float32, MutAnyOrigin]
comptime Obs = Pointer[Float32, MutAnyOrigin]


# ============================================================================
# Helpers
# ============================================================================

@always_inline
def _in_bounds(y: Int, x: Int) -> Bool:
    return 0 <= y and y < MAP_H and 0 <= x and x < MAP_W


@always_inline
def _tile_off(local_y: Int, local_x: Int) -> Int:
    """Byte offset of the start of one tile's 83-channel block."""
    return local_y * VIEW_W * TILE_CHANNELS + local_x * TILE_CHANNELS


@always_inline
def _put_mob_array(
    s: State,
    obs: Obs,
    array_base: Int,
    max_n: Int,
    field_stride: Int,
    py: Int,
    px: Int,
    half_h: Int,
    half_w: Int,
    class_idx: Int,
):
    """For each alive mob in the array starting at `array_base` (this
    must already include the floor offset), if its position lies inside
    the view window, set its (class_idx, type_id) channel to 1.0.

    Works for both 6-field mobs and 8-field projectiles via
    `field_stride`."""
    for i in range(max_n):
        var mob_base = array_base + i * field_stride
        var mask = Int(s[unsafe_offset=mob_base + MOB_MASK])
        if mask == 0:
            continue
        var my = Int(s[unsafe_offset=mob_base + MOB_FY])
        var mx = Int(s[unsafe_offset=mob_base + MOB_FX])
        var lv = my - py + half_h
        var lx = mx - px + half_w
        if 0 <= lv and lv < VIEW_H and 0 <= lx and lx < VIEW_W:
            var type_id = Int(s[unsafe_offset=mob_base + MOB_TYPE_ID])
            if 0 <= type_id and type_id < OBS_MOB_TYPES_PER_CLASS:
                var ch = (
                    OBS_CH_MOB_BASE
                    + class_idx * OBS_MOB_TYPES_PER_CLASS
                    + type_id
                )
                obs[unsafe_offset=_tile_off(lv, lx) + ch] = Float32(1.0)


@always_inline
def _is_boss_vulnerable(s: State, floor: Int) -> Bool:
    """Boss is vulnerable when no melee + no ranged mobs are alive on the
    current floor AND the boss spawn cooldown has elapsed."""
    if Int(s[unsafe_offset=S_BOSS_TIMESTEPS]) > 0:
        return False
    for i in range(MAX_MELEE_MOBS):
        if Int(s[unsafe_offset=s_melee_mob(floor, i, MOB_MASK)]) != 0:
            return False
    for i in range(MAX_RANGED_MOBS):
        if Int(s[unsafe_offset=s_ranged_mob(floor, i, MOB_MASK)]) != 0:
            return False
    return True


# ============================================================================
# Main encoder
# ============================================================================

@always_inline
def encode_symbolic_obs(s: State, obs: Obs):
    """Fill `obs[0 : OBS_DIM)` with the Craftax-Full symbolic observation."""
    for i in range(OBS_DIM):
        obs[unsafe_offset=i] = Float32(0.0)

    var floor = Int(s[unsafe_offset=S_PLAYER_LEVEL])
    var py = Int(s[unsafe_offset=S_PLAYER_POS])
    var px = Int(s[unsafe_offset=S_PLAYER_POS + 1])
    var half_h = VIEW_H // 2
    var half_w = VIEW_W // 2

    # ------------------------------------------------------------------
    # 1. Per-tile block / item / light channels.
    # ------------------------------------------------------------------
    for lv in range(VIEW_H):
        var wy = py - half_h + lv
        for lx in range(VIEW_W):
            var wx = px - half_w + lx
            var tb = _tile_off(lv, lx)

            var in_b = _in_bounds(wy, wx)
            var blk_id: Int
            var item_id: Int
            var lit: Bool

            if in_b:
                blk_id = Int(s[unsafe_offset=s_map(floor, wy, wx)])
                item_id = Int(s[unsafe_offset=s_item_map(floor, wy, wx)])
                var light_val = s[unsafe_offset=s_light_map(floor, wy, wx)]
                lit = light_val > Float32(LIGHT_VISIBILITY_THRESHOLD)
            else:
                blk_id = BLOCK_OUT_OF_BOUNDS
                item_id = ITEM_NONE
                lit = False  # light_map pads with 0.0 → unlit

            if lit:
                if 0 <= blk_id and blk_id < NUM_BLOCK_TYPES:
                    obs[unsafe_offset=tb + OBS_CH_BLOCK_BASE + blk_id] = Float32(1.0)
                if 0 <= item_id and item_id < NUM_ITEM_TYPES:
                    obs[unsafe_offset=tb + OBS_CH_ITEM_BASE + item_id] = Float32(1.0)
                obs[unsafe_offset=tb + OBS_CH_LIGHT] = Float32(1.0)
            # else: leave all channels at 0 — matches reference darkness mask.

    # ------------------------------------------------------------------
    # 2. Mob channels per class. Walk the 5 arrays directly; the
    #    inner helper skips dead slots and clips to the view window.
    # ------------------------------------------------------------------
    var melee_base = (
        S_MELEE_MOBS_BASE + floor * MAX_MELEE_MOBS * MOB_FIELDS
    )
    var passive_base = (
        S_PASSIVE_MOBS_BASE + floor * MAX_PASSIVE_MOBS * MOB_FIELDS
    )
    var ranged_base = (
        S_RANGED_MOBS_BASE + floor * MAX_RANGED_MOBS * MOB_FIELDS
    )
    var mob_proj_base = (
        S_MOB_PROJECTILES_BASE + floor * MAX_MOB_PROJECTILES * PROJ_FIELDS
    )
    var player_proj_base = (
        S_PLAYER_PROJECTILES_BASE
        + floor * MAX_PLAYER_PROJECTILES * PROJ_FIELDS
    )

    _put_mob_array(
        s, obs, melee_base, MAX_MELEE_MOBS, MOB_FIELDS,
        py, px, half_h, half_w, OBS_MOB_CLASS_MELEE,
    )
    _put_mob_array(
        s, obs, passive_base, MAX_PASSIVE_MOBS, MOB_FIELDS,
        py, px, half_h, half_w, OBS_MOB_CLASS_PASSIVE,
    )
    _put_mob_array(
        s, obs, ranged_base, MAX_RANGED_MOBS, MOB_FIELDS,
        py, px, half_h, half_w, OBS_MOB_CLASS_RANGED,
    )
    _put_mob_array(
        s, obs, mob_proj_base, MAX_MOB_PROJECTILES, PROJ_FIELDS,
        py, px, half_h, half_w, OBS_MOB_CLASS_MOB_PROJ,
    )
    _put_mob_array(
        s, obs, player_proj_base, MAX_PLAYER_PROJECTILES, PROJ_FIELDS,
        py, px, half_h, half_w, OBS_MOB_CLASS_PLAYER_PROJ,
    )

    # Apply darkness mask to mob channels — reference multiplies the
    # whole view by the binary light mask before appending the light
    # channel itself.
    for lv in range(VIEW_H):
        for lx in range(VIEW_W):
            var tb = _tile_off(lv, lx)
            if obs[unsafe_offset=tb + OBS_CH_LIGHT] == Float32(0.0):
                for c in range(OBS_MOB_CLASSES * OBS_MOB_TYPES_PER_CLASS):
                    obs[unsafe_offset=tb + OBS_CH_MOB_BASE + c] = Float32(0.0)

    # ------------------------------------------------------------------
    # 3. Scalar tail — 51 floats.
    # ------------------------------------------------------------------
    var off = OBS_VIEW_SIZE

    # inventory (16) — order matches reference renderer.
    obs[unsafe_offset=off + 0]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_WOOD)])))     * Float32(0.1)
    obs[unsafe_offset=off + 1]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_STONE)])))    * Float32(0.1)
    obs[unsafe_offset=off + 2]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_COAL)])))     * Float32(0.1)
    obs[unsafe_offset=off + 3]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_IRON)])))     * Float32(0.1)
    obs[unsafe_offset=off + 4]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_DIAMOND)])))  * Float32(0.1)
    obs[unsafe_offset=off + 5]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_SAPPHIRE)]))) * Float32(0.1)
    obs[unsafe_offset=off + 6]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_RUBY)])))     * Float32(0.1)
    obs[unsafe_offset=off + 7]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_SAPLING)])))  * Float32(0.1)
    obs[unsafe_offset=off + 8]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_TORCHES)])))  * Float32(0.1)
    obs[unsafe_offset=off + 9]  = sqrt(Float32(Int(s[unsafe_offset=s_inv(INV_ARROWS)])))   * Float32(0.1)
    obs[unsafe_offset=off + 10] = Float32(Int(s[unsafe_offset=s_inv(INV_BOOKS)]))   * Float32(0.5)
    obs[unsafe_offset=off + 11] = Float32(Int(s[unsafe_offset=s_inv(INV_PICKAXE)])) * Float32(0.25)
    obs[unsafe_offset=off + 12] = Float32(Int(s[unsafe_offset=s_inv(INV_SWORD)]))   * Float32(0.25)
    obs[unsafe_offset=off + 13] = s[unsafe_offset=S_SWORD_ENCHANT]
    obs[unsafe_offset=off + 14] = s[unsafe_offset=S_BOW_ENCHANT]
    obs[unsafe_offset=off + 15] = Float32(Int(s[unsafe_offset=s_inv(INV_BOW)]))
    off += OBS_INV_SIZE  # +16

    # potions (6)
    for k in range(NUM_POTIONS):
        var v = Float32(Int(s[unsafe_offset=s_inv(INV_POTIONS_BASE + k)]))
        obs[unsafe_offset=off + k] = sqrt(v) * Float32(0.1)
    off += NUM_POTIONS  # +6

    # intrinsics (9)
    obs[unsafe_offset=off + 0] = Float32(Int(s[unsafe_offset=s_intrinsic(INTRINSIC_HEALTH)])) * Float32(0.1)
    obs[unsafe_offset=off + 1] = Float32(Int(s[unsafe_offset=s_intrinsic(INTRINSIC_FOOD)]))   * Float32(0.1)
    obs[unsafe_offset=off + 2] = Float32(Int(s[unsafe_offset=s_intrinsic(INTRINSIC_DRINK)]))  * Float32(0.1)
    obs[unsafe_offset=off + 3] = Float32(Int(s[unsafe_offset=s_intrinsic(INTRINSIC_ENERGY)])) * Float32(0.1)
    obs[unsafe_offset=off + 4] = Float32(Int(s[unsafe_offset=s_intrinsic(INTRINSIC_MANA)]))   * Float32(0.1)
    obs[unsafe_offset=off + 5] = Float32(Int(s[unsafe_offset=s_attribute(ATTR_XP)]))          * Float32(0.1)
    obs[unsafe_offset=off + 6] = Float32(Int(s[unsafe_offset=s_attribute(ATTR_DEXTERITY)]))   * Float32(0.1)
    obs[unsafe_offset=off + 7] = Float32(Int(s[unsafe_offset=s_attribute(ATTR_STRENGTH)]))    * Float32(0.1)
    obs[unsafe_offset=off + 8] = Float32(Int(s[unsafe_offset=s_attribute(ATTR_INTELLIGENCE)])) * Float32(0.1)
    off += OBS_INTRINSICS_SIZE  # +9

    # direction one-hot (4) — Mojo dirs are 0..3, no `-1` shift.
    var d = Int(s[unsafe_offset=S_PLAYER_DIR])
    for k in range(NUM_DIRECTIONS):
        obs[unsafe_offset=off + k] = Float32(1.0) if k == d else Float32(0.0)
    off += OBS_DIRECTION_SIZE  # +4

    # armour (4) — tier-encoded each, /2 to normalize.
    obs[unsafe_offset=off + 0] = Float32(Int(s[unsafe_offset=s_inv(INV_ARMOUR_HEAD)])) * Float32(0.5)
    obs[unsafe_offset=off + 1] = Float32(Int(s[unsafe_offset=s_inv(INV_ARMOUR_BODY)])) * Float32(0.5)
    obs[unsafe_offset=off + 2] = Float32(Int(s[unsafe_offset=s_inv(INV_ARMOUR_LEGS)])) * Float32(0.5)
    obs[unsafe_offset=off + 3] = Float32(Int(s[unsafe_offset=s_inv(INV_ARMOUR_FEET)])) * Float32(0.5)
    off += OBS_ARMOUR_SIZE  # +4

    # armour enchants (4)
    for k in range(NUM_ARMOUR_ENCHANTS):
        obs[unsafe_offset=off + k] = s[unsafe_offset=s_armour_enchant(k)]
    off += OBS_ARMOUR_ENCH_SIZE  # +4

    # special (8)
    var floor_cleared = (
        Int(s[unsafe_offset=s_monsters_killed(floor)]) >= MONSTERS_KILLED_TO_CLEAR_LEVEL
    )
    obs[unsafe_offset=off + 0] = s[unsafe_offset=S_LIGHT_LEVEL]
    obs[unsafe_offset=off + 1] = Float32(Int(s[unsafe_offset=s_intrinsic(INTRINSIC_IS_SLEEPING)]))
    obs[unsafe_offset=off + 2] = Float32(Int(s[unsafe_offset=s_intrinsic(INTRINSIC_IS_RESTING)]))
    obs[unsafe_offset=off + 3] = s[unsafe_offset=s_learned_spell(0)]
    obs[unsafe_offset=off + 4] = s[unsafe_offset=s_learned_spell(1)]
    obs[unsafe_offset=off + 5] = Float32(floor) * Float32(0.1)
    obs[unsafe_offset=off + 6] = Float32(1.0) if floor_cleared else Float32(0.0)
    obs[unsafe_offset=off + 7] = (
        Float32(1.0) if _is_boss_vulnerable(s, floor) else Float32(0.0)
    )
