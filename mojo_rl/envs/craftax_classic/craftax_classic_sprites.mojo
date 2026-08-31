"""Craftax-Classic sprite sheet — PNG textures packed into one buffer.

Loads the 16×16 RGBA sprites under `assets/` (copied from the original
Craftax repo) and packs them into a horizontal sprite sheet that the
renderer uploads as a single SDL3 texture, then samples via src-rect.

Layout: SHEET_WIDTH = SPRITE_SIZE × NUM_SPRITES, SHEET_HEIGHT = SPRITE_SIZE.
Each sprite occupies `[idx * 16 .. (idx + 1) * 16) × [0 .. 16)` in the sheet.

Sprite indices below are comptime constants used by the renderer to compute
the src-rect for any tile or icon. Block sprite indices intentionally match
the BLOCK_* constants in `constants.mojo` for the 17 block types, so a
block_id can be used directly as a sprite index without a translation table.
"""

from std.memory import alloc, unsafe_memset, Pointer
from mojo_rl.io.image import resize_nearest_pil
from mojo_rl.io.png import load_png_file, to_rgba
from mojo_rl.nn.core.ptr import mptr


# ============================================================================
# Sheet geometry
# ============================================================================

comptime SPRITE_SIZE: Int = 16
comptime SPRITE_BPP: Int = 4  # RGBA8

# Sprite indices. Blocks 0..16 mirror BLOCK_* IDs; everything else follows.
# Blocks (match BLOCK_* in constants.mojo):
comptime SPR_INVALID: Int = 0          # blank/debug
comptime SPR_OOB: Int = 1              # opaque black
comptime SPR_GRASS: Int = 2
comptime SPR_WATER: Int = 3
comptime SPR_STONE: Int = 4
comptime SPR_TREE: Int = 5
comptime SPR_WOOD: Int = 6
comptime SPR_PATH: Int = 7
comptime SPR_COAL: Int = 8
comptime SPR_IRON: Int = 9
comptime SPR_DIAMOND: Int = 10
comptime SPR_TABLE: Int = 11
comptime SPR_FURNACE: Int = 12
comptime SPR_SAND: Int = 13
comptime SPR_LAVA: Int = 14
comptime SPR_PLANT_YOUNG: Int = 15
comptime SPR_PLANT_RIPE: Int = 16

# Mobs.
comptime SPR_ZOMBIE: Int = 17
comptime SPR_COW: Int = 18
comptime SPR_SKELETON: Int = 19
comptime SPR_ARROW_UP: Int = 20
comptime SPR_ARROW_DOWN: Int = 21
comptime SPR_ARROW_LEFT: Int = 22
comptime SPR_ARROW_RIGHT: Int = 23

# Player (5 states).
comptime SPR_PLAYER_UP: Int = 24
comptime SPR_PLAYER_DOWN: Int = 25
comptime SPR_PLAYER_LEFT: Int = 26
comptime SPR_PLAYER_RIGHT: Int = 27
comptime SPR_PLAYER_SLEEP: Int = 28

# Inventory icons.
comptime SPR_INV_WOOD: Int = 29          # log
comptime SPR_INV_SAPLING: Int = 30
comptime SPR_INV_WOOD_PICKAXE: Int = 31
comptime SPR_INV_STONE_PICKAXE: Int = 32
comptime SPR_INV_IRON_PICKAXE: Int = 33
comptime SPR_INV_WOOD_SWORD: Int = 34
comptime SPR_INV_STONE_SWORD: Int = 35
comptime SPR_INV_IRON_SWORD: Int = 36

# Intrinsic bar icons.
comptime SPR_ICON_HEALTH: Int = 37
comptime SPR_ICON_FOOD: Int = 38
comptime SPR_ICON_DRINK: Int = 39
comptime SPR_ICON_ENERGY: Int = 40

comptime NUM_SPRITES: Int = 41
comptime SHEET_WIDTH: Int = SPRITE_SIZE * NUM_SPRITES   # 656
comptime SHEET_HEIGHT: Int = SPRITE_SIZE                # 16
comptime SHEET_BYTES: Int = SHEET_WIDTH * SHEET_HEIGHT * SPRITE_BPP


# ============================================================================
# Loader — native PNG decode (io/png.mojo), no interpreter
# ============================================================================
#
# We can't use `load_png` directly because it returns a TextureData with a
# List[UInt8] payload; we'd then copy that into the sheet buffer. To save
# one copy and keep the heap simpler, we open each PNG with PIL and blit
# its bytes into the sheet in a single pass per sprite.


def _blit_sprite_to_sheet(
    sheet: Pointer[UInt8, MutUntrackedOrigin],
    slot_idx: Int,
    ref raw_bytes: List[UInt8],
    src_w: Int,
    src_h: Int,
) raises:
    """Copy a SPRITE_SIZE×SPRITE_SIZE region from `raw_bytes` into the
    sheet at the given slot index. Source bytes must be RGBA8 row-major.
    If the source is larger than SPRITE_SIZE, only the top-left region is
    used (Craftax classic sprites are all 16×16 so this never triggers)."""
    var copy_w = src_w if src_w < SPRITE_SIZE else SPRITE_SIZE
    var copy_h = src_h if src_h < SPRITE_SIZE else SPRITE_SIZE
    var dst_x = slot_idx * SPRITE_SIZE
    for y in range(copy_h):
        for x in range(copy_w):
            var src_off = (y * src_w + x) * SPRITE_BPP
            var dst_off = (y * SHEET_WIDTH + (dst_x + x)) * SPRITE_BPP
            sheet[unsafe_offset=dst_off + 0] = raw_bytes[src_off + 0]
            sheet[unsafe_offset=dst_off + 1] = raw_bytes[src_off + 1]
            sheet[unsafe_offset=dst_off + 2] = raw_bytes[src_off + 2]
            sheet[unsafe_offset=dst_off + 3] = raw_bytes[src_off + 3]


def _load_one(
    sheet: Pointer[UInt8, MutUntrackedOrigin],
    slot_idx: Int,
    asset_dir: String,
    filename: String,
) raises:
    """Open `<asset_dir>/<filename>`, convert to RGBA, blit to sheet[slot_idx]."""
    var img = load_png_file(asset_dir + "/" + filename)
    var raw = to_rgba(img)
    _blit_sprite_to_sheet(sheet, slot_idx, raw, img.width, img.height)


def build_sprite_sheet(
    asset_dir: String,
) raises -> Pointer[UInt8, MutUntrackedOrigin]:
    """Allocate and populate the full Craftax-Classic sprite sheet.

    Returns a heap buffer the caller owns (must be freed via `.free()`).
    Sheet layout matches the comptime SPR_* indices above.
    """
    var sheet = alloc[UInt8]({count = SHEET_BYTES}).unsafe_leak()
    # Zero the sheet first so SPR_INVALID is transparent and SPR_OOB stays
    # at solid black-with-alpha (we set its alpha explicitly below).
    unsafe_memset(sheet, UInt8(0), SHEET_BYTES)
    # OOB sprite: opaque dark slate so it's clearly the void.
    for y in range(SPRITE_SIZE):
        for x in range(SPRITE_SIZE):
            var off = (
                y * SHEET_WIDTH + (SPR_OOB * SPRITE_SIZE + x)
            ) * SPRITE_BPP
            sheet[unsafe_offset=off + 0] = UInt8(15)
            sheet[unsafe_offset=off + 1] = UInt8(15)
            sheet[unsafe_offset=off + 2] = UInt8(25)
            sheet[unsafe_offset=off + 3] = UInt8(255)

    # Block sprites (indices match BLOCK_*).
    _load_one(sheet, SPR_GRASS, asset_dir, "grass.png")
    _load_one(sheet, SPR_WATER, asset_dir, "water.png")
    _load_one(sheet, SPR_STONE, asset_dir, "stone.png")
    _load_one(sheet, SPR_TREE, asset_dir, "tree.png")
    _load_one(sheet, SPR_WOOD, asset_dir, "wood.png")
    _load_one(sheet, SPR_PATH, asset_dir, "path.png")
    _load_one(sheet, SPR_COAL, asset_dir, "coal.png")
    _load_one(sheet, SPR_IRON, asset_dir, "iron.png")
    _load_one(sheet, SPR_DIAMOND, asset_dir, "diamond.png")
    _load_one(sheet, SPR_TABLE, asset_dir, "table.png")
    _load_one(sheet, SPR_FURNACE, asset_dir, "furnace.png")
    _load_one(sheet, SPR_SAND, asset_dir, "sand.png")
    _load_one(sheet, SPR_LAVA, asset_dir, "lava.png")
    _load_one(sheet, SPR_PLANT_YOUNG, asset_dir, "plant-young.png")
    _load_one(sheet, SPR_PLANT_RIPE, asset_dir, "plant-ripe.png")

    # Mobs.
    _load_one(sheet, SPR_ZOMBIE, asset_dir, "zombie.png")
    _load_one(sheet, SPR_COW, asset_dir, "cow.png")
    _load_one(sheet, SPR_SKELETON, asset_dir, "skeleton.png")
    _load_one(sheet, SPR_ARROW_UP, asset_dir, "arrow-up.png")
    _load_one(sheet, SPR_ARROW_DOWN, asset_dir, "arrow-down.png")
    _load_one(sheet, SPR_ARROW_LEFT, asset_dir, "arrow-left.png")
    _load_one(sheet, SPR_ARROW_RIGHT, asset_dir, "arrow-right.png")

    # Player.
    _load_one(sheet, SPR_PLAYER_UP, asset_dir, "player-up.png")
    _load_one(sheet, SPR_PLAYER_DOWN, asset_dir, "player-down.png")
    _load_one(sheet, SPR_PLAYER_LEFT, asset_dir, "player-left.png")
    _load_one(sheet, SPR_PLAYER_RIGHT, asset_dir, "player-right.png")
    _load_one(sheet, SPR_PLAYER_SLEEP, asset_dir, "player-sleep.png")

    # Inventory icons.
    _load_one(sheet, SPR_INV_WOOD, asset_dir, "log.png")
    _load_one(sheet, SPR_INV_SAPLING, asset_dir, "sapling.png")
    _load_one(sheet, SPR_INV_WOOD_PICKAXE, asset_dir, "wood_pickaxe.png")
    _load_one(sheet, SPR_INV_STONE_PICKAXE, asset_dir, "stone_pickaxe.png")
    _load_one(sheet, SPR_INV_IRON_PICKAXE, asset_dir, "iron_pickaxe.png")
    _load_one(sheet, SPR_INV_WOOD_SWORD, asset_dir, "wood_sword.png")
    _load_one(sheet, SPR_INV_STONE_SWORD, asset_dir, "stone_sword.png")
    _load_one(sheet, SPR_INV_IRON_SWORD, asset_dir, "iron_sword.png")

    # Intrinsic icons.
    _load_one(sheet, SPR_ICON_HEALTH, asset_dir, "health.png")
    _load_one(sheet, SPR_ICON_FOOD, asset_dir, "food.png")
    _load_one(sheet, SPR_ICON_DRINK, asset_dir, "drink.png")
    _load_one(sheet, SPR_ICON_ENERGY, asset_dir, "energy.png")

    return sheet


# ============================================================================
# Agent pixel-obs atlas — downsampled RGBA float32 sprites for the kernel
# ============================================================================
#
# The agent's pixel observation is rendered at `block_pixel_size` (typically 10
# in Craftax-spec), much smaller than the 16×16 PNGs. We pre-resize every
# sprite to `(BPS, BPS, 4) float32 in [0, 1]` and pack them into a single
# flat buffer, so the per-pixel render kernel can look up `(sprite_idx,
# ly, lx) → (r, g, b, a)` with one offset arithmetic.
#
# Atlas layout (flat float32):
#   [0 .. NUM_SPRITES * BPS * BPS * 4)     main sprites in slot order
# The OOB slot is set to opaque dark slate by hand (matches the renderer).


@always_inline
def agent_atlas_size(block_pixel_size: Int) -> Int:
    """Number of float32 entries in the agent pixel-obs atlas."""
    return NUM_SPRITES * block_pixel_size * block_pixel_size * 4


def _blit_resized_to_atlas(
    atlas: Pointer[Float32, MutUntrackedOrigin],
    slot_idx: Int,
    block_pixel_size: Int,
    ref raw_bytes: List[UInt8],
) raises:
    """Copy a `BPS×BPS×RGBA` region (already resized) into atlas slot."""
    var bps = block_pixel_size
    var slot_base = slot_idx * bps * bps * 4
    for y in range(bps):
        for x in range(bps):
            var src_off = (y * bps + x) * 4
            var dst_off = slot_base + (y * bps + x) * 4
            atlas[unsafe_offset=dst_off + 0] = Float32(Int(raw_bytes[src_off + 0])) / Float32(255.0)
            atlas[unsafe_offset=dst_off + 1] = Float32(Int(raw_bytes[src_off + 1])) / Float32(255.0)
            atlas[unsafe_offset=dst_off + 2] = Float32(Int(raw_bytes[src_off + 2])) / Float32(255.0)
            atlas[unsafe_offset=dst_off + 3] = Float32(Int(raw_bytes[src_off + 3])) / Float32(255.0)


def _load_one_resized(
    atlas: Pointer[Float32, MutUntrackedOrigin],
    slot_idx: Int,
    block_pixel_size: Int,
    asset_dir: String,
    filename: String,
) raises:
    """Open `<asset_dir>/<filename>`, RGBA-convert, nearest-resize to
    `BPS×BPS`, and write float32 [0,1] into atlas[slot_idx].

    ⚠ `resize_nearest_pil` REPRODUCES PILLOW'S ARITHMETIC, not the textbook
    formula — Pillow walks a floating-point accumulator, and the obvious
    `floor((x + 0.5) * in / out)` picks a different source pixel on 93 of 600
    random size pairs. A sprite atlas built with the wrong one looks almost
    right, which is why `tests/io/test_resize_nearest.mojo` exists."""
    var img = load_png_file(asset_dir + "/" + filename)
    var rgba = to_rgba(img)
    var small = List[UInt8]()
    small.resize(block_pixel_size * block_pixel_size * 4, 0)
    resize_nearest_pil(
        mptr(rgba.unsafe_ptr()),
        img.height,
        img.width,
        mptr(small.unsafe_ptr()),
        block_pixel_size,
        block_pixel_size,
        4,
    )
    _blit_resized_to_atlas(atlas, slot_idx, block_pixel_size, small)


def build_agent_atlas(
    asset_dir: String,
    block_pixel_size: Int,
) raises -> Pointer[Float32, MutUntrackedOrigin]:
    """Build the float32 RGBA atlas at the agent's small block_pixel_size.

    Returns a heap buffer owned by caller (must be freed via `.free()`).
    All non-empty slots are populated; OOB is set to opaque dark slate;
    SPR_INVALID is fully transparent.
    """
    var bps = block_pixel_size
    var size = NUM_SPRITES * bps * bps * 4
    var atlas = alloc[Float32]({count = size}).unsafe_leak()
    # Zero (transparent) by default.
    for i in range(size):
        atlas[unsafe_offset=i] = Float32(0.0)
    # OOB slot — opaque dark slate.
    for y in range(bps):
        for x in range(bps):
            var off = SPR_OOB * bps * bps * 4 + (y * bps + x) * 4
            atlas[unsafe_offset=off + 0] = Float32(15.0) / Float32(255.0)
            atlas[unsafe_offset=off + 1] = Float32(15.0) / Float32(255.0)
            atlas[unsafe_offset=off + 2] = Float32(25.0) / Float32(255.0)
            atlas[unsafe_offset=off + 3] = Float32(1.0)

    # Blocks (indices match BLOCK_*).
    _load_one_resized(atlas, SPR_GRASS, bps, asset_dir, "grass.png")
    _load_one_resized(atlas, SPR_WATER, bps, asset_dir, "water.png")
    _load_one_resized(atlas, SPR_STONE, bps, asset_dir, "stone.png")
    _load_one_resized(atlas, SPR_TREE, bps, asset_dir, "tree.png")
    _load_one_resized(atlas, SPR_WOOD, bps, asset_dir, "wood.png")
    _load_one_resized(atlas, SPR_PATH, bps, asset_dir, "path.png")
    _load_one_resized(atlas, SPR_COAL, bps, asset_dir, "coal.png")
    _load_one_resized(atlas, SPR_IRON, bps, asset_dir, "iron.png")
    _load_one_resized(atlas, SPR_DIAMOND, bps, asset_dir, "diamond.png")
    _load_one_resized(atlas, SPR_TABLE, bps, asset_dir, "table.png")
    _load_one_resized(atlas, SPR_FURNACE, bps, asset_dir, "furnace.png")
    _load_one_resized(atlas, SPR_SAND, bps, asset_dir, "sand.png")
    _load_one_resized(atlas, SPR_LAVA, bps, asset_dir, "lava.png")
    _load_one_resized(atlas, SPR_PLANT_YOUNG, bps, asset_dir, "plant-young.png")
    _load_one_resized(atlas, SPR_PLANT_RIPE, bps, asset_dir, "plant-ripe.png")

    # Mobs.
    _load_one_resized(atlas, SPR_ZOMBIE, bps, asset_dir, "zombie.png")
    _load_one_resized(atlas, SPR_COW, bps, asset_dir, "cow.png")
    _load_one_resized(atlas, SPR_SKELETON, bps, asset_dir, "skeleton.png")
    _load_one_resized(atlas, SPR_ARROW_UP, bps, asset_dir, "arrow-up.png")
    _load_one_resized(atlas, SPR_ARROW_DOWN, bps, asset_dir, "arrow-down.png")
    _load_one_resized(atlas, SPR_ARROW_LEFT, bps, asset_dir, "arrow-left.png")
    _load_one_resized(atlas, SPR_ARROW_RIGHT, bps, asset_dir, "arrow-right.png")

    # Player.
    _load_one_resized(atlas, SPR_PLAYER_UP, bps, asset_dir, "player-up.png")
    _load_one_resized(atlas, SPR_PLAYER_DOWN, bps, asset_dir, "player-down.png")
    _load_one_resized(atlas, SPR_PLAYER_LEFT, bps, asset_dir, "player-left.png")
    _load_one_resized(atlas, SPR_PLAYER_RIGHT, bps, asset_dir, "player-right.png")
    _load_one_resized(atlas, SPR_PLAYER_SLEEP, bps, asset_dir, "player-sleep.png")

    # Inventory icons.
    _load_one_resized(atlas, SPR_INV_WOOD, bps, asset_dir, "log.png")
    _load_one_resized(atlas, SPR_INV_SAPLING, bps, asset_dir, "sapling.png")
    _load_one_resized(atlas, SPR_INV_WOOD_PICKAXE, bps, asset_dir, "wood_pickaxe.png")
    _load_one_resized(atlas, SPR_INV_STONE_PICKAXE, bps, asset_dir, "stone_pickaxe.png")
    _load_one_resized(atlas, SPR_INV_IRON_PICKAXE, bps, asset_dir, "iron_pickaxe.png")
    _load_one_resized(atlas, SPR_INV_WOOD_SWORD, bps, asset_dir, "wood_sword.png")
    _load_one_resized(atlas, SPR_INV_STONE_SWORD, bps, asset_dir, "stone_sword.png")
    _load_one_resized(atlas, SPR_INV_IRON_SWORD, bps, asset_dir, "iron_sword.png")

    # Intrinsic icons.
    _load_one_resized(atlas, SPR_ICON_HEALTH, bps, asset_dir, "health.png")
    _load_one_resized(atlas, SPR_ICON_FOOD, bps, asset_dir, "food.png")
    _load_one_resized(atlas, SPR_ICON_DRINK, bps, asset_dir, "drink.png")
    _load_one_resized(atlas, SPR_ICON_ENERGY, bps, asset_dir, "energy.png")

    return atlas
