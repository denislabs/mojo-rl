"""Craftax-Full sprite sheet — PNG textures packed into one buffer.

Same packing strategy as `craftax_classic_sprites.mojo`, scaled up to cover
Full Craftax's 9 floors: all 37 block types, item overlays (torch / ladder /
blocked ladder), 24 mob species (8 per class × 3 classes), 8 projectile
species + 4 directional arrow overlays, player (5 poses), 4-tier
pickaxes / swords / armour, potions, spell icons, and 9 intrinsic icons.

Sheet layout: SHEET_WIDTH = SPRITE_SIZE × NUM_SPRITES, SHEET_HEIGHT = SPRITE_SIZE.
Slot `idx` occupies `[idx*16 .. (idx+1)*16) × [0 .. 16)` and is sampled via
src-rect by the renderer.

Block sprites occupy slots 0..36 so a `block_id` is its own sprite index
(matches the BLOCK_* enum). Everything else follows the layout below.
"""

from std.memory import alloc, memset, UnsafePointer
from std.python import Python, PythonObject


# ============================================================================
# Sheet geometry
# ============================================================================

comptime SPRITE_SIZE: Int = 16
comptime SPRITE_BPP: Int = 4  # RGBA8

# --- Block sprites (slots 0..36 — direct block_id index) ------------------
comptime SPR_BLOCK_BASE: Int = 0
# Slot k = block k; we don't enumerate the 37 individual names here — the
# renderer just blits with block_id as the slot index.

# --- Item overlay sprites (slots 37..41 — direct item_id index) -----------
comptime SPR_ITEM_BASE: Int = 37        # ITEM_NONE is transparent
comptime SPR_ITEM_TORCH: Int = 38       # ITEM_TORCH
comptime SPR_ITEM_LADDER_DOWN: Int = 39
comptime SPR_ITEM_LADDER_UP: Int = 40
comptime SPR_ITEM_LADDER_DOWN_BLOCKED: Int = 41

# --- Mob sprites: 8 species per class, 3 classes (slots 42..65) ----------
comptime SPR_PASSIVE_BASE: Int = 42  # passive species 0..7
comptime SPR_MELEE_BASE: Int = 50    # melee species 0..7
comptime SPR_RANGED_BASE: Int = 58   # ranged species 0..7

# --- Projectile sprites: 8 species (slots 66..73) ------------------------
comptime SPR_PROJ_BASE: Int = 66
# Directional arrow overlays (slot 74..77 — used for projectile type 0/4 if
# we want to draw a rotated arrow). Order matches DIR_LEFT/RIGHT/UP/DOWN.
comptime SPR_ARROW_LEFT: Int = 74
comptime SPR_ARROW_RIGHT: Int = 75
comptime SPR_ARROW_UP: Int = 76
comptime SPR_ARROW_DOWN: Int = 77

# --- Player sprites (slots 78..82) ---------------------------------------
comptime SPR_PLAYER_LEFT: Int = 78
comptime SPR_PLAYER_RIGHT: Int = 79
comptime SPR_PLAYER_UP: Int = 80
comptime SPR_PLAYER_DOWN: Int = 81
comptime SPR_PLAYER_SLEEP: Int = 82

# --- Inventory icons -----------------------------------------------------
# Pickaxe by tier (0..4): tier 0 is empty/transparent.
comptime SPR_PICKAXE_BASE: Int = 83  # tier 0 = empty, 1..4 = wood/stone/iron/diamond
# Sword by tier (0..4).
comptime SPR_SWORD_BASE: Int = 88   # tier 0 = empty, 1..4 = wood/stone/iron/diamond
# Bow (single).
comptime SPR_BOW: Int = 93
# Armour: 4 pieces × 3 tiers (0=none, 1=iron, 2=diamond) → 12 slots, but
# tier 0 is empty so we only populate 8.
# Layout: SPR_ARMOUR_BASE + piece * 3 + tier. piece ∈ {0=head,1=body,2=legs,3=feet}.
comptime SPR_ARMOUR_BASE: Int = 94
# Materials (we re-use block sprites for stone/coal/iron/diamond/sapphire/ruby
# via the block_id slots) but log and sapling need dedicated icons.
comptime SPR_INV_LOG: Int = 106
comptime SPR_INV_SAPLING: Int = 107
# Tools / consumables.
comptime SPR_INV_TORCH: Int = 108
comptime SPR_INV_BOOK: Int = 109
# Potions (6).
comptime SPR_POTION_BASE: Int = 110  # red, green, blue, pink, cyan, yellow
# Spell icons (re-use fireball / iceball projectile textures at smaller size).
comptime SPR_SPELL_FIREBALL: Int = 116
comptime SPR_SPELL_ICEBALL: Int = 117

# --- Intrinsic bar icons (slots 118..126) --------------------------------
comptime SPR_ICON_HEALTH: Int = 118
comptime SPR_ICON_FOOD: Int = 119
comptime SPR_ICON_DRINK: Int = 120
comptime SPR_ICON_ENERGY: Int = 121
comptime SPR_ICON_MANA: Int = 122
comptime SPR_ICON_XP: Int = 123
comptime SPR_ICON_DEX: Int = 124
comptime SPR_ICON_STR: Int = 125
comptime SPR_ICON_INT: Int = 126

comptime NUM_SPRITES: Int = 127

comptime SHEET_WIDTH: Int = SPRITE_SIZE * NUM_SPRITES   # 127 × 16 = 2032
comptime SHEET_HEIGHT: Int = SPRITE_SIZE
comptime SHEET_BYTES: Int = SHEET_WIDTH * SHEET_HEIGHT * SPRITE_BPP


# ============================================================================
# Loader helpers (PIL via std.python)
# ============================================================================


def _blit_sprite_to_sheet(
    sheet: UnsafePointer[UInt8, MutUntrackedOrigin],
    slot_idx: Int,
    raw_bytes: PythonObject,
    src_w: Int,
    src_h: Int,
) raises:
    """Copy a SPRITE_SIZE×SPRITE_SIZE region from `raw_bytes` (RGBA8 row-major)
    into the sheet at the given slot index. Source larger than SPRITE_SIZE
    has its top-left region used."""
    var copy_w = src_w if src_w < SPRITE_SIZE else SPRITE_SIZE
    var copy_h = src_h if src_h < SPRITE_SIZE else SPRITE_SIZE
    var dst_x = slot_idx * SPRITE_SIZE
    for y in range(copy_h):
        for x in range(copy_w):
            var src_off = (y * src_w + x) * SPRITE_BPP
            var dst_off = (y * SHEET_WIDTH + (dst_x + x)) * SPRITE_BPP
            sheet[dst_off + 0] = UInt8(Int(py=raw_bytes[src_off + 0]))
            sheet[dst_off + 1] = UInt8(Int(py=raw_bytes[src_off + 1]))
            sheet[dst_off + 2] = UInt8(Int(py=raw_bytes[src_off + 2]))
            sheet[dst_off + 3] = UInt8(Int(py=raw_bytes[src_off + 3]))


def _load_one(
    sheet: UnsafePointer[UInt8, MutUntrackedOrigin],
    slot_idx: Int,
    asset_dir: String,
    filename: String,
    pil: PythonObject,
) raises:
    """Open `<asset_dir>/<filename>`, convert to RGBA, blit to sheet[slot_idx]."""
    var path = asset_dir + "/" + filename
    var img = pil.open(path).convert("RGBA")
    var w = Int(py=img.width)
    var h = Int(py=img.height)
    var raw = img.tobytes()
    _blit_sprite_to_sheet(sheet, slot_idx, raw, w, h)


def _load_blocks(
    sheet: UnsafePointer[UInt8, MutUntrackedOrigin],
    asset_dir: String,
    pil: PythonObject,
) raises:
    """Block sprites — slots 0..36, indexed by BLOCK_* enum.

    Slot 0 = INVALID (transparent), slot 1 = OUT_OF_BOUNDS (dark slate set
    in `build_sprite_sheet`), slot 18 = DARKNESS (stays transparent)."""
    var names = [
        "grass.png", "water.png", "stone.png", "tree.png",
        "wood.png", "path.png", "coal.png", "iron.png",
        "diamond.png", "table.png", "furnace.png", "sand.png",
        "lava.png", "plant-young.png", "plant-ripe.png", "wall.png",
    ]
    for k in range(len(names)):
        _load_one(sheet, 2 + k, asset_dir, names[k], pil)
    var more = [
        "wall_moss.png", "stalagmite.png", "sapphire.png", "ruby.png",
        "chest.png", "fountain.png", "fire_grass.png", "ice_grass.png",
        "gravel.png", "fire_tree.png", "ice_shrub.png",
        "enchantment_table_fire.png", "enchantment_table_ice.png",
        "necromancer.png", "grave.png", "grave2.png", "grave3.png",
        "necromancer_vulnerable.png",
    ]
    for k in range(len(more)):
        _load_one(sheet, 19 + k, asset_dir, more[k], pil)


def _load_items_and_player(
    sheet: UnsafePointer[UInt8, MutUntrackedOrigin],
    asset_dir: String,
    pil: PythonObject,
) raises:
    """Item overlays, directional arrows, and player poses."""
    # Items (slots 38..41); slot 37 stays transparent.
    _load_one(sheet, SPR_ITEM_TORCH, asset_dir, "torch_on_path.png", pil)
    _load_one(sheet, SPR_ITEM_LADDER_DOWN, asset_dir, "ladder_down.png", pil)
    _load_one(sheet, SPR_ITEM_LADDER_UP, asset_dir, "ladder_up.png", pil)
    _load_one(
        sheet, SPR_ITEM_LADDER_DOWN_BLOCKED, asset_dir,
        "ladder_down_blocked.png", pil,
    )
    # Directional arrow overlays.
    _load_one(sheet, SPR_ARROW_LEFT, asset_dir, "arrow-left.png", pil)
    _load_one(sheet, SPR_ARROW_RIGHT, asset_dir, "arrow-right.png", pil)
    _load_one(sheet, SPR_ARROW_UP, asset_dir, "arrow-up.png", pil)
    _load_one(sheet, SPR_ARROW_DOWN, asset_dir, "arrow-down.png", pil)
    # Player poses.
    _load_one(sheet, SPR_PLAYER_LEFT, asset_dir, "player-left.png", pil)
    _load_one(sheet, SPR_PLAYER_RIGHT, asset_dir, "player-right.png", pil)
    _load_one(sheet, SPR_PLAYER_UP, asset_dir, "player-up.png", pil)
    _load_one(sheet, SPR_PLAYER_DOWN, asset_dir, "player-down.png", pil)
    _load_one(sheet, SPR_PLAYER_SLEEP, asset_dir, "player-sleep.png", pil)


def _load_mobs(
    sheet: UnsafePointer[UInt8, MutUntrackedOrigin],
    asset_dir: String,
    pil: PythonObject,
) raises:
    """Mob + projectile sprites — 8 species per class × 4 classes."""
    # Passive (8): cow/bat/snail + 5 aliases to cow.
    var passive = [
        "cow.png", "bat.png", "snail.png",
        "cow.png", "cow.png", "cow.png", "cow.png", "cow.png",
    ]
    for k in range(8):
        _load_one(sheet, SPR_PASSIVE_BASE + k, asset_dir, passive[k], pil)
    # Melee (8) — 8 distinct species.
    var melee = [
        "zombie.png", "gnome_warrior.png", "orc_soldier.png", "lizard.png",
        "knight.png", "troll.png", "pigman.png", "frost_troll.png",
    ]
    for k in range(8):
        _load_one(sheet, SPR_MELEE_BASE + k, asset_dir, melee[k], pil)
    # Ranged (8).
    var ranged = [
        "skeleton.png", "gnome_archer.png", "orc_mage.png", "kobold.png",
        "knight_archer.png", "deep_thing.png", "fire_elemental.png",
        "ice_elemental.png",
    ]
    for k in range(8):
        _load_one(sheet, SPR_RANGED_BASE + k, asset_dir, ranged[k], pil)
    # Projectile species (arrow/dagger/fireball/iceball + variants share png).
    var proj = [
        "arrow-up.png", "dagger.png", "fireball.png", "iceball.png",
        "arrow-up.png", "slimeball.png", "fireball.png", "iceball.png",
    ]
    for k in range(8):
        _load_one(sheet, SPR_PROJ_BASE + k, asset_dir, proj[k], pil)


def _load_inventory(
    sheet: UnsafePointer[UInt8, MutUntrackedOrigin],
    asset_dir: String,
    pil: PythonObject,
) raises:
    """Tools, armour, materials, potions, spells."""
    var pickaxe_names = [
        "wood_pickaxe.png", "stone_pickaxe.png",
        "iron_pickaxe.png", "diamond_pickaxe.png",
    ]
    for k in range(4):
        _load_one(sheet, SPR_PICKAXE_BASE + 1 + k, asset_dir,
                  pickaxe_names[k], pil)
    var sword_names = [
        "wood_sword.png", "stone_sword.png",
        "iron_sword.png", "diamond_sword.png",
    ]
    for k in range(4):
        _load_one(sheet, SPR_SWORD_BASE + 1 + k, asset_dir,
                  sword_names[k], pil)
    _load_one(sheet, SPR_BOW, asset_dir, "bow.png", pil)
    # Armour: 4 pieces × 2 tiers (iron, diamond).
    var armour_names = [
        "iron_helmet.png", "diamond_helmet.png",
        "iron_chestplate.png", "diamond_chestplate.png",
        "iron_pants.png", "diamond_pants.png",
        "iron_boots.png", "diamond_boots.png",
    ]
    for piece in range(4):
        for tier in range(2):
            _load_one(
                sheet, SPR_ARMOUR_BASE + piece * 3 + 1 + tier,
                asset_dir, armour_names[piece * 2 + tier], pil,
            )
    _load_one(sheet, SPR_INV_LOG, asset_dir, "log.png", pil)
    _load_one(sheet, SPR_INV_SAPLING, asset_dir, "sapling.png", pil)
    _load_one(sheet, SPR_INV_TORCH, asset_dir, "torch_in_inventory.png", pil)
    _load_one(sheet, SPR_INV_BOOK, asset_dir, "book.png", pil)
    # Potions.
    var potion_names = [
        "potion_red.png", "potion_green.png", "potion_blue.png",
        "potion_pink.png", "potion_cyan.png", "potion_yellow.png",
    ]
    for k in range(6):
        _load_one(sheet, SPR_POTION_BASE + k, asset_dir, potion_names[k], pil)
    # Spell icons (re-use projectile textures).
    _load_one(sheet, SPR_SPELL_FIREBALL, asset_dir, "fireball.png", pil)
    _load_one(sheet, SPR_SPELL_ICEBALL, asset_dir, "iceball.png", pil)


def _load_intrinsic_icons(
    sheet: UnsafePointer[UInt8, MutUntrackedOrigin],
    asset_dir: String,
    pil: PythonObject,
) raises:
    """Intrinsic bar icons (9 slots)."""
    var names = [
        "health.png", "food.png", "drink.png", "energy.png", "mana.png",
        "xp.png", "dexterity.png", "strength.png", "intelligence.png",
    ]
    for k in range(9):
        _load_one(sheet, SPR_ICON_HEALTH + k, asset_dir, names[k], pil)


def build_sprite_sheet(
    asset_dir: String,
) raises -> UnsafePointer[UInt8, MutUntrackedOrigin]:
    """Allocate and populate the Craftax-Full sprite sheet.

    Returns a heap buffer the caller owns (must be freed via `.free()`).
    Sheet layout matches the comptime SPR_* indices above.
    """
    var sheet = alloc[UInt8](SHEET_BYTES)
    # Default: fully transparent. BLOCK_INVALID (slot 0) and ITEM_NONE
    # (slot 37) inherit transparency this way without an explicit load.
    memset(sheet, UInt8(0), SHEET_BYTES)

    # BLOCK_OUT_OF_BOUNDS (slot 1) — opaque dark slate (matches Classic).
    for y in range(SPRITE_SIZE):
        for x in range(SPRITE_SIZE):
            var off = (
                y * SHEET_WIDTH + (1 * SPRITE_SIZE + x)
            ) * SPRITE_BPP
            sheet[off + 0] = UInt8(15)
            sheet[off + 1] = UInt8(15)
            sheet[off + 2] = UInt8(25)
            sheet[off + 3] = UInt8(255)

    var pil = Python.import_module("PIL.Image")
    _load_blocks(sheet, asset_dir, pil)
    _load_items_and_player(sheet, asset_dir, pil)
    _load_mobs(sheet, asset_dir, pil)
    _load_inventory(sheet, asset_dir, pil)
    _load_intrinsic_icons(sheet, asset_dir, pil)
    return sheet


# ============================================================================
# Agent pixel-obs atlas — RGBA float32 sprites at the agent's tiny block size
# ============================================================================
#
# Same idea as `craftax_classic_sprites.build_agent_atlas`: pre-resize every
# sprite to `(BPS, BPS, 4) float32 in [0, 1]` so the per-pixel render kernel
# can look up `(sprite_idx, ly, lx) → (r, g, b, a)` with one offset arithmetic.
# OOB sprite is set to opaque dark slate by hand; INVALID stays transparent.


@always_inline
def agent_atlas_size(block_pixel_size: Int) -> Int:
    """Number of float32 entries in the agent pixel-obs atlas."""
    return NUM_SPRITES * block_pixel_size * block_pixel_size * 4


def _blit_resized_to_atlas(
    atlas: UnsafePointer[Float32, MutUntrackedOrigin],
    slot_idx: Int,
    block_pixel_size: Int,
    raw_bytes: PythonObject,
) raises:
    """Copy a `BPS×BPS×RGBA` region (already nearest-resized) into atlas slot."""
    var bps = block_pixel_size
    var slot_base = slot_idx * bps * bps * 4
    for y in range(bps):
        for x in range(bps):
            var src_off = (y * bps + x) * 4
            var dst_off = slot_base + (y * bps + x) * 4
            atlas[dst_off + 0] = Float32(Int(py=raw_bytes[src_off + 0])) / Float32(255.0)
            atlas[dst_off + 1] = Float32(Int(py=raw_bytes[src_off + 1])) / Float32(255.0)
            atlas[dst_off + 2] = Float32(Int(py=raw_bytes[src_off + 2])) / Float32(255.0)
            atlas[dst_off + 3] = Float32(Int(py=raw_bytes[src_off + 3])) / Float32(255.0)


def _load_one_resized(
    atlas: UnsafePointer[Float32, MutUntrackedOrigin],
    slot_idx: Int,
    block_pixel_size: Int,
    asset_dir: String,
    filename: String,
    pil: PythonObject,
) raises:
    """Open `<asset_dir>/<filename>`, RGBA-convert, nearest-resize to
    `BPS×BPS`, write float32 [0,1] into atlas[slot_idx]."""
    var path = asset_dir + "/" + filename
    var img = pil.open(path).convert("RGBA")
    var size = Python.tuple(block_pixel_size, block_pixel_size)
    img = img.resize(size, resample=pil.NEAREST)
    var raw = img.tobytes()
    _blit_resized_to_atlas(atlas, slot_idx, block_pixel_size, raw)


def _atlas_load_blocks(
    atlas: UnsafePointer[Float32, MutUntrackedOrigin],
    bps: Int,
    asset_dir: String,
    pil: PythonObject,
) raises:
    var names_a = [
        "grass.png", "water.png", "stone.png", "tree.png",
        "wood.png", "path.png", "coal.png", "iron.png",
        "diamond.png", "table.png", "furnace.png", "sand.png",
        "lava.png", "plant-young.png", "plant-ripe.png", "wall.png",
    ]
    for k in range(len(names_a)):
        _load_one_resized(atlas, 2 + k, bps, asset_dir, names_a[k], pil)
    var names_b = [
        "wall_moss.png", "stalagmite.png", "sapphire.png", "ruby.png",
        "chest.png", "fountain.png", "fire_grass.png", "ice_grass.png",
        "gravel.png", "fire_tree.png", "ice_shrub.png",
        "enchantment_table_fire.png", "enchantment_table_ice.png",
        "necromancer.png", "grave.png", "grave2.png", "grave3.png",
        "necromancer_vulnerable.png",
    ]
    for k in range(len(names_b)):
        _load_one_resized(atlas, 19 + k, bps, asset_dir, names_b[k], pil)


def _atlas_load_items_and_player(
    atlas: UnsafePointer[Float32, MutUntrackedOrigin],
    bps: Int,
    asset_dir: String,
    pil: PythonObject,
) raises:
    _load_one_resized(atlas, SPR_ITEM_TORCH, bps, asset_dir, "torch_on_path.png", pil)
    _load_one_resized(atlas, SPR_ITEM_LADDER_DOWN, bps, asset_dir, "ladder_down.png", pil)
    _load_one_resized(atlas, SPR_ITEM_LADDER_UP, bps, asset_dir, "ladder_up.png", pil)
    _load_one_resized(
        atlas, SPR_ITEM_LADDER_DOWN_BLOCKED, bps, asset_dir,
        "ladder_down_blocked.png", pil,
    )
    _load_one_resized(atlas, SPR_ARROW_LEFT, bps, asset_dir, "arrow-left.png", pil)
    _load_one_resized(atlas, SPR_ARROW_RIGHT, bps, asset_dir, "arrow-right.png", pil)
    _load_one_resized(atlas, SPR_ARROW_UP, bps, asset_dir, "arrow-up.png", pil)
    _load_one_resized(atlas, SPR_ARROW_DOWN, bps, asset_dir, "arrow-down.png", pil)
    _load_one_resized(atlas, SPR_PLAYER_LEFT, bps, asset_dir, "player-left.png", pil)
    _load_one_resized(atlas, SPR_PLAYER_RIGHT, bps, asset_dir, "player-right.png", pil)
    _load_one_resized(atlas, SPR_PLAYER_UP, bps, asset_dir, "player-up.png", pil)
    _load_one_resized(atlas, SPR_PLAYER_DOWN, bps, asset_dir, "player-down.png", pil)
    _load_one_resized(atlas, SPR_PLAYER_SLEEP, bps, asset_dir, "player-sleep.png", pil)


def _atlas_load_mobs(
    atlas: UnsafePointer[Float32, MutUntrackedOrigin],
    bps: Int,
    asset_dir: String,
    pil: PythonObject,
) raises:
    var passive = [
        "cow.png", "bat.png", "snail.png",
        "cow.png", "cow.png", "cow.png", "cow.png", "cow.png",
    ]
    for k in range(8):
        _load_one_resized(atlas, SPR_PASSIVE_BASE + k, bps, asset_dir, passive[k], pil)
    var melee = [
        "zombie.png", "gnome_warrior.png", "orc_soldier.png", "lizard.png",
        "knight.png", "troll.png", "pigman.png", "frost_troll.png",
    ]
    for k in range(8):
        _load_one_resized(atlas, SPR_MELEE_BASE + k, bps, asset_dir, melee[k], pil)
    var ranged = [
        "skeleton.png", "gnome_archer.png", "orc_mage.png", "kobold.png",
        "knight_archer.png", "deep_thing.png", "fire_elemental.png",
        "ice_elemental.png",
    ]
    for k in range(8):
        _load_one_resized(atlas, SPR_RANGED_BASE + k, bps, asset_dir, ranged[k], pil)
    var proj = [
        "arrow-up.png", "dagger.png", "fireball.png", "iceball.png",
        "arrow-up.png", "slimeball.png", "fireball.png", "iceball.png",
    ]
    for k in range(8):
        _load_one_resized(atlas, SPR_PROJ_BASE + k, bps, asset_dir, proj[k], pil)


def _atlas_load_inventory(
    atlas: UnsafePointer[Float32, MutUntrackedOrigin],
    bps: Int,
    asset_dir: String,
    pil: PythonObject,
) raises:
    var pickaxe_names = [
        "wood_pickaxe.png", "stone_pickaxe.png",
        "iron_pickaxe.png", "diamond_pickaxe.png",
    ]
    for k in range(4):
        _load_one_resized(atlas, SPR_PICKAXE_BASE + 1 + k, bps, asset_dir,
                          pickaxe_names[k], pil)
    var sword_names = [
        "wood_sword.png", "stone_sword.png",
        "iron_sword.png", "diamond_sword.png",
    ]
    for k in range(4):
        _load_one_resized(atlas, SPR_SWORD_BASE + 1 + k, bps, asset_dir,
                          sword_names[k], pil)
    _load_one_resized(atlas, SPR_BOW, bps, asset_dir, "bow.png", pil)
    var armour_names = [
        "iron_helmet.png", "diamond_helmet.png",
        "iron_chestplate.png", "diamond_chestplate.png",
        "iron_pants.png", "diamond_pants.png",
        "iron_boots.png", "diamond_boots.png",
    ]
    for piece in range(4):
        for tier in range(2):
            _load_one_resized(
                atlas, SPR_ARMOUR_BASE + piece * 3 + 1 + tier, bps,
                asset_dir, armour_names[piece * 2 + tier], pil,
            )
    _load_one_resized(atlas, SPR_INV_LOG, bps, asset_dir, "log.png", pil)
    _load_one_resized(atlas, SPR_INV_SAPLING, bps, asset_dir, "sapling.png", pil)
    _load_one_resized(atlas, SPR_INV_TORCH, bps, asset_dir, "torch_in_inventory.png", pil)
    _load_one_resized(atlas, SPR_INV_BOOK, bps, asset_dir, "book.png", pil)
    var potion_names = [
        "potion_red.png", "potion_green.png", "potion_blue.png",
        "potion_pink.png", "potion_cyan.png", "potion_yellow.png",
    ]
    for k in range(6):
        _load_one_resized(atlas, SPR_POTION_BASE + k, bps, asset_dir,
                          potion_names[k], pil)
    _load_one_resized(atlas, SPR_SPELL_FIREBALL, bps, asset_dir, "fireball.png", pil)
    _load_one_resized(atlas, SPR_SPELL_ICEBALL, bps, asset_dir, "iceball.png", pil)


def _atlas_load_intrinsic_icons(
    atlas: UnsafePointer[Float32, MutUntrackedOrigin],
    bps: Int,
    asset_dir: String,
    pil: PythonObject,
) raises:
    var names = [
        "health.png", "food.png", "drink.png", "energy.png", "mana.png",
        "xp.png", "dexterity.png", "strength.png", "intelligence.png",
    ]
    for k in range(9):
        _load_one_resized(atlas, SPR_ICON_HEALTH + k, bps, asset_dir, names[k], pil)


def build_agent_atlas(
    asset_dir: String,
    block_pixel_size: Int,
) raises -> UnsafePointer[Float32, MutUntrackedOrigin]:
    """Build the float32 RGBA atlas at the agent's small block_pixel_size.

    Returns a heap buffer owned by caller (must be freed via `.free()`).
    All non-empty slots are populated; OOB is set to opaque dark slate;
    BLOCK_INVALID (slot 0) and ITEM_NONE (slot 37) are fully transparent.
    """
    var bps = block_pixel_size
    var size = NUM_SPRITES * bps * bps * 4
    var atlas = alloc[Float32](size)
    for i in range(size):
        atlas[i] = Float32(0.0)
    # OOB slot — opaque dark slate.
    for y in range(bps):
        for x in range(bps):
            var off = 1 * bps * bps * 4 + (y * bps + x) * 4
            atlas[off + 0] = Float32(15.0) / Float32(255.0)
            atlas[off + 1] = Float32(15.0) / Float32(255.0)
            atlas[off + 2] = Float32(25.0) / Float32(255.0)
            atlas[off + 3] = Float32(1.0)

    var pil = Python.import_module("PIL.Image")
    _atlas_load_blocks(atlas, bps, asset_dir, pil)
    _atlas_load_items_and_player(atlas, bps, asset_dir, pil)
    _atlas_load_mobs(atlas, bps, asset_dir, pil)
    _atlas_load_inventory(atlas, bps, asset_dir, pil)
    _atlas_load_intrinsic_icons(atlas, bps, asset_dir, pil)
    return atlas
