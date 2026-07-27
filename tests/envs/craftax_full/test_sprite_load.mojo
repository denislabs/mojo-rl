"""Smoke test: Craftax-Full build_sprite_sheet loads all sprites without error.

Doesn't open an SDL window — just verifies PIL is available, every PNG
asset exists in `assets/`, and the resulting sheet has non-zero alpha pixels
in each slot we populated.
"""

from std.memory import alloc
from mojo_rl.envs.craftax_full.craftax_full_sprites import (
    build_sprite_sheet,
    SPRITE_SIZE,
    NUM_SPRITES,
    SHEET_WIDTH,
    SHEET_HEIGHT,
    SPRITE_BPP,
    SPR_ITEM_TORCH,
    SPR_ITEM_LADDER_DOWN,
    SPR_PASSIVE_BASE,
    SPR_MELEE_BASE,
    SPR_RANGED_BASE,
    SPR_PROJ_BASE,
    SPR_PLAYER_DOWN,
    SPR_PICKAXE_BASE,
    SPR_SWORD_BASE,
    SPR_ARMOUR_BASE,
    SPR_POTION_BASE,
    SPR_ICON_HEALTH,
    SPR_ICON_MANA,
    SPR_ICON_INT,
)
from mojo_rl.envs.craftax_full.constants import (
    BLOCK_GRASS,
    BLOCK_WATER,
    BLOCK_LAVA,
    BLOCK_CHEST,
    BLOCK_NECROMANCER,
    BLOCK_NECROMANCER_VULNERABLE,
)


def _slot_has_content(
    sheet: UnsafePointer[UInt8, MutAnyOrigin], slot: Int
) -> Bool:
    """Return True if any pixel in `slot` has alpha > 0."""
    for y in range(SPRITE_SIZE):
        for x in range(SPRITE_SIZE):
            var off = (y * SHEET_WIDTH + (slot * SPRITE_SIZE + x)) * SPRITE_BPP
            if Int(sheet[off + 3]) > 0:
                return True
    return False


def main() raises:
    print("Craftax-Full sprite-load smoke test")
    print("=" * 50)

    var sheet = build_sprite_sheet(String("mojo_rl/envs/craftax_full/assets"))
    print(
        "Sheet allocated:",
        SHEET_WIDTH,
        "x",
        SHEET_HEIGHT,
        "px (",
        NUM_SPRITES,
        "sprites)",
    )

    var passed = 0
    var failed = 0
    # Sample representative slots that must have content.
    var slots = [
        BLOCK_GRASS,
        BLOCK_WATER,
        BLOCK_LAVA,
        BLOCK_CHEST,
        BLOCK_NECROMANCER,
        BLOCK_NECROMANCER_VULNERABLE,
        SPR_ITEM_TORCH,
        SPR_ITEM_LADDER_DOWN,
        SPR_PASSIVE_BASE + 0,  # cow
        SPR_PASSIVE_BASE + 1,  # bat
        SPR_PASSIVE_BASE + 2,  # snail
        SPR_MELEE_BASE + 0,  # zombie
        SPR_MELEE_BASE + 7,  # frost troll
        SPR_RANGED_BASE + 0,  # skeleton
        SPR_RANGED_BASE + 7,  # ice elemental
        SPR_PROJ_BASE + 0,  # arrow
        SPR_PROJ_BASE + 5,  # slimeball
        SPR_PLAYER_DOWN,
        SPR_PICKAXE_BASE + 4,  # diamond pickaxe
        SPR_SWORD_BASE + 4,  # diamond sword
        SPR_ARMOUR_BASE + 0 * 3 + 1,  # iron helmet
        SPR_ARMOUR_BASE + 3 * 3 + 2,  # diamond boots
        SPR_POTION_BASE + 5,  # yellow potion
        SPR_ICON_HEALTH,
        SPR_ICON_MANA,
        SPR_ICON_INT,
    ]
    for slot in slots:
        if _slot_has_content(sheet.as_unsafe_any_origin(), slot):
            passed += 1
            print("  PASS slot", slot, "has content")
        else:
            failed += 1
            print("  FAIL slot", slot, "is empty")

    sheet.free()
    print()
    print("Passed:", passed, "Failed:", failed)
    if failed > 0:
        raise Error("sprite-load FAILED")
    print("sprite-load PASS")
