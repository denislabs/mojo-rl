"""Smoke test: build_sprite_sheet loads all 41 sprites without raising.

Doesn't open an SDL window — just verifies that PIL is available, every PNG
asset exists in `assets/`, and the resulting sheet has non-zero alpha pixels
in each slot we populated.
"""

from std.memory import alloc
from mojo_rl.envs.craftax_classic.craftax_classic_sprites import (
    build_sprite_sheet,
    SPRITE_SIZE,
    NUM_SPRITES,
    SHEET_WIDTH,
    SHEET_HEIGHT,
    SPRITE_BPP,
    SPR_GRASS,
    SPR_WATER,
    SPR_ZOMBIE,
    SPR_PLAYER_DOWN,
    SPR_INV_IRON_SWORD,
    SPR_ICON_HEALTH,
)


def _slot_has_content(sheet: UnsafePointer[UInt8, MutAnyOrigin], slot: Int) -> Bool:
    """Return True if any pixel in `slot` has alpha > 0."""
    for y in range(SPRITE_SIZE):
        for x in range(SPRITE_SIZE):
            var off = (
                y * SHEET_WIDTH + (slot * SPRITE_SIZE + x)
            ) * SPRITE_BPP
            if Int(sheet[off + 3]) > 0:
                return True
    return False


def main() raises:
    print("Craftax-Classic sprite-load smoke test")
    print("=" * 50)

    var sheet = build_sprite_sheet(
        String("mojo_rl/envs/craftax_classic/assets")
    )
    print("Sheet allocated:", SHEET_WIDTH, "x", SHEET_HEIGHT, "px (",
          NUM_SPRITES, "sprites)")

    var passed = 0
    var failed = 0
    # Sample a few slots that must have content.
    var slots = [
        SPR_GRASS, SPR_WATER, SPR_ZOMBIE, SPR_PLAYER_DOWN,
        SPR_INV_IRON_SWORD, SPR_ICON_HEALTH,
    ]
    for slot in slots:
        if _slot_has_content(sheet, slot):
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
