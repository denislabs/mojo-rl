"""TIA lookup tables — sprite masks, collision, playfield.

Simplified versions of CuLE's tables.cpp. Instead of full precomputed
mask arrays (which are large), we compute masks on-the-fly from the
TIA state registers. This is simpler and uses less memory per instance,
which is better for GPU where we want minimal per-thread state.

Ported from CuLE (BSD-3): cule/atari/tables.hpp, tables.cpp
"""

from .atari_state import AtariState
from .flags import (
    TIA_P0_REFLECT, TIA_P1_REFLECT, TIA_PF_REFLECT,
    TIA_M0_LOCK, TIA_M1_LOCK,
    TIA_VDELP0, TIA_VDELP1, TIA_VDELBL,
    FRAME_WIDTH,
)


@always_inline
fn player_mask(state: AtariState, player: Int, pixel: Int) -> Bool:
    """Check if player sprite is visible at this pixel position.

    Considers position, NUSIZ (copies/size), reflection, and VDEL.
    """
    var pos: Int
    var grp: UInt8
    var nusiz: UInt8
    var reflect: Bool

    if player == 0:
        pos = Int(state.pos_p0)
        grp = state.grp0 if (state.tia_flags & TIA_VDELP0) == 0 else state.grp0_old
        nusiz = state.nusiz0
        reflect = (state.tia_flags & TIA_P0_REFLECT) != 0
    else:
        pos = Int(state.pos_p1)
        grp = state.grp1 if (state.tia_flags & TIA_VDELP1) == 0 else state.grp1_old
        nusiz = state.nusiz1
        reflect = (state.tia_flags & TIA_P1_REFLECT) != 0

    if grp == 0:
        return False

    var size_mode = Int(nusiz & 0x07)
    var stretch = 1  # pixels per bit
    if size_mode == 5:
        stretch = 2  # double-size
    elif size_mode == 7:
        stretch = 4  # quad-size

    # Check main copy and up to 2 additional copies
    var num_copies = 1
    var copy_spacing = 0
    if size_mode == 1:
        num_copies = 2
        copy_spacing = 16
    elif size_mode == 2:
        num_copies = 2
        copy_spacing = 32
    elif size_mode == 3:
        num_copies = 3
        copy_spacing = 16
    elif size_mode == 4:
        num_copies = 2
        copy_spacing = 64
    elif size_mode == 6:
        num_copies = 3
        copy_spacing = 32

    for copy in range(num_copies):
        var copy_pos = (pos + copy * copy_spacing) % FRAME_WIDTH
        var rel = pixel - copy_pos
        if rel < 0:
            rel += FRAME_WIDTH
        if rel < 0 or rel >= 8 * stretch:
            continue

        var bit_idx = rel // stretch
        if reflect:
            bit_idx = 7 - bit_idx

        if (grp >> UInt8(7 - bit_idx)) & 1:
            return True

    return False


@always_inline
fn missile_mask(state: AtariState, missile: Int, pixel: Int) -> Bool:
    """Check if missile is visible at this pixel position."""
    var enabled: UInt8
    var pos: Int
    var nusiz: UInt8
    var locked: Bool

    if missile == 0:
        enabled = state.enam0
        pos = Int(state.pos_m0)
        nusiz = state.nusiz0
        locked = (state.tia_flags & TIA_M0_LOCK) != 0
    else:
        enabled = state.enam1
        pos = Int(state.pos_m1)
        nusiz = state.nusiz1
        locked = (state.tia_flags & TIA_M1_LOCK) != 0

    if (enabled & 0x02) == 0 or locked:
        return False

    # Missile size from NUSIZ bits 4-5
    var size_bits = Int((nusiz >> 4) & 0x03)
    var size = 1 << size_bits  # 1, 2, 4, or 8 pixels wide

    var rel = pixel - pos
    if rel < 0:
        rel += FRAME_WIDTH
    return rel >= 0 and rel < size


@always_inline
fn ball_mask(state: AtariState, pixel: Int) -> Bool:
    """Check if ball is visible at this pixel position."""
    var enabled = state.enabl if (state.tia_flags & TIA_VDELBL) == 0 else state.enabl_old
    if (enabled & 0x02) == 0:
        return False

    # Ball size from CTRLPF bits 4-5
    var size_bits = Int((state.ctrlpf >> 4) & 0x03)
    var size = 1 << size_bits  # 1, 2, 4, or 8 pixels wide

    var pos = Int(state.pos_bl)
    var rel = pixel - pos
    if rel < 0:
        rel += FRAME_WIDTH
    return rel >= 0 and rel < size


@always_inline
fn playfield_mask(state: AtariState, pixel: Int) -> Bool:
    """Check if playfield is visible at this pixel position.

    The playfield is 40 bits wide (20 bits repeated or reflected).
    PF0 uses bits 4-7 (4 bits), PF1 uses bits 7-0 (8 bits reversed),
    PF2 uses bits 0-7 (8 bits).

    For mid-scanline PF writes (e.g. Pong score digits), the left half
    (pixels 0-79) uses the midpoint PF snapshot (pf0_mid/pf1_mid/pf2_mid)
    captured at ~cycle 49, and the right half uses the final PF values.
    For non-score scanlines, both snapshots are identical.
    """
    var x = pixel >> 2  # 4 clocks per playfield bit, so 160 pixels / 4 = 40 pf pixels

    # Select PF register source: midpoint snapshot for left, final for right
    var pf0 = state.pf0_mid if pixel < 80 else state.pf0
    var pf1 = state.pf1_mid if pixel < 80 else state.pf1
    var pf2 = state.pf2_mid if pixel < 80 else state.pf2

    # Determine if we're in the left or right half
    var pf_bit: Int
    if x < 20:
        pf_bit = x
    else:
        if (state.tia_flags & TIA_PF_REFLECT) != 0:
            pf_bit = 39 - x  # Mirror: 20->19, 21->18, ..., 39->0
        else:
            pf_bit = x - 20  # Repeat

    # Map pf_bit (0-19) to the actual register bits
    # PF0: bits 4-7 map to pf_bit 0-3
    # PF1: bits 7-0 map to pf_bit 4-11
    # PF2: bits 0-7 map to pf_bit 12-19
    if pf_bit < 4:
        # PF0: bit (pf_bit + 4)
        return ((pf0 >> UInt8(pf_bit + 4)) & 1) != 0
    elif pf_bit < 12:
        # PF1: bit (11 - pf_bit) = reversed order
        return ((pf1 >> UInt8(11 - pf_bit)) & 1) != 0
    else:
        # PF2: bit (pf_bit - 12)
        return ((pf2 >> UInt8(pf_bit - 12)) & 1) != 0


@always_inline
fn collision_mask() -> UInt16:
    """Return initial (empty) collision mask."""
    return UInt16(0)
