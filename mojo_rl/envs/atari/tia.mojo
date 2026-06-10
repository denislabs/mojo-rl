"""Atari 2600 TIA (Television Interface Adapter) emulation.

The TIA generates video output and handles collision detection.
This is a simplified version focused on collision-only emulation
for RL (no pixel rendering in the hot path). Frame rendering is
done separately in preprocessing.mojo when pixel observations are needed.

For RL with RAM observations, we only need collision detection
(games read collision registers to determine game state).

Ported from CuLE (BSD-3): cule/atari/tia.hpp
"""

from .atari_state import AtariState
from .flags import (
    TIA_VBLANK,
    TIA_VSYNC,
    TIA_HMOVE,
    TIA_PADDLE_GROUND,
    TIA_M0_LOCK,
    TIA_M1_LOCK,
    TIA_P0_REFLECT,
    TIA_P1_REFLECT,
    TIA_PF_REFLECT,
    TIA_PF_SCORE,
    TIA_PF_PRIORITY,
    TIA_VDELP0,
    TIA_VDELP1,
    TIA_VDELBL,
    CLOCKS_PER_LINE,
    HBLANK_CLOCKS,
    TOTAL_SCANLINES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CX_M0P1,
    CX_M0P0,
    CX_M1P0,
    CX_M1P1,
    CX_P0PF,
    CX_P0BL,
    CX_P1PF,
    CX_P1BL,
    CX_M0PF,
    CX_M0BL,
    CX_M1PF,
    CX_M1BL,
    CX_BLPF,
    CX_P0P1,
    CX_M0M1,
    FLAG_CON_FIRE,
    FLAG_SWAP_PORTS,
    FLAG_PADDLES,
)


# ============================================================================
# TIA Register Read
# ============================================================================


@always_inline
def tia_read(state: AtariState, addr: UInt8) -> UInt8:
    """Read a TIA register. Only collision and input registers are readable,
    and they only drive bits 7/6 — the low 6 bits leak the previous data-bus
    byte (Stella TIA::peek `noise = dataBusState & 0x3F`). For a zero-page
    read the last bus byte is the operand, so e.g. Haunted House's
    `SBC $0f` on unmapped TIA $0F reads back 15 — its divide-by-15 divisor.
    """
    var reg = addr & 0x0F
    var noise = state.data_bus & 0x3F
    var val: UInt8 = 0

    # Collision registers (0x00-0x07)
    if reg == 0x00:  # CXM0P
        if (state.collision & CX_M0P1) != 0:
            val = val | 0x80
        if (state.collision & CX_M0P0) != 0:
            val = val | 0x40
    elif reg == 0x01:  # CXM1P
        if (state.collision & CX_M1P0) != 0:
            val = val | 0x80
        if (state.collision & CX_M1P1) != 0:
            val = val | 0x40
    elif reg == 0x02:  # CXP0FB
        if (state.collision & CX_P0PF) != 0:
            val = val | 0x80
        if (state.collision & CX_P0BL) != 0:
            val = val | 0x40
    elif reg == 0x03:  # CXP1FB
        if (state.collision & CX_P1PF) != 0:
            val = val | 0x80
        if (state.collision & CX_P1BL) != 0:
            val = val | 0x40
    elif reg == 0x04:  # CXM0FB
        if (state.collision & CX_M0PF) != 0:
            val = val | 0x80
        if (state.collision & CX_M0BL) != 0:
            val = val | 0x40
    elif reg == 0x05:  # CXM1FB
        if (state.collision & CX_M1PF) != 0:
            val = val | 0x80
        if (state.collision & CX_M1BL) != 0:
            val = val | 0x40
    elif reg == 0x06:  # CXBLPF
        if (state.collision & CX_BLPF) != 0:
            val = val | 0x80
    elif reg == 0x07:  # CXPPMM
        if (state.collision & CX_P0P1) != 0:
            val = val | 0x80
        if (state.collision & CX_M0M1) != 0:
            val = val | 0x40

    # Input ports (0x08-0x0D)
    elif reg == 0x08:  # INPT0 - Paddle 0 (Player 0 paddle)
        # Paddle: bit 7 = 1 when capacitor charge >= paddle position
        if state.paddle_charge >= state.paddle_pos:
            val = 0x80
    elif reg == 0x09:  # INPT1 - Paddle 1 (right paddle in Pong)
        # Same paddle input — Pong reads INPT1 for the human player
        if state.paddle_charge >= state.paddle_pos:
            val = 0x80
    elif reg == 0x0A:  # INPT2 - Paddle 2
        val = 0x80
    elif reg == 0x0B:  # INPT3 - Paddle 3
        val = 0x80
    elif reg == 0x0C:  # INPT4 - left-port fire button
        if (state.sys_flags & FLAG_PADDLES) != 0:
            val = 0x80  # Paddle cart: fire is a SWCHA button, INPT4 floats
        elif (state.sys_flags & FLAG_SWAP_PORTS) != 0:
            val = 0x80  # Player 1 is on the right port
        elif (state.sys_flags & FLAG_CON_FIRE) != 0:
            val = 0x00  # Button pressed (bit 7 = 0)
        else:
            val = 0x80  # Not pressed (bit 7 = 1)
    elif reg == 0x0D:  # INPT5 - right-port fire button
        if (state.sys_flags & FLAG_PADDLES) != 0:
            val = 0x80
        elif (state.sys_flags & FLAG_SWAP_PORTS) != 0 and (
            state.sys_flags & FLAG_CON_FIRE
        ) != 0:
            val = 0x00
        else:
            val = 0x80
    # 0x0E/0x0F: unmapped — pure bus noise

    return val | noise


# ============================================================================
# TIA Register Write
# ============================================================================


@always_inline
def _resp_pos(clock: Int) -> UInt8:
    """Convert TIA clock to pixel position (0-159) for RESPx strobe.

    state.clock is set at instruction START, but the RESP write happens
    on the last cycle of the store instruction. For STA zeropage (3 cycles):
      - Write occurs 2 CPU cycles after start = +6 TIA clocks
      - TIA RESP hardware delay = +5 TIA clocks
      - Total offset from instruction start = +11 TIA clocks
    """
    # ALE: if hpos < HBLANK then pos=3, else pos = ((hpos - HBLANK) + 5) % 160
    # Our clock is at instruction start, so add 6 for STA write cycle
    var hpos = clock + 6  # Approximate TIA clock at write time
    if hpos < HBLANK_CLOCKS:
        return 3  # ALE default when in HBLANK
    return UInt8(((hpos - HBLANK_CLOCKS) + 5) % FRAME_WIDTH)


@always_inline
def tia_write(mut state: AtariState, addr: UInt8, value: UInt8):
    """Write a TIA register."""
    var reg = addr & 0x3F

    if reg == 0x00:  # VSYNC
        if (value & 0x02) != 0:
            state.tia_flags = state.tia_flags | TIA_VSYNC
        else:
            state.tia_flags = state.tia_flags & ~TIA_VSYNC

    elif reg == 0x01:  # VBLANK
        if (value & 0x02) != 0:
            state.tia_flags = state.tia_flags | TIA_VBLANK
        else:
            state.tia_flags = state.tia_flags & ~TIA_VBLANK
        # Bit 7: paddle capacitor grounding (real hardware latches this)
        # When set: ground capacitors (charge=0, stays grounded)
        # When clear: release ground (capacitors begin charging)
        if (value & 0x80) != 0:
            state.paddle_charge = 0
            state.tia_flags = state.tia_flags | TIA_PADDLE_GROUND
        else:
            state.tia_flags = state.tia_flags & ~TIA_PADDLE_GROUND

    elif reg == 0x02:  # WSYNC — halt CPU until end of scanline
        state.wsync = True

    elif reg == 0x04:  # NUSIZ0
        state.nusiz0 = value & 0x37

    elif reg == 0x05:  # NUSIZ1
        state.nusiz1 = value & 0x37

    elif reg == 0x06:  # COLUP0
        state.colup0 = value & 0xFE  # Low bit ignored

    elif reg == 0x07:  # COLUP1
        state.colup1 = value & 0xFE

    elif reg == 0x08:  # COLUPF
        state.colupf = value & 0xFE

    elif reg == 0x09:  # COLUBK
        state.colubk = value & 0xFE

    elif reg == 0x0A:  # CTRLPF
        state.ctrlpf = value
        if (value & 0x01) != 0:
            state.tia_flags = state.tia_flags | TIA_PF_REFLECT
        else:
            state.tia_flags = state.tia_flags & ~TIA_PF_REFLECT
        if (value & 0x02) != 0:
            state.tia_flags = state.tia_flags | TIA_PF_SCORE
        else:
            state.tia_flags = state.tia_flags & ~TIA_PF_SCORE
        if (value & 0x04) != 0:
            state.tia_flags = state.tia_flags | TIA_PF_PRIORITY
        else:
            state.tia_flags = state.tia_flags & ~TIA_PF_PRIORITY

    elif reg == 0x0B:  # REFP0
        if (value & 0x08) != 0:
            state.tia_flags = state.tia_flags | TIA_P0_REFLECT
        else:
            state.tia_flags = state.tia_flags & ~TIA_P0_REFLECT

    elif reg == 0x0C:  # REFP1
        if (value & 0x08) != 0:
            state.tia_flags = state.tia_flags | TIA_P1_REFLECT
        else:
            state.tia_flags = state.tia_flags & ~TIA_P1_REFLECT

    elif reg == 0x0D:  # PF0
        state.pf0 = value

    elif reg == 0x0E:  # PF1
        state.pf1 = value

    elif reg == 0x0F:  # PF2
        state.pf2 = value

    elif reg == 0x10:  # RESP0 — reset player 0 position
        state.pos_p0 = _resp_pos(Int(state.clock))

    elif reg == 0x11:  # RESP1
        state.pos_p1 = _resp_pos(Int(state.clock))

    elif reg == 0x12:  # RESM0
        state.pos_m0 = _resp_pos(Int(state.clock))

    elif reg == 0x13:  # RESM1
        state.pos_m1 = _resp_pos(Int(state.clock))

    elif reg == 0x14:  # RESBL
        state.pos_bl = _resp_pos(Int(state.clock))

    elif reg == 0x1B:  # GRP0
        # Vertical delay: writing GRP0 clocks the delayed copy of GRP1.
        state.grp1_old = state.grp1
        state.grp0 = value

    elif reg == 0x1C:  # GRP1
        # Vertical delay: writing GRP1 clocks the delayed copies of GRP0 and
        # the ball enable (the VDEL "old" latches are clocked by the *other*
        # player's graphics write, not by each object's own write).
        state.grp0_old = state.grp0
        state.enabl_old = state.enabl
        state.grp1 = value

    elif reg == 0x1D:  # ENAM0
        state.enam0 = value & 0x02

    elif reg == 0x1E:  # ENAM1
        state.enam1 = value & 0x02

    elif reg == 0x1F:  # ENABL
        # New ball-enable value; the delayed copy (enabl_old) is latched on
        # GRP1 writes above, not here (NMOS TIA vertical-delay behavior).
        state.enabl = value & 0x02

    elif reg == 0x20:  # HMP0
        state.hm_p0 = value >> 4

    elif reg == 0x21:  # HMP1
        state.hm_p1 = value >> 4

    elif reg == 0x22:  # HMM0
        state.hm_m0 = value >> 4

    elif reg == 0x23:  # HMM1
        state.hm_m1 = value >> 4

    elif reg == 0x24:  # HMBL
        state.hm_bl = value >> 4

    elif reg == 0x25:  # VDELP0
        if (value & 0x01) != 0:
            state.tia_flags = state.tia_flags | TIA_VDELP0
        else:
            state.tia_flags = state.tia_flags & ~TIA_VDELP0

    elif reg == 0x26:  # VDELP1
        if (value & 0x01) != 0:
            state.tia_flags = state.tia_flags | TIA_VDELP1
        else:
            state.tia_flags = state.tia_flags & ~TIA_VDELP1

    elif reg == 0x27:  # VDELBL
        if (value & 0x01) != 0:
            state.tia_flags = state.tia_flags | TIA_VDELBL
        else:
            state.tia_flags = state.tia_flags & ~TIA_VDELBL

    elif reg == 0x28:  # RESMP0
        if (value & 0x02) != 0:
            state.tia_flags = state.tia_flags | TIA_M0_LOCK
        else:
            state.tia_flags = state.tia_flags & ~TIA_M0_LOCK

    elif reg == 0x29:  # RESMP1
        if (value & 0x02) != 0:
            state.tia_flags = state.tia_flags | TIA_M1_LOCK
        else:
            state.tia_flags = state.tia_flags & ~TIA_M1_LOCK

    elif reg == 0x2A:  # HMOVE — apply horizontal motion
        _apply_hmove(state)

    elif reg == 0x2B:  # HMCLR — clear all motion registers
        state.hm_p0 = 0
        state.hm_p1 = 0
        state.hm_m0 = 0
        state.hm_m1 = 0
        state.hm_bl = 0

    elif reg == 0x2C:  # CXCLR — clear collision latches
        state.collision = 0


@always_inline
def _hm_to_signed(hm: UInt8) -> Int:
    """Convert 4-bit horizontal motion value to signed displacement.

    The 4-bit value is in the upper nibble format:
    0000 = no motion, 0001-0111 = left 1-7, 1000-1111 = right 8-1
    After >> 4 in the write, we have the raw 4-bit value.
    """
    var val = Int(hm)
    if val >= 8:
        return val - 16  # 8->-8, 9->-7, ..., 15->-1
    return val  # 0->0, 1->1, ..., 7->7


@always_inline
def _clamp_pos(pos: Int) -> UInt8:
    """Clamp position to [0, 159] with wrapping."""
    var p = pos % FRAME_WIDTH
    if p < 0:
        p += FRAME_WIDTH
    return UInt8(p)


@always_inline
def _apply_hmove(mut state: AtariState):
    """Apply horizontal motion (HMOVE register write)."""
    state.pos_p0 = _clamp_pos(Int(state.pos_p0) - _hm_to_signed(state.hm_p0))
    state.pos_p1 = _clamp_pos(Int(state.pos_p1) - _hm_to_signed(state.hm_p1))
    state.pos_m0 = _clamp_pos(Int(state.pos_m0) - _hm_to_signed(state.hm_m0))
    state.pos_m1 = _clamp_pos(Int(state.pos_m1) - _hm_to_signed(state.hm_m1))
    state.pos_bl = _clamp_pos(Int(state.pos_bl) - _hm_to_signed(state.hm_bl))
    state.tia_flags = state.tia_flags | TIA_HMOVE
