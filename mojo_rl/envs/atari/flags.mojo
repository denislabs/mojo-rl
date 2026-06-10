"""Atari 2600 constants, flags, and action definitions.

Ported from CuLE (BSD-3): cule/atari/flags.hpp, cule/atari/actions.hpp
"""


# ============================================================================
# Actions (18 standard Atari actions + RESET)
# ============================================================================

comptime ACTION_NOOP: UInt8 = 0
comptime ACTION_FIRE: UInt8 = 1
comptime ACTION_UP: UInt8 = 2
comptime ACTION_RIGHT: UInt8 = 3
comptime ACTION_LEFT: UInt8 = 4
comptime ACTION_DOWN: UInt8 = 5
comptime ACTION_UPRIGHT: UInt8 = 6
comptime ACTION_UPLEFT: UInt8 = 7
comptime ACTION_DOWNRIGHT: UInt8 = 8
comptime ACTION_DOWNLEFT: UInt8 = 9
comptime ACTION_UPFIRE: UInt8 = 10
comptime ACTION_RIGHTFIRE: UInt8 = 11
comptime ACTION_LEFTFIRE: UInt8 = 12
comptime ACTION_DOWNFIRE: UInt8 = 13
comptime ACTION_UPRIGHTFIRE: UInt8 = 14
comptime ACTION_UPLEFTFIRE: UInt8 = 15
comptime ACTION_DOWNRIGHTFIRE: UInt8 = 16
comptime ACTION_DOWNLEFTFIRE: UInt8 = 17
comptime ACTION_RESET: UInt8 = 18
comptime NUM_TOTAL_ACTIONS: Int = 18  # Excluding RESET

# ============================================================================
# ROM Formats (cartridge mapper types)
# ============================================================================

comptime ROM_2K: UInt8 = 0
comptime ROM_4K: UInt8 = 1
comptime ROM_F8: UInt8 = 2  # 8K, two 4K banks
comptime ROM_F8SC: UInt8 = 3  # 8K + 128 bytes super chip RAM
comptime ROM_F6: UInt8 = 4  # 16K, four 4K banks
comptime ROM_FE: UInt8 = 5
comptime ROM_E0: UInt8 = 6  # 8K Parker Bros
comptime ROM_CV: UInt8 = 7
comptime ROM_3E: UInt8 = 8
comptime ROM_3F: UInt8 = 9
comptime ROM_UA: UInt8 = 10
comptime ROM_F6SC: UInt8 = 11  # 16K + super chip RAM
comptime ROM_AUTO: UInt8 = 0xFF  # Resolve mapper from ROM size at init

# ============================================================================
# 6502 CPU Status Flags
# ============================================================================

comptime FLAG_C: UInt8 = 0x01  # Carry
comptime FLAG_Z: UInt8 = 0x02  # Zero
comptime FLAG_I: UInt8 = 0x04  # IRQ disable
comptime FLAG_D: UInt8 = 0x08  # Decimal mode
comptime FLAG_B: UInt8 = 0x10  # Break
comptime FLAG_V: UInt8 = 0x40  # Overflow
comptime FLAG_N: UInt8 = 0x80  # Negative

# ============================================================================
# System Flags (packed into a UInt32)
# ============================================================================

# Controller direction flags
comptime FLAG_CON_UP: UInt32 = 1 << 0
comptime FLAG_CON_DOWN: UInt32 = 1 << 1
comptime FLAG_CON_LEFT: UInt32 = 1 << 2
comptime FLAG_CON_RIGHT: UInt32 = 1 << 3
comptime FLAG_CON_FIRE: UInt32 = 1 << 4

# Console switches
comptime FLAG_CON_SELECT: UInt32 = 1 << 5
comptime FLAG_CON_RESET: UInt32 = 1 << 6
comptime FLAG_CON_COLOR: UInt32 = 1 << 7
comptime FLAG_CON_LEFT_DIFF: UInt32 = 1 << 8
comptime FLAG_CON_RIGHT_DIFF: UInt32 = 1 << 9

# System state flags
comptime FLAG_ALE_STARTED: UInt32 = 1 << 10
comptime FLAG_ALE_TERMINAL: UInt32 = 1 << 11

# Bank switching state (bits 16-19 for current bank)
comptime BANK_SHIFT: Int = 16
comptime BANK_MASK: UInt32 = 0xF << 16

# ============================================================================
# TIA Flags (packed into a UInt32)
# ============================================================================

comptime TIA_VBLANK: UInt32 = 1 << 0
comptime TIA_VSYNC: UInt32 = 1 << 1
comptime TIA_HMOVE: UInt32 = 1 << 2
comptime TIA_M0_LOCK: UInt32 = 1 << 3  # RESMP0
comptime TIA_M1_LOCK: UInt32 = 1 << 4  # RESMP1
comptime TIA_P0_REFLECT: UInt32 = 1 << 5
comptime TIA_P1_REFLECT: UInt32 = 1 << 6
comptime TIA_PF_REFLECT: UInt32 = 1 << 7
comptime TIA_PF_SCORE: UInt32 = 1 << 8
comptime TIA_PF_PRIORITY: UInt32 = 1 << 9
comptime TIA_VDELP0: UInt32 = 1 << 10
comptime TIA_VDELP1: UInt32 = 1 << 11
comptime TIA_VDELBL: UInt32 = 1 << 12
comptime TIA_COSMIC_ARK: UInt32 = 1 << 13  # Cosmic Ark M0 bug
comptime TIA_PADDLE_GROUND: UInt32 = 1 << 14  # Paddle capacitors grounded (VBLANK bit 7)

# ============================================================================
# TIA Registers (addresses)
# ============================================================================

# Write registers
comptime TIA_VSYNC_REG: UInt8 = 0x00
comptime TIA_VBLANK_REG: UInt8 = 0x01
comptime TIA_WSYNC: UInt8 = 0x02
comptime TIA_RSYNC: UInt8 = 0x03
comptime TIA_NUSIZ0: UInt8 = 0x04
comptime TIA_NUSIZ1: UInt8 = 0x05
comptime TIA_COLUP0: UInt8 = 0x06
comptime TIA_COLUP1: UInt8 = 0x07
comptime TIA_COLUPF: UInt8 = 0x08
comptime TIA_COLUBK: UInt8 = 0x09
comptime TIA_CTRLPF: UInt8 = 0x0A
comptime TIA_REFP0: UInt8 = 0x0B
comptime TIA_REFP1: UInt8 = 0x0C
comptime TIA_PF0: UInt8 = 0x0D
comptime TIA_PF1: UInt8 = 0x0E
comptime TIA_PF2: UInt8 = 0x0F
comptime TIA_RESP0: UInt8 = 0x10
comptime TIA_RESP1: UInt8 = 0x11
comptime TIA_RESM0: UInt8 = 0x12
comptime TIA_RESM1: UInt8 = 0x13
comptime TIA_RESBL: UInt8 = 0x14
comptime TIA_AUDC0: UInt8 = 0x15
comptime TIA_AUDC1: UInt8 = 0x16
comptime TIA_AUDF0: UInt8 = 0x17
comptime TIA_AUDF1: UInt8 = 0x18
comptime TIA_AUDV0: UInt8 = 0x19
comptime TIA_AUDV1: UInt8 = 0x1A
comptime TIA_GRP0: UInt8 = 0x1B
comptime TIA_GRP1: UInt8 = 0x1C
comptime TIA_ENAM0: UInt8 = 0x1D
comptime TIA_ENAM1: UInt8 = 0x1E
comptime TIA_ENABL: UInt8 = 0x1F
comptime TIA_HMP0: UInt8 = 0x20
comptime TIA_HMP1: UInt8 = 0x21
comptime TIA_HMM0: UInt8 = 0x22
comptime TIA_HMM1: UInt8 = 0x23
comptime TIA_HMBL: UInt8 = 0x24
comptime TIA_VDELP0_REG: UInt8 = 0x25
comptime TIA_VDELP1_REG: UInt8 = 0x26
comptime TIA_VDELBL_REG: UInt8 = 0x27
comptime TIA_RESMP0: UInt8 = 0x28
comptime TIA_RESMP1: UInt8 = 0x29
comptime TIA_HMOVE_REG: UInt8 = 0x2A
comptime TIA_HMCLR: UInt8 = 0x2B
comptime TIA_CXCLR: UInt8 = 0x2C

# Read registers
comptime TIA_CXM0P: UInt8 = 0x00
comptime TIA_CXM1P: UInt8 = 0x01
comptime TIA_CXP0FB: UInt8 = 0x02
comptime TIA_CXP1FB: UInt8 = 0x03
comptime TIA_CXM0FB: UInt8 = 0x04
comptime TIA_CXM1FB: UInt8 = 0x05
comptime TIA_CXBLPF: UInt8 = 0x06
comptime TIA_CXPPMM: UInt8 = 0x07
comptime TIA_INPT0: UInt8 = 0x08
comptime TIA_INPT1: UInt8 = 0x09
comptime TIA_INPT2: UInt8 = 0x0A
comptime TIA_INPT3: UInt8 = 0x0B
comptime TIA_INPT4: UInt8 = 0x0C
comptime TIA_INPT5: UInt8 = 0x0D

# ============================================================================
# RIOT Registers (addresses relative to 0x280)
# ============================================================================

comptime RIOT_SWCHA: UInt8 = 0x00  # Port A data (joystick)
comptime RIOT_SWACNT: UInt8 = 0x01  # Port A DDR
comptime RIOT_SWCHB: UInt8 = 0x02  # Port B data (console switches)
comptime RIOT_SWBCNT: UInt8 = 0x03  # Port B DDR
comptime RIOT_INTIM: UInt8 = 0x04  # Timer output
comptime RIOT_INSTAT: UInt8 = 0x05  # Timer status

# Timer write registers
comptime RIOT_TIM1T: UInt8 = 0x14  # Set timer / 1
comptime RIOT_TIM8T: UInt8 = 0x15  # Set timer / 8
comptime RIOT_TIM64T: UInt8 = 0x16  # Set timer / 64
comptime RIOT_T1024T: UInt8 = 0x17  # Set timer / 1024

# ============================================================================
# Collision bit masks
# ============================================================================

comptime CX_M0P1: UInt16 = 1 << 0  # Missile 0 - Player 1
comptime CX_M0P0: UInt16 = 1 << 1  # Missile 0 - Player 0
comptime CX_M1P0: UInt16 = 1 << 2  # Missile 1 - Player 0
comptime CX_M1P1: UInt16 = 1 << 3  # Missile 1 - Player 1
comptime CX_P0PF: UInt16 = 1 << 4  # Player 0 - Playfield
comptime CX_P0BL: UInt16 = 1 << 5  # Player 0 - Ball
comptime CX_P1PF: UInt16 = 1 << 6  # Player 1 - Playfield
comptime CX_P1BL: UInt16 = 1 << 7  # Player 1 - Ball
comptime CX_M0PF: UInt16 = 1 << 8  # Missile 0 - Playfield
comptime CX_M0BL: UInt16 = 1 << 9  # Missile 0 - Ball
comptime CX_M1PF: UInt16 = 1 << 10  # Missile 1 - Playfield
comptime CX_M1BL: UInt16 = 1 << 11  # Missile 1 - Ball
comptime CX_BLPF: UInt16 = 1 << 12  # Ball - Playfield
comptime CX_P0P1: UInt16 = 1 << 13  # Player 0 - Player 1
comptime CX_M0M1: UInt16 = 1 << 14  # Missile 0 - Missile 1

# ============================================================================
# Display constants
# ============================================================================

comptime FRAME_WIDTH: Int = 160  # TIA output width
comptime FRAME_HEIGHT: Int = 210  # Visible scanlines (NTSC)
comptime TOTAL_SCANLINES: Int = 262  # Total scanlines per frame (NTSC)
comptime CLOCKS_PER_LINE: Int = 228  # Color clocks per scanline
comptime HBLANK_CLOCKS: Int = 68  # Horizontal blank clocks
comptime CPU_CLOCKS_PER_LINE: Int = 76  # CPU cycles per scanline (228/3)

# Preprocessed observation dimensions
comptime OBS_WIDTH: Int = 84
comptime OBS_HEIGHT: Int = 84

# Per-instruction TIA write log capacity (cycle-accurate path). A single 6502
# instruction performs at most one TIA store; a couple slots covers RMW dummies.
comptime TIA_WRITE_LOG_CAP: Int = 4

# ============================================================================
# Game IDs
# ============================================================================

comptime GAME_PONG: Int = 0
comptime GAME_BREAKOUT: Int = 1
comptime GAME_SPACE_INVADERS: Int = 2
comptime GAME_SEAQUEST: Int = 3
comptime GAME_QBERT: Int = 4
comptime GAME_BEAMRIDER: Int = 5
comptime GAME_ENDURO: Int = 6
comptime GAME_FREEWAY: Int = 7
comptime GAME_MONTEZUMA: Int = 8
comptime GAME_ASTEROIDS: Int = 9

# ============================================================================
# Max ROM size
# ============================================================================

comptime MAX_ROM_SIZE: Int = 16384  # 16KB max cartridge
comptime RAM_SIZE: Int = 128  # 128 bytes system RAM
