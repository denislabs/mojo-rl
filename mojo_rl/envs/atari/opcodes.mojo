"""MOS 6502 opcode table — all 256 entries.

Each entry maps an opcode byte to:
  - instruction: which operation to perform
  - addr_mode: how to resolve the operand address
  - cycles: base cycle count
  - size: instruction size in bytes (1, 2, or 3)

Ported from CuLE (BSD-3): cule/atari/opcodes.hpp, opcodes.cpp
"""

from std.collections import InlineArray

# ============================================================================
# Addressing Modes
# ============================================================================

comptime ADDR_IMPLIED: UInt8 = 0
comptime ADDR_ACCUMULATOR: UInt8 = 1
comptime ADDR_IMMEDIATE: UInt8 = 2
comptime ADDR_ZERO_PAGE: UInt8 = 3
comptime ADDR_ZERO_PAGE_X: UInt8 = 4
comptime ADDR_ZERO_PAGE_Y: UInt8 = 5
comptime ADDR_ABSOLUTE: UInt8 = 6
comptime ADDR_ABSOLUTE_X: UInt8 = 7
comptime ADDR_ABSOLUTE_Y: UInt8 = 8
comptime ADDR_INDIRECT: UInt8 = 9
comptime ADDR_INDIRECT_X: UInt8 = 10  # (Indirect,X) aka (zp,X)
comptime ADDR_INDIRECT_Y: UInt8 = 11  # (Indirect),Y aka (zp),Y
comptime ADDR_RELATIVE: UInt8 = 12
comptime ADDR_INVALID: UInt8 = 13

# ============================================================================
# Instructions
# ============================================================================

# Load/Store
comptime OP_LDA: UInt8 = 0
comptime OP_LDX: UInt8 = 1
comptime OP_LDY: UInt8 = 2
comptime OP_STA: UInt8 = 3
comptime OP_STX: UInt8 = 4
comptime OP_STY: UInt8 = 5

# Arithmetic
comptime OP_ADC: UInt8 = 6
comptime OP_SBC: UInt8 = 7

# Compare
comptime OP_CMP: UInt8 = 8
comptime OP_CPX: UInt8 = 9
comptime OP_CPY: UInt8 = 10

# Increment/Decrement
comptime OP_INC: UInt8 = 11
comptime OP_INX: UInt8 = 12
comptime OP_INY: UInt8 = 13
comptime OP_DEC: UInt8 = 14
comptime OP_DEX: UInt8 = 15
comptime OP_DEY: UInt8 = 16

# Shifts/Rotates
comptime OP_ASL: UInt8 = 17
comptime OP_LSR: UInt8 = 18
comptime OP_ROL: UInt8 = 19
comptime OP_ROR: UInt8 = 20

# Logic
comptime OP_AND: UInt8 = 21
comptime OP_ORA: UInt8 = 22
comptime OP_EOR: UInt8 = 23
comptime OP_BIT: UInt8 = 24

# Branch
comptime OP_BCC: UInt8 = 25
comptime OP_BCS: UInt8 = 26
comptime OP_BEQ: UInt8 = 27
comptime OP_BMI: UInt8 = 28
comptime OP_BNE: UInt8 = 29
comptime OP_BPL: UInt8 = 30
comptime OP_BVC: UInt8 = 31
comptime OP_BVS: UInt8 = 32

# Jump/Call
comptime OP_JMP: UInt8 = 33
comptime OP_JSR: UInt8 = 34
comptime OP_RTS: UInt8 = 35
comptime OP_RTI: UInt8 = 36

# Stack
comptime OP_PHA: UInt8 = 37
comptime OP_PHP: UInt8 = 38
comptime OP_PLA: UInt8 = 39
comptime OP_PLP: UInt8 = 40

# Flag
comptime OP_CLC: UInt8 = 41
comptime OP_CLD: UInt8 = 42
comptime OP_CLI: UInt8 = 43
comptime OP_CLV: UInt8 = 44
comptime OP_SEC: UInt8 = 45
comptime OP_SED: UInt8 = 46
comptime OP_SEI: UInt8 = 47

# Transfer
comptime OP_TAX: UInt8 = 48
comptime OP_TAY: UInt8 = 49
comptime OP_TSX: UInt8 = 50
comptime OP_TXA: UInt8 = 51
comptime OP_TXS: UInt8 = 52
comptime OP_TYA: UInt8 = 53

# Misc
comptime OP_NOP: UInt8 = 54
comptime OP_BRK: UInt8 = 55

# Illegal opcodes (commonly used by Atari games)
comptime OP_LAX: UInt8 = 56  # LDA + LDX
comptime OP_SAX: UInt8 = 57  # STA & STX
comptime OP_DCP: UInt8 = 58  # DEC + CMP
comptime OP_ISB: UInt8 = 59  # INC + SBC
comptime OP_SLO: UInt8 = 60  # ASL + ORA
comptime OP_RLA: UInt8 = 61  # ROL + AND
comptime OP_SRE: UInt8 = 62  # LSR + EOR
comptime OP_RRA: UInt8 = 63  # ROR + ADC
comptime OP_ANC: UInt8 = 64  # AND + set C
comptime OP_ALR: UInt8 = 65  # AND + LSR
comptime OP_ARR: UInt8 = 66  # AND + ROR
comptime OP_XAA: UInt8 = 67  # unstable
comptime OP_AHX: UInt8 = 68  # unstable
comptime OP_TAS: UInt8 = 69  # unstable
comptime OP_SHX: UInt8 = 70  # unstable
comptime OP_SHY: UInt8 = 71  # unstable
comptime OP_LAS: UInt8 = 72
comptime OP_AXS: UInt8 = 73
comptime OP_KIL: UInt8 = 74  # halt CPU

# ============================================================================
# Opcode Entry
# ============================================================================


struct OpcodeEntry(Copyable, Movable, ImplicitlyCopyable, RegisterPassable):
    var instruction: UInt8
    var addr_mode: UInt8
    var cycles: UInt8
    var size: UInt8

    fn __init__(
        out self,
        instruction: UInt8,
        addr_mode: UInt8,
        cycles: UInt8,
        size: UInt8,
    ):
        self.instruction = instruction
        self.addr_mode = addr_mode
        self.cycles = cycles
        self.size = size


# Helper to create entries
@always_inline
fn _op(inst: UInt8, mode: UInt8, cyc: UInt8, sz: UInt8) -> OpcodeEntry:
    return OpcodeEntry(inst, mode, cyc, sz)


# Addressing mode sizes
@always_inline
fn addr_mode_size(mode: UInt8) -> UInt8:
    if mode == ADDR_IMPLIED or mode == ADDR_ACCUMULATOR:
        return 1
    elif (
        mode == ADDR_IMMEDIATE
        or mode == ADDR_ZERO_PAGE
        or mode == ADDR_ZERO_PAGE_X
        or mode == ADDR_ZERO_PAGE_Y
        or mode == ADDR_INDIRECT_X
        or mode == ADDR_INDIRECT_Y
        or mode == ADDR_RELATIVE
    ):
        return 2
    else:  # ABSOLUTE, ABSOLUTE_X, ABSOLUTE_Y, INDIRECT
        return 3


# ============================================================================
# Full 256-entry opcode table
# Matches CuLE's opcodes.cpp exactly
# ============================================================================


fn _build_opcode_table() -> InlineArray[OpcodeEntry, 256]:
    var t = InlineArray[OpcodeEntry, 256](uninitialized=True)

    # Initialize all as KIL (illegal halt) — catches unimplemented opcodes
    for i in range(256):
        t[i] = _op(OP_KIL, ADDR_IMPLIED, 2, 1)

    # 0x00 - 0x0F
    t[0x00] = _op(OP_BRK, ADDR_IMPLIED, 7, 1)
    t[0x01] = _op(OP_ORA, ADDR_INDIRECT_X, 6, 2)
    t[0x03] = _op(OP_SLO, ADDR_INDIRECT_X, 8, 2)
    t[0x04] = _op(OP_NOP, ADDR_ZERO_PAGE, 3, 2)
    t[0x05] = _op(OP_ORA, ADDR_ZERO_PAGE, 3, 2)
    t[0x06] = _op(OP_ASL, ADDR_ZERO_PAGE, 5, 2)
    t[0x07] = _op(OP_SLO, ADDR_ZERO_PAGE, 5, 2)
    t[0x08] = _op(OP_PHP, ADDR_IMPLIED, 3, 1)
    t[0x09] = _op(OP_ORA, ADDR_IMMEDIATE, 2, 2)
    t[0x0A] = _op(OP_ASL, ADDR_ACCUMULATOR, 2, 1)
    t[0x0B] = _op(OP_ANC, ADDR_IMMEDIATE, 2, 2)
    t[0x0C] = _op(OP_NOP, ADDR_ABSOLUTE, 4, 3)
    t[0x0D] = _op(OP_ORA, ADDR_ABSOLUTE, 4, 3)
    t[0x0E] = _op(OP_ASL, ADDR_ABSOLUTE, 6, 3)
    t[0x0F] = _op(OP_SLO, ADDR_ABSOLUTE, 6, 3)

    # 0x10 - 0x1F
    t[0x10] = _op(OP_BPL, ADDR_RELATIVE, 2, 2)
    t[0x11] = _op(OP_ORA, ADDR_INDIRECT_Y, 5, 2)
    t[0x13] = _op(OP_SLO, ADDR_INDIRECT_Y, 8, 2)
    t[0x14] = _op(OP_NOP, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x15] = _op(OP_ORA, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x16] = _op(OP_ASL, ADDR_ZERO_PAGE_X, 6, 2)
    t[0x17] = _op(OP_SLO, ADDR_ZERO_PAGE_X, 6, 2)
    t[0x18] = _op(OP_CLC, ADDR_IMPLIED, 2, 1)
    t[0x19] = _op(OP_ORA, ADDR_ABSOLUTE_Y, 4, 3)
    t[0x1A] = _op(OP_NOP, ADDR_IMPLIED, 2, 1)
    t[0x1B] = _op(OP_SLO, ADDR_ABSOLUTE_Y, 7, 3)
    t[0x1C] = _op(OP_NOP, ADDR_ABSOLUTE_X, 4, 3)
    t[0x1D] = _op(OP_ORA, ADDR_ABSOLUTE_X, 4, 3)
    t[0x1E] = _op(OP_ASL, ADDR_ABSOLUTE_X, 7, 3)
    t[0x1F] = _op(OP_SLO, ADDR_ABSOLUTE_X, 7, 3)

    # 0x20 - 0x2F
    t[0x20] = _op(OP_JSR, ADDR_ABSOLUTE, 6, 3)
    t[0x21] = _op(OP_AND, ADDR_INDIRECT_X, 6, 2)
    t[0x23] = _op(OP_RLA, ADDR_INDIRECT_X, 8, 2)
    t[0x24] = _op(OP_BIT, ADDR_ZERO_PAGE, 3, 2)
    t[0x25] = _op(OP_AND, ADDR_ZERO_PAGE, 3, 2)
    t[0x26] = _op(OP_ROL, ADDR_ZERO_PAGE, 5, 2)
    t[0x27] = _op(OP_RLA, ADDR_ZERO_PAGE, 5, 2)
    t[0x28] = _op(OP_PLP, ADDR_IMPLIED, 4, 1)
    t[0x29] = _op(OP_AND, ADDR_IMMEDIATE, 2, 2)
    t[0x2A] = _op(OP_ROL, ADDR_ACCUMULATOR, 2, 1)
    t[0x2B] = _op(OP_ANC, ADDR_IMMEDIATE, 2, 2)
    t[0x2C] = _op(OP_BIT, ADDR_ABSOLUTE, 4, 3)
    t[0x2D] = _op(OP_AND, ADDR_ABSOLUTE, 4, 3)
    t[0x2E] = _op(OP_ROL, ADDR_ABSOLUTE, 6, 3)
    t[0x2F] = _op(OP_RLA, ADDR_ABSOLUTE, 6, 3)

    # 0x30 - 0x3F
    t[0x30] = _op(OP_BMI, ADDR_RELATIVE, 2, 2)
    t[0x31] = _op(OP_AND, ADDR_INDIRECT_Y, 5, 2)
    t[0x33] = _op(OP_RLA, ADDR_INDIRECT_Y, 8, 2)
    t[0x34] = _op(OP_NOP, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x35] = _op(OP_AND, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x36] = _op(OP_ROL, ADDR_ZERO_PAGE_X, 6, 2)
    t[0x37] = _op(OP_RLA, ADDR_ZERO_PAGE_X, 6, 2)
    t[0x38] = _op(OP_SEC, ADDR_IMPLIED, 2, 1)
    t[0x39] = _op(OP_AND, ADDR_ABSOLUTE_Y, 4, 3)
    t[0x3A] = _op(OP_NOP, ADDR_IMPLIED, 2, 1)
    t[0x3B] = _op(OP_RLA, ADDR_ABSOLUTE_Y, 7, 3)
    t[0x3C] = _op(OP_NOP, ADDR_ABSOLUTE_X, 4, 3)
    t[0x3D] = _op(OP_AND, ADDR_ABSOLUTE_X, 4, 3)
    t[0x3E] = _op(OP_ROL, ADDR_ABSOLUTE_X, 7, 3)
    t[0x3F] = _op(OP_RLA, ADDR_ABSOLUTE_X, 7, 3)

    # 0x40 - 0x4F
    t[0x40] = _op(OP_RTI, ADDR_IMPLIED, 6, 1)
    t[0x41] = _op(OP_EOR, ADDR_INDIRECT_X, 6, 2)
    t[0x43] = _op(OP_SRE, ADDR_INDIRECT_X, 8, 2)
    t[0x44] = _op(OP_NOP, ADDR_ZERO_PAGE, 3, 2)
    t[0x45] = _op(OP_EOR, ADDR_ZERO_PAGE, 3, 2)
    t[0x46] = _op(OP_LSR, ADDR_ZERO_PAGE, 5, 2)
    t[0x47] = _op(OP_SRE, ADDR_ZERO_PAGE, 5, 2)
    t[0x48] = _op(OP_PHA, ADDR_IMPLIED, 3, 1)
    t[0x49] = _op(OP_EOR, ADDR_IMMEDIATE, 2, 2)
    t[0x4A] = _op(OP_LSR, ADDR_ACCUMULATOR, 2, 1)
    t[0x4B] = _op(OP_ALR, ADDR_IMMEDIATE, 2, 2)
    t[0x4C] = _op(OP_JMP, ADDR_ABSOLUTE, 3, 3)
    t[0x4D] = _op(OP_EOR, ADDR_ABSOLUTE, 4, 3)
    t[0x4E] = _op(OP_LSR, ADDR_ABSOLUTE, 6, 3)
    t[0x4F] = _op(OP_SRE, ADDR_ABSOLUTE, 6, 3)

    # 0x50 - 0x5F
    t[0x50] = _op(OP_BVC, ADDR_RELATIVE, 2, 2)
    t[0x51] = _op(OP_EOR, ADDR_INDIRECT_Y, 5, 2)
    t[0x53] = _op(OP_SRE, ADDR_INDIRECT_Y, 8, 2)
    t[0x54] = _op(OP_NOP, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x55] = _op(OP_EOR, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x56] = _op(OP_LSR, ADDR_ZERO_PAGE_X, 6, 2)
    t[0x57] = _op(OP_SRE, ADDR_ZERO_PAGE_X, 6, 2)
    t[0x58] = _op(OP_CLI, ADDR_IMPLIED, 2, 1)
    t[0x59] = _op(OP_EOR, ADDR_ABSOLUTE_Y, 4, 3)
    t[0x5A] = _op(OP_NOP, ADDR_IMPLIED, 2, 1)
    t[0x5B] = _op(OP_SRE, ADDR_ABSOLUTE_Y, 7, 3)
    t[0x5C] = _op(OP_NOP, ADDR_ABSOLUTE_X, 4, 3)
    t[0x5D] = _op(OP_EOR, ADDR_ABSOLUTE_X, 4, 3)
    t[0x5E] = _op(OP_LSR, ADDR_ABSOLUTE_X, 7, 3)
    t[0x5F] = _op(OP_SRE, ADDR_ABSOLUTE_X, 7, 3)

    # 0x60 - 0x6F
    t[0x60] = _op(OP_RTS, ADDR_IMPLIED, 6, 1)
    t[0x61] = _op(OP_ADC, ADDR_INDIRECT_X, 6, 2)
    t[0x63] = _op(OP_RRA, ADDR_INDIRECT_X, 8, 2)
    t[0x64] = _op(OP_NOP, ADDR_ZERO_PAGE, 3, 2)
    t[0x65] = _op(OP_ADC, ADDR_ZERO_PAGE, 3, 2)
    t[0x66] = _op(OP_ROR, ADDR_ZERO_PAGE, 5, 2)
    t[0x67] = _op(OP_RRA, ADDR_ZERO_PAGE, 5, 2)
    t[0x68] = _op(OP_PLA, ADDR_IMPLIED, 4, 1)
    t[0x69] = _op(OP_ADC, ADDR_IMMEDIATE, 2, 2)
    t[0x6A] = _op(OP_ROR, ADDR_ACCUMULATOR, 2, 1)
    t[0x6B] = _op(OP_ARR, ADDR_IMMEDIATE, 2, 2)
    t[0x6C] = _op(OP_JMP, ADDR_INDIRECT, 5, 3)
    t[0x6D] = _op(OP_ADC, ADDR_ABSOLUTE, 4, 3)
    t[0x6E] = _op(OP_ROR, ADDR_ABSOLUTE, 6, 3)
    t[0x6F] = _op(OP_RRA, ADDR_ABSOLUTE, 6, 3)

    # 0x70 - 0x7F
    t[0x70] = _op(OP_BVS, ADDR_RELATIVE, 2, 2)
    t[0x71] = _op(OP_ADC, ADDR_INDIRECT_Y, 5, 2)
    t[0x73] = _op(OP_RRA, ADDR_INDIRECT_Y, 8, 2)
    t[0x74] = _op(OP_NOP, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x75] = _op(OP_ADC, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x76] = _op(OP_ROR, ADDR_ZERO_PAGE_X, 6, 2)
    t[0x77] = _op(OP_RRA, ADDR_ZERO_PAGE_X, 6, 2)
    t[0x78] = _op(OP_SEI, ADDR_IMPLIED, 2, 1)
    t[0x79] = _op(OP_ADC, ADDR_ABSOLUTE_Y, 4, 3)
    t[0x7A] = _op(OP_NOP, ADDR_IMPLIED, 2, 1)
    t[0x7B] = _op(OP_RRA, ADDR_ABSOLUTE_Y, 7, 3)
    t[0x7C] = _op(OP_NOP, ADDR_ABSOLUTE_X, 4, 3)
    t[0x7D] = _op(OP_ADC, ADDR_ABSOLUTE_X, 4, 3)
    t[0x7E] = _op(OP_ROR, ADDR_ABSOLUTE_X, 7, 3)
    t[0x7F] = _op(OP_RRA, ADDR_ABSOLUTE_X, 7, 3)

    # 0x80 - 0x8F
    t[0x80] = _op(OP_NOP, ADDR_IMMEDIATE, 2, 2)
    t[0x81] = _op(OP_STA, ADDR_INDIRECT_X, 6, 2)
    t[0x82] = _op(OP_NOP, ADDR_IMMEDIATE, 2, 2)
    t[0x83] = _op(OP_SAX, ADDR_INDIRECT_X, 6, 2)
    t[0x84] = _op(OP_STY, ADDR_ZERO_PAGE, 3, 2)
    t[0x85] = _op(OP_STA, ADDR_ZERO_PAGE, 3, 2)
    t[0x86] = _op(OP_STX, ADDR_ZERO_PAGE, 3, 2)
    t[0x87] = _op(OP_SAX, ADDR_ZERO_PAGE, 3, 2)
    t[0x88] = _op(OP_DEY, ADDR_IMPLIED, 2, 1)
    t[0x89] = _op(OP_NOP, ADDR_IMMEDIATE, 2, 2)
    t[0x8A] = _op(OP_TXA, ADDR_IMPLIED, 2, 1)
    t[0x8B] = _op(OP_XAA, ADDR_IMMEDIATE, 2, 2)
    t[0x8C] = _op(OP_STY, ADDR_ABSOLUTE, 4, 3)
    t[0x8D] = _op(OP_STA, ADDR_ABSOLUTE, 4, 3)
    t[0x8E] = _op(OP_STX, ADDR_ABSOLUTE, 4, 3)
    t[0x8F] = _op(OP_SAX, ADDR_ABSOLUTE, 4, 3)

    # 0x90 - 0x9F
    t[0x90] = _op(OP_BCC, ADDR_RELATIVE, 2, 2)
    t[0x91] = _op(OP_STA, ADDR_INDIRECT_Y, 6, 2)
    t[0x93] = _op(OP_AHX, ADDR_INDIRECT_Y, 6, 2)
    t[0x94] = _op(OP_STY, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x95] = _op(OP_STA, ADDR_ZERO_PAGE_X, 4, 2)
    t[0x96] = _op(OP_STX, ADDR_ZERO_PAGE_Y, 4, 2)
    t[0x97] = _op(OP_SAX, ADDR_ZERO_PAGE_Y, 4, 2)
    t[0x98] = _op(OP_TYA, ADDR_IMPLIED, 2, 1)
    t[0x99] = _op(OP_STA, ADDR_ABSOLUTE_Y, 5, 3)
    t[0x9A] = _op(OP_TXS, ADDR_IMPLIED, 2, 1)
    t[0x9B] = _op(OP_TAS, ADDR_ABSOLUTE_Y, 5, 3)
    t[0x9C] = _op(OP_SHY, ADDR_ABSOLUTE_X, 5, 3)
    t[0x9D] = _op(OP_STA, ADDR_ABSOLUTE_X, 5, 3)
    t[0x9E] = _op(OP_SHX, ADDR_ABSOLUTE_Y, 5, 3)
    t[0x9F] = _op(OP_AHX, ADDR_ABSOLUTE_Y, 5, 3)

    # 0xA0 - 0xAF
    t[0xA0] = _op(OP_LDY, ADDR_IMMEDIATE, 2, 2)
    t[0xA1] = _op(OP_LDA, ADDR_INDIRECT_X, 6, 2)
    t[0xA2] = _op(OP_LDX, ADDR_IMMEDIATE, 2, 2)
    t[0xA3] = _op(OP_LAX, ADDR_INDIRECT_X, 6, 2)
    t[0xA4] = _op(OP_LDY, ADDR_ZERO_PAGE, 3, 2)
    t[0xA5] = _op(OP_LDA, ADDR_ZERO_PAGE, 3, 2)
    t[0xA6] = _op(OP_LDX, ADDR_ZERO_PAGE, 3, 2)
    t[0xA7] = _op(OP_LAX, ADDR_ZERO_PAGE, 3, 2)
    t[0xA8] = _op(OP_TAY, ADDR_IMPLIED, 2, 1)
    t[0xA9] = _op(OP_LDA, ADDR_IMMEDIATE, 2, 2)
    t[0xAA] = _op(OP_TAX, ADDR_IMPLIED, 2, 1)
    t[0xAB] = _op(OP_LAX, ADDR_IMMEDIATE, 2, 2)
    t[0xAC] = _op(OP_LDY, ADDR_ABSOLUTE, 4, 3)
    t[0xAD] = _op(OP_LDA, ADDR_ABSOLUTE, 4, 3)
    t[0xAE] = _op(OP_LDX, ADDR_ABSOLUTE, 4, 3)
    t[0xAF] = _op(OP_LAX, ADDR_ABSOLUTE, 4, 3)

    # 0xB0 - 0xBF
    t[0xB0] = _op(OP_BCS, ADDR_RELATIVE, 2, 2)
    t[0xB1] = _op(OP_LDA, ADDR_INDIRECT_Y, 5, 2)
    t[0xB3] = _op(OP_LAX, ADDR_INDIRECT_Y, 5, 2)
    t[0xB4] = _op(OP_LDY, ADDR_ZERO_PAGE_X, 4, 2)
    t[0xB5] = _op(OP_LDA, ADDR_ZERO_PAGE_X, 4, 2)
    t[0xB6] = _op(OP_LDX, ADDR_ZERO_PAGE_Y, 4, 2)
    t[0xB7] = _op(OP_LAX, ADDR_ZERO_PAGE_Y, 4, 2)
    t[0xB8] = _op(OP_CLV, ADDR_IMPLIED, 2, 1)
    t[0xB9] = _op(OP_LDA, ADDR_ABSOLUTE_Y, 4, 3)
    t[0xBA] = _op(OP_TSX, ADDR_IMPLIED, 2, 1)
    t[0xBB] = _op(OP_LAS, ADDR_ABSOLUTE_Y, 4, 3)
    t[0xBC] = _op(OP_LDY, ADDR_ABSOLUTE_X, 4, 3)
    t[0xBD] = _op(OP_LDA, ADDR_ABSOLUTE_X, 4, 3)
    t[0xBE] = _op(OP_LDX, ADDR_ABSOLUTE_Y, 4, 3)
    t[0xBF] = _op(OP_LAX, ADDR_ABSOLUTE_Y, 4, 3)

    # 0xC0 - 0xCF
    t[0xC0] = _op(OP_CPY, ADDR_IMMEDIATE, 2, 2)
    t[0xC1] = _op(OP_CMP, ADDR_INDIRECT_X, 6, 2)
    t[0xC2] = _op(OP_NOP, ADDR_IMMEDIATE, 2, 2)
    t[0xC3] = _op(OP_DCP, ADDR_INDIRECT_X, 8, 2)
    t[0xC4] = _op(OP_CPY, ADDR_ZERO_PAGE, 3, 2)
    t[0xC5] = _op(OP_CMP, ADDR_ZERO_PAGE, 3, 2)
    t[0xC6] = _op(OP_DEC, ADDR_ZERO_PAGE, 5, 2)
    t[0xC7] = _op(OP_DCP, ADDR_ZERO_PAGE, 5, 2)
    t[0xC8] = _op(OP_INY, ADDR_IMPLIED, 2, 1)
    t[0xC9] = _op(OP_CMP, ADDR_IMMEDIATE, 2, 2)
    t[0xCA] = _op(OP_DEX, ADDR_IMPLIED, 2, 1)
    t[0xCB] = _op(OP_AXS, ADDR_IMMEDIATE, 2, 2)
    t[0xCC] = _op(OP_CPY, ADDR_ABSOLUTE, 4, 3)
    t[0xCD] = _op(OP_CMP, ADDR_ABSOLUTE, 4, 3)
    t[0xCE] = _op(OP_DEC, ADDR_ABSOLUTE, 6, 3)
    t[0xCF] = _op(OP_DCP, ADDR_ABSOLUTE, 6, 3)

    # 0xD0 - 0xDF
    t[0xD0] = _op(OP_BNE, ADDR_RELATIVE, 2, 2)
    t[0xD1] = _op(OP_CMP, ADDR_INDIRECT_Y, 5, 2)
    t[0xD3] = _op(OP_DCP, ADDR_INDIRECT_Y, 8, 2)
    t[0xD4] = _op(OP_NOP, ADDR_ZERO_PAGE_X, 4, 2)
    t[0xD5] = _op(OP_CMP, ADDR_ZERO_PAGE_X, 4, 2)
    t[0xD6] = _op(OP_DEC, ADDR_ZERO_PAGE_X, 6, 2)
    t[0xD7] = _op(OP_DCP, ADDR_ZERO_PAGE_X, 6, 2)
    t[0xD8] = _op(OP_CLD, ADDR_IMPLIED, 2, 1)
    t[0xD9] = _op(OP_CMP, ADDR_ABSOLUTE_Y, 4, 3)
    t[0xDA] = _op(OP_NOP, ADDR_IMPLIED, 2, 1)
    t[0xDB] = _op(OP_DCP, ADDR_ABSOLUTE_Y, 7, 3)
    t[0xDC] = _op(OP_NOP, ADDR_ABSOLUTE_X, 4, 3)
    t[0xDD] = _op(OP_CMP, ADDR_ABSOLUTE_X, 4, 3)
    t[0xDE] = _op(OP_DEC, ADDR_ABSOLUTE_X, 7, 3)
    t[0xDF] = _op(OP_DCP, ADDR_ABSOLUTE_X, 7, 3)

    # 0xE0 - 0xEF
    t[0xE0] = _op(OP_CPX, ADDR_IMMEDIATE, 2, 2)
    t[0xE1] = _op(OP_SBC, ADDR_INDIRECT_X, 6, 2)
    t[0xE2] = _op(OP_NOP, ADDR_IMMEDIATE, 2, 2)
    t[0xE3] = _op(OP_ISB, ADDR_INDIRECT_X, 8, 2)
    t[0xE4] = _op(OP_CPX, ADDR_ZERO_PAGE, 3, 2)
    t[0xE5] = _op(OP_SBC, ADDR_ZERO_PAGE, 3, 2)
    t[0xE6] = _op(OP_INC, ADDR_ZERO_PAGE, 5, 2)
    t[0xE7] = _op(OP_ISB, ADDR_ZERO_PAGE, 5, 2)
    t[0xE8] = _op(OP_INX, ADDR_IMPLIED, 2, 1)
    t[0xE9] = _op(OP_SBC, ADDR_IMMEDIATE, 2, 2)
    t[0xEA] = _op(OP_NOP, ADDR_IMPLIED, 2, 1)
    t[0xEB] = _op(OP_SBC, ADDR_IMMEDIATE, 2, 2)  # Illegal SBC #imm
    t[0xEC] = _op(OP_CPX, ADDR_ABSOLUTE, 4, 3)
    t[0xED] = _op(OP_SBC, ADDR_ABSOLUTE, 4, 3)
    t[0xEE] = _op(OP_INC, ADDR_ABSOLUTE, 6, 3)
    t[0xEF] = _op(OP_ISB, ADDR_ABSOLUTE, 6, 3)

    # 0xF0 - 0xFF
    t[0xF0] = _op(OP_BEQ, ADDR_RELATIVE, 2, 2)
    t[0xF1] = _op(OP_SBC, ADDR_INDIRECT_Y, 5, 2)
    t[0xF3] = _op(OP_ISB, ADDR_INDIRECT_Y, 8, 2)
    t[0xF4] = _op(OP_NOP, ADDR_ZERO_PAGE_X, 4, 2)
    t[0xF5] = _op(OP_SBC, ADDR_ZERO_PAGE_X, 4, 2)
    t[0xF6] = _op(OP_INC, ADDR_ZERO_PAGE_X, 6, 2)
    t[0xF7] = _op(OP_ISB, ADDR_ZERO_PAGE_X, 6, 2)
    t[0xF8] = _op(OP_SED, ADDR_IMPLIED, 2, 1)
    t[0xF9] = _op(OP_SBC, ADDR_ABSOLUTE_Y, 4, 3)
    t[0xFA] = _op(OP_NOP, ADDR_IMPLIED, 2, 1)
    t[0xFB] = _op(OP_ISB, ADDR_ABSOLUTE_Y, 7, 3)
    t[0xFC] = _op(OP_NOP, ADDR_ABSOLUTE_X, 4, 3)
    t[0xFD] = _op(OP_SBC, ADDR_ABSOLUTE_X, 4, 3)
    t[0xFE] = _op(OP_INC, ADDR_ABSOLUTE_X, 7, 3)
    t[0xFF] = _op(OP_ISB, ADDR_ABSOLUTE_X, 7, 3)

    return t^


# Global opcode table
comptime OPCODE_TABLE = _build_opcode_table()
