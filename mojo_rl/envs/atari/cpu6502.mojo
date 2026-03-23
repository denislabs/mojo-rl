"""MOS 6502 CPU emulation for Atari 2600.

Implements the full fetch-decode-execute cycle with all official
opcodes and commonly-used illegal opcodes.

The memory map for the Atari 2600:
  0x0000-0x007F: TIA registers (active on A12=0, A7=0)
  0x0080-0x00FF: RIOT RAM (128 bytes, active on A12=0, A7=1)
  0x0280-0x0297: RIOT registers (active on A12=0, A7=1, A9=1)
  0x1000-0x1FFF: Cartridge ROM (active on A12=1)

Ported from CuLE (BSD-3): cule/atari/m6502.hpp, mmc.hpp, stack.hpp
"""

from .atari_state import AtariState
from .flags import (
    FLAG_C,
    FLAG_Z,
    FLAG_I,
    FLAG_D,
    FLAG_B,
    FLAG_V,
    FLAG_N,
    RAM_SIZE,
)
from .opcodes import (
    OPCODE_TABLE,
    OpcodeEntry,
    ADDR_IMPLIED,
    ADDR_ACCUMULATOR,
    ADDR_IMMEDIATE,
    ADDR_ZERO_PAGE,
    ADDR_ZERO_PAGE_X,
    ADDR_ZERO_PAGE_Y,
    ADDR_ABSOLUTE,
    ADDR_ABSOLUTE_X,
    ADDR_ABSOLUTE_Y,
    ADDR_INDIRECT,
    ADDR_INDIRECT_X,
    ADDR_INDIRECT_Y,
    ADDR_RELATIVE,
    OP_LDA,
    OP_LDX,
    OP_LDY,
    OP_STA,
    OP_STX,
    OP_STY,
    OP_ADC,
    OP_SBC,
    OP_CMP,
    OP_CPX,
    OP_CPY,
    OP_INC,
    OP_INX,
    OP_INY,
    OP_DEC,
    OP_DEX,
    OP_DEY,
    OP_ASL,
    OP_LSR,
    OP_ROL,
    OP_ROR,
    OP_AND,
    OP_ORA,
    OP_EOR,
    OP_BIT,
    OP_BCC,
    OP_BCS,
    OP_BEQ,
    OP_BMI,
    OP_BNE,
    OP_BPL,
    OP_BVC,
    OP_BVS,
    OP_JMP,
    OP_JSR,
    OP_RTS,
    OP_RTI,
    OP_PHA,
    OP_PHP,
    OP_PLA,
    OP_PLP,
    OP_CLC,
    OP_CLD,
    OP_CLI,
    OP_CLV,
    OP_SEC,
    OP_SED,
    OP_SEI,
    OP_TAX,
    OP_TAY,
    OP_TSX,
    OP_TXA,
    OP_TXS,
    OP_TYA,
    OP_NOP,
    OP_BRK,
    OP_LAX,
    OP_SAX,
    OP_DCP,
    OP_ISB,
    OP_SLO,
    OP_RLA,
    OP_SRE,
    OP_RRA,
    OP_ANC,
    OP_ALR,
    OP_ARR,
    OP_AXS,
    OP_KIL,
)
from .ram import read_ram, write_ram
from .tia import tia_read, tia_write
from .riot import riot_read, riot_write, riot_update_timer
from .cartridge import rom_read, rom_write


# ============================================================================
# Memory Access
# ============================================================================


@always_inline
def mem_read(
    state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    addr: UInt16,
) -> UInt8:
    """Read a byte from the Atari 2600 memory map."""
    var a = Int(addr) & 0x1FFF  # 13-bit address space

    if a & 0x1000:  # Cartridge ROM
        return rom_read(state, rom, rom_size, UInt16(a))
    elif a & 0x0080:  # RIOT area
        if a & 0x0200:  # RIOT registers (0x0280-0x0297)
            return riot_read(state, UInt8(a & 0xFF))
        else:  # RAM (0x0080-0x00FF)
            return read_ram(state.ram, Int(a & 0x7F))
    else:  # TIA
        return tia_read(state, UInt8(a & 0x0F))


@always_inline
def mem_write(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    addr: UInt16,
    value: UInt8,
):
    """Write a byte to the Atari 2600 memory map."""
    var a = Int(addr) & 0x1FFF

    if a & 0x1000:  # Cartridge ROM (may trigger bank switch)
        rom_write(state, rom, rom_size, UInt16(a), value)
    elif a & 0x0080:  # RIOT area
        if a & 0x0200:  # RIOT registers
            riot_write(state, UInt8(a & 0xFF), value)
        else:  # RAM
            write_ram(state.ram, Int(a & 0x7F), value)
    else:  # TIA
        tia_write(state, UInt8(a & 0x3F), value)


# ============================================================================
# Stack Operations
# ============================================================================


@always_inline
def push_byte(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    value: UInt8,
):
    """Push a byte onto the stack."""
    mem_write(state, rom, rom_size, UInt16(0x0100) + UInt16(state.sp), value)
    state.sp -= 1


@always_inline
def pull_byte(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
) -> UInt8:
    """Pull a byte from the stack."""
    state.sp += 1
    return mem_read(state, rom, rom_size, UInt16(0x0100) + UInt16(state.sp))


@always_inline
def push_word(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    value: UInt16,
):
    """Push a 16-bit word onto the stack (high byte first)."""
    push_byte(state, rom, rom_size, UInt8((value >> 8) & 0xFF))
    push_byte(state, rom, rom_size, UInt8(value & 0xFF))


@always_inline
def pull_word(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
) -> UInt16:
    """Pull a 16-bit word from the stack (low byte first)."""
    var lo = UInt16(pull_byte(state, rom, rom_size))
    var hi = UInt16(pull_byte(state, rom, rom_size))
    return (hi << 8) | lo


# ============================================================================
# Flag Helpers
# ============================================================================


@always_inline
def set_flag(mut state: AtariState, flag: UInt8, value: Bool):
    if value:
        state.flags = state.flags | flag
    else:
        state.flags = state.flags & ~flag


@always_inline
def get_flag(state: AtariState, flag: UInt8) -> Bool:
    return (state.flags & flag) != 0


@always_inline
def update_nz(mut state: AtariState, value: UInt8):
    """Update N and Z flags based on value."""
    set_flag(state, FLAG_Z, value == 0)
    set_flag(state, FLAG_N, (value & 0x80) != 0)


# ============================================================================
# Addressing Mode Resolution
# ============================================================================


@always_inline
def resolve_operand_addr(
    state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    mode: UInt8,
    pc_after_opcode: UInt16,
) -> UInt16:
    """Resolve the effective address for the given addressing mode.

    pc_after_opcode points to the first operand byte (PC+1 after opcode fetch).
    """
    if mode == ADDR_ZERO_PAGE:
        return UInt16(mem_read(state, rom, rom_size, pc_after_opcode))

    elif mode == ADDR_ZERO_PAGE_X:
        var base = mem_read(state, rom, rom_size, pc_after_opcode)
        return UInt16((base + state.x) & 0xFF)

    elif mode == ADDR_ZERO_PAGE_Y:
        var base = mem_read(state, rom, rom_size, pc_after_opcode)
        return UInt16((base + state.y) & 0xFF)

    elif mode == ADDR_ABSOLUTE:
        var lo = UInt16(mem_read(state, rom, rom_size, pc_after_opcode))
        var hi = UInt16(mem_read(state, rom, rom_size, pc_after_opcode + 1))
        return (hi << 8) | lo

    elif mode == ADDR_ABSOLUTE_X:
        var lo = UInt16(mem_read(state, rom, rom_size, pc_after_opcode))
        var hi = UInt16(mem_read(state, rom, rom_size, pc_after_opcode + 1))
        return ((hi << 8) | lo) + UInt16(state.x)

    elif mode == ADDR_ABSOLUTE_Y:
        var lo = UInt16(mem_read(state, rom, rom_size, pc_after_opcode))
        var hi = UInt16(mem_read(state, rom, rom_size, pc_after_opcode + 1))
        return ((hi << 8) | lo) + UInt16(state.y)

    elif mode == ADDR_INDIRECT:
        var lo = UInt16(mem_read(state, rom, rom_size, pc_after_opcode))
        var hi = UInt16(mem_read(state, rom, rom_size, pc_after_opcode + 1))
        var ptr = (hi << 8) | lo
        # 6502 indirect JMP bug: doesn't cross page boundary
        var ptr_hi = (ptr & 0xFF00) | ((ptr + 1) & 0x00FF)
        var addr_lo = UInt16(mem_read(state, rom, rom_size, ptr))
        var addr_hi = UInt16(mem_read(state, rom, rom_size, ptr_hi))
        return (addr_hi << 8) | addr_lo

    elif mode == ADDR_INDIRECT_X:
        var base = mem_read(state, rom, rom_size, pc_after_opcode)
        var zp = (base + state.x) & 0xFF
        var lo = UInt16(mem_read(state, rom, rom_size, UInt16(zp)))
        var hi = UInt16(mem_read(state, rom, rom_size, UInt16((zp + 1) & 0xFF)))
        return (hi << 8) | lo

    elif mode == ADDR_INDIRECT_Y:
        var zp = mem_read(state, rom, rom_size, pc_after_opcode)
        var lo = UInt16(mem_read(state, rom, rom_size, UInt16(zp)))
        var hi = UInt16(mem_read(state, rom, rom_size, UInt16((zp + 1) & 0xFF)))
        return ((hi << 8) | lo) + UInt16(state.y)

    # IMMEDIATE, RELATIVE, IMPLIED, ACCUMULATOR — addr is just PC
    return pc_after_opcode


# ============================================================================
# Instruction Execution
# ============================================================================


@always_inline
def execute_one(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
) -> UInt8:
    """Execute one instruction. Returns the number of CPU cycles consumed."""
    var opcode_byte = mem_read(state, rom, rom_size, state.pc)
    var table = materialize[OPCODE_TABLE]()
    var entry = table[Int(opcode_byte)]
    var inst = entry.instruction
    var mode = entry.addr_mode
    var cycles = entry.cycles
    var size = entry.size

    var operand_pc = state.pc + 1  # Points to first operand byte
    var addr = resolve_operand_addr(state, rom, rom_size, mode, operand_pc)

    # Advance PC past this instruction
    state.pc = state.pc + UInt16(size)

    # ====== Load/Store ======
    if inst == OP_LDA:
        if mode == ADDR_IMMEDIATE:
            state.a = mem_read(state, rom, rom_size, addr)
        else:
            state.a = mem_read(state, rom, rom_size, addr)
        update_nz(state, state.a)

    elif inst == OP_LDX:
        if mode == ADDR_IMMEDIATE:
            state.x = mem_read(state, rom, rom_size, addr)
        else:
            state.x = mem_read(state, rom, rom_size, addr)
        update_nz(state, state.x)

    elif inst == OP_LDY:
        if mode == ADDR_IMMEDIATE:
            state.y = mem_read(state, rom, rom_size, addr)
        else:
            state.y = mem_read(state, rom, rom_size, addr)
        update_nz(state, state.y)

    elif inst == OP_STA:
        mem_write(state, rom, rom_size, addr, state.a)

    elif inst == OP_STX:
        mem_write(state, rom, rom_size, addr, state.x)

    elif inst == OP_STY:
        mem_write(state, rom, rom_size, addr, state.y)

    # ====== Arithmetic ======
    elif inst == OP_ADC:
        var operand = mem_read(state, rom, rom_size, addr)
        _adc(state, operand)

    elif inst == OP_SBC:
        var operand = mem_read(state, rom, rom_size, addr)
        _sbc(state, operand)

    # ====== Compare ======
    elif inst == OP_CMP:
        _compare(state, state.a, mem_read(state, rom, rom_size, addr))

    elif inst == OP_CPX:
        _compare(state, state.x, mem_read(state, rom, rom_size, addr))

    elif inst == OP_CPY:
        _compare(state, state.y, mem_read(state, rom, rom_size, addr))

    # ====== Inc/Dec ======
    elif inst == OP_INC:
        var val = mem_read(state, rom, rom_size, addr) + 1
        mem_write(state, rom, rom_size, addr, val)
        update_nz(state, val)

    elif inst == OP_INX:
        state.x = state.x + 1
        update_nz(state, state.x)

    elif inst == OP_INY:
        state.y = state.y + 1
        update_nz(state, state.y)

    elif inst == OP_DEC:
        var val = mem_read(state, rom, rom_size, addr) - 1
        mem_write(state, rom, rom_size, addr, val)
        update_nz(state, val)

    elif inst == OP_DEX:
        state.x = state.x - 1
        update_nz(state, state.x)

    elif inst == OP_DEY:
        state.y = state.y - 1
        update_nz(state, state.y)

    # ====== Shifts ======
    elif inst == OP_ASL:
        if mode == ADDR_ACCUMULATOR:
            set_flag(state, FLAG_C, (state.a & 0x80) != 0)
            state.a = state.a << 1
            update_nz(state, state.a)
        else:
            var val = mem_read(state, rom, rom_size, addr)
            set_flag(state, FLAG_C, (val & 0x80) != 0)
            val = val << 1
            mem_write(state, rom, rom_size, addr, val)
            update_nz(state, val)

    elif inst == OP_LSR:
        if mode == ADDR_ACCUMULATOR:
            set_flag(state, FLAG_C, (state.a & 0x01) != 0)
            state.a = state.a >> 1
            update_nz(state, state.a)
        else:
            var val = mem_read(state, rom, rom_size, addr)
            set_flag(state, FLAG_C, (val & 0x01) != 0)
            val = val >> 1
            mem_write(state, rom, rom_size, addr, val)
            update_nz(state, val)

    elif inst == OP_ROL:
        var carry_in = UInt8(1) if get_flag(state, FLAG_C) else UInt8(0)
        if mode == ADDR_ACCUMULATOR:
            set_flag(state, FLAG_C, (state.a & 0x80) != 0)
            state.a = (state.a << 1) | carry_in
            update_nz(state, state.a)
        else:
            var val = mem_read(state, rom, rom_size, addr)
            set_flag(state, FLAG_C, (val & 0x80) != 0)
            val = (val << 1) | carry_in
            mem_write(state, rom, rom_size, addr, val)
            update_nz(state, val)

    elif inst == OP_ROR:
        var carry_in = UInt8(0x80) if get_flag(state, FLAG_C) else UInt8(0)
        if mode == ADDR_ACCUMULATOR:
            set_flag(state, FLAG_C, (state.a & 0x01) != 0)
            state.a = (state.a >> 1) | carry_in
            update_nz(state, state.a)
        else:
            var val = mem_read(state, rom, rom_size, addr)
            set_flag(state, FLAG_C, (val & 0x01) != 0)
            val = (val >> 1) | carry_in
            mem_write(state, rom, rom_size, addr, val)
            update_nz(state, val)

    # ====== Logic ======
    elif inst == OP_AND:
        state.a = state.a & mem_read(state, rom, rom_size, addr)
        update_nz(state, state.a)

    elif inst == OP_ORA:
        state.a = state.a | mem_read(state, rom, rom_size, addr)
        update_nz(state, state.a)

    elif inst == OP_EOR:
        state.a = state.a ^ mem_read(state, rom, rom_size, addr)
        update_nz(state, state.a)

    elif inst == OP_BIT:
        var val = mem_read(state, rom, rom_size, addr)
        set_flag(state, FLAG_Z, (state.a & val) == 0)
        set_flag(state, FLAG_N, (val & 0x80) != 0)
        set_flag(state, FLAG_V, (val & 0x40) != 0)

    # ====== Branch ======
    elif inst == OP_BCC:
        if not get_flag(state, FLAG_C):
            _branch(state, rom, rom_size, addr)
            cycles += 1

    elif inst == OP_BCS:
        if get_flag(state, FLAG_C):
            _branch(state, rom, rom_size, addr)
            cycles += 1

    elif inst == OP_BEQ:
        if get_flag(state, FLAG_Z):
            _branch(state, rom, rom_size, addr)
            cycles += 1

    elif inst == OP_BMI:
        if get_flag(state, FLAG_N):
            _branch(state, rom, rom_size, addr)
            cycles += 1

    elif inst == OP_BNE:
        if not get_flag(state, FLAG_Z):
            _branch(state, rom, rom_size, addr)
            cycles += 1

    elif inst == OP_BPL:
        if not get_flag(state, FLAG_N):
            _branch(state, rom, rom_size, addr)
            cycles += 1

    elif inst == OP_BVC:
        if not get_flag(state, FLAG_V):
            _branch(state, rom, rom_size, addr)
            cycles += 1

    elif inst == OP_BVS:
        if get_flag(state, FLAG_V):
            _branch(state, rom, rom_size, addr)
            cycles += 1

    # ====== Jump/Call ======
    elif inst == OP_JMP:
        state.pc = addr

    elif inst == OP_JSR:
        push_word(state, rom, rom_size, state.pc - 1)
        state.pc = addr

    elif inst == OP_RTS:
        state.pc = pull_word(state, rom, rom_size) + 1

    elif inst == OP_RTI:
        state.flags = pull_byte(state, rom, rom_size) | 0x20  # Bit 5 always set
        state.pc = pull_word(state, rom, rom_size)

    # ====== Stack ======
    elif inst == OP_PHA:
        push_byte(state, rom, rom_size, state.a)

    elif inst == OP_PHP:
        push_byte(state, rom, rom_size, state.flags | FLAG_B | 0x20)

    elif inst == OP_PLA:
        state.a = pull_byte(state, rom, rom_size)
        update_nz(state, state.a)

    elif inst == OP_PLP:
        state.flags = pull_byte(state, rom, rom_size) | 0x20
        state.flags = state.flags & ~FLAG_B

    # ====== Flags ======
    elif inst == OP_CLC:
        set_flag(state, FLAG_C, False)

    elif inst == OP_CLD:
        set_flag(state, FLAG_D, False)

    elif inst == OP_CLI:
        set_flag(state, FLAG_I, False)

    elif inst == OP_CLV:
        set_flag(state, FLAG_V, False)

    elif inst == OP_SEC:
        set_flag(state, FLAG_C, True)

    elif inst == OP_SED:
        set_flag(state, FLAG_D, True)

    elif inst == OP_SEI:
        set_flag(state, FLAG_I, True)

    # ====== Transfer ======
    elif inst == OP_TAX:
        state.x = state.a
        update_nz(state, state.x)

    elif inst == OP_TAY:
        state.y = state.a
        update_nz(state, state.y)

    elif inst == OP_TSX:
        state.x = state.sp
        update_nz(state, state.x)

    elif inst == OP_TXA:
        state.a = state.x
        update_nz(state, state.a)

    elif inst == OP_TXS:
        state.sp = state.x

    elif inst == OP_TYA:
        state.a = state.y
        update_nz(state, state.a)

    # ====== Misc ======
    elif inst == OP_NOP:
        pass

    elif inst == OP_BRK:
        state.pc = state.pc + 1
        push_word(state, rom, rom_size, state.pc)
        push_byte(state, rom, rom_size, state.flags | FLAG_B | 0x20)
        set_flag(state, FLAG_I, True)
        var lo = UInt16(mem_read(state, rom, rom_size, UInt16(0xFFFE)))
        var hi = UInt16(mem_read(state, rom, rom_size, UInt16(0xFFFF)))
        state.pc = (hi << 8) | lo

    # ====== Illegal opcodes ======
    elif inst == OP_LAX:
        state.a = mem_read(state, rom, rom_size, addr)
        state.x = state.a
        update_nz(state, state.a)

    elif inst == OP_SAX:
        mem_write(state, rom, rom_size, addr, state.a & state.x)

    elif inst == OP_DCP:
        var val = mem_read(state, rom, rom_size, addr) - 1
        mem_write(state, rom, rom_size, addr, val)
        _compare(state, state.a, val)

    elif inst == OP_ISB:
        var val = mem_read(state, rom, rom_size, addr) + 1
        mem_write(state, rom, rom_size, addr, val)
        _sbc(state, val)

    elif inst == OP_SLO:
        var val = mem_read(state, rom, rom_size, addr)
        set_flag(state, FLAG_C, (val & 0x80) != 0)
        val = val << 1
        mem_write(state, rom, rom_size, addr, val)
        state.a = state.a | val
        update_nz(state, state.a)

    elif inst == OP_RLA:
        var carry_in = UInt8(1) if get_flag(state, FLAG_C) else UInt8(0)
        var val = mem_read(state, rom, rom_size, addr)
        set_flag(state, FLAG_C, (val & 0x80) != 0)
        val = (val << 1) | carry_in
        mem_write(state, rom, rom_size, addr, val)
        state.a = state.a & val
        update_nz(state, state.a)

    elif inst == OP_SRE:
        var val = mem_read(state, rom, rom_size, addr)
        set_flag(state, FLAG_C, (val & 0x01) != 0)
        val = val >> 1
        mem_write(state, rom, rom_size, addr, val)
        state.a = state.a ^ val
        update_nz(state, state.a)

    elif inst == OP_RRA:
        var carry_in = UInt8(0x80) if get_flag(state, FLAG_C) else UInt8(0)
        var val = mem_read(state, rom, rom_size, addr)
        set_flag(state, FLAG_C, (val & 0x01) != 0)
        val = (val >> 1) | carry_in
        mem_write(state, rom, rom_size, addr, val)
        _adc(state, val)

    elif inst == OP_ANC:
        state.a = state.a & mem_read(state, rom, rom_size, addr)
        update_nz(state, state.a)
        set_flag(state, FLAG_C, (state.a & 0x80) != 0)

    elif inst == OP_ALR:
        state.a = state.a & mem_read(state, rom, rom_size, addr)
        set_flag(state, FLAG_C, (state.a & 0x01) != 0)
        state.a = state.a >> 1
        update_nz(state, state.a)

    elif inst == OP_ARR:
        state.a = state.a & mem_read(state, rom, rom_size, addr)
        var carry_in = UInt8(0x80) if get_flag(state, FLAG_C) else UInt8(0)
        state.a = (state.a >> 1) | carry_in
        update_nz(state, state.a)
        set_flag(state, FLAG_C, (state.a & 0x40) != 0)
        set_flag(
            state, FLAG_V, ((state.a & 0x40) ^ ((state.a & 0x20) << 1)) != 0
        )

    elif inst == OP_AXS:
        var val = state.a & state.x
        var operand = mem_read(state, rom, rom_size, addr)
        var result = Int(val) - Int(operand)
        state.x = UInt8(result & 0xFF)
        set_flag(state, FLAG_C, result >= 0)
        update_nz(state, state.x)

    elif inst == OP_KIL:
        # Halt — in practice just do nothing
        pass

    return cycles


# ============================================================================
# ALU Helpers
# ============================================================================


@always_inline
def _adc(mut state: AtariState, operand: UInt8):
    """Add with carry."""
    var carry = UInt16(1) if get_flag(state, FLAG_C) else UInt16(0)
    var a16 = UInt16(state.a)
    var op16 = UInt16(operand)
    var result = a16 + op16 + carry

    set_flag(state, FLAG_C, result > 0xFF)
    # Overflow: positive + positive = negative, or negative + negative = positive
    set_flag(state, FLAG_V, ((~(a16 ^ op16)) & (a16 ^ result) & 0x80) != 0)
    state.a = UInt8(result & 0xFF)
    update_nz(state, state.a)


@always_inline
def _sbc(mut state: AtariState, operand: UInt8):
    """Subtract with borrow (SBC = ADC with complement)."""
    _adc(state, ~operand)


@always_inline
def _compare(mut state: AtariState, reg: UInt8, operand: UInt8):
    """Compare register with memory value."""
    var result = Int(reg) - Int(operand)
    set_flag(state, FLAG_C, reg >= operand)
    set_flag(state, FLAG_Z, reg == operand)
    set_flag(state, FLAG_N, (UInt8(result & 0xFF) & 0x80) != 0)


@always_inline
def _branch(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    addr: UInt16,
):
    """Execute a branch. addr points to the relative offset byte."""
    var offset = mem_read(state, rom, rom_size, addr)
    var signed_offset: Int
    if (offset & 0x80) != 0:
        signed_offset = Int(offset) - 256
    else:
        signed_offset = Int(offset)
    state.pc = UInt16(Int(state.pc) + signed_offset)


# ============================================================================
# Reset / Init
# ============================================================================


def cpu_reset(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
):
    """Perform CPU reset — load PC from reset vector ($FFFC-$FFFD)."""
    state.sp = 0xFD
    state.flags = 0x20 | FLAG_I  # IRQ disabled after reset
    var lo = UInt16(mem_read(state, rom, rom_size, UInt16(0xFFFC)))
    var hi = UInt16(mem_read(state, rom, rom_size, UInt16(0xFFFD)))
    state.pc = (hi << 8) | lo


# ============================================================================
# Run One Frame
# ============================================================================


def _run_scanline(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    overflow: Int = 0,
) -> Int:
    """Execute one scanline's worth of CPU cycles with clock tracking.

    Updates state.clock before each instruction so that RESP0/RESP1
    writes capture the correct beam position. This is critical because
    the Atari 2600 positions sprites by timing writes to RESPx.

    Handles WSYNC: when the game writes to WSYNC, the CPU halts until
    the end of the current scanline. This is the primary mechanism
    games use to synchronize sprite positioning timing.

    1 CPU cycle = 3 TIA color clocks. 76 CPU cycles = 228 clocks/line.
    HBLANK occupies clocks 0-67, visible area is clocks 68-227.

    Args:
        state: AtariState.
        rom: ROM data pointer.
        rom_size: ROM size in bytes.
        overflow: CPU cycles carried over from the previous scanline's
            last instruction (0-6). The beam is already this many cycles
            into the current scanline. Matches ALE's myClocksToEndOfScanLine
            approach for accurate frame timing.

    Returns: total CPU cycles consumed (including overflow). The caller
        should compute the new overflow as (return_value - CPU_CLOCKS_PER_LINE).
    """
    var line_cycles: Int = overflow
    var saved_mid = False

    # If WSYNC carried over from previous scanline (instruction that set WSYNC
    # overflowed past the scanline boundary), consume remaining cycles now.
    if state.wsync:
        state.wsync = False
        if line_cycles < CPU_CLOCKS_PER_LINE:
            riot_update_timer(state, UInt32(CPU_CLOCKS_PER_LINE - line_cycles))
            line_cycles = CPU_CLOCKS_PER_LINE

    while line_cycles < CPU_CLOCKS_PER_LINE:
        # Update TIA clock position BEFORE instruction executes,
        # so any RESP writes during this instruction see the correct beam pos
        state.clock = UInt16(line_cycles * 3)

        # Snapshot PF registers between left-digit and right-digit PF writes.
        # In Pong's score kernel: left-digit PF is written at cycles 0-15,
        # right-digit PF is written at cycles ~47-56. Capturing at cycle 36
        # ensures we have the left-digit values. For non-score scanlines,
        # PF is set early and stable, so snapshot equals final values.
        if not saved_mid and line_cycles >= 36:
            state.pf0_mid = state.pf0
            state.pf1_mid = state.pf1
            state.pf2_mid = state.pf2
            saved_mid = True

        var cycles = Int(execute_one(state, rom, rom_size))
        riot_update_timer(state, UInt32(cycles))
        line_cycles += cycles

        # WSYNC: game requested halt until end of scanline
        # Consume remaining cycles — next instruction runs at start of next line
        if state.wsync:
            state.wsync = False
            if line_cycles < CPU_CLOCKS_PER_LINE:
                riot_update_timer(
                    state, UInt32(CPU_CLOCKS_PER_LINE - line_cycles)
                )
                line_cycles = CPU_CLOCKS_PER_LINE
            # If line_cycles >= CPU_CLOCKS_PER_LINE, WSYNC extends into next
            # scanline — keep wsync=True so next _run_scanline handles it.
            elif line_cycles > CPU_CLOCKS_PER_LINE:
                state.wsync = True

    # If midpoint was never reached (e.g. WSYNC before cycle 49), use final PF
    if not saved_mid:
        state.pf0_mid = state.pf0
        state.pf1_mid = state.pf1
        state.pf2_mid = state.pf2

    return line_cycles


def run_frame(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
):
    """Execute one full frame (~262 scanlines × 76 CPU cycles).

    This is the main emulation entry point. After this call:
    - state.ram has been updated by the game logic
    - state.collision has been updated by TIA scanline processing
    - Game-specific reward/lives/terminal should be extracted from RAM
    """
    var overflow = Int(state.cpu_cycles)  # Carry from previous frame
    for _ in range(TOTAL_SCANLINES):
        # Only charge paddle capacitor when not grounded (VBLANK bit 7 clear)
        if (
            state.paddle_charge < 255
            and (state.tia_flags & TIA_PADDLE_GROUND) == 0
        ):
            state.paddle_charge += 1
        var total = _run_scanline(state, rom, rom_size, overflow)
        overflow = total - CPU_CLOCKS_PER_LINE

    state.cpu_cycles = UInt32(overflow)  # Persist for next frame
    state.frame_number += 1


def run_frame_with_video(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    frame_buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Execute one frame with scanline-by-scanline pixel rendering.

    The Atari 2600 is a "racing the beam" system: the CPU updates TIA
    registers mid-frame. Each scanline may have different graphics.

    For mid-scanline PF writes (e.g. score digits in Pong), _run_scanline
    captures a PF snapshot at the beam midpoint (~cycle 49). The renderer
    uses this snapshot for left-half pixels and the final PF for right-half.

    Cycle overflow from each scanline's last instruction is carried forward
    to the next scanline (matching ALE's myClocksToEndOfScanLine approach).
    This prevents VSYNC drift that causes vertical display shaking.

    Args:
        state: Emulator state (modified in place).
        rom: ROM data pointer.
        rom_size: ROM size in bytes.
        frame_buf: Output BGRA buffer (160×210×4 = 134400 bytes).
    """
    from .frame_render import render_scanline_with_collision_bgra
    from .flags import TIA_VBLANK, TIA_VSYNC, FRAME_HEIGHT as FH

    # Use state.scanline to track visible line position across frames.
    # This handles the case where VSYNC falls near the 262-scanline loop
    # boundary — visible_line persists correctly across calls.
    var visible_line = Int(state.scanline)
    var overflow = Int(state.cpu_cycles)  # Carry from previous frame

    for _ in range(TOTAL_SCANLINES):
        # Only charge paddle capacitor when not grounded (VBLANK bit 7 clear)
        if (
            state.paddle_charge < 255
            and (state.tia_flags & TIA_PADDLE_GROUND) == 0
        ):
            state.paddle_charge += 1

        var total = _run_scanline(state, rom, rom_size, overflow)
        overflow = total - CPU_CLOCKS_PER_LINE

        # Detect VSYNC — marks the start of a new frame
        if (state.tia_flags & TIA_VSYNC) != 0:
            visible_line = 0  # Reset visible line counter

        # Combined render + collision in one pass (halves mask computations)
        # Render when VBLANK is clear (no need for saw_vsync gate since
        # visible_line is reset by VSYNC and persists across calls)
        if (state.tia_flags & TIA_VBLANK) == 0 and visible_line < FH:
            render_scanline_with_collision_bgra(state, visible_line, frame_buf)
            visible_line += 1

    state.scanline = UInt16(visible_line)  # Persist for next frame
    state.cpu_cycles = UInt32(overflow)  # Persist overflow for next frame
    state.frame_number += 1


# Import here to avoid circular dependency
from .flags import (
    TOTAL_SCANLINES,
    CPU_CLOCKS_PER_LINE,
    FRAME_WIDTH,
    TIA_PADDLE_GROUND,
)
