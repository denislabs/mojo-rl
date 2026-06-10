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
    TIA_WRITE_LOG_CAP,
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
        var reg = UInt8(a & 0x3F)

        # Record the write at its exact color clock for the cycle-accurate
        # per-clock tick loop. pending_tia_write_clock was set by execute_one.
        if state.tia_log_count < TIA_WRITE_LOG_CAP:
            var i = state.tia_log_count
            state.tia_log_clock[i] = state.pending_tia_write_clock
            state.tia_log_reg[i] = reg
            state.tia_log_value[i] = value
            state.tia_log_count = i + 1

        tia_write(state, reg, value)


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

    # A 6502 store writes on its LAST cycle, so a TIA write lands (cycles-1)
    # CPU cycles = 3*(cycles-1) color clocks after the instruction START. We
    # store this as an OFFSET (not absolute) so the per-clock tick loop can
    # match it against a per-instruction local clock index without any
    # frame-absolute counter (which would overflow UInt16 at the 2x cap).
    # entry.cycles already includes the fixed extra cycle store modes pay.
    state.tia_log_count = 0
    state.pending_tia_write_clock = (Int(cycles) - 1) * 3

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
    """Add with carry. Supports NMOS 6502 decimal (BCD) mode.

    Games like Space Invaders compute their BCD score with the D flag set,
    so ignoring decimal mode yields garbage scores (and any value derived
    from BCD arithmetic). Decimal-mode result + C are correct BCD; the
    N/V/Z flags follow the quirky NMOS behavior (Stella's algorithm).
    """
    var carry = 1 if get_flag(state, FLAG_C) else 0

    if get_flag(state, FLAG_D):
        var a = Int(state.a)
        var value = Int(operand)
        var lo = (a & 0x0F) + (value & 0x0F) + carry
        var hi = (a & 0xF0) + (value & 0xF0)
        if lo > 0x09:
            hi += 0x10
            lo += 0x06
        # NMOS quirks: Z from the binary sum; N/V from the high nibble.
        set_flag(state, FLAG_Z, ((a + value + carry) & 0xFF) == 0)
        set_flag(state, FLAG_N, (hi & 0x80) != 0)
        set_flag(
            state,
            FLAG_V,
            ((a ^ value) & 0x80) == 0 and ((a ^ hi) & 0x80) != 0,
        )
        if hi > 0x90:
            hi += 0x60
        set_flag(state, FLAG_C, (hi & 0xFF00) != 0)
        state.a = UInt8((lo & 0x0F) | (hi & 0xF0))
        return

    var a16 = UInt16(state.a)
    var op16 = UInt16(operand)
    var result = a16 + op16 + UInt16(carry)

    set_flag(state, FLAG_C, result > 0xFF)
    # Overflow: positive + positive = negative, or negative + negative = positive
    set_flag(state, FLAG_V, ((~(a16 ^ op16)) & (a16 ^ result) & 0x80) != 0)
    state.a = UInt8(result & 0xFF)
    update_nz(state, state.a)


@always_inline
def _sbc(mut state: AtariState, operand: UInt8):
    """Subtract with borrow. Supports NMOS 6502 decimal (BCD) mode.

    In binary mode SBC == ADC with the complemented operand. In decimal
    mode the flags are identical to binary SBC (NMOS), but the accumulator
    result is BCD-adjusted.
    """
    if get_flag(state, FLAG_D):
        var carry = 1 if get_flag(state, FLAG_C) else 0
        var borrow = 1 - carry
        var a = Int(state.a)
        var value = Int(operand)

        # Flags from the plain binary subtraction (NMOS: same as binary mode).
        var bin = a - value - borrow
        set_flag(state, FLAG_C, bin >= 0)
        set_flag(
            state,
            FLAG_V,
            ((a ^ value) & 0x80) != 0 and ((a ^ bin) & 0x80) != 0,
        )
        update_nz(state, UInt8(bin & 0xFF))

        # BCD-adjusted accumulator result.
        var lo = (a & 0x0F) - (value & 0x0F) - borrow
        var hi = (a & 0xF0) - (value & 0xF0)
        if (lo & 0x10) != 0:
            lo -= 0x06
            hi -= 0x10
        if (hi & 0x100) != 0:
            hi -= 0x60
        state.a = UInt8((lo & 0x0F) | (hi & 0xF0))
        return

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

    # If midpoint was never reached (e.g. WSYNC before cycle 36), use final PF
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
    """Execute one full frame headlessly (no video output).

    This is the main RL emulation entry point. It runs the SAME cycle-accurate
    CPU/TIA lockstep as run_frame_video — per-color-clock object ticks and TIA
    collision latches — just without writing pixels. Collisions are gameplay
    for most carts (SI laser kills, Breakout brick breaks, Pong paddle bounces
    all read CX* registers), so the headless path must compute them; the old
    scanline-batched run_frame never did, which silently broke those games in
    RAM-observation training. After this call:
    - state.ram has been updated by the game logic
    - state.collision has been latched per color clock
    - Game-specific reward/lives/terminal should be extracted from RAM
    """
    # Dummy buffer: never written (RENDER=False skips all pixel writes).
    var dummy = InlineArray[UInt8, 4](fill=0)
    run_frame_cycle_accurate[RENDER=False](
        state, rom, rom_size, dummy.unsafe_ptr()
    )


# Import here to avoid circular dependency
from .flags import (
    TOTAL_SCANLINES,
    CPU_CLOCKS_PER_LINE,
    FRAME_WIDTH,
    TIA_PADDLE_GROUND,
)


# ============================================================================
# Cycle-accurate TIA frame runner — the single rendering path.
# Drives the per-color-clock object counters in state.ctia from the exact write
# clocks logged during execute_one, and latches per-clock collisions — Stella's
# model. Playfield/colors are read live from AtariState (applied immediately by
# tia_write); only sprite positions/enables/motion flow through the counters.
# ============================================================================

from .tia_cycle import (
    resx_counter,
    playfield_bit,
    LIT_P0,
    LIT_P1,
    LIT_M0,
    LIT_M1,
    LIT_BL,
    DELAY_PF,
    DELAY_GRP,
    DELAY_ENAM,
    DELAY_ENABL,
    DELAY_HMP,
    DELAY_HMM,
    DELAY_HMBL,
    DELAY_REFP,
    DELAY_HMCLR,
)
from .flags import (
    HBLANK_CLOCKS,
    CLOCKS_PER_LINE as CPL,
    FRAME_HEIGHT as FH2,
    TIA_VBLANK as VBL,
    TIA_VSYNC as VSY,
    TIA_VDELP0,
    TIA_VDELP1,
    TIA_VDELBL,
    TIA_PF_PRIORITY,
    TIA_PF_SCORE,
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
)
from .frame_render import _write_pixel_bgra


@always_inline
def _cycle_reg_delay(reg: UInt8) -> Int:
    """Color-clock latency for a TIA register write (TIA.cxx Delay enum).

    Strobes (RESPx/RESMx/RESBL), NUSIZ, CTRLPF apply immediately (delay 0)."""
    if reg == 0x0D or reg == 0x0E or reg == 0x0F:  # PF0/PF1/PF2
        return DELAY_PF
    if reg == 0x1B or reg == 0x1C:  # GRP0/GRP1
        return DELAY_GRP
    if reg == 0x1D or reg == 0x1E:  # ENAM0/ENAM1
        return DELAY_ENAM
    if reg == 0x1F:  # ENABL
        return DELAY_ENABL
    if reg == 0x20 or reg == 0x21:  # HMP0/HMP1
        return DELAY_HMP
    if reg == 0x22 or reg == 0x23:  # HMM0/HMM1
        return DELAY_HMM
    if reg == 0x24:  # HMBL
        return DELAY_HMBL
    if reg == 0x0B or reg == 0x0C:  # REFP0/REFP1
        return DELAY_REFP
    if reg == 0x2A:  # HMOVE
        return 6
    if reg == 0x2B:  # HMCLR
        return DELAY_HMCLR
    return 0


@always_inline
def _cycle_apply_reg(
    mut state: AtariState, reg: UInt8, value: UInt8, hctr: Int, in_hblank: Bool
):
    """Route a (delayed) TIA register write to the cycle-accurate counters.

    Position strobes use resx_counter() at the apply clock; graphics/enable use
    the VDEL-resolved values that tia_write already latched into AtariState."""
    if reg == 0x04:  # NUSIZ0
        state.ctia.p0.set_nusiz(value)
        state.ctia.m0.set_nusiz(value)
    elif reg == 0x05:  # NUSIZ1
        state.ctia.p1.set_nusiz(value)
        state.ctia.m1.set_nusiz(value)
    elif reg == 0x06:  # COLUP0
        state.ctia.colup0 = value
    elif reg == 0x07:  # COLUP1
        state.ctia.colup1 = value
    elif reg == 0x08:  # COLUPF
        state.ctia.colupf = value
    elif reg == 0x09:  # COLUBK
        state.ctia.colubk = value
    elif reg == 0x0A:  # CTRLPF (reflect/score/priority + ball width)
        state.ctia.ctrlpf = value
        state.ctia.bl.set_width_from_ctrlpf(value)
    elif reg == 0x0D:  # PF0
        state.ctia.pf0 = value
    elif reg == 0x0E:  # PF1
        state.ctia.pf1 = value
    elif reg == 0x0F:  # PF2
        state.ctia.pf2 = value
    elif reg == 0x0B:  # REFP0
        state.ctia.p0.set_reflect((value & 0x08) != 0)
    elif reg == 0x0C:  # REFP1
        state.ctia.p1.set_reflect((value & 0x08) != 0)
    elif reg == 0x10:  # RESP0
        state.ctia.p0.resp(resx_counter(hctr, in_hblank))
    elif reg == 0x11:  # RESP1
        state.ctia.p1.resp(resx_counter(hctr, in_hblank))
    elif reg == 0x12:  # RESM0
        state.ctia.m0.resm(resx_counter(hctr, in_hblank))
    elif reg == 0x13:  # RESM1
        state.ctia.m1.resm(resx_counter(hctr, in_hblank))
    elif reg == 0x14:  # RESBL
        state.ctia.bl.resbl(resx_counter(hctr, in_hblank))
    elif reg == 0x1B:  # GRP0 (sets P0 new pattern; shuffles P1 old=new)
        state.ctia.p0.set_grp_new(value)
        state.ctia.p1.shuffle()
    elif reg == 0x1C:  # GRP1 (sets P1 new; shuffles P0 old=new + ball, Stella)
        state.ctia.p1.set_grp_new(value)
        state.ctia.p0.shuffle()
        state.ctia.bl.shuffle()
    elif reg == 0x1D:  # ENAM0
        state.ctia.m0.set_enam(value)
    elif reg == 0x1E:  # ENAM1
        state.ctia.m1.set_enam(value)
    elif reg == 0x1F:  # ENABL
        state.ctia.bl.set_enabl_new((value & 0x02) != 0)
    elif reg == 0x20:  # HMP0
        state.ctia.p0.set_hmp(value)
    elif reg == 0x21:  # HMP1
        state.ctia.p1.set_hmp(value)
    elif reg == 0x22:  # HMM0
        state.ctia.m0.set_hmm(value)
    elif reg == 0x23:  # HMM1
        state.ctia.m1.set_hmm(value)
    elif reg == 0x24:  # HMBL
        state.ctia.bl.set_hmbl(value)
    elif reg == 0x25:  # VDELP0
        state.ctia.p0.set_vdel((value & 0x01) != 0)
    elif reg == 0x26:  # VDELP1
        state.ctia.p1.set_vdel((value & 0x01) != 0)
    elif reg == 0x27:  # VDELBL
        state.ctia.bl.set_vdel((value & 0x01) != 0)
    elif reg == 0x28:  # RESMP0
        state.ctia.m0.set_resmp(value)
    elif reg == 0x29:  # RESMP1
        state.ctia.m1.set_resmp(value)
    elif reg == 0x2A:  # HMOVE
        state.ctia.start_hmove()
    elif reg == 0x2B:  # HMCLR
        state.ctia.p0.set_hmp(0x00)
        state.ctia.p1.set_hmp(0x00)
        state.ctia.m0.set_hmm(0x00)
        state.ctia.m1.set_hmm(0x00)
        state.ctia.bl.set_hmbl(0x00)


@always_inline
def _cycle_pixel(
    mut state: AtariState,
    render_row: Int,
    pixel: Int,
    lit: UInt8,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    palette: InlineArray[UInt32, 256],
):
    """Latch per-clock collisions and render one pixel from the counter lit-mask
    plus the live playfield."""
    if (state.tia_flags & VBL) != 0:
        if render_row >= 0:
            _write_pixel_bgra(
                buf,
                (render_row * FRAME_WIDTH + pixel) * 4,
                state.ctia.colubk,
                palette,
            )
        return

    var p0 = (lit & UInt8(1 << LIT_P0)) != 0
    var p1 = (lit & UInt8(1 << LIT_P1)) != 0
    var m0 = (lit & UInt8(1 << LIT_M0)) != 0
    var m1 = (lit & UInt8(1 << LIT_M1)) != 0
    var bl = (lit & UInt8(1 << LIT_BL)) != 0
    # Playfield: render from the beam-accurate SHADOW (state.ctia.pf*), which is
    # driven through the DelayQueue at each write's exact color clock
    # (instr_start + (cycles-1)*3 + DELAY_PF ≈ the eol path's +8 write delay).
    # Reading the immediate live state.pf* instead applies PF writes too early
    # (at instruction start), shifting Breakout's walls. One source (the shadow)
    # for both render+collide keeps them consistent (no phantom brick).
    var reflect = (state.ctia.ctrlpf & 0x01) != 0
    var pf = playfield_bit(
        state.ctia.pf0, state.ctia.pf1, state.ctia.pf2, reflect, pixel
    )
    if m0 and p1:
        state.collision = state.collision | CX_M0P1
    if m0 and p0:
        state.collision = state.collision | CX_M0P0
    if m1 and p0:
        state.collision = state.collision | CX_M1P0
    if m1 and p1:
        state.collision = state.collision | CX_M1P1
    if p0 and pf:
        state.collision = state.collision | CX_P0PF
    if p0 and bl:
        state.collision = state.collision | CX_P0BL
    if p1 and pf:
        state.collision = state.collision | CX_P1PF
    if p1 and bl:
        state.collision = state.collision | CX_P1BL
    if m0 and pf:
        state.collision = state.collision | CX_M0PF
    if m0 and bl:
        state.collision = state.collision | CX_M0BL
    if m1 and pf:
        state.collision = state.collision | CX_M1PF
    if m1 and bl:
        state.collision = state.collision | CX_M1BL
    if bl and pf:
        state.collision = state.collision | CX_BLPF
    if p0 and p1:
        state.collision = state.collision | CX_P0P1
    if m0 and m1:
        state.collision = state.collision | CX_M0M1

    if render_row < 0:
        return

    # --- Render with TIA priority (beam-accurate shadow regs) ---
    var color_idx: UInt8
    var pf_pri = (state.ctia.ctrlpf & 0x04) != 0  # CTRLPF bit2 = priority
    var pf_score = (state.ctia.ctrlpf & 0x02) != 0  # CTRLPF bit1 = score
    if pf_pri:
        if pf or bl:
            if pf_score and pf:
                color_idx = state.ctia.colup0 if pixel < 80 else state.ctia.colup1
            else:
                color_idx = state.ctia.colupf
        elif p0 or m0:
            color_idx = state.ctia.colup0
        elif p1 or m1:
            color_idx = state.ctia.colup1
        else:
            color_idx = state.ctia.colubk
    else:
        if p0 or m0:
            color_idx = state.ctia.colup0
        elif p1 or m1:
            color_idx = state.ctia.colup1
        elif pf or bl:
            if pf_score and pf:
                color_idx = state.ctia.colup0 if pixel < 80 else state.ctia.colup1
            else:
                color_idx = state.ctia.colupf
        else:
            color_idx = state.ctia.colubk

    _write_pixel_bgra(
        buf, (render_row * FRAME_WIDTH + pixel) * 4, color_idx, palette
    )




def run_frame_cycle_accurate[
    RENDER: Bool = True
](
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    frame_buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Cycle-accurate frame: CPU and TIA in lockstep, per-color-clock collision.

    One VSYNC-aligned frame. The TIA advances exactly 3 color clocks per CPU
    cycle; each instruction's logged TIA writes are replayed at their exact clock
    through the DelayQueue, driving the object counters in state.ctia. Per clock
    we tick the objects, latch collisions, and render the pixel.

    RENDER=False skips all pixel writes (frame_buf is never dereferenced — pass
    a null pointer) while keeping the identical CPU/TIA/collision behavior:
    the headless mode for RL training, where games still need the TIA collision
    latches (SI laser kills, Breakout brick breaks) but no video.
    """
    from .palette import NTSC_PALETTE

    var palette = materialize[NTSC_PALETTE]()
    # Seed the beam-accurate PF/color shadow from the live regs (kept current by
    # tia_write) so registers written before this frame / not rewritten this
    # frame still render correctly; mid-line writes then update it at exact clock.
    state.ctia.pf0 = state.pf0
    state.ctia.pf1 = state.pf1
    state.ctia.pf2 = state.pf2
    state.ctia.ctrlpf = state.ctrlpf
    state.ctia.colup0 = state.colup0
    state.ctia.colup1 = state.colup1
    state.ctia.colupf = state.colupf
    state.ctia.colubk = state.colubk
    # Seed missile position/enable/size from the eol state. Breakout draws its
    # side WALLS (and SI its laser BEAM) with missiles positioned ONCE during
    # setup — which runs through the eol path (env.reset), invisible to the cycle
    # counters. Without this seed the missile counters free-run at the wrong
    # phase and the walls/beam render shifted into the playfield / off-screen.
    # counter = (160 - pos) % 160 places the decode so the missile renders at
    # `pos` (decode@156 + render offset, mod the 160-clock counter cycle).
    state.ctia.m0.set_nusiz(state.nusiz0)
    state.ctia.m0.set_enam(state.enam0)
    state.ctia.m0.counter = (FRAME_WIDTH - Int(state.pos_m0)) % FRAME_WIDTH
    state.ctia.m1.set_nusiz(state.nusiz1)
    state.ctia.m1.set_enam(state.enam1)
    state.ctia.m1.counter = (FRAME_WIDTH - Int(state.pos_m1)) % FRAME_WIDTH
    # Seed player VDEL + GRP double-buffer from eol state (VDELPx / GRP may be
    # set during the eol reset frames, invisible to the cycle path otherwise).
    state.ctia.p0.set_vdel((state.tia_flags & TIA_VDELP0) != 0)
    state.ctia.p0.grp_new = state.grp0
    state.ctia.p0.grp_old = state.grp0_old
    state.ctia.p1.set_vdel((state.tia_flags & TIA_VDELP1) != 0)
    state.ctia.p1.grp_new = state.grp1
    state.ctia.p1.grp_old = state.grp1_old
    # Seed ball enable double-buffer (VDELBL) + width + position from eol state.
    state.ctia.bl.set_width_from_ctrlpf(state.ctrlpf)
    state.ctia.bl.enabl_new = (state.enabl & 0x02) != 0
    state.ctia.bl.enabl_old = (state.enabl_old & 0x02) != 0
    state.ctia.bl.set_vdel((state.tia_flags & TIA_VDELBL) != 0)
    state.ctia.bl.counter = (FRAME_WIDTH - Int(state.pos_bl)) % FRAME_WIDTH
    var visible_line = 0
    var rendered_any = False
    var prev_vsync = (state.tia_flags & VSY) != 0
    # Frame-geometry diagnostics: total lines this frame + the line at which
    # VBLANK was first released (counted from frame start ≈ VSYNC).
    var total_lines = 0
    var ystart = -1
    var done = False
    comptime MAX_CLOCKS: Int = TOTAL_SCANLINES * CPL * 2
    var clocks = 0
    var due_reg = List[UInt8]()
    var due_val = List[UInt8]()

    while not done and clocks < MAX_CLOCKS:
        # --- one CPU instruction ---
        var start_hctr = state.ctia.hctr
        state.clock = UInt16(start_hctr)
        var cyc: Int
        var wsync_pad = False
        if state.wsync:
            state.wsync = False
            # CPU idle to end of line: pad TIA clocks to the next line boundary.
            cyc = 0
            wsync_pad = True
        else:
            cyc = Int(execute_one(state, rom, rom_size))
            riot_update_timer(state, UInt32(cyc))

        var nclk = (CPL - (start_hctr % CPL)) if wsync_pad else cyc * 3
        if wsync_pad:
            # The CPU is halted by WSYNC but TIME still passes: advance the RIOT
            # interval timer by the padded CPU cycles (nclk/3), exactly like
            # run_frame. Omitting this lets the RIOT timer lag every
            # line → the game's frame/scanline count and input timing drift
            # (SI vertical shaking; Breakout paddle-position-dependent glitches).
            riot_update_timer(state, UInt32(nclk // 3))

        for j in range(nclk):
            var hctr = state.ctia.hctr
            var pixel = hctr - HBLANK_CLOCKS
            # Extended HBLANK after HMOVE: the 8 clocks past the normal 68 stay
            # "blank" so objects skip 8 frame ticks, canceling the comb's 8
            # baseline ticks (net HMOVE motion = hmm_clocks - 8). resx strobes in
            # this window also use the hblank counter (Stella myHstate==blank).
            var hbe = 76 if state.ctia.extended_hblank else HBLANK_CLOCKS
            var in_hblank = hctr < hbe

            # Replay this instruction's TIA writes at their exact offset clock.
            if not wsync_pad:
                for li in range(state.tia_log_count):
                    if state.tia_log_clock[li] == j:
                        var r = state.tia_log_reg[li]
                        state.ctia.dq.push(
                            r, state.tia_log_value[li], _cycle_reg_delay(r)
                        )

            # Apply writes whose delay elapsed this clock. The queue is empty
            # on the vast majority of clocks — skip the List churn entirely.
            if state.ctia.dq.count != 0:
                due_reg.clear()
                due_val.clear()
                state.ctia.dq.cycle_collect(due_reg, due_val)
                for k in range(len(due_reg)):
                    _cycle_apply_reg(
                        state, due_reg[k], due_val[k], hctr, in_hblank
                    )

            # Tick objects + render/collide per color clock from the SAME state
            # (Stella-style): render and collision both use the cycle counters +
            # shadow PF, so they are always consistent (what breaks is what the
            # ball visibly touches; the laser kills what it visibly overlaps).
            var lit = state.ctia.tick(in_hblank)
            var render_row = -1
            comptime if RENDER:
                render_row = visible_line if visible_line < FH2 else -1
            if pixel >= 0 and pixel < FRAME_WIDTH:
                comptime if RENDER:
                    _cycle_pixel(
                        state, render_row, pixel, lit, frame_buf, palette
                    )
                else:
                    # Headless: every collision pair involves at least one
                    # object (PF alone collides with nothing), so a clock
                    # with no object lit can latch nothing — skip it.
                    if lit != 0:
                        _cycle_pixel(
                            state, render_row, pixel, lit, frame_buf, palette
                        )

            # Advance the line clock.
            var nh = hctr + 1
            clocks += 1
            if nh >= CPL:
                nh = 0
                # Extended HBLANK is a per-line flag (Stella clears at hctr 0).
                state.ctia.extended_hblank = False
                # WSYNC satisfied by this very crossing: if the instruction that
                # just ran asserted WSYNC AND its own clocks already carried the
                # beam across the line boundary, the halt-to-next-line is already
                # complete. Without this, the next iteration's wsync_pad would add
                # a SECOND full scanline (eol lands exactly on the 76-cycle line
                # end and pads zero) — the source of Breakout's ~5 extra brick
                # scanlines (vertical stretch) and the Space Invaders shake.
                if state.wsync and not wsync_pad:
                    state.wsync = False
                # End of scanline bookkeeping (mirrors run_frame).
                if (
                    state.paddle_charge < 255
                    and (state.tia_flags & TIA_PADDLE_GROUND) == 0
                ):
                    state.paddle_charge += 1
                var vsync_now = (state.tia_flags & VSY) != 0
                var vsync_rising = vsync_now and not prev_vsync
                prev_vsync = vsync_now
                if vsync_rising:
                    if rendered_any:
                        done = True
                    else:
                        visible_line = 0
                        total_lines = 0
                        ystart = -1
                total_lines += 1
                if (state.tia_flags & VBL) == 0 and visible_line < FH2:
                    if ystart < 0:
                        ystart = total_lines - 1
                    visible_line += 1
                    rendered_any = True
            state.ctia.hctr = nh
            if done:
                break

    state.scanline = UInt16(visible_line)
    state.dbg_frame_lines = UInt16(total_lines)
    state.dbg_ystart = UInt16(ystart) if ystart >= 0 else 0
    state.frame_number += 1


def run_frame_video(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    frame_buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render one VSYNC-aligned frame into frame_buf (BGRA).

    Thin alias for the cycle-accurate, Stella-style per-color-clock TIA path —
    the single rendering path (the legacy end-of-line renderer was removed)."""
    run_frame_cycle_accurate(state, rom, rom_size, frame_buf)
