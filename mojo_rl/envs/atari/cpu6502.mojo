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
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    addr: UInt16,
) -> UInt8:
    """Read a byte from the Atari 2600 memory map.

    state is mut because cartridge hotspot READS switch banks (F8/F6/E0),
    exactly like real hardware and Stella/ALE, and because every access
    drives the data bus (Stella System: myDataBusState = result after each
    peek) — TIA reads leak the previous bus byte in their low 6 bits.
    """
    var a = Int(addr) & 0x1FFF  # 13-bit address space

    var v: UInt8
    if a & 0x1000:  # Cartridge ROM
        # Pass the FULL 16-bit address: the FE mapper banks on A13.
        v = rom_read(state, rom, rom_size, addr)
    elif a & 0x0080:  # RIOT area
        if a & 0x0200:  # RIOT registers (0x0280-0x0297)
            v = riot_read(state, UInt8(a & 0xFF))
        else:  # RAM (0x0080-0x00FF)
            v = read_ram(state.ram, Int(a & 0x7F))
    else:  # TIA
        v = tia_read(state, UInt8(a & 0x0F))
    state.data_bus = v
    return v


@always_inline
def mem_write(
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    addr: UInt16,
    value: UInt8,
):
    """Write a byte to the Atari 2600 memory map."""
    var a = Int(addr) & 0x1FFF
    state.data_bus = value  # writes drive the bus (Stella System::poke)

    if a & 0x1000:  # Cartridge ROM (may trigger bank switch)
        rom_write(state, rom, rom_size, addr, value)
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
    rom: Pointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    value: UInt8,
):
    """Push a byte onto the stack."""
    mem_write(state, rom, rom_size, UInt16(0x0100) + UInt16(state.sp), value)
    state.sp -= 1


@always_inline
def pull_byte(
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
) -> UInt8:
    """Pull a byte from the stack."""
    state.sp += 1
    return mem_read(state, rom, rom_size, UInt16(0x0100) + UInt16(state.sp))


@always_inline
def push_word(
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    value: UInt16,
):
    """Push a 16-bit word onto the stack (high byte first)."""
    push_byte(state, rom, rom_size, UInt8((value >> 8) & 0xFF))
    push_byte(state, rom, rom_size, UInt8(value & 0xFF))


@always_inline
def pull_word(
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
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
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
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
def _fetch_opcode_entry[
    ot_o: MutOrigin
](
    opcode_byte: UInt8, op_table: Pointer[OpcodeEntry, ot_o]
) -> OpcodeEntry:
    """Opcode-table lookup via a caller-provided table pointer.

    The 256-entry table is passed in rather than read from the module-level
    comptime `OPCODE_TABLE`: a host global is NOT present in a GPU device module
    (Metal fails pipeline creation with `Undefined symbols: global_constant`;
    CUDA would need it in device/constant memory), and `materialize` inside the
    kernel still references that host global. So CPU entry points materialize
    the table once per frame and pass its pointer; the GPU driver uploads the
    table to a device buffer and passes the device pointer. Same table contents
    either way → the CPU trajectory checksum is unchanged.
    """
    return op_table[unsafe_offset=Int(opcode_byte)]


@no_inline
def execute_one[
    ot_o: MutOrigin
](
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    op_table: Pointer[OpcodeEntry, ot_o],
) -> UInt8:
    """Execute one instruction. Returns the number of CPU cycles consumed.

    `@no_inline` (compile-size hygiene): fully inlined (this function plus
    its `@always_inline` mem_read/mem_write/resolve_operand_addr internals)
    the opcode dispatch was the dominant share of `run_frame_cycle_accurate`'s
    ~220K lines of LLVM IR — and that frame runner is instantiated twice
    (RENDER=True/False) in pixel-training binaries. Outlined, the dispatch is
    ONE shared copy and the frame runners shrink ~an order of magnitude.
    (Historical note: the Rainbow Atari pixel example's -O3 compile blowup
    that motivated this was ultimately root-caused to `NStepTransition`'s
    by-value InlineArray obs — see deep_agents/data/n_step_replay.mojo —
    not the emulator; this boundary is kept as cheap IR hygiene.)
    Runtime cost is one real call per emulated instruction, noise against
    the 9–21 TIA color-clock ticks each instruction drives (measured: no
    fps regression on the Pong benchmark, trajectory checksum identical).
    """
    var opcode_byte = mem_read(state, rom, rom_size, state.pc)
    var entry = _fetch_opcode_entry(opcode_byte, op_table)
    var inst = entry.instruction
    var mode = entry.addr_mode
    var cycles = entry.cycles
    var size = entry.size

    # A 6502 store writes on its LAST cycle. We log the offset of that cycle's
    # START ((cycles-1)*3 color clocks after the instruction start) so the
    # per-clock tick loop can match it against a per-instruction local clock
    # index; the loop then adds STORE_CYCLE_CLOCKS when pushing into the
    # DelayQueue, because the bus write takes effect at the END of the write
    # cycle (Stella M6502::poke does incrementCycles(1) BEFORE System::poke).
    # Berzerk strobes RESP0 right at the hblank edge — 3 clocks early shifts
    # the player 3px left into the electrified wall (constant P0PF = instant
    # death, random play scored 0 vs ALE's ~160/episode).
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
    rom: Pointer[UInt8, ImmutAnyOrigin],
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
    rom: Pointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
):
    """Perform CPU reset — load PC from reset vector ($FFFC-$FFFD)."""
    state.sp = 0xFD
    state.flags = 0x20 | FLAG_I  # IRQ disabled after reset
    var lo = UInt16(mem_read(state, rom, rom_size, UInt16(0xFFFC)))
    var hi = UInt16(mem_read(state, rom, rom_size, UInt16(0xFFFD)))
    state.pc = (hi << 8) | lo

    # Scan unbanked carts ONCE for INTIM wait-loop sites (see
    # _intim_wait_skip_cycles): `AD lo hi D0 FB` where the absolute address
    # decodes to the RIOT INTIM register (mirrors included, same decode as
    # mem_read/riot_read). The static scan is what makes the runner's
    # per-instruction fast-forward probe two PC compares instead of five
    # mem_reads. ≤4K carts have no bankswitch, so ROM offsets ≡ PC & mask.
    state.ff_site0 = 0xFFFF
    state.ff_site1 = 0xFFFF
    if rom_size <= 4096:
        for i in range(rom_size - 4):
            if (
                rom[unsafe_offset=i] != 0xAD
                or rom[unsafe_offset=i + 3] != 0xD0
                or rom[unsafe_offset=i + 4] != 0xFB
            ):
                continue
            var a = ((Int(rom[unsafe_offset=i + 2]) << 8) | Int(rom[unsafe_offset=i + 1])) & 0x1FFF
            if (
                (a & 0x1000) == 0
                and (a & 0x0080) != 0
                and (a & 0x0200) != 0
                and (a & 0x07) == 4
            ):
                if state.ff_site0 == 0xFFFF:
                    state.ff_site0 = UInt16(i)
                elif state.ff_site1 == 0xFFFF:
                    state.ff_site1 = UInt16(i)


# ============================================================================
# Run One Frame
# ============================================================================


def _run_scanline(
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
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
    # Stale diagnostic path (reachable only from diag_atari_si): materialize the
    # opcode table once and pass its pointer, mirroring the headless runner.
    var _optab = materialize[OPCODE_TABLE]()

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

        var cycles = Int(execute_one(state, rom, rom_size, _optab.unsafe_ptr()))
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
    rom: Pointer[UInt8, ImmutAnyOrigin],
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
    # Materialize the opcode table once per frame and pass its pointer down (the
    # frame runner can no longer read the comptime global directly — see
    # _fetch_opcode_entry). Once per ~20k instructions, so negligible on CPU.
    var optab = materialize[OPCODE_TABLE]()
    run_frame_cycle_accurate[RENDER=False](
        state, rom, rom_size, dummy.unsafe_ptr(), optab.unsafe_ptr()
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
    DQ_CAP,
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
def _cycle_pixel[
    o: MutOrigin
](
    mut state: AtariState,
    render_row: Int,
    pixel: Int,
    lit: UInt8,
    buf: Pointer[UInt8, o],
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




# Measurement builds only: flip to True to make run_frame_cycle_accurate
# accumulate bulk/per-clock clock counts + the selective-ticking ceiling
# into state.dbg_prof_* (read them with a probe; they accumulate across
# frames). False compiles the counters out of the hot loop.
comptime ATARI_PROFILE = False


@always_inline
def _intim_wait_skip_cycles(state: AtariState, rom_size: Int) -> Int:
    """INTIM wait-loop fast-forward: if PC sits on a pre-scanned
    `LDA <INTIM abs> / BNE -5` timer-poll site (`AD lo hi D0 FB` — the
    standard vblank/overscan wait; Pong has two), return the CPU cycles
    of the taken iterations to skip as one pseudo-instruction; 0 means
    execute normally.

    Sites are found by a one-time ROM scan in `cpu_reset` (unbanked ≤4K
    carts only — static instruction stream, no hotspots), so this probe
    is two PC compares per instruction, not memory reads.

    Exactness argument: a taken iteration only (a) reads INTIM
    (side-effect free), (b) overwrites A and N/Z (both rewritten by the
    next iteration — we always leave ≥1 taken iteration to run for
    real), and (c) consumes cycles. So skipping K iterations ≡ advancing
    the RIOT timer + TIA by K·L cycles, which the caller does through
    the same per-instruction span machinery as any real instruction
    (bulk where safe, per-clock otherwise — collisions/HMOVE stay
    exact). K is sized so every skipped read returns nonzero (the
    reference would also take the branch), from the closed-form
    decrement schedule.

    Guards: VSYNC inactive and DelayQueue empty (no frame-end edge can
    occur inside the span, which would end the frame with pre-paid timer
    cycles); timer not yet expired. Capped — long waits simply
    re-trigger at the next loop top.
    """
    if (state.pc & 0x1000) == 0:
        return 0  # PC not in cart space
    var off = UInt16(Int(state.pc) & (rom_size - 1))
    if off != state.ff_site0 and off != state.ff_site1:
        return 0
    if state.timer_value == 0:
        return 0
    if (state.tia_flags & VSY) != 0 or state.ctia.dq.count != 0:
        return 0

    # Taken-iteration length: LDA abs (4) + BNE taken (3, +1 if the branch
    # target crosses a page vs the instruction after the BNE).
    var loop_cyc = 7
    if ((state.pc + 5) >> 8) != (state.pc >> 8):
        loop_cyc = 8

    # Cycles until INTIM reaches 0 (per riot_update_timer's schedule: the
    # next decrement lands after interval - timer_clocks cycles, then one
    # per interval). Reads sample the timer at loop-top (the runner applies
    # riot_update_timer per whole instruction), so iteration j (1-based,
    # loop-top offset (j-1)·L) is taken iff (j-1)·L < t_zero.
    var t_first = Int(state.timer_interval) - Int(state.timer_clocks)
    var t_zero = t_first + (Int(state.timer_value) - 1) * Int(
        state.timer_interval
    )
    var taken = (t_zero - 1) // loop_cyc + 1
    var skip = taken - 1  # leave one taken iteration to execute for real
    if skip < 4:
        return 0  # not worth a pseudo-instruction
    if skip * loop_cyc > 2048:
        skip = 2048 // loop_cyc  # cap; the loop re-triggers if still waiting
    return skip * loop_cyc


@no_inline
def run_frame_cycle_accurate[
    RENDER: Bool = True,
    UNIFORM: Bool = False,
    fb_o: MutOrigin = MutAnyOrigin,
    ot_o: MutOrigin = MutAnyOrigin,
](
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    frame_buf: Pointer[UInt8, fb_o],
    op_table: Pointer[OpcodeEntry, ot_o],
):
    """Cycle-accurate frame: CPU and TIA in lockstep, per-color-clock collision.

    `UNIFORM` (comptime, default False): when True, the branchy bulk
    span-skipping fast path is dead-code-eliminated and every color clock takes
    the per-clock reference path. Bit-identical to the default (the bulk path is
    defined to mirror per-clock exactly), so the CPU trajectory checksum is
    unchanged — it exists to test, on the GPU, whether *uniform* work across a
    warp beats the divergent bulk path (audit risk #2). CPU/default builds
    should leave it False.

    `@no_inline` (compile-size hygiene): the largest function in the
    program. Historically ~220K lines of LLVM IR per instantiation when the
    405-line `execute_one` opcode dispatch was `@always_inline`d across every
    instruction of every scanline; `_step_obs_pixel` instantiates this runner
    twice (RENDER=True/False), and without this boundary `-O3` fused both
    copies into one ~440K-line function. `execute_one` is now `@no_inline`
    too (one shared dispatch copy), shrinking each instantiation ~10×; both
    boundaries stay. (The Rainbow-Atari -O3 compile OOM once blamed on this
    was root-caused to NStepTransition's by-value InlineArray obs in
    deep_agents — the emulator was a red herring; most of its residual IR
    bulk was live debug_assert bounds checks, see `-D ASSERT=none`.)
    Called once per emulated frame: a non-inlined call is
    runtime-negligible against the thousands of instructions it runs.

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
    # Flip to True (module comptime below) for a measurement build; the
    # counters quantify bulk vs per-clock clock consumption and the
    # selective-ticking ceiling (active objects per per-clock tick).
    var prof_bulk = 0
    var prof_pc = 0
    var prof_pc_target = 0
    var prof_active = 0
    # Sub-span granularity: how many `while consumed < span` iterations
    # (each pays a horizon-min + advance_objects) the bulk clocks split into.
    var prof_bulk_spans = 0
    var prof_bulk_visible_spans = 0
    # Cached lit-free budget: visible ticks (counted from the CycleTIA's
    # flushed-state-plus-pending position) during which no object can be lit,
    # so bulk sub-spans can defer the 5-object advance + horizon min. Always
    # decremented per accumulated tick (vblank included — those ticks move
    # the counters too); recomputed on FLUSHED state when exhausted, and
    # invalidated whenever the per-clock path runs (writes/strobes/movement
    # can change any object's window).
    var safe_budget = 0
    var clocks = 0
    # Fixed drain buffers for matured DelayQueue writes — no per-frame heap
    # (kernel-safe). At most DQ_CAP writes can fire in a single color clock.
    var due_reg = InlineArray[UInt8, DQ_CAP](fill=0)
    var due_val = InlineArray[UInt8, DQ_CAP](fill=0)
    # Bulk fast-path throttle: after a blocked attempt (an object's render
    # window is ahead), run this many per-clock iterations before retrying.
    var skip_bulk = 0

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
            # INTIM wait-loop fast-forward: skip whole taken poll iterations
            # as one pseudo-instruction (no TIA writes, PC unchanged); the
            # span machinery below advances the TIA exactly as for any
            # instruction of `cyc` cycles. See _intim_wait_skip_cycles.
            var ff = _intim_wait_skip_cycles(state, rom_size)
            if ff > 0:
                cyc = ff
                state.tia_log_count = 0
            else:
                cyc = Int(execute_one(state, rom, rom_size, op_table))
            riot_update_timer(state, UInt32(cyc))

        var nclk = (CPL - (start_hctr % CPL)) if wsync_pad else cyc * 3
        if wsync_pad:
            # The CPU is halted by WSYNC but TIME still passes: advance the RIOT
            # interval timer by the padded CPU cycles (nclk/3), exactly like
            # run_frame. Omitting this lets the RIOT timer lag every
            # line → the game's frame/scanline count and input timing drift
            # (SI vertical shaking; Breakout paddle-position-dependent glitches).
            riot_update_timer(state, UInt32(nclk // 3))

        var j = 0
        while j < nclk:
            # ---- Bulk fast path: advance through event-free clocks without
            # the per-clock loop. Preconditions: DelayQueue empty, no HMOVE
            # movement in progress, and no logged TIA write pushes before
            # `limit` — so TIA flags and all object configs are static across
            # the span. Then:
            #   - HBLANK clocks do nothing at all (movement is off): skip O(1)
            #   - visible clocks only advance the object counters; as long as
            #     no object can be lit (lit_safe_horizon), or VBLANK blanks
            #     collision latching anyway, no collision can latch —
            #     advance the counters exactly via advance_n.
            #   - video: every covered in-range pixel is exactly what
            #     _cycle_pixel(lit=0) produces (PF/BK priority, or COLUBK
            #     under VBLANK) — fill via the same function, no object ticks.
            # Line wraps replicate the per-clock bookkeeping below verbatim.
            if (
                not UNIFORM
                and skip_bulk == 0
                and state.ctia.dq.count == 0
                and not state.ctia.movement_in_progress
            ):
                var limit = nclk
                if not wsync_pad:
                    for li in range(state.tia_log_count):
                        var c = state.tia_log_clock[li]
                        if c >= j and c < limit:
                            limit = c
                var vbl_on = (state.tia_flags & VBL) != 0
                var span = limit - j
                var consumed = 0
                while consumed < span and not done:
                    comptime if ATARI_PROFILE:
                        prof_bulk_spans += 1
                    var bh = state.ctia.hctr
                    var bhbe = (
                        76 if state.ctia.extended_hblank else HBLANK_CLOCKS
                    )
                    var k: Int
                    if bh < bhbe:
                        # HBLANK: no ticks, no collisions.
                        k = min(bhbe - bh, span - consumed)
                    else:
                        comptime if ATARI_PROFILE:
                            prof_bulk_visible_spans += 1
                        k = min(CPL - bh, span - consumed)
                        if not vbl_on:
                            if safe_budget < k:
                                # Horizon is defined on flushed counter
                                # state — flush, then recompute the budget.
                                state.ctia.flush_pending()
                                safe_budget = state.ctia.lit_safe_horizon()
                                if safe_budget <= 0:
                                    break  # render window -> per-clock
                                if safe_budget < k:
                                    k = safe_budget
                        # Defer the 5-object advance: accumulate the ticks;
                        # flush happens before anything can observe object
                        # state (per-clock entry / dq applies / frame end).
                        state.ctia.pending_ticks += k
                        safe_budget -= k
                    comptime if RENDER:
                        var brow = (
                            visible_line if visible_line < FH2 else -1
                        )
                        var pend = min(
                            bh + k - HBLANK_CLOCKS, FRAME_WIDTH
                        )
                        for pix in range(max(bh - HBLANK_CLOCKS, 0), pend):
                            _cycle_pixel(
                                state, brow, pix, 0, frame_buf, palette
                            )
                    consumed += k
                    var bnh = bh + k
                    if bnh >= CPL:
                        bnh = 0
                        state.ctia.extended_hblank = False
                        if state.wsync and not wsync_pad:
                            state.wsync = False
                        if (
                            state.paddle_charge < 255
                            and (state.tia_flags & TIA_PADDLE_GROUND) == 0
                        ):
                            state.paddle_charge += 1
                        var bv_now = (state.tia_flags & VSY) != 0
                        var bv_rising = bv_now and not prev_vsync
                        prev_vsync = bv_now
                        if bv_rising:
                            if rendered_any:
                                done = True
                            else:
                                visible_line = 0
                                total_lines = 0
                                ystart = -1
                        total_lines += 1
                        if (
                            state.tia_flags & VBL
                        ) == 0 and visible_line < FH2:
                            if ystart < 0:
                                ystart = total_lines - 1
                            visible_line += 1
                            rendered_any = True
                    state.ctia.hctr = bnh
                j += consumed
                clocks += consumed
                comptime if ATARI_PROFILE:
                    prof_bulk += consumed
                if done:
                    break
                if consumed > 0:
                    continue
                skip_bulk = 4
            elif skip_bulk > 0:
                skip_bulk -= 1

            comptime if ATARI_PROFILE:
                prof_pc += 1
                var hbe_p = 76 if state.ctia.extended_hblank else HBLANK_CLOCKS
                if (
                    state.ctia.dq.count == 0
                    and not state.ctia.movement_in_progress
                    and state.ctia.hctr >= hbe_p
                ):
                    # Visible clock forced per-clock by a lit window — the
                    # selective-ticking target. Count objects that would
                    # still need a real per-clock tick.
                    prof_pc_target += 1
                    if state.ctia.p0.lit_horizon() == 0:
                        prof_active += 1
                    if state.ctia.p1.lit_horizon() == 0:
                        prof_active += 1
                    if state.ctia.m0.lit_horizon() == 0:
                        prof_active += 1
                    if state.ctia.m1.lit_horizon() == 0:
                        prof_active += 1
                    if state.ctia.bl.lit_horizon() == 0:
                        prof_active += 1

            # Per-clock path: everything below reads or mutates live object
            # state (dq applies/strobes, movement ticks, ctia.tick), so the
            # deferred bulk ticks must land first, and the cached budget is
            # no longer valid after whatever happens here.
            state.ctia.flush_pending()
            safe_budget = 0

            var hctr = state.ctia.hctr
            var pixel = hctr - HBLANK_CLOCKS
            # Extended HBLANK after HMOVE: the 8 clocks past the normal 68 stay
            # "blank" so objects skip 8 frame ticks, canceling the comb's 8
            # baseline ticks (net HMOVE motion = hmm_clocks - 8). resx strobes in
            # this window also use the hblank counter (Stella myHstate==blank).
            var hbe = 76 if state.ctia.extended_hblank else HBLANK_CLOCKS
            var in_hblank = hctr < hbe

            # Replay this instruction's TIA writes at their exact offset clock.
            # STORE_CYCLE_CLOCKS: the write takes effect at the END of its CPU
            # cycle (Stella increments the system clock before the poke), i.e.
            # 3 color clocks after the logged cycle-start offset. Adding it to
            # the queue delay (instead of the log clock) lets the apply land
            # past this instruction's last clock — the queue persists across
            # instructions and WSYNC padding.
            comptime STORE_CYCLE_CLOCKS = 3
            if not wsync_pad:
                for li in range(state.tia_log_count):
                    if state.tia_log_clock[li] == j:
                        var r = state.tia_log_reg[li]
                        state.ctia.dq.push(
                            r,
                            state.tia_log_value[li],
                            _cycle_reg_delay(r) + STORE_CYCLE_CLOCKS,
                        )

            # Apply writes whose delay elapsed this clock. The queue is empty
            # on the vast majority of clocks — skip the List churn entirely.
            if state.ctia.dq.count != 0:
                var ndue = state.ctia.dq.cycle_collect(due_reg, due_val)
                for k in range(ndue):
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
            j += 1

    # Deferred bulk ticks must be applied before the frame ends: the next
    # frame's prologue re-seeds object state, and anything inspecting the
    # CycleTIA between frames must see flushed counters.
    state.ctia.flush_pending()

    state.scanline = UInt16(visible_line)
    state.dbg_frame_lines = UInt16(total_lines)
    state.dbg_ystart = UInt16(ystart) if ystart >= 0 else 0
    state.frame_number += 1
    # Unconditional flush keeps the locals "used" when ATARI_PROFILE=False
    # (they are constant 0 then — this is 4 dead stores per frame).
    state.dbg_prof_bulk_clocks += prof_bulk
    state.dbg_prof_perclock += prof_pc
    state.dbg_prof_perclock_target += prof_pc_target
    state.dbg_prof_active_ticks += prof_active
    state.dbg_prof_bulk_spans += prof_bulk_spans
    state.dbg_prof_bulk_visible_spans += prof_bulk_visible_spans


def run_frame_video(
    mut state: AtariState,
    rom: Pointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    frame_buf: Pointer[UInt8, MutAnyOrigin],
):
    """Render one VSYNC-aligned frame into frame_buf (BGRA).

    Thin alias for the cycle-accurate, Stella-style per-color-clock TIA path —
    the single rendering path (the legacy end-of-line renderer was removed)."""
    var optab = materialize[OPCODE_TABLE]()
    run_frame_cycle_accurate(
        state, rom, rom_size, frame_buf, optab.unsafe_ptr()
    )
