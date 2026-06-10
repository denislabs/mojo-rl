"""Atari 2600 cartridge / ROM access with bank switching.

Supported mappers (state.mapper, resolved by init_bank):
  - 2K:   Direct mapping (mirrors to fill 4K)
  - 4K:   Direct mapping
  - F8:   8K, two 4K banks (hotspots $0FF8/$0FF9)
  - F6:   16K, four 4K banks (hotspots $0FF6-$0FF9)
  - F8SC/F6SC: F8/F6 + 128B Superchip RAM (write $1000-$107F, read $1080-$10FF)
  - E0:   8K Parker Bros — four 1K segments, segments 0-2 switchable among
          eight 1K slices via hotspots $0FE0-$0FF7, segment 3 fixed to slice 7
  - FE:   8K Activision — bank selected by A13 of the CPU address ($Dxxx vs
          $Fxxx), no hotspots
Bank-switch hotspots trigger on ANY access (read or write), like real
hardware and Stella/ALE — reads must go through rom_read with a mut state.

Size-based detection cannot distinguish F8 / E0 / FE / F8SC at 8K (or
F6 / F6SC at 16K); games needing a non-default mapper carry an override in
the game registry (AtariGame.mapper()), baked from running ALE's
Cartridge::autodetectType signatures over the ROM set.

Ported from ALE/Stella: emucore/Cart{F8,F6,F8SC,F6SC,E0,FE}.cxx
"""

from .atari_state import AtariState
from .flags import (
    ROM_2K,
    ROM_4K,
    ROM_F8,
    ROM_F8SC,
    ROM_F6,
    ROM_F6SC,
    ROM_E0,
    ROM_FE,
    ROM_AUTO,
)


@always_inline
def rom_read(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    addr: UInt16,
) -> UInt8:
    """Read a byte from cartridge space (bit 12 of addr already confirmed set).

    addr is the FULL 16-bit CPU address: FE needs A13, which the 13-bit
    system mask would destroy. May switch banks (hotspot reads), hence mut.
    """
    var offset = Int(addr) & 0x0FFF  # 12-bit offset within the 4K window
    var m = state.mapper

    if m == ROM_4K:
        return rom[offset]
    elif m == ROM_2K:
        return rom[offset & 0x07FF]
    elif m == ROM_F8 or m == ROM_F8SC:
        if m == ROM_F8SC and offset < 0x100:
            # Superchip RAM (read port $80-$FF; reading the write port is
            # undefined on hardware — return the RAM byte).
            return state.sc_ram[offset & 0x7F]
        if offset == 0x0FF8:
            state.current_bank = 0
        elif offset == 0x0FF9:
            state.current_bank = 1
        return rom[Int(state.current_bank) * 4096 + offset]
    elif m == ROM_F6 or m == ROM_F6SC:
        if m == ROM_F6SC and offset < 0x100:
            return state.sc_ram[offset & 0x7F]
        if offset == 0x0FF6:
            state.current_bank = 0
        elif offset == 0x0FF7:
            state.current_bank = 1
        elif offset == 0x0FF8:
            state.current_bank = 2
        elif offset == 0x0FF9:
            state.current_bank = 3
        return rom[Int(state.current_bank) * 4096 + offset]
    elif m == ROM_E0:
        if offset >= 0x0FE0 and offset <= 0x0FE7:
            state.e0_slices[0] = UInt8(offset & 0x0007)
        elif offset >= 0x0FE8 and offset <= 0x0FEF:
            state.e0_slices[1] = UInt8(offset & 0x0007)
        elif offset >= 0x0FF0 and offset <= 0x0FF7:
            state.e0_slices[2] = UInt8(offset & 0x0007)
        return rom[
            (Int(state.e0_slices[offset >> 10]) << 10) + (offset & 0x03FF)
        ]
    elif m == ROM_FE:
        # Bank = A13 of the CPU address: $Fxxx (A13=1) → bank 0,
        # $Dxxx (A13=0) → bank 1 (Stella CartFE::peek).
        if (Int(addr) & 0x2000) == 0:
            return rom[offset + 4096]
        return rom[offset]

    # Unknown mapper: direct 4K access.
    return rom[offset]


@always_inline
def rom_write(
    mut state: AtariState,
    rom: UnsafePointer[UInt8, ImmutAnyOrigin],
    rom_size: Int,
    addr: UInt16,
    value: UInt8,
):
    """Write to cartridge address space — bank-switch hotspots + SC RAM."""
    var offset = Int(addr) & 0x0FFF
    var m = state.mapper

    if m == ROM_F8 or m == ROM_F8SC:
        if m == ROM_F8SC and offset < 0x80:
            state.sc_ram[offset] = value
            return
        if offset == 0x0FF8:
            state.current_bank = 0
        elif offset == 0x0FF9:
            state.current_bank = 1
    elif m == ROM_F6 or m == ROM_F6SC:
        if m == ROM_F6SC and offset < 0x80:
            state.sc_ram[offset] = value
            return
        if offset == 0x0FF6:
            state.current_bank = 0
        elif offset == 0x0FF7:
            state.current_bank = 1
        elif offset == 0x0FF8:
            state.current_bank = 2
        elif offset == 0x0FF9:
            state.current_bank = 3
    elif m == ROM_E0:
        if offset >= 0x0FE0 and offset <= 0x0FE7:
            state.e0_slices[0] = UInt8(offset & 0x0007)
        elif offset >= 0x0FE8 and offset <= 0x0FEF:
            state.e0_slices[1] = UInt8(offset & 0x0007)
        elif offset >= 0x0FF0 and offset <= 0x0FF7:
            state.e0_slices[2] = UInt8(offset & 0x0007)
    # 2K/4K/FE: writes to ROM space are no-ops.


def detect_rom_format(
    rom: UnsafePointer[UInt8, ImmutAnyOrigin], rom_size: Int
) -> UInt8:
    """Auto-detect ROM format from size (F8/F6 defaults at 8K/16K).

    Games whose 8K/16K image is NOT plain F8/F6 (E0, FE, F8SC, F6SC) must
    override via the registry — size alone cannot distinguish them.
    """
    if rom_size <= 2048:
        return ROM_2K
    elif rom_size <= 4096:
        return ROM_4K
    elif rom_size <= 8192:
        return ROM_F8
    elif rom_size <= 16384:
        return ROM_F6
    else:
        return ROM_4K  # Fallback


def init_bank(mut state: AtariState, rom_size: Int, mapper: UInt8 = ROM_AUTO):
    """Resolve the mapper and set the power-on bank configuration."""
    if mapper == ROM_AUTO:
        if rom_size <= 2048:
            state.mapper = ROM_2K
        elif rom_size <= 4096:
            state.mapper = ROM_4K
        elif rom_size <= 8192:
            state.mapper = ROM_F8
        else:
            state.mapper = ROM_F6
    else:
        state.mapper = mapper

    state.sc_ram = InlineArray[UInt8, 128](fill=0)
    # E0 power-on slices (Stella CartE0::reset): 4, 5, 6 + fixed 7.
    state.e0_slices = [4, 5, 6, 7]

    var m = state.mapper
    if m == ROM_F8 or m == ROM_F8SC:
        # Start in bank 1 (contains the reset vector).
        state.current_bank = 1
    elif m == ROM_F6 or m == ROM_F6SC:
        # Start in the last bank (contains the reset vector).
        state.current_bank = 3
    else:
        state.current_bank = 0
