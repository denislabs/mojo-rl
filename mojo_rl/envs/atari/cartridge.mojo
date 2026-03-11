"""Atari 2600 cartridge / ROM access with bank switching.

Supports the most common mapper types:
  - 2K: Direct mapping (mirrors to fill 4K)
  - 4K: Direct mapping
  - F8: 8K, two 4K banks (switched via $1FF8/$1FF9)
  - F6: 16K, four 4K banks (switched via $1FF6-$1FF9)

Ported from CuLE (BSD-3): cule/atari/accessors.hpp, mmc.hpp
"""

from .atari_state import AtariState
from .flags import ROM_2K, ROM_4K, ROM_F8, ROM_F6


@always_inline
fn rom_read(
    state: AtariState,
    rom: UnsafePointer[UInt8],
    rom_size: Int,
    addr: UInt16,
) -> UInt8:
    """Read a byte from cartridge ROM.

    addr is the full 13-bit address (bit 12 already confirmed set).
    For bank-switched cartridges, the current_bank selects the 4K window.
    """
    var offset = Int(addr) & 0x0FFF  # 12-bit offset within 4K window

    if rom_size <= 2048:
        # 2K ROM: mirror within 2K
        return rom[offset & 0x07FF]
    elif rom_size <= 4096:
        # 4K ROM: direct access
        return rom[offset]
    else:
        # Bank-switched: use current_bank to select 4K window
        var bank_offset = Int(state.current_bank) * 4096
        var rom_addr = bank_offset + offset
        if rom_addr < rom_size:
            return rom[rom_addr]
        return 0


@always_inline
fn rom_write(
    mut state: AtariState,
    rom: UnsafePointer[UInt8],
    rom_size: Int,
    addr: UInt16,
    value: UInt8,
):
    """Write to cartridge address space — handles bank switching.

    ROM is read-only, but accessing certain addresses triggers bank switches.
    """
    var offset = Int(addr) & 0x0FFF

    if rom_size <= 4096:
        return  # No bank switching for 2K/4K

    if rom_size <= 8192:
        # F8: 8K, two 4K banks
        if offset == 0x0FF8:
            state.current_bank = 0
        elif offset == 0x0FF9:
            state.current_bank = 1

    elif rom_size <= 16384:
        # F6: 16K, four 4K banks
        if offset == 0x0FF6:
            state.current_bank = 0
        elif offset == 0x0FF7:
            state.current_bank = 1
        elif offset == 0x0FF8:
            state.current_bank = 2
        elif offset == 0x0FF9:
            state.current_bank = 3


@always_inline
fn rom_read_triggers_bankswitch(
    mut state: AtariState,
    rom_size: Int,
    addr: UInt16,
):
    """Check if a ROM read triggers a bank switch (for F8/F6 mappers).

    Some bank switching is triggered by reads, not writes.
    This should be called on every ROM read in the memory map.
    """
    var offset = Int(addr) & 0x0FFF

    if rom_size <= 4096:
        return

    if rom_size <= 8192:  # F8
        if offset == 0x0FF8:
            state.current_bank = 0
        elif offset == 0x0FF9:
            state.current_bank = 1
    elif rom_size <= 16384:  # F6
        if offset == 0x0FF6:
            state.current_bank = 0
        elif offset == 0x0FF7:
            state.current_bank = 1
        elif offset == 0x0FF8:
            state.current_bank = 2
        elif offset == 0x0FF9:
            state.current_bank = 3


fn detect_rom_format(rom: UnsafePointer[UInt8], rom_size: Int) -> UInt8:
    """Auto-detect ROM format from size."""
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


fn init_bank(mut state: AtariState, rom_size: Int):
    """Initialize bank to the correct starting bank for the ROM format."""
    if rom_size <= 4096:
        state.current_bank = 0
    elif rom_size <= 8192:
        # F8: Start in bank 1 (contains reset vector)
        state.current_bank = 1
    elif rom_size <= 16384:
        # F6: Start in last bank (bank 3, contains reset vector)
        state.current_bank = 3
