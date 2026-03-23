"""Atari 2600 RAM access (128 bytes).

CuLE stores RAM as 32 x UInt32 for packed access. We use a simple
InlineArray[UInt8, 128] for clarity and direct byte addressing.

Ported from CuLE (BSD-3): cule/atari/ram.hpp
"""

from .flags import RAM_SIZE


@always_inline
def read_ram(ram: InlineArray[UInt8, RAM_SIZE], addr: Int) -> UInt8:
    """Read a byte from the 128-byte RAM."""
    return ram[addr & 0x7F]


@always_inline
def write_ram(mut ram: InlineArray[UInt8, RAM_SIZE], addr: Int, value: UInt8):
    """Write a byte to the 128-byte RAM."""
    ram[addr & 0x7F] = value
