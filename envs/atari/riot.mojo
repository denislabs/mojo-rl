"""Atari 2600 RIOT (6532) chip emulation — RAM, I/O, Timer.

The RIOT provides:
  - 128 bytes of RAM (handled in ram.mojo)
  - Two 8-bit I/O ports (joystick, console switches)
  - A programmable interval timer

Ported from CuLE (BSD-3): cule/atari/m6532.hpp
"""

from .atari_state import AtariState
from .flags import (
    FLAG_CON_UP, FLAG_CON_DOWN, FLAG_CON_LEFT, FLAG_CON_RIGHT, FLAG_CON_FIRE,
    FLAG_CON_SELECT, FLAG_CON_RESET, FLAG_CON_COLOR,
    FLAG_CON_LEFT_DIFF, FLAG_CON_RIGHT_DIFF,
    ACTION_NOOP, ACTION_FIRE, ACTION_UP, ACTION_RIGHT, ACTION_LEFT,
    ACTION_DOWN, ACTION_UPRIGHT, ACTION_UPLEFT, ACTION_DOWNRIGHT,
    ACTION_DOWNLEFT, ACTION_UPFIRE, ACTION_RIGHTFIRE, ACTION_LEFTFIRE,
    ACTION_DOWNFIRE, ACTION_UPRIGHTFIRE, ACTION_UPLEFTFIRE,
    ACTION_DOWNRIGHTFIRE, ACTION_DOWNLEFTFIRE, ACTION_RESET,
)
from .ram import read_ram, write_ram


@always_inline
fn set_action(mut state: AtariState, action: UInt8):
    """Map an ALE action to joystick flags in sys_flags."""
    # Clear all controller direction flags
    state.sys_flags = state.sys_flags & ~(
        FLAG_CON_UP | FLAG_CON_DOWN | FLAG_CON_LEFT | FLAG_CON_RIGHT | FLAG_CON_FIRE
    )

    # Set flags based on action
    if action == ACTION_UP or action == ACTION_UPRIGHT or action == ACTION_UPLEFT or action == ACTION_UPFIRE or action == ACTION_UPRIGHTFIRE or action == ACTION_UPLEFTFIRE:
        state.sys_flags = state.sys_flags | FLAG_CON_UP
    if action == ACTION_DOWN or action == ACTION_DOWNRIGHT or action == ACTION_DOWNLEFT or action == ACTION_DOWNFIRE or action == ACTION_DOWNRIGHTFIRE or action == ACTION_DOWNLEFTFIRE:
        state.sys_flags = state.sys_flags | FLAG_CON_DOWN
    if action == ACTION_LEFT or action == ACTION_UPLEFT or action == ACTION_DOWNLEFT or action == ACTION_LEFTFIRE or action == ACTION_UPLEFTFIRE or action == ACTION_DOWNLEFTFIRE:
        state.sys_flags = state.sys_flags | FLAG_CON_LEFT
    if action == ACTION_RIGHT or action == ACTION_UPRIGHT or action == ACTION_DOWNRIGHT or action == ACTION_RIGHTFIRE or action == ACTION_UPRIGHTFIRE or action == ACTION_DOWNRIGHTFIRE:
        state.sys_flags = state.sys_flags | FLAG_CON_RIGHT
    if action == ACTION_FIRE or action == ACTION_UPFIRE or action == ACTION_RIGHTFIRE or action == ACTION_LEFTFIRE or action == ACTION_DOWNFIRE or action == ACTION_UPRIGHTFIRE or action == ACTION_UPLEFTFIRE or action == ACTION_DOWNRIGHTFIRE or action == ACTION_DOWNLEFTFIRE:
        state.sys_flags = state.sys_flags | FLAG_CON_FIRE
    if action == ACTION_RESET:
        state.sys_flags = state.sys_flags | FLAG_CON_RESET


@always_inline
fn riot_read_swcha(state: AtariState) -> UInt8:
    """Read SWCHA — joystick port.

    Each direction pulls a pin low (0). Default is 0xFF (nothing pressed).
    Bits 7-4: Player 1 (right/left/down/up)
    Bits 3-0: Player 0 (right/left/down/up)

    CuLE only uses Player 0 (bits 7-4), but we follow the convention
    where bits are active-low.
    """
    var value: UInt8 = 0xFF  # All pins high (unpressed)

    if (state.sys_flags & FLAG_CON_UP) != 0:
        value = value & ~UInt8(0x10)    # Bit 4 low
    if (state.sys_flags & FLAG_CON_DOWN) != 0:
        value = value & ~UInt8(0x20)    # Bit 5 low
    if (state.sys_flags & FLAG_CON_LEFT) != 0:
        value = value & ~UInt8(0x40)    # Bit 6 low
    if (state.sys_flags & FLAG_CON_RIGHT) != 0:
        value = value & ~UInt8(0x80)    # Bit 7 low

    return value


@always_inline
fn riot_read_swchb(state: AtariState) -> UInt8:
    """Read SWCHB — console switches.

    Bit 0: RESET (active low)
    Bit 1: SELECT (active low)
    Bit 3: B/W-Color (1=color)
    Bit 6: Left difficulty (0=B, 1=A)
    Bit 7: Right difficulty (0=B, 1=A)
    """
    var value: UInt8 = 0x0B  # Default: color TV, no reset/select

    if (state.sys_flags & FLAG_CON_RESET) != 0:
        value = value & ~UInt8(0x01)
    if (state.sys_flags & FLAG_CON_SELECT) != 0:
        value = value & ~UInt8(0x02)
    if (state.sys_flags & FLAG_CON_LEFT_DIFF) != 0:
        value = value | UInt8(0x40)
    if (state.sys_flags & FLAG_CON_RIGHT_DIFF) != 0:
        value = value | UInt8(0x80)

    return value


@always_inline
fn riot_read(state: AtariState, addr: UInt8) -> UInt8:
    """Read a RIOT register or RAM.

    Address space: 0x0280-0x0297 for registers, 0x0080-0x00FF for RAM.
    The addr parameter is the low byte (already masked).
    """
    var reg = addr & 0x07

    if reg == 0x00:  # SWCHA
        return riot_read_swcha(state)
    elif reg == 0x02:  # SWCHB
        return riot_read_swchb(state)
    elif reg == 0x04:  # INTIM - timer value
        return UInt8(state.timer_value & 0xFF)
    elif reg == 0x05:  # INSTAT - timer status
        # Bit 7: timer underflow, Bit 6: underflow since last INTIM read
        if state.timer_value == 0:
            return 0xC0
        return 0x00
    else:
        return 0


@always_inline
fn riot_write(mut state: AtariState, addr: UInt8, value: UInt8):
    """Write a RIOT register. Handles timer setup."""
    var reg = addr & 0x1F

    if reg == 0x14:  # TIM1T — set timer, interval = 1
        state.timer_value = UInt32(value)
        state.timer_interval = 1
        state.timer_clocks = 0
    elif reg == 0x15:  # TIM8T — set timer, interval = 8
        state.timer_value = UInt32(value)
        state.timer_interval = 8
        state.timer_clocks = 0
    elif reg == 0x16:  # TIM64T — set timer, interval = 64
        state.timer_value = UInt32(value)
        state.timer_interval = 64
        state.timer_clocks = 0
    elif reg == 0x17:  # T1024T — set timer, interval = 1024
        state.timer_value = UInt32(value)
        state.timer_interval = 1024
        state.timer_clocks = 0


@always_inline
fn riot_update_timer(mut state: AtariState, cycles: UInt32):
    """Advance the RIOT timer by the given number of CPU cycles.

    Each CPU cycle = 3 TIA clocks. Timer counts down at (3 * cycles / interval).
    After underflow, timer counts at 1x (every clock).
    """
    state.timer_clocks += cycles * 3

    while state.timer_clocks >= state.timer_interval:
        state.timer_clocks -= state.timer_interval
        if state.timer_value > 0:
            state.timer_value -= 1
        else:
            # After underflow, count at 1x rate
            state.timer_interval = 1
            # Timer wraps to 0xFF
            state.timer_value = 0xFF
