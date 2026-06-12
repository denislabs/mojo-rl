"""Atari 2600 RIOT (6532) chip emulation — RAM, I/O, Timer.

The RIOT provides:
  - 128 bytes of RAM (handled in ram.mojo)
  - Two 8-bit I/O ports (joystick, console switches)
  - A programmable interval timer

Ported from CuLE (BSD-3): cule/atari/m6532.hpp
"""

from .atari_state import AtariState
from .flags import (
    FLAG_CON_UP,
    FLAG_CON_DOWN,
    FLAG_CON_LEFT,
    FLAG_CON_RIGHT,
    FLAG_CON_FIRE,
    FLAG_CON_SELECT,
    FLAG_CON_RESET,
    FLAG_CON_COLOR,
    FLAG_CON_LEFT_DIFF,
    FLAG_CON_RIGHT_DIFF,
    FLAG_SWAP_PORTS,
    FLAG_PADDLES,
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_UP,
    ACTION_RIGHT,
    ACTION_LEFT,
    ACTION_DOWN,
    ACTION_UPRIGHT,
    ACTION_UPLEFT,
    ACTION_DOWNRIGHT,
    ACTION_DOWNLEFT,
    ACTION_UPFIRE,
    ACTION_RIGHTFIRE,
    ACTION_LEFTFIRE,
    ACTION_DOWNFIRE,
    ACTION_UPRIGHTFIRE,
    ACTION_UPLEFTFIRE,
    ACTION_DOWNRIGHTFIRE,
    ACTION_DOWNLEFTFIRE,
    ACTION_RESET,
)
from .ram import read_ram, write_ram


@always_inline
def set_action(mut state: AtariState, action: UInt8):
    """Map an ALE action to joystick flags in sys_flags."""
    # Clear all controller flags (including RESET so it doesn't stick)
    state.sys_flags = state.sys_flags & ~(
        FLAG_CON_UP
        | FLAG_CON_DOWN
        | FLAG_CON_LEFT
        | FLAG_CON_RIGHT
        | FLAG_CON_FIRE
        | FLAG_CON_RESET
    )

    # Set flags based on action
    if (
        action == ACTION_UP
        or action == ACTION_UPRIGHT
        or action == ACTION_UPLEFT
        or action == ACTION_UPFIRE
        or action == ACTION_UPRIGHTFIRE
        or action == ACTION_UPLEFTFIRE
    ):
        state.sys_flags = state.sys_flags | FLAG_CON_UP
    if (
        action == ACTION_DOWN
        or action == ACTION_DOWNRIGHT
        or action == ACTION_DOWNLEFT
        or action == ACTION_DOWNFIRE
        or action == ACTION_DOWNRIGHTFIRE
        or action == ACTION_DOWNLEFTFIRE
    ):
        state.sys_flags = state.sys_flags | FLAG_CON_DOWN
    if (
        action == ACTION_LEFT
        or action == ACTION_UPLEFT
        or action == ACTION_DOWNLEFT
        or action == ACTION_LEFTFIRE
        or action == ACTION_UPLEFTFIRE
        or action == ACTION_DOWNLEFTFIRE
    ):
        state.sys_flags = state.sys_flags | FLAG_CON_LEFT
    if (
        action == ACTION_RIGHT
        or action == ACTION_UPRIGHT
        or action == ACTION_DOWNRIGHT
        or action == ACTION_RIGHTFIRE
        or action == ACTION_UPRIGHTFIRE
        or action == ACTION_DOWNRIGHTFIRE
    ):
        state.sys_flags = state.sys_flags | FLAG_CON_RIGHT
    if (
        action == ACTION_FIRE
        or action == ACTION_UPFIRE
        or action == ACTION_RIGHTFIRE
        or action == ACTION_LEFTFIRE
        or action == ACTION_DOWNFIRE
        or action == ACTION_UPRIGHTFIRE
        or action == ACTION_UPLEFTFIRE
        or action == ACTION_DOWNRIGHTFIRE
        or action == ACTION_DOWNLEFTFIRE
    ):
        state.sys_flags = state.sys_flags | FLAG_CON_FIRE
    if action == ACTION_RESET:
        state.sys_flags = state.sys_flags | FLAG_CON_RESET

    # Update paddle position for paddle-based games (Pong, Breakout, etc.).
    # The paddle is read as INPT0/INPT1 (driven by paddle_pos), so movement
    # actions must adjust paddle_pos.
    #
    # Paddle carts (FLAG_PADDLES, ALE applyActionPaddles): only RIGHT
    # (decrease position) and LEFT (increase) move the paddle; UP/DOWN do
    # nothing at all in ALE's paddle mode.
    #
    # Legacy path (flag clear, GameDef envs): two directions map to "up"
    # and "down":
    #   - UP/RIGHT   move the paddle up   (decrease position)
    #   - DOWN/LEFT  move the paddle down (increase position)
    # RIGHT/LEFT are included because the minimal action sets for paddle games
    # (e.g. PongDef.map_action) emit RIGHT/LEFT, not UP/DOWN — without this an
    # agent's movement actions would never move the paddle.
    comptime PADDLE_DELTA: Int = 3
    var paddles_mode = (state.sys_flags & FLAG_PADDLES) != 0
    var move_up = (state.sys_flags & FLAG_CON_RIGHT) != 0 or (
        not paddles_mode and (state.sys_flags & FLAG_CON_UP) != 0
    )
    var move_down = (state.sys_flags & FLAG_CON_LEFT) != 0 or (
        not paddles_mode and (state.sys_flags & FLAG_CON_DOWN) != 0
    )
    if move_up:
        if Int(state.paddle_pos) >= PADDLE_DELTA:
            state.paddle_pos = UInt8(Int(state.paddle_pos) - PADDLE_DELTA)
        else:
            state.paddle_pos = 0
    if move_down:
        if Int(state.paddle_pos) + PADDLE_DELTA <= 255:
            state.paddle_pos = UInt8(Int(state.paddle_pos) + PADDLE_DELTA)
        else:
            state.paddle_pos = 255


@always_inline
def riot_read_swcha(state: AtariState) -> UInt8:
    """Read SWCHA — joystick port.

    Each direction pulls a pin low (0). Default is 0xFF (nothing pressed).
    Bits 7-4: Player 1 (right/left/down/up)
    Bits 3-0: Player 0 (right/left/down/up)

    CuLE only uses Player 0 (bits 7-4), but we follow the convention
    where bits are active-low.
    """
    var value: UInt8 = 0xFF  # All pins high (unpressed)

    # Swapped-port carts (Stella Console.SwapPorts, e.g. Wizard of Wor)
    # read player 1 from the RIGHT port = SWCHA low nibble.
    var shift: UInt8 = 0 if (state.sys_flags & FLAG_SWAP_PORTS) != 0 else 4

    if (state.sys_flags & FLAG_PADDLES) != 0:
        # Paddle cart: the joystick direction pins are the paddle BUTTONS
        # (Stella Paddles — pin Four = paddle 0 fire, pin Three = paddle 1
        # fire). FIRE grounds pin Four (= SWCHA D7 on the left port, ALE
        # PaddleZeroFire); directions never reach SWCHA.
        if (state.sys_flags & FLAG_CON_FIRE) != 0:
            value = value & ~(UInt8(0x08) << shift)
        return value

    if (state.sys_flags & FLAG_CON_UP) != 0:
        value = value & ~(UInt8(0x01) << shift)
    if (state.sys_flags & FLAG_CON_DOWN) != 0:
        value = value & ~(UInt8(0x02) << shift)
    if (state.sys_flags & FLAG_CON_LEFT) != 0:
        value = value & ~(UInt8(0x04) << shift)
    if (state.sys_flags & FLAG_CON_RIGHT) != 0:
        value = value & ~(UInt8(0x08) << shift)

    return value


@always_inline
def riot_read_swchb(state: AtariState) -> UInt8:
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


# @no_inline: see tia_read — compile-memory boundary.
@no_inline
def riot_read(state: AtariState, addr: UInt8) -> UInt8:
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


# @no_inline: see tia_read — compile-memory boundary.
@no_inline
def riot_write(mut state: AtariState, addr: UInt8, value: UInt8):
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
def riot_update_timer(mut state: AtariState, cycles: UInt32):
    """Advance the RIOT timer by the given number of CPU cycles.

    The 6532 runs at the CPU clock: TIM1T/TIM8T/TIM64T/T1024T intervals are
    1/8/64/1024 CPU cycles (ALE M6532: `delta >> myIntervalShift` with delta in
    CPU cycles). Counting color clocks against these intervals (the old
    `cycles * 3`) ran the timer 3x too fast: the VBLANK wait expired before the
    kernel's variable game-logic finished, INTIM free-ran past 0, and the
    wait-for-INTIM loop sampled a wrapping counter — frame length then tracked
    logic time (SI: 245-286 lines instead of a constant 262 = vertical shake).
    After underflow, the timer counts at 1x (every CPU cycle), wrapping 0xFF.
    """
    state.timer_clocks += cycles

    while state.timer_clocks >= state.timer_interval:
        state.timer_clocks -= state.timer_interval
        if state.timer_value > 0:
            state.timer_value -= 1
        else:
            # After underflow, count at 1x rate (per CPU cycle)
            state.timer_interval = 1
            # Timer wraps to 0xFF
            state.timer_value = 0xFF
