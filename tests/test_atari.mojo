"""Basic test for the Atari emulator core.

Tests that the 6502 CPU, memory map, and game definitions compile
and produce sensible output.
"""

from mojo_rl.envs.atari.atari_state import AtariState
from mojo_rl.envs.atari.flags import (
    RAM_SIZE,
    ACTION_NOOP,
    FLAG_C,
    FLAG_Z,
    FLAG_N,
)
from mojo_rl.envs.atari.ram import read_ram, write_ram
from mojo_rl.envs.atari.cpu6502 import (
    mem_read,
    mem_write,
    execute_one,
    cpu_reset,
    set_flag,
    get_flag,
    update_nz,
)
from mojo_rl.envs.atari.opcodes import OPCODE_TABLE, OP_LDA, OP_NOP
from mojo_rl.envs.atari.riot import set_action, riot_update_timer
from mojo_rl.envs.atari.cartridge import detect_rom_format, init_bank
from std.memory import alloc
from mojo_rl.envs.atari.frame_render import render_frame_bgra, FRAME_BUF_SIZE
from mojo_rl.envs.atari.games.pong import PongDef
from mojo_rl.envs.atari.games.breakout import BreakoutDef
from mojo_rl.envs.atari.games.helpers import (
    get_decimal_score,
    get_decimal_score_2,
)


fn test_state_init():
    print("Test: AtariState init...")
    var state = AtariState()
    assert_true(state.pc == 0)
    assert_true(state.sp == 0xFD)
    assert_true(state.a == 0)
    assert_true(state.flags == 0x20)
    assert_true(state.ram[0] == 0)
    assert_true(state.ram[127] == 0)
    print("  PASSED")


fn test_ram_rw():
    print("Test: RAM read/write...")
    var state = AtariState()
    write_ram(state.ram, 0x42, 0xAB)
    assert_true(read_ram(state.ram, 0x42) == 0xAB)
    # Test wrapping
    write_ram(state.ram, 0x82, 0xCD)
    assert_true(read_ram(state.ram, 0x02) == 0xCD)
    print("  PASSED")


fn test_flags():
    print("Test: CPU flags...")
    var state = AtariState()
    set_flag(state, FLAG_C, True)
    assert_true(get_flag(state, FLAG_C))
    set_flag(state, FLAG_C, False)
    assert_true(not get_flag(state, FLAG_C))

    update_nz(state, 0)
    assert_true(get_flag(state, FLAG_Z))
    assert_true(not get_flag(state, FLAG_N))

    update_nz(state, 0x80)
    assert_true(not get_flag(state, FLAG_Z))
    assert_true(get_flag(state, FLAG_N))

    update_nz(state, 0x42)
    assert_true(not get_flag(state, FLAG_Z))
    assert_true(not get_flag(state, FLAG_N))
    print("  PASSED")


fn test_opcode_table():
    print("Test: Opcode table...")
    var table = materialize[OPCODE_TABLE]()
    # NOP = 0xEA
    var nop = table[0xEA]
    assert_true(nop.instruction == OP_NOP)
    assert_true(nop.cycles == 2)
    assert_true(nop.size == 1)

    # LDA #imm = 0xA9
    var lda = table[0xA9]
    assert_true(lda.instruction == OP_LDA)
    assert_true(lda.cycles == 2)
    assert_true(lda.size == 2)
    print("  PASSED")


fn test_pong_game():
    print("Test: Pong game definition...")
    var ram = InlineArray[UInt8, RAM_SIZE](fill=0)

    # Player score = 5, CPU score = 3
    ram[14] = 5
    ram[13] = 3
    assert_true(PongDef.get_score(ram) == 2)  # 5 - 3
    assert_true(not PongDef.is_terminal(ram))

    # Player wins at 21
    ram[14] = 21
    assert_true(PongDef.is_terminal(ram))
    print("  PASSED")


fn test_breakout_game():
    print("Test: Breakout game definition...")
    var ram = InlineArray[UInt8, RAM_SIZE](fill=0)

    # Score = 123: RAM[77] = 0x23 (tens=2, ones=3), RAM[76] = 0x01 (hundreds=1)
    ram[77] = 0x23
    ram[76] = 0x01
    assert_true(BreakoutDef.get_score(ram) == 123)

    # Lives
    ram[57] = 3
    assert_true(BreakoutDef.get_lives(ram) == 3)
    assert_true(not BreakoutDef.is_terminal(ram))

    ram[57] = 0
    assert_true(BreakoutDef.is_terminal(ram))
    print("  PASSED")


fn test_bcd_helpers():
    print("Test: BCD score helpers...")
    var ram = InlineArray[UInt8, RAM_SIZE](fill=0)

    # Single byte: 0x42 = 42
    ram[0] = 0x42
    assert_true(get_decimal_score(ram, 0) == 42)

    # Two bytes: 0x56 (ones/tens) + 0x78 (hundreds/thousands) = 7856
    ram[0] = 0x56
    ram[1] = 0x78
    assert_true(get_decimal_score_2(ram, 0, 1) == 7856)

    # Two bytes, no high byte
    assert_true(get_decimal_score_2(ram, 0, -1) == 56)
    print("  PASSED")


fn test_action_mapping():
    print("Test: Action mapping...")
    var state = AtariState()
    set_action(state, ACTION_NOOP)
    # NOOP should have no direction flags
    assert_true((state.sys_flags & 0x1F) == 0)
    print("  PASSED")


fn test_palette():
    print("Test: Palette...")
    from mojo_rl.envs.atari.palette import (
        palette_r,
        palette_g,
        palette_b,
        palette_grayscale,
    )

    # Color index 0 = black (0x000000)
    assert_true(palette_r(0) == 0)
    assert_true(palette_g(0) == 0)
    assert_true(palette_b(0) == 0)
    assert_true(palette_grayscale(0) == 0)
    # Color index 2 = gray (0x4A4A4A)
    assert_true(palette_r(2) == 0x4A)
    assert_true(palette_g(2) == 0x4A)
    assert_true(palette_b(2) == 0x4A)
    print("  PASSED")


fn test_frame_render():
    print("Test: Frame rendering...")
    var state = AtariState()

    # Set a background color (index 0x1E = gold group, entry 15 = 0xFCFC00)
    state.colubk = 0x1E

    # Allocate frame buffer
    var buf = alloc[UInt8](FRAME_BUF_SIZE)

    # Render a frame (just background, no objects active)
    render_frame_bgra(state, buf)

    # Check first pixel is the background color in BGRA format
    # Color 0x1E = palette index 30 = 0xECECEC (gray)
    from mojo_rl.envs.atari.palette import palette_r, palette_g, palette_b

    var expected_r = palette_r(0x1E)
    var expected_g = palette_g(0x1E)
    var expected_b = palette_b(0x1E)

    # BGRA layout: [B, G, R, A]
    assert_true(buf[0] == expected_b)  # B
    assert_true(buf[1] == expected_g)  # G
    assert_true(buf[2] == expected_r)  # R
    assert_true(buf[3] == 0xFF)  # A

    # Check a pixel in the middle of the frame
    var mid = (105 * 160 + 80) * 4  # y=105, x=80
    assert_true(buf[mid + 0] == expected_b)
    assert_true(buf[mid + 1] == expected_g)
    assert_true(buf[mid + 2] == expected_r)
    assert_true(buf[mid + 3] == 0xFF)

    buf.free()
    print("  PASSED")


fn test_frame_render_player():
    print("Test: Frame rendering with player sprite...")
    var state = AtariState()

    # Set colors
    state.colubk = 0x00  # Black background
    state.colup0 = 0x42  # Player 0 = red-ish (Pink group)

    # Enable player 0: full 8-pixel bar at position 80
    state.grp0 = 0xFF  # All 8 bits set
    state.pos_p0 = 80  # Position at center

    var buf = alloc[UInt8](FRAME_BUF_SIZE)
    render_frame_bgra(state, buf)

    # Pixel at x=80 (player position) should have player color
    from mojo_rl.envs.atari.palette import palette_r, palette_g, palette_b

    var p0_r = palette_r(0x42)
    var p0_g = palette_g(0x42)
    var p0_b = palette_b(0x42)

    # Check pixel at (80, 0) — first scanline
    var offset = 80 * 4
    assert_true(buf[offset + 0] == p0_b)  # B
    assert_true(buf[offset + 1] == p0_g)  # G
    assert_true(buf[offset + 2] == p0_r)  # R

    # Pixel at x=0 should be background (black)
    assert_true(buf[0] == 0)  # B = 0
    assert_true(buf[1] == 0)  # G = 0
    assert_true(buf[2] == 0)  # R = 0

    buf.free()
    print("  PASSED")


fn main():
    print("=== Atari 2600 Emulator Tests ===")
    test_state_init()
    test_ram_rw()
    test_flags()
    test_opcode_table()
    test_pong_game()
    test_breakout_game()
    test_bcd_helpers()
    test_action_mapping()
    test_palette()
    test_frame_render()
    test_frame_render_player()
    print("=== All tests passed! ===")


fn assert_true(cond: Bool):
    if not cond:
        print("ASSERTION FAILED!")
