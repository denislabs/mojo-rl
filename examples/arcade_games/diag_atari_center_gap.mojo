"""Headless detector for the Breakout px72-87 center repeat-gap (vertical step).

Drives the paddle hard RIGHT (the user's trigger), serves the ball, and each
frame scans the brick band for scanlines where a center 8px column (px72-79 or
px80-87) is DARK while both of its neighbors are LIT — i.e. an otherwise-solid
brick row with a hole punched at the screen-center repeat boundary. That hole is
the vertical step/stretch the user reported. Compare the fixed cycle path's
count against the eol baseline (legit ball-break gaps only).

Usage:
    pixi run -e apple mojo run -I . examples/arcade_games/diag_atari_center_gap.mojo
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.cpu6502 import run_frame_video
from mojo_rl.envs.atari.flags import (
    ACTION_RIGHT,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FLAG_CON_RIGHT,
)
from mojo_rl.envs.atari.riot import set_action
from std.memory import alloc


def px_on(buf: Pointer[UInt8, MutAnyOrigin], x: Int, y: Int) -> Bool:
    if x < 0 or x >= FRAME_WIDTH or y < 0 or y >= FRAME_HEIGHT:
        return False
    var off = (y * FRAME_WIDTH + x) * 4
    return (Int(buf[off]) + Int(buf[off + 1]) + Int(buf[off + 2])) > 24


def band_lit(buf: Pointer[UInt8, MutAnyOrigin], x0: Int, y: Int) -> Bool:
    var n = 0
    for dx in range(8):
        if px_on(buf, x0 + dx, y):
            n += 1
    return n >= 6


def main() raises:
    var rom_data = load_rom("roms/breakout.bin")
    var env = AtariEnvironment(
        rom_data.data.value(), rom_data.size, frame_skip=1, max_frames=0
    )
    env.reset()
    # `.as_unsafe_any_origin()` at the SOURCE, not at each call: `alloc`
    # hands back `MutUntrackedOrigin`, and every helper below (and
    # `run_frame_video`) wants an Any origin. Converting once here is one
    # edit instead of one per call site.
    var buf = (
        alloc[UInt8]({count = FRAME_WIDTH * FRAME_HEIGHT * 4})
        .unsafe_leak()
        .as_unsafe_any_origin()
    )

    var gap_frames = 0
    var gap_scanlines = 0
    comptime MAX_STEPS = 1500

    for _ in range(MAX_STEPS):
        set_action(env.state, ACTION_RIGHT)
        env.state.paddle_pos = 8
        env.state.sys_flags = env.state.sys_flags | FLAG_CON_RIGHT
        run_frame_video(env.state,
            env.rom.as_unsafe_any_origin(),
            env.rom_size,
            buf,)

        var frame_gaps = 0
        for y in range(50, 100):
            # Center column hole: px72-79 OR px80-87 dark with both sides lit.
            var l = band_lit(buf, 64, y)
            var c1 = band_lit(buf, 72, y)
            var c2 = band_lit(buf, 80, y)
            var r = band_lit(buf, 88, y)
            if l and not c1 and c2:
                frame_gaps += 1
            elif c1 and not c2 and r:
                frame_gaps += 1
        if frame_gaps > 0:
            gap_frames += 1
            gap_scanlines += frame_gaps

    print("=== center-gap detector ===")
    print("frames scanned:      " + String(MAX_STEPS))
    print("frames with a gap:   " + String(gap_frames))
    print("total gap scanlines: " + String(gap_scanlines))
