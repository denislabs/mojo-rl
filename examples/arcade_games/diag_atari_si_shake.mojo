"""Headless vertical-stability check for Space Invaders (the "shake").

The shake = the whole image jumps vertically between frames. The score digits
at the top sit at a FIXED scanline, so the topmost lit row should be constant.
We render N frames and count how often that row (and the total visible scanline
count) changes — a stable image has near-zero changes.

Usage:
    pixi run -e apple mojo run -I . examples/arcade_games/diag_atari_si_shake.mojo
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.cpu6502 import run_frame_video
from mojo_rl.envs.atari.flags import (
    ACTION_NOOP,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)
from mojo_rl.envs.atari.riot import set_action
from std.memory import alloc


def px_on(buf: UnsafePointer[UInt8, MutAnyOrigin], x: Int, y: Int) -> Bool:
    var off = (y * FRAME_WIDTH + x) * 4
    return (Int(buf[off]) + Int(buf[off + 1]) + Int(buf[off + 2])) > 24


def top_lit_row(buf: UnsafePointer[UInt8, MutAnyOrigin]) -> Int:
    for y in range(FRAME_HEIGHT):
        for x in range(FRAME_WIDTH):
            if px_on(buf, x, y):
                return y
    return -1


def main() raises:
    var rom_data = load_rom("roms/space_invaders.bin")
    var env = AtariEnvironment(
        rom_data.data.value(), rom_data.size, frame_skip=1, max_frames=0
    )
    env.reset()
    var buf = alloc[UInt8](FRAME_WIDTH * FRAME_HEIGHT * 4)

    # Warm up past the title/attract transition.
    for _ in range(120):
        set_action(env.state, ACTION_NOOP)
        run_frame_video(env.state, env.rom, env.rom_size, buf)

    var prev_top = -2
    var prev_scan = -2
    var top_changes = 0
    var scan_changes = 0
    comptime N = 400
    for _ in range(N):
        set_action(env.state, ACTION_NOOP)
        run_frame_video(env.state, env.rom, env.rom_size, buf)
        var t = top_lit_row(buf)
        var s = Int(env.state.scanline)
        if prev_top >= 0 and t != prev_top:
            top_changes += 1
        if prev_scan >= 0 and s != prev_scan:
            scan_changes += 1
        prev_top = t
        prev_scan = s

    print("=== SI vertical stability ===")
    print("frames:                " + String(N))
    print("top-row changes:       " + String(top_changes) + " (0 = no shake)")
    print("scanline-count changes:" + String(scan_changes))
    print("final top row:         " + String(prev_top))
    print("final scanline count:  " + String(prev_scan))
