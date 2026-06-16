"""Headless vertical-stability check for Space Invaders (the "shake").

The shake = the whole image jumps vertically between frames. The score digits
at the top sit at a FIXED scanline, so the topmost lit row should be constant.
We render frames in BOTH attract mode and real gameplay (RESET to start, then
fire/move), and track three per-frame geometry stats:

  top lit row        — where content lands in the framebuffer (jumps = shake)
  dbg_frame_lines    — total scanlines VSYNC→VSYNC (frame timing stability)
  dbg_ystart         — line (from VSYNC) where the game released VBLANK

If top row jumps while frame_lines is constant and ystart jitters, the shake
is an ALIGNMENT artifact: rows are anchored to VBLANK-release instead of
VSYNC (a real TV anchors to VSYNC; VBLANK only blanks the beam).

Usage:
    pixi run -e apple mojo run -I . examples/arcade_games/diag_atari_si_shake.mojo
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.cpu6502 import run_frame_video
from mojo_rl.envs.atari.flags import (
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_RESET,
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


struct GeomStats(Copyable, Movable):
    var top_changes: Int
    var line_changes: Int
    var ystart_changes: Int
    var top_min: Int
    var top_max: Int
    var lines_min: Int
    var lines_max: Int
    var ystart_min: Int
    var ystart_max: Int
    var prev_top: Int
    var prev_lines: Int
    var prev_ystart: Int

    def __init__(out self):
        self.top_changes = 0
        self.line_changes = 0
        self.ystart_changes = 0
        self.top_min = 99999
        self.top_max = -1
        self.lines_min = 99999
        self.lines_max = -1
        self.ystart_min = 99999
        self.ystart_max = -1
        self.prev_top = -2
        self.prev_lines = -2
        self.prev_ystart = -2

    def record(mut self, top: Int, lines: Int, ystart: Int):
        if self.prev_top >= 0 and top != self.prev_top:
            self.top_changes += 1
        if self.prev_lines >= 0 and lines != self.prev_lines:
            self.line_changes += 1
        if self.prev_ystart >= 0 and ystart != self.prev_ystart:
            self.ystart_changes += 1
        self.prev_top = top
        self.prev_lines = lines
        self.prev_ystart = ystart
        self.top_min = min(self.top_min, top)
        self.top_max = max(self.top_max, top)
        self.lines_min = min(self.lines_min, lines)
        self.lines_max = max(self.lines_max, lines)
        self.ystart_min = min(self.ystart_min, ystart)
        self.ystart_max = max(self.ystart_max, ystart)

    def show(self, name: String, n: Int):
        print("--- " + name + " (" + String(n) + " frames) ---")
        print(
            "top row:     changes="
            + String(self.top_changes)
            + " range=["
            + String(self.top_min)
            + ","
            + String(self.top_max)
            + "]   (0 changes = no shake)"
        )
        print(
            "frame lines: changes="
            + String(self.line_changes)
            + " range=["
            + String(self.lines_min)
            + ","
            + String(self.lines_max)
            + "]"
        )
        print(
            "ystart:      changes="
            + String(self.ystart_changes)
            + " range=["
            + String(self.ystart_min)
            + ","
            + String(self.ystart_max)
            + "]"
        )


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

    print("=== SI vertical stability ===")

    # Phase 1: attract mode, NOOP.
    var attract = GeomStats()
    comptime N_ATTRACT = 300
    for _ in range(N_ATTRACT):
        set_action(env.state, ACTION_NOOP)
        run_frame_video(env.state, env.rom, env.rom_size, buf)
        attract.record(
            top_lit_row(buf),
            Int(env.state.dbg_frame_lines),
            Int(env.state.dbg_ystart),
        )
    attract.show("attract (NOOP)", N_ATTRACT)

    # Start the game: hold the console RESET switch a few frames.
    for _ in range(4):
        set_action(env.state, ACTION_RESET)
        run_frame_video(env.state, env.rom, env.rom_size, buf)
    for _ in range(30):
        set_action(env.state, ACTION_NOOP)
        run_frame_video(env.state, env.rom, env.rom_size, buf)

    # Phase 2: gameplay — alternate fire and movement like a player would.
    var play = GeomStats()
    comptime N_PLAY = 900
    for i in range(N_PLAY):
        var phase = (i // 15) % 4
        var a = ACTION_FIRE if phase == 0 else (
            ACTION_LEFT if phase == 1 else (
                ACTION_FIRE if phase == 2 else ACTION_RIGHT
            )
        )
        set_action(env.state, a)
        run_frame_video(env.state, env.rom, env.rom_size, buf)
        play.record(
            top_lit_row(buf),
            Int(env.state.dbg_frame_lines),
            Int(env.state.dbg_ystart),
        )
    play.show("gameplay (fire+move)", N_PLAY)
    print("final score: " + String(env.state.score))
