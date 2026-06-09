"""Headless diagnostic for Atari Space Invaders emulator accuracy.

Dumps:
  - INPT4 (fire button) read with and without FIRE held
  - Per-visible-scanline COLUBK / COLUP0 / COLUPF distribution over a frame
  - VSYNC/VBLANK transitions and visible-line count per frame

Run:
    pixi run -e apple mojo run -I . examples/arcade_games/diag_atari_si.mojo
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.riot import set_action
from mojo_rl.envs.atari.cpu6502 import _run_scanline
from mojo_rl.envs.atari.tia import tia_read
from mojo_rl.envs.atari.games.space_invaders import SpaceInvadersDef
from mojo_rl.envs.atari.flags import (
    ACTION_NOOP,
    ACTION_FIRE,
    TOTAL_SCANLINES,
    CPU_CLOCKS_PER_LINE,
    FRAME_HEIGHT,
    TIA_VBLANK,
    TIA_VSYNC,
    TIA_PADDLE_GROUND,
)


def diag_frame(mut env: AtariEnvironment, label: String):
    """Run one frame manually, recording per-scanline color + structure."""
    var visible_line = 0
    var overflow = Int(env.state.cpu_cycles)
    var rendered_any = False
    var prev_vsync = (env.state.tia_flags & TIA_VSYNC) != 0

    var visible_count = 0
    var vsync_lines = 0
    var vblank_lines = 0
    var bk_min = 255
    var bk_max = 0
    var first_bk = -1
    var nonzero_bk = 0

    for _ in range(TOTAL_SCANLINES * 2):
        if (
            env.state.paddle_charge < 255
            and (env.state.tia_flags & TIA_PADDLE_GROUND) == 0
        ):
            env.state.paddle_charge += 1

        var total = _run_scanline(env.state, env.rom, env.rom_size, overflow)
        overflow = total - CPU_CLOCKS_PER_LINE

        var vsync_now = (env.state.tia_flags & TIA_VSYNC) != 0
        var vsync_rising = vsync_now and not prev_vsync
        prev_vsync = vsync_now
        if vsync_now:
            vsync_lines += 1
        if vsync_rising:
            if rendered_any:
                break
            visible_line = 0
        if (env.state.tia_flags & TIA_VBLANK) != 0:
            vblank_lines += 1

        if (env.state.tia_flags & TIA_VBLANK) == 0 and visible_line < FRAME_HEIGHT:
            var bk = Int(env.state.colubk)
            if first_bk < 0:
                first_bk = bk
            if bk < bk_min:
                bk_min = bk
            if bk > bk_max:
                bk_max = bk
            if bk != 0:
                nonzero_bk += 1
            visible_count += 1
            visible_line += 1
            rendered_any = True

    env.state.scanline = UInt16(visible_line)
    env.state.cpu_cycles = UInt32(overflow)
    env.state.frame_number += 1

    print(
        label
        + " visible="
        + String(visible_count)
        + " vsync_lines="
        + String(vsync_lines)
        + " vblank_lines="
        + String(vblank_lines)
        + " COLUBK[first="
        + String(first_bk)
        + " min="
        + String(bk_min)
        + " max="
        + String(bk_max)
        + " nonzero="
        + String(nonzero_bk)
        + "] COLUP0="
        + String(Int(env.state.colup0))
        + " COLUPF="
        + String(Int(env.state.colubk))
        + " score="
        + String(SpaceInvadersDef.get_score(env.state.ram))
    )


def main() raises:
    var rom_path = "roms/space_invaders.bin"
    var rom_data = load_rom(rom_path)
    print("ROM: " + String(rom_data.size) + " bytes")

    var env = AtariEnvironment(
        rom_data.data.value(), rom_data.size, frame_skip=1, max_frames=0
    )
    env.reset()

    # --- Fire button read test ---
    set_action(env.state, ACTION_NOOP)
    var inpt4_noop = Int(tia_read(env.state, 0x0C))
    set_action(env.state, ACTION_FIRE)
    var inpt4_fire = Int(tia_read(env.state, 0x0C))
    print(
        "INPT4 (fire button bit7): NOOP="
        + String(inpt4_noop)
        + " (expect 128 = released)  FIRE="
        + String(inpt4_fire)
        + " (expect 0 = pressed)"
    )

    # --- Per-frame color / structure dump over ~20 frames ---
    print("")
    print("Per-frame structure (look for frame-to-frame COLUBK flicker):")
    set_action(env.state, ACTION_NOOP)
    for i in range(20):
        diag_frame(env, "frame " + String(i))
