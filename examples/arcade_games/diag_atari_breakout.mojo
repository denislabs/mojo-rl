"""Headless Breakout phantom-brick chaser.

Drives Breakout without a window and hunts for the "broke a brick I wasn't
supposed to break" bug. It serves the ball via the paddle trigger (SWCHA D7 —
note: ACTION_FIRE/INPT4 does NOT serve Breakout, only the paddle button does),
drives the paddle directly to track the ball, and each frame scans the rendered
frame for the brick grid. It flags a cell only when it is removed AND scored
(score delta > 0) AND was 8-connected enclosed by intact bricks — a physically
impossible removal.

Crucially, `brick_cell` requires a brick to fill most of its 8px width so the
~2px ball is NOT mistaken for a brick when it sits inside an empty cell — that
mistake makes a ball rising through a column gap (legally hitting a brick from
below) look like an enclosed-brick break.

Result (60k frames, multiple full games): 0 enclosed-brick removals. The cases
that look impossible are the ball reaching a brick through a gap.

Usage:
    pixi run -e apple mojo run -I . examples/arcade_games/diag_atari_breakout.mojo
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.cpu6502 import run_frame_video
from mojo_rl.envs.atari.games.breakout import BreakoutDef
from mojo_rl.envs.atari.flags import (
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_LEFT,
    ACTION_RIGHT,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)
from mojo_rl.envs.atari.riot import set_action
from std.memory import alloc


def px_on(buf: Pointer[UInt8, MutAnyOrigin], x: Int, y: Int) -> Bool:
    """True if pixel (x,y) is non-background (any color channel lit)."""
    var off = (y * FRAME_WIDTH + x) * 4
    var b = Int(buf[off + 0])
    var g = Int(buf[off + 1])
    var r = Int(buf[off + 2])
    # Breakout background is black; bricks/walls/ball/paddle are colored.
    return (r + g + b) > 60


def brick_cell(buf: Pointer[UInt8, MutAnyOrigin], col: Int, row: Int) -> Bool:
    """Sample the centre of brick cell (col,row). 18 cols x 6 rows.

    Breakout brick field: x ~ [8,152), y ~ [57,93). Each brick ~8px wide, 6px
    tall. A real brick fills its full 8px width; the ball is only ~2px wide, so
    requiring most of the cell's width to be lit rejects the ball (which would
    otherwise masquerade as a brick when it sits inside an empty cell).
    """
    var by = 57 + row * 6 + 3
    var x0 = 8 + col * 8
    if x0 < 0 or x0 + 8 > FRAME_WIDTH or by < 0 or by >= FRAME_HEIGHT:
        return False
    var lit = 0
    for dx in range(8):
        if px_on(buf, x0 + dx, by):
            lit += 1
    return lit >= 6


def find_ball_y(buf: Pointer[UInt8, MutAnyOrigin], ball_x: Int) -> Int:
    """Scan column ball_x below the brick field for the ball; -1 if not found."""
    if ball_x < 0 or ball_x >= FRAME_WIDTH:
        return -1
    for y in range(93, 188):
        if px_on(buf, ball_x, y):
            return y
    return -1


def find_paddle_center(buf: Pointer[UInt8, MutAnyOrigin]) -> Int:
    """Find the paddle's screen-x center by scanning the bottom band."""
    var best_len = 0
    var best_center = -1
    for y in range(185, 205):
        var run = 0
        var run_start = 0
        for x in range(8, 152):
            if px_on(buf, x, y):
                if run == 0:
                    run_start = x
                run += 1
            else:
                if run > best_len:
                    best_len = run
                    best_center = run_start + run // 2
                run = 0
        if run > best_len:
            best_len = run
            best_center = run_start + run // 2
    return best_center


def dump_ascii(buf: Pointer[UInt8, MutAnyOrigin], label: String):
    """Coarse ASCII dump of the whole frame (every 2px x, 4px y)."""
    print("--- ASCII frame: " + label + " ---")
    for y in range(0, FRAME_HEIGHT, 4):
        var line = String("")
        for x in range(0, FRAME_WIDTH, 2):
            line += "#" if px_on(buf, x, y) else "."
        print(String(y) + ": " + line)


def main() raises:
    var rom_path = "roms/breakout.bin"
    print("Loading ROM: " + rom_path)
    var rom_data = load_rom(rom_path)
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

    comptime COLS = 18
    comptime ROWS = 6
    var prev = InlineArray[Bool, COLS * ROWS](fill=False)
    var have_prev = False

    var phantom_count = 0
    var flicker_count = 0
    var prev_score = 0
    var step = 0
    comptime MAX_STEPS = 60000

    # --- PROBE PHASE: find which action serves the ball ---
    # Try each pure action held for 120 frames; report if the ball appears
    # (enabl/enabl_old nonzero) or the score rises.
    for pi in range(4):
        var pa = ACTION_NOOP
        var pn = String("NOOP")
        if pi == 1:
            pa = ACTION_FIRE
            pn = "FIRE"
        elif pi == 2:
            pa = ACTION_LEFT
            pn = "LEFT"
        elif pi == 3:
            pa = ACTION_RIGHT
            pn = "RIGHT"
        env.reset()
        var served = False
        for _ in range(120):
            set_action(env.state, pa)
            run_frame_video(env.state,
            env.rom.as_unsafe_any_origin(),
            env.rom_size,
            buf,)
            if Int(env.state.enabl) != 0 or Int(env.state.enabl_old) != 0:
                served = True
        print(
            "PROBE hold "
            + pn
            + ": served="
            + String(served)
            + " enabl="
            + String(Int(env.state.enabl))
            + " enabl_old="
            + String(Int(env.state.enabl_old))
            + " pos_bl="
            + String(Int(env.state.pos_bl))
            + " score="
            + String(BreakoutDef.get_score(env.state.ram))
            + " lives="
            + String(BreakoutDef.get_lives(env.state.ram))
        )
    print("--- end probe ---")
    env.reset()

    # Auto-player state. The serve trigger is the paddle button on SWCHA D7
    # (== FLAG_CON_RIGHT line); paddle position is driven DIRECTLY via paddle_pos
    # so the trigger and the knob are decoupled. We learn the sign of the
    # paddle_pos→screen-x mapping at runtime from observed motion.
    from mojo_rl.envs.atari.flags import FLAG_CON_RIGHT

    var paddle_target: Int = 128  # paddle_pos we command
    var last_paddle_pos: Int = 128
    var last_center: Int = -1
    var sign: Int = -1  # assume decreasing paddle_pos moves paddle right; refine

    while step < MAX_STEPS:
        var ball_x = Int(env.state.pos_bl)
        var ball_y = find_ball_y(buf, ball_x)
        var in_play = (
            Int(env.state.enabl) != 0
            or Int(env.state.enabl_old) != 0
            or ball_y >= 0
        )

        # Clear all controller flags (paddle_pos untouched by NOOP).
        set_action(env.state, ACTION_NOOP)

        if not in_play:
            # Pulse the serve trigger (SWCHA D7) without moving the knob.
            env.state.sys_flags = env.state.sys_flags | FLAG_CON_RIGHT
        else:
            # Track the ball: move paddle_pos toward making paddle_center==ball_x.
            var pc = find_paddle_center(buf)
            # Refine the sign from the last commanded move, if we have data.
            if last_center >= 0 and pc >= 0 and paddle_target != last_paddle_pos:
                var dpos = paddle_target - last_paddle_pos
                var dcen = pc - last_center
                if dcen != 0:
                    sign = 1 if (dcen * dpos) > 0 else -1
            if pc >= 0:
                var err = ball_x - pc
                if err > 2:
                    paddle_target += sign * 6
                elif err < -2:
                    paddle_target -= sign * 6
            last_center = pc
        last_paddle_pos = paddle_target
        if paddle_target < 0:
            paddle_target = 0
        if paddle_target > 255:
            paddle_target = 255
        env.state.paddle_pos = UInt8(paddle_target)

        run_frame_video(env.state,
            env.rom.as_unsafe_any_origin(),
            env.rom_size,
            buf,)
        step += 1

        if step == 60 or step == 300 or step == 1200:
            dump_ascii(buf, "step " + String(step))
            print(
                "  ball pos_bl="
                + String(Int(env.state.pos_bl))
                + " enabl="
                + String(Int(env.state.enabl))
                + " paddle_pos="
                + String(Int(env.state.paddle_pos))
                + " score="
                + String(BreakoutDef.get_score(env.state.ram))
                + " lives="
                + String(BreakoutDef.get_lives(env.state.ram))
            )

        env.state.terminal = BreakoutDef.is_terminal(env.state.ram)
        if env.state.terminal:
            env.reset()
            have_prev = False
            prev_score = 0
            continue

        var cur_score = BreakoutDef.get_score(env.state.ram)
        var score_delta = cur_score - prev_score
        prev_score = cur_score

        # Build current brick grid.
        var cur = InlineArray[Bool, COLS * ROWS](fill=False)
        for r in range(ROWS):
            for c in range(COLS):
                cur[r * COLS + c] = brick_cell(buf, c, r)

        if have_prev:
            for r in range(ROWS):
                for c in range(COLS):
                    var idx = r * COLS + c
                    # A cell that just turned off...
                    if prev[idx] and not cur[idx]:
                        # TRULY surrounded = all 4 orthogonal neighbours still ON
                        # AFTER the break. The ball cannot be adjacent to such a
                        # cell, so removing it is physically impossible. (Normal
                        # hits from above/below leave the up/down neighbour empty
                        # where the ball came from.)
                        # 8-connected enclosure: all 8 neighbours (incl. the 4
                        # diagonals) still ON. A ball cannot even corner-clip a
                        # cell whose diagonals are all bricks, so removing it is
                        # genuinely impossible.
                        var interior = (
                            c > 0
                            and c < COLS - 1
                            and r > 0
                            and r < ROWS - 1
                        )
                        if interior:
                            for dr in range(-1, 2):
                                for dc in range(-1, 2):
                                    if dr == 0 and dc == 0:
                                        continue
                                    if not cur[(r + dr) * COLS + (c + dc)]:
                                        interior = False

                        if interior and score_delta <= 0:
                            # Surrounded cell vanished but no brick was scored →
                            # 1-frame render flicker, not a real removal.
                            flicker_count += 1
                        if interior and score_delta > 0:
                            phantom_count += 1
                            var ball_x = Int(env.state.pos_bl)
                            var ball_y = find_ball_y(buf, ball_x)
                            print("")
                            print("### PHANTOM BRICK #" + String(phantom_count))
                            print(
                                "  step="
                                + String(step)
                                + " cell=(col "
                                + String(c)
                                + ", row "
                                + String(r)
                                + ")"
                            )
                            print(
                                "  ball pos_bl="
                                + String(ball_x)
                                + " ball_y(screen)="
                                + String(ball_y)
                                + " enabl="
                                + String(Int(env.state.enabl))
                                + " enabl_old="
                                + String(Int(env.state.enabl_old))
                            )
                            var cell_top = 57 + r * 6
                            var cell_left = 8 + c * 8
                            print(
                                "  cell pixel box: x["
                                + String(cell_left)
                                + ","
                                + String(cell_left + 8)
                                + ") y["
                                + String(cell_top)
                                + ","
                                + String(cell_top + 6)
                                + ")"
                            )
                            print(
                                "  collision=0x"
                                + hex(Int(env.state.collision))
                                + " ctrlpf=0x"
                                + hex(Int(env.state.ctrlpf))
                                + " score_delta="
                                + String(score_delta)
                            )
                            # Dump the brick grid (X=brick, .=gone, *=this cell).
                            print("  grid (this frame):")
                            for rr in range(ROWS):
                                var gl = String("    ")
                                for cc in range(COLS):
                                    if rr == r and cc == c:
                                        gl += "*"
                                    else:
                                        gl += "X" if cur[rr * COLS + cc] else "."
                                print(gl)

        for i in range(COLS * ROWS):
            prev[i] = cur[i]
        have_prev = True

        if step % 5000 == 0:
            print(
                "step="
                + String(step)
                + " score="
                + String(BreakoutDef.get_score(env.state.ram))
                + " lives="
                + String(BreakoutDef.get_lives(env.state.ram))
                + " phantoms="
                + String(phantom_count)
                + " flickers="
                + String(flicker_count)
            )

    print("")
    print(
        "DONE. steps="
        + String(step)
        + " final_score="
        + String(BreakoutDef.get_score(env.state.ram))
        + " phantom_total="
        + String(phantom_count)
        + " flicker_total="
        + String(flicker_count)
    )
    buf.free()

