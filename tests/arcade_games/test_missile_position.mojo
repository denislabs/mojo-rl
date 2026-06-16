"""Divergence-free unit test: does the cycle MissileCounter render a missile at
the SAME pixel as the proven-correct end-of-line model (_resp_pos)?

Breakout draws its side walls (and Space Invaders its laser beam) with MISSILES,
so a position mismatch here = missing walls / beam-misses. We sweep RESM strobe
clocks and compare the cycle counter's first-lit pixel to eol's _resp_pos.

Run: pixi run -e apple mojo run -I . tests/arcade_games/test_missile_position.mojo
"""

from mojo_rl.envs.atari.tia_cycle import MissileCounter, resx_counter
from mojo_rl.envs.atari.tia import _resp_pos
from mojo_rl.envs.atari.flags import HBLANK_CLOCKS, FRAME_WIDTH


def cycle_missile_first_lit(write_hctr: Int) -> Int:
    """First visible pixel (0-159) lit by a width-1 missile RESM'd at write_hctr.

    Mirrors run_frame_cycle_accurate EXACTLY: the counter is a free-running
    beam-synchronized counter. RESM is applied at its actual hctr (preserving the
    counter PHASE relative to the beam), then the counter ticks once per VISIBLE
    color clock (hctr 68-227) and is NOT ticked during HBLANK. We run three lines
    and read the steady-state (third line) position."""
    var m = MissileCounter()
    m.set_nusiz(0)  # width 1, one copy
    m.set_enam(0x02)  # enabled
    var in_hblank = write_hctr < HBLANK_CLOCKS
    var first = -1
    for line in range(3):
        for hctr in range(228):  # full 228-clock line
            if line == 0 and hctr == write_hctr:
                m.resm(resx_counter(write_hctr, in_hblank))
            if hctr >= HBLANK_CLOCKS:  # counter ticks only in visible region
                var lit = m.tick()
                if line == 2 and lit and first < 0:
                    first = hctr - HBLANK_CLOCKS
    return first


def main() raises:
    # The cycle MissileCounter must track the RESM strobe position like the
    # proven end-of-line model. eol's _resp_pos is a heuristic (+5, "approximate"
    # per its docstring); the cycle path uses Stella's exact decode@156/offset-4,
    # so a consistent <=1px difference is expected and acceptable (the wall/beam
    # missiles are several px wide). A LARGER drift means the position mechanism
    # is broken (the bug this guards against).
    var fails = 0
    for start in range(64, 200, 4):  # visible-region strobes
        var write_hctr = start + 6
        var eol = Int(_resp_pos(start))
        var cyc = cycle_missile_first_lit(write_hctr)
        var d = eol - cyc
        if d < 0:
            d = -d
        if d > 1:
            print(
                "FAIL start=" + String(start) + " eol=" + String(eol)
                + " cycle=" + String(cyc) + " diff=" + String(eol - cyc)
            )
            fails += 1
    if fails == 0:
        print("PASS: cycle missile position tracks eol within 1px across sweep")
    else:
        print("FAILED: " + String(fails) + " positions drift >1px")
