"""Stage 1 tests for cycle-accurate TIA primitives (tia_cycle.mojo).

Validates the building blocks before they are wired into a renderer:
  - resx_counter() strobe mapping vs Stella TIA::resxCounter
  - DelayQueue ordering / per-entry latency
  - BallCounter decode + lit-window width + counter wrap + HMBL/CTRLPF decode

Run: pixi run -e apple mojo run -I . tests/arcade_games/test_tia_cycle.mojo
"""

from mojo_rl.envs.atari.tia_cycle import (
    DelayQueue,
    BallCounter,
    PlayerCounter,
    MissileCounter,
    decode_copy,
    resx_counter,
    tia_write_clock,
    RESX_HBLANK,
    RESX_LATE_HBLANK,
    RESX_FRAME,
)


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("FAIL: " + msg)
    print("  ok: " + msg)


def test_resx_counter() raises:
    print("test_resx_counter")
    expect(resx_counter(10, True) == RESX_HBLANK, "normal hblank -> 159")
    expect(resx_counter(80, True) == RESX_LATE_HBLANK, "late hblank -> 158")
    expect(resx_counter(100, False) == RESX_FRAME, "visible -> 157")
    expect(resx_counter(72, True) == RESX_HBLANK, "hctr 72 hblank -> 159")
    expect(resx_counter(73, True) == RESX_LATE_HBLANK, "hctr 73 -> 158")


def test_write_clock() raises:
    print("test_write_clock")
    # STA zp = 3 cycles -> write 2 CPU cycles (6 color clocks) after start.
    expect(tia_write_clock(30, 3) == 36, "STA zp: 30 + 6 = 36 (legacy +6)")
    # STA abs = 4 cycles -> +9 color clocks.
    expect(tia_write_clock(30, 4) == 39, "STA abs: 30 + 9 = 39")
    # STA abs,X = 5 cycles -> +12.
    expect(tia_write_clock(0, 5) == 12, "STA abs,X: +12")
    # STA (zp),Y = 6 cycles -> +15.
    expect(tia_write_clock(100, 6) == 115, "STA (zp),Y: +15")


def test_delay_queue() raises:
    print("test_delay_queue")
    var q = DelayQueue()
    # delay=0 fires on the first collect; delay=1 on the second, etc.
    q.push(0x1F, 0xAA, 0)
    q.push(0x1B, 0xBB, 2)
    expect(q.pending() == 2, "two pending after push")

    var r0 = List[UInt8]()
    var v0 = List[UInt8]()
    q.cycle_collect(r0, v0)
    expect(len(r0) == 1, "delay0 fires on cycle 1")
    expect(r0[0] == 0x1F and v0[0] == 0xAA, "delay0 reg/value correct")

    var r1 = List[UInt8]()
    var v1 = List[UInt8]()
    q.cycle_collect(r1, v1)
    expect(len(r1) == 0, "nothing on cycle 2 (delay2 not yet)")

    var r2 = List[UInt8]()
    var v2 = List[UInt8]()
    q.cycle_collect(r2, v2)
    expect(len(r2) == 1, "delay2 fires on cycle 3")
    expect(r2[0] == 0x1B and v2[0] == 0xBB, "delay2 reg/value correct")
    expect(q.pending() == 0, "queue empty after all fired")


def _lit_window_len(mut b: BallCounter, ticks: Int) -> Int:
    """Count lit clocks over `ticks` ticks (assumes a single contiguous window)."""
    var lit = 0
    for _ in range(ticks):
        if b.tick():
            lit += 1
    return lit


def test_ball_decode_width() raises:
    print("test_ball_decode_width")
    # Natural decode path: start a few clocks before 156, sweep a full line.
    for w_bits in range(4):
        var ctrlpf = UInt8(w_bits << 4)
        var b = BallCounter()
        b.set_width_from_ctrlpf(ctrlpf)
        b.set_enabl_new(True)
        b.counter = 150  # will pass through 156 during the sweep
        var expected = 1 << w_bits
        var lit = _lit_window_len(b, 160)
        expect(
            lit == expected,
            "width bits=" + String(w_bits) + " -> " + String(expected)
            + " lit clocks (got " + String(lit) + ")",
        )


def test_ball_resbl_width() raises:
    print("test_ball_resbl_width")
    # RESBL in the visible frame (strobe counter 157) then sweep: ball must light
    # for exactly `width` contiguous clocks.
    for w_bits in range(4):
        var b = BallCounter()
        b.set_width_from_ctrlpf(UInt8(w_bits << 4))
        b.set_enabl_new(True)
        b.resbl(RESX_FRAME)
        var expected = 1 << w_bits
        var lit = _lit_window_len(b, 24)
        expect(
            lit == expected,
            "resbl width bits=" + String(w_bits) + " -> " + String(expected)
            + " (got " + String(lit) + ")",
        )


def test_ball_counter_wrap() raises:
    print("test_ball_counter_wrap")
    var b = BallCounter()
    b.counter = 0
    for _ in range(H_pixel_ticks()):
        _ = b.tick()
    expect(b.counter == 0, "counter wraps back to 0 after 160 ticks")


def H_pixel_ticks() -> Int:
    return 160


def test_ball_hmbl_decode() raises:
    print("test_ball_hmbl_decode")
    var b = BallCounter()
    b.set_hmbl(0x00)  # (0>>4)^0x08 = 8 -> no movement
    expect(b.hmm_clocks == 8, "HMBL 0x00 -> 8 (centre, no move)")
    b.set_hmbl(0x70)  # (7)^8 = 15 -> max left
    expect(b.hmm_clocks == 15, "HMBL 0x70 -> 15")
    b.set_hmbl(0x80)  # (8)^8 = 0 -> max right
    expect(b.hmm_clocks == 0, "HMBL 0x80 -> 0")


def _player_lit(mut p: PlayerCounter, ticks: Int) -> Int:
    var lit = 0
    for _ in range(ticks):
        if p.tick():
            lit += 1
    return lit


def _missile_lit(mut m: MissileCounter, ticks: Int) -> Int:
    var lit = 0
    for _ in range(ticks):
        if m.tick():
            lit += 1
    return lit


def test_decode_copy() raises:
    print("test_decode_copy")
    expect(decode_copy(0, 156) == 1, "single copy decodes at 156")
    expect(decode_copy(0, 12) == 0, "nusiz0 no copy at 12")
    expect(decode_copy(1, 12) == 2, "nusiz1 second copy at 12 (+16)")
    expect(decode_copy(2, 28) == 2, "nusiz2 second copy at 28 (+32)")
    expect(decode_copy(3, 12) == 2 and decode_copy(3, 28) == 3, "nusiz3 +16,+32")
    expect(decode_copy(4, 60) == 2, "nusiz4 second copy at 60 (+64)")
    expect(decode_copy(6, 28) == 2 and decode_copy(6, 60) == 3, "nusiz6 +32,+64")


def test_player_pattern_width() raises:
    print("test_player_pattern_width")
    # Full pattern, single copy: divider 1->8 lit, double(5)->16, quad(7)->32.
    for nz in [0, 5, 7]:
        var p = PlayerCounter()
        p.set_nusiz(UInt8(nz))
        p.set_grp_new(0xFF)
        p.counter = 150
        var lit = _player_lit(p, 160)
        var stretch = 1 if nz == 0 else (2 if nz == 5 else 4)
        expect(
            lit == 8 * stretch,
            "nusiz=" + String(nz) + " full pattern -> " + String(8 * stretch)
            + " lit (got " + String(lit) + ")",
        )
    # Empty pattern -> never lit.
    var pe = PlayerCounter()
    pe.set_grp_new(0x00)
    pe.counter = 150
    expect(_player_lit(pe, 160) == 0, "empty GRP -> 0 lit")
    # One bit set -> exactly 1 lit clock at divider 1.
    var p1 = PlayerCounter()
    p1.set_grp_new(0x80)  # leftmost pixel only (non-reflected: GRP bit 7 first)
    p1.counter = 150
    expect(_player_lit(p1, 160) == 1, "single GRP bit -> 1 lit")


def test_player_two_copies() raises:
    print("test_player_two_copies")
    var p = PlayerCounter()
    p.set_nusiz(1)  # two copies close
    p.set_grp_new(0xFF)
    p.counter = 0
    # Sweep a full line: two 8-wide copies = 16 lit clocks.
    expect(_player_lit(p, 170) == 16, "two copies full pattern -> 16 lit")


def test_missile_width() raises:
    print("test_missile_width")
    for wb in range(4):
        var m = MissileCounter()
        m.set_nusiz(UInt8(wb << 4))  # width bits
        m.set_enam(0x02)  # enabled
        m.counter = 150
        var expected = 1 << wb
        var lit = _missile_lit(m, 160)
        expect(
            lit == expected,
            "missile width bits=" + String(wb) + " -> " + String(expected)
            + " (got " + String(lit) + ")",
        )
    # Disabled -> never lit; RESMP -> suppressed.
    var md = MissileCounter()
    md.set_nusiz(0x00)
    md.counter = 150
    expect(_missile_lit(md, 160) == 0, "disabled missile -> 0 lit")
    var mr = MissileCounter()
    mr.set_nusiz(0x00)
    mr.set_enam(0x02)
    mr.set_resmp(0x02)  # locked to player -> decode suppressed
    mr.counter = 150
    expect(_missile_lit(mr, 160) == 0, "resmp-locked missile -> 0 lit")


def main():
    try:
        test_resx_counter()
        test_write_clock()
        test_decode_copy()
        test_delay_queue()
        test_ball_decode_width()
        test_ball_resbl_width()
        test_ball_counter_wrap()
        test_ball_hmbl_decode()
        test_player_pattern_width()
        test_player_two_copies()
        test_missile_width()
        print("ALL TIA-CYCLE TESTS PASSED")
    except e:
        print(String(e))
        print("TESTS FAILED")
