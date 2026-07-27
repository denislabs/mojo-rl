"""Starpilot reset (add_spawners) parity vs the Qt-free C++ probe.

Validates the heavy reset that pre-schedules the whole episode's enemy waves — the
RNG crux of starpilot. Checks the spawner count, per-type counts, and an
ORDER-INDEPENDENT checksum of the spawner set (so std::sort ordering is irrelevant),
per mode × seed. Ground truth = `scratchpad/starpilot_reset_probe.cpp`.
Asset-free/fast. See `docs/PROCGEN_STARPILOT_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, TestSuite

from mojo_rl.envs.procgen.games import (
    StarpilotGame,
    FLYER,
    METEOR,
    CLOUD,
    TURRET,
    FAST_FLYER,
)
from mojo_rl.envs.procgen.games.starpilot import DIST_EASY, DIST_HARD


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var n_spawn: Int
    var fly: Int
    var met: Int
    var cld: Int
    var tur: Int
    var fast: Int
    var set_sig: Int


def test_starpilot_reset_parity() raises:
    var c = List[Expect]()
    # From starpilot_reset_probe: Easy (FLYER-only) + Hard (full mix) × 5 seeds.
    c.append(Expect(DIST_EASY, 0, 67, 67, 0, 0, 0, 0, -670687))
    c.append(Expect(DIST_HARD, 0, 45, 28, 2, 7, 1, 7, -300387))
    c.append(Expect(DIST_EASY, 1, 81, 81, 0, 0, 0, 0, -1336043))
    c.append(Expect(DIST_HARD, 1, 56, 43, 3, 2, 2, 6, -1689046))
    c.append(Expect(DIST_EASY, 7, 77, 77, 0, 0, 0, 0, -1400641))
    c.append(Expect(DIST_HARD, 7, 60, 24, 4, 0, 7, 25, -1893928))
    c.append(Expect(DIST_EASY, 42, 73, 73, 0, 0, 0, 0, -983129))
    c.append(Expect(DIST_HARD, 42, 49, 30, 3, 6, 6, 4, -931666))
    c.append(Expect(DIST_EASY, 123, 81, 81, 0, 0, 0, 0, -1648282))
    c.append(Expect(DIST_HARD, 123, 52, 23, 5, 3, 7, 14, -1275614))

    var easy = StarpilotGame(DIST_EASY)
    var hard = StarpilotGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _check(easy, e)
        else:
            _check(hard, e)


def _check(mut g: StarpilotGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)
    assert_equal(len(g.spawners), e.n_spawn, "n_spawn" + tag)

    var sig = 0
    var fly = 0
    var met = 0
    var cld = 0
    var tur = 0
    var fast = 0
    for i in range(len(g.spawners)):
        ref s = g.spawners[i]
        sig += (
            s.type * 1
            + s.spawn_time * 7
            + Int(floor(s.x * 100)) * 13
            + Int(floor(s.y * 100)) * 17
            + Int(floor(s.vx * 10000)) * 19
            + Int(floor(s.vy * 10000)) * 23
            + Int(s.health) * 29
            + s.image_theme * 31
            + s.fire_time * 41
        )
        if s.type == FLYER:
            fly += 1
        elif s.type == METEOR:
            met += 1
        elif s.type == CLOUD:
            cld += 1
        elif s.type == TURRET:
            tur += 1
        elif s.type == FAST_FLYER:
            fast += 1
    assert_equal(fly, e.fly, "fly" + tag)
    assert_equal(met, e.met, "met" + tag)
    assert_equal(cld, e.cld, "cld" + tag)
    assert_equal(tur, e.tur, "tur" + tag)
    assert_equal(fast, e.fast, "fast" + tag)
    assert_equal(sig, e.set_sig, "set_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
