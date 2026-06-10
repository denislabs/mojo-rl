"""Property tests for the TIA bulk fast path (tia_cycle advance_n/lit_horizon).

The headless cycle loop's bulk path relies on two contracts per object counter:

  1. advance_n(n) leaves the object in EXACTLY the state n tick() calls would
     (lit output discarded), for any reachable state and static config.
  2. lit_horizon() = h means the next h ticks all produce lit == False.

Both are exercised with a randomized state walk (config mutations + strobes +
mixed advance sizes) against the per-tick reference. Any divergence here would
silently break collision timing in bulk-advanced spans.

Usage:
    pixi run mojo run -I . tests/arcade_games/test_tia_advance_n.mojo
"""

from mojo_rl.envs.atari.tia_cycle import (
    BallCounter,
    MissileCounter,
    PlayerCounter,
)


comptime ROUNDS = 20000


@always_inline
def _lcg(mut rng: UInt64) -> UInt64:
    rng = rng * 6364136223846793005 + 1442695040888963407
    return rng >> 24


def test_ball() raises:
    var rng: UInt64 = 0x9E3779B97F4A7C15
    var gold = BallCounter()
    for round in range(ROUNDS):
        var r = _lcg(rng)
        if r % 7 == 0:
            gold.set_width_from_ctrlpf(UInt8((r >> 8) & 0x30))
        if r % 11 == 0:
            gold.set_enabl_new(((r >> 4) & 1) != 0)
        if r % 13 == 0:
            gold.set_vdel(((r >> 5) & 1) != 0)
        if r % 17 == 0:
            gold.shuffle()
        if r % 19 == 0:
            gold.resbl(157 + Int((r >> 10) % 3))

        # lit_horizon contract: no lit within the horizon.
        var h = min(gold.lit_horizon(), 400)
        var probe = gold.copy()
        for i in range(h):
            if probe.tick():
                raise Error(
                    "ball lit at tick "
                    + String(i)
                    + " within horizon "
                    + String(h)
                    + " (round "
                    + String(round)
                    + ")"
                )

        # advance_n contract: identical end state to n ticks.
        var v = 1 + Int((r >> 16) % 400)
        var fast = gold.copy()
        fast.advance_n(v)
        for _ in range(v):
            _ = gold.tick()
        if (
            fast.counter != gold.counter
            or fast.render_counter != gold.render_counter
            or fast.is_rendering != gold.is_rendering
        ):
            raise Error(
                "ball advance_n("
                + String(v)
                + ") diverged at round "
                + String(round)
                + ": counter "
                + String(fast.counter)
                + "/"
                + String(gold.counter)
                + " rc "
                + String(fast.render_counter)
                + "/"
                + String(gold.render_counter)
            )
    print("ball: advance_n + lit_horizon OK (" + String(ROUNDS) + " rounds)")


def test_missile() raises:
    var rng: UInt64 = 0xDEADBEEFCAFEBABE
    var gold = MissileCounter()
    for round in range(ROUNDS):
        var r = _lcg(rng)
        if r % 7 == 0:
            gold.set_nusiz(UInt8((r >> 8) & 0x37))
        if r % 11 == 0:
            gold.set_enam(UInt8((r >> 4) & 0x02))
        if r % 13 == 0:
            gold.set_resmp(UInt8((r >> 5) & 0x02))
        if r % 19 == 0:
            gold.resm(157 + Int((r >> 10) % 3))

        var h = min(gold.lit_horizon(), 400)
        var probe = gold.copy()
        for i in range(h):
            if probe.tick():
                raise Error(
                    "missile lit at tick "
                    + String(i)
                    + " within horizon "
                    + String(h)
                    + " (round "
                    + String(round)
                    + ")"
                )

        var v = 1 + Int((r >> 16) % 400)
        var fast = gold.copy()
        fast.advance_n(v)
        for _ in range(v):
            _ = gold.tick()
        if (
            fast.counter != gold.counter
            or fast.render_counter != gold.render_counter
            or fast.is_rendering != gold.is_rendering
            or fast.copy_num != gold.copy_num
        ):
            raise Error(
                "missile advance_n("
                + String(v)
                + ") diverged at round "
                + String(round)
                + ": counter "
                + String(fast.counter)
                + "/"
                + String(gold.counter)
            )
    print(
        "missile: advance_n + lit_horizon OK (" + String(ROUNDS) + " rounds)"
    )


def test_player() raises:
    var rng: UInt64 = 0x123456789ABCDEF1
    var gold = PlayerCounter()
    for round in range(ROUNDS):
        var r = _lcg(rng)
        if r % 7 == 0:
            gold.set_nusiz(UInt8((r >> 8) & 0x07))
        if r % 11 == 0:
            gold.set_grp_new(UInt8((r >> 4) & 0xFF))
        if r % 13 == 0:
            gold.set_vdel(((r >> 5) & 1) != 0)
        if r % 17 == 0:
            gold.shuffle()
        if r % 23 == 0:
            gold.set_reflect(((r >> 6) & 1) != 0)
        if r % 19 == 0:
            gold.resp(157 + Int((r >> 10) % 3))

        var h = min(gold.lit_horizon(), 400)
        var probe = gold.copy()
        for i in range(h):
            if probe.tick():
                raise Error(
                    "player lit at tick "
                    + String(i)
                    + " within horizon "
                    + String(h)
                    + " (round "
                    + String(round)
                    + ")"
                )

        var v = 1 + Int((r >> 16) % 400)
        var fast = gold.copy()
        fast.advance_n(v)
        for _ in range(v):
            _ = gold.tick()
        if (
            fast.counter != gold.counter
            or fast.render_counter != gold.render_counter
            or fast.sample_counter != gold.sample_counter
            or fast.is_rendering != gold.is_rendering
            or fast.copy_num != gold.copy_num
        ):
            raise Error(
                "player advance_n("
                + String(v)
                + ") diverged at round "
                + String(round)
                + ": counter "
                + String(fast.counter)
                + "/"
                + String(gold.counter)
                + " sample "
                + String(fast.sample_counter)
                + "/"
                + String(gold.sample_counter)
            )
    print("player: advance_n + lit_horizon OK (" + String(ROUNDS) + " rounds)")


def main() raises:
    test_ball()
    test_missile()
    test_player()
    print("ALL advance_n property tests PASSED")
