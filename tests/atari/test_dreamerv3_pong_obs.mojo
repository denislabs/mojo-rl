"""DreamerV3-Atari P0 env gate — grayscale-96 SINGLE-frame obs (OBS_MODE=3) +
sticky actions + random no-op starts (CPU).

Pins the P0 deliverable (docs/DREAMERV3_ATARI_PONG_SCOPE.md): an ADDITIVE
`OBS_MODE=3` on `AtariEnv` emitting the DreamerV3 Atari observation — one
grayscale frame area-resized to 96×96 = 9216 floats, NO frame stacking (the RSSM
carries temporal state) — plus the Machado protocol knobs `sticky_prob` and
`noop_max`. The existing modes (RAM / gray-84 / RGB-96) are untouched; defaults
(sticky_prob=0, noop_max=0) keep behavior identical.

Checks:
  1. obs_dim == 9216 (= 96·96) and reset obs is that length, all in [0,1].
  2. reset obs is not all-zero (a frame was actually rendered).
  3. Stepping changes the observation and stays in [0,1]; Pong doesn't end early.
  4. sticky_prob=1.0 → the executed action is pinned to the first one: obs
     trajectory is INVARIANT to the requested action after step 0 (the env keeps
     repeating the previous ALE action).
  5. noop_max>0 is constructible and the env still resets+steps cleanly.

Run:
    pixi run -e apple mojo run -I . tests/atari/test_dreamerv3_pong_obs.mojo
"""

from std.math import abs
from std.testing import assert_true, assert_equal

from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


comptime DT = DType.float32
comptime GRAY96 = 96 * 96  # 9216


def _in_unit(v: Scalar[DT]) -> Bool:
    return v >= Scalar[DT](0.0) and v <= Scalar[DT](1.0)


def test_gray96_single_frame_obs() raises:
    print("test gray-96 single-frame obs (Pong, OBS_MODE=3) ...")
    var rom = load_rom("roms/pong.bin")
    var env = AtariEnv[3, DT](AtariGame.PONG, rom.data.value(), rom.size)

    # (1) dims
    assert_equal(env.obs_dim(), GRAY96, "obs_dim == 9216 (single 96x96 frame)")

    var obs0 = env.reset_obs_list()
    assert_equal(len(obs0), GRAY96, "reset obs length == 9216")
    var all_unit = True
    var any_nonzero = False
    for i in range(GRAY96):
        if not _in_unit(obs0[i]):
            all_unit = False
        if obs0[i] > Scalar[DT](0.0):
            any_nonzero = True
    assert_true(all_unit, "reset obs all in [0,1]")
    assert_true(any_nonzero, "reset obs not all zero (frame rendered)")

    # (3) step and inspect
    var changed = False
    var obs_unit = True
    var early_term = False
    for t in range(12):
        var res = env.step_obs(1)  # any valid action
        var obs = res[0].copy()
        # Dense scan: Pong's moving pixels (ball/paddles/score) are small and
        # localized, so a sparse stride misses them.
        for i in range(GRAY96):
            if not _in_unit(obs[i]):
                obs_unit = False
            if abs(obs[i] - obs0[i]) > Scalar[DT](1e-9):
                changed = True
        if res[2] and t < 3:
            early_term = True
    assert_true(changed, "observation advances when stepping")
    assert_true(obs_unit, "stepped obs stays in [0,1]")
    assert_true(not early_term, "Pong does not end in the first few steps")
    env.close()
    _ = env^
    print("  ok")


def test_sticky_actions_pin_the_action() raises:
    print("test sticky_prob=1.0 pins the executed action ...")
    var rom = load_rom("roms/pong.bin")
    # sticky_prob=1.0 → after the first step, the env ALWAYS repeats the previous
    # ALE action regardless of what we request. So two envs that agree on step 0
    # but then request DIFFERENT actions must produce IDENTICAL trajectories.
    var a = AtariEnv[3, DT](
        AtariGame.PONG, rom.data.value(), rom.size, sticky_prob=1.0
    )
    var b = AtariEnv[3, DT](
        AtariGame.PONG, rom.data.value(), rom.size, sticky_prob=1.0
    )
    _ = a.reset_obs_list()
    _ = b.reset_obs_list()
    # Step 0: both request the same action (2), which becomes the pinned action.
    _ = a.step_obs(2)
    _ = b.step_obs(2)
    var identical = True
    for _t in range(8):
        var oa = a.step_obs(2)[0].copy()  # keep requesting 2
        var ob = b.step_obs(5)[0].copy()  # request something else — ignored
        for i in range(GRAY96):
            if abs(oa[i] - ob[i]) > Scalar[DT](1e-9):
                identical = False
    assert_true(
        identical, "sticky_prob=1.0 → trajectory invariant to requested action"
    )
    a.close()
    b.close()
    _ = a^
    _ = b^
    print("  ok")


def test_noop_starts_construct_and_step() raises:
    print("test noop_max>0 constructs + resets + steps ...")
    var rom = load_rom("roms/pong.bin")
    var env = AtariEnv[3, DT](
        AtariGame.PONG,
        rom.data.value(),
        rom.size,
        sticky_prob=0.25,
        noop_max=30,
    )
    var obs = env.reset_obs_list()
    assert_equal(len(obs), GRAY96, "reset obs length == 9216 with noop starts")
    var ok = True
    for i in range(GRAY96):
        if not _in_unit(obs[i]):
            ok = False
    assert_true(ok, "obs in [0,1] after random no-op start")
    var res = env.step_obs(1)
    assert_equal(len(res[0]), GRAY96, "step obs length == 9216")
    env.close()
    _ = env^
    print("  ok")


def main() raises:
    print("=" * 60)
    print("DreamerV3-Atari P0 env gate (gray-96 + sticky + no-ops)")
    print("=" * 60)
    test_gray96_single_frame_obs()
    test_sticky_actions_pin_the_action()
    test_noop_starts_construct_and_step()
    print("ALL DREAMERV3 ATARI P0 ENV GATES PASSED")
