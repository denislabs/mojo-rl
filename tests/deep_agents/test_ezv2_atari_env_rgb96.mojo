"""EZv2-Atari Stage-1 env parity — RGB-96 obs mode + parity flags (CPU).

Pins the Stage-1 deliverable (see docs/EZV2_ATARI_PARITY.md §A): an ADDITIVE
`OBS_MODE=2` on `AtariEnv` emitting the EfficientZero-V2 Atari observation —
RGB, area-resized to 96×96, 4 frames stacked → `[12,96,96]` = 110592 floats —
plus the per-env parity flags (full 18-action set, reward clip, episodic life).
The grayscale-84 path (OBS_MODE=1) is untouched and is instantiated here only
to confirm no compile regression.

Checks:
  1. obs_dim == 110592 (= 4·3·96·96) and reset obs is that length, all in [0,1].
  2. full_action_set → num_actions == 18; default → Pong minimal set (6).
  3. Stepping changes the observation (frame stack advances) and stays in [0,1].
  4. clip_reward → returned reward ∈ {−1,0,1}; episode_reward stays raw.
  5. RGB is preserved (not collapsed to gray): some pixel has R≠G or G≠B.
  6. episodic_life is inert on Pong (lives≡0): no spurious early termination.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_env_rgb96.mojo
"""

from std.math import abs
from std.testing import assert_true, assert_equal

from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame
from mojo_rl.nn.constants import LAYOUT_NHWC


comptime DT = DType.float32
comptime RGB_STACK = 4 * 3 * 96 * 96  # 110592
comptime PLANE = 96 * 96  # 9216


def _in_unit(v: Scalar[DT]) -> Bool:
    return v >= Scalar[DT](0.0) and v <= Scalar[DT](1.0)


def test_rgb96_obs_and_flags() raises:
    print("test RGB-96 obs + parity flags (Pong) ...")
    var rom = load_rom("roms/pong.bin")

    # OBS_MODE=2, EZv2 flags on.
    var env = AtariEnv[2, DT](
        AtariGame.PONG,
        rom.data.value(),
        rom.size,
        clip_reward=True,
        episodic_life=True,
        full_action_set=True,
    )

    # (1) dims
    assert_equal(env.obs_dim(), RGB_STACK, "obs_dim == 110592")
    # (2) full action set
    assert_equal(env.num_actions(), 18, "full_action_set → 18 actions")

    var obs0 = env.reset_obs_list()
    assert_equal(len(obs0), RGB_STACK, "reset obs length == 110592")
    var all_unit = True
    var any_nonzero = False
    for i in range(RGB_STACK):
        if not _in_unit(obs0[i]):
            all_unit = False
        if obs0[i] > Scalar[DT](0.0):
            any_nonzero = True
    assert_true(all_unit, "reset obs all in [0,1]")
    assert_true(any_nonzero, "reset obs not all zero (frame rendered)")

    # (3)+(4)+(5)+(6): step and inspect
    var changed = False
    var rgb_differs = False
    var clip_ok = True
    var obs_unit = True
    var early_term = False
    for t in range(12):
        var res = env.step_obs(1)  # FIRE / serve-ish; any valid action
        var obs = res[0].copy()
        var reward = res[1]
        var done = res[2]
        # clip: reward must be one of -1, 0, 1
        if not (
            abs(reward - Scalar[DT](-1.0)) < Scalar[DT](1e-6)
            or abs(reward) < Scalar[DT](1e-6)
            or abs(reward - Scalar[DT](1.0)) < Scalar[DT](1e-6)
        ):
            clip_ok = False
        # obs differs from the initial frame somewhere (broad sample)?
        for i in range(0, RGB_STACK, 97):
            if not _in_unit(obs[i]):
                obs_unit = False
            if abs(obs[i] - obs0[i]) > Scalar[DT](1e-9):
                changed = True
        # RGB preserved: within the most-recent frame (channels 9,10,11 =
        # R,G,B of frame 3 in chronological order — last written slot), look
        # for a pixel where R != G or G != B.
        # frame-major layout: channel c occupies obs[c*PLANE : (c+1)*PLANE].
        for p in range(0, PLANE, 257):  # sparse scan
            var r = obs[9 * PLANE + p]
            var g = obs[10 * PLANE + p]
            var b = obs[11 * PLANE + p]
            if abs(r - g) > Scalar[DT](1e-6) or abs(g - b) > Scalar[DT](1e-6):
                rgb_differs = True
        if done and t < 3:
            early_term = True  # Pong should not end in the first few steps

    assert_true(changed, "observation advances when stepping")
    assert_true(obs_unit, "stepped obs stays in [0,1]")
    assert_true(clip_ok, "clip_reward → reward ∈ {−1,0,1}")
    assert_true(rgb_differs, "RGB channels preserved (some R≠G or G≠B)")
    assert_true(not early_term, "episodic_life inert on Pong (no early term)")
    # episode_reward stays raw (Pong rewards are already ±1, so == clipped sum,
    # but the field type is Float64 accumulation — just assert it is finite).
    assert_true(
        env.episode_reward == env.episode_reward, "episode_reward finite"
    )
    env.close()
    _ = env^
    print("  ok")


def test_defaults_minimal_actions() raises:
    print("test defaults (flags off) → Pong minimal action set ...")
    var rom = load_rom("roms/pong.bin")
    var env = AtariEnv[2, DT](AtariGame.PONG, rom.data.value(), rom.size)
    assert_equal(env.num_actions(), 6, "Pong minimal set has 6 actions")
    assert_equal(env.obs_dim(), RGB_STACK, "obs_dim still 110592 (mode 2)")
    var obs = env.reset_obs_list()
    assert_equal(len(obs), RGB_STACK, "reset obs length")
    env.close()
    _ = env^
    print("  ok")


def test_gray84_still_compiles() raises:
    print("test gray-84 path (OBS_MODE=1) unchanged + compiles ...")
    var rom = load_rom("roms/pong.bin")
    var env = AtariEnv[1, DT](AtariGame.PONG, rom.data.value(), rom.size)
    assert_equal(env.obs_dim(), 4 * 84 * 84, "gray obs_dim == 28224")
    assert_equal(env.num_actions(), 6, "gray Pong minimal set 6")
    var obs = env.reset_obs_list()
    assert_equal(len(obs), 4 * 84 * 84, "gray reset obs length")
    var res = env.step_obs(1)
    assert_equal(len(res[0]), 4 * 84 * 84, "gray step obs length")
    env.close()
    _ = env^
    print("  ok")


def test_rgb96_nhwc_parity() raises:
    """channels_last: AtariEnv[...,LAYOUT_NHWC] must emit the SAME logical obs as
    NCHW, reordered to [96,96,12]. Two envs stepped identically (deterministic
    emulation) → exact equality nhwc[p*12+ch] == nchw[ch*PLANE+p]."""
    print("test RGB-96 NHWC vs NCHW obs parity ...")
    comptime CH = 12
    var rn = load_rom("roms/pong.bin")
    var en = AtariEnv[2, DT](AtariGame.PONG, rn.data.value(), rn.size)
    var rh = load_rom("roms/pong.bin")
    var eh = AtariEnv[2, DT, LAYOUT_NHWC](AtariGame.PONG, rh.data.value(), rh.size)

    var bad = 0
    var on = en.reset_obs_list()
    var oh = eh.reset_obs_list()
    for ch in range(CH):
        for p in range(PLANE):
            if abs(on[ch * PLANE + p] - oh[p * CH + ch]) > Scalar[DT](1e-9):
                bad += 1
    for t in range(5):
        var sn = en.step_obs(1)[0].copy()
        var sh = eh.step_obs(1)[0].copy()
        for ch in range(CH):
            for p in range(PLANE):
                if abs(sn[ch * PLANE + p] - sh[p * CH + ch]) > Scalar[DT](1e-9):
                    bad += 1
    assert_true(bad == 0, "NHWC obs == NCHW obs reordered (exact)")
    en.close()
    eh.close()
    _ = en^
    _ = eh^
    print("  ok (mismatches=", bad, ")")


def main() raises:
    print("=" * 70)
    print("EZv2-Atari Stage-1 env parity: RGB-96 obs + flags (CPU)")
    print("=" * 70)
    test_rgb96_obs_and_flags()
    test_defaults_minimal_actions()
    test_gray84_still_compiles()
    test_rgb96_nhwc_parity()
    print("=" * 70)
    print("PASSED")
    print("=" * 70)
