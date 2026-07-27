"""LunarLander copy/move semantics regression gate.

Guards the fix for a real bug: the copy AND move constructors used to build a
FRESH physics state and call `_reset_cpu()` — so passing an env by value (or
storing it in a container) silently wiped mid-episode state, regenerated the
terrain, and advanced the RNG counter. A copy must continue from the SAME
mid-episode state as the original.

Run: pixi run mojo run -I . tests/envs/test_lunar_lander_copy_semantics.mojo
"""

from std.testing import assert_equal, assert_true

from mojo_rl.envs.lunar_lander import LunarLander


def main() raises:
    print("=" * 70)
    print("LunarLander copy/move semantics")
    print("=" * 70)

    var env = LunarLander[DType.float32](seed=42)
    _ = env.reset_obs_list()

    # Advance mid-episode with a deterministic action sequence.
    var done = False
    for i in range(40):
        if done:
            break
        var result = env.step_obs(2 if i % 3 == 0 else 0)
        done = result[2]
    assert_true(not done, "episode ended during warm-up; shorten the rollout")

    var orig_obs = env.get_obs_list()

    # ── Copy must preserve mid-episode state bit-for-bit ─────────────────
    var copied = env.copy()
    var copy_obs = copied.get_obs_list()
    assert_equal(len(copy_obs), len(orig_obs), "obs dim mismatch")
    for k in range(len(orig_obs)):
        assert_equal(
            copy_obs[k],
            orig_obs[k],
            "copy reset obs[" + String(k) + "] (copy-ctor reset bug)",
        )

    # Stepping the copy and the original with the same action must produce
    # identical physics (terrain + bodies + shapes all came across).
    var r_orig = env.step_obs(2)
    var r_copy = copied.step_obs(2)
    for k in range(len(r_orig[0])):
        assert_equal(
            r_copy[0][k],
            r_orig[0][k],
            "copy diverged from original after 1 step (obs["
            + String(k)
            + "])",
        )
    assert_equal(r_copy[1], r_orig[1], "copy reward diverged")
    assert_equal(r_copy[2], r_orig[2], "copy done flag diverged")
    print("  copy preserves mid-episode state + physics: OK")

    # ── Move must transfer state verbatim ────────────────────────────────
    var before = copied.get_obs_list()
    var moved = copied^
    var after = moved.get_obs_list()
    for k in range(len(before)):
        assert_equal(
            after[k],
            before[k],
            "move reset obs[" + String(k) + "] (move-ctor reset bug)",
        )
    print("  move transfers state verbatim: OK")

    print("LUNAR LANDER COPY/MOVE SEMANTICS OK")
