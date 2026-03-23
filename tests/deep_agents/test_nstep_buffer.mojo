"""Test NStepBuffer (CPU) for correctness."""

from mojo_rl.deep_agents.core.replay import NStepBuffer, NStepTransition
from mojo_rl.nn.constants import dtype


def test_3step_basic() raises:
    """Test 3-step return: R_3 = r0 + γ*r1 + γ²*r2."""
    print("Test 3-step basic...")
    var buf = NStepBuffer[3, 2](gamma=0.99)

    var obs0 = InlineArray[Scalar[dtype], 2](fill=Scalar[dtype](0))
    var obs1 = InlineArray[Scalar[dtype], 2](fill=Scalar[dtype](0))
    obs0[0] = Scalar[dtype](1.0)
    obs0[1] = Scalar[dtype](2.0)

    # Step 0: reward=1.0, not done → no emission
    var r0 = buf.add(obs0, Scalar[dtype](0), Scalar[dtype](1.0), obs1, False)
    if r0.valid:
        print("  FAIL: should not emit after 1 step")
        return
    print("  Step 0: buffered (count=1)")

    # Step 1: reward=2.0, not done → no emission
    obs1[0] = Scalar[dtype](3.0)
    var r1 = buf.add(obs1, Scalar[dtype](1), Scalar[dtype](2.0), obs1, False)
    if r1.valid:
        print("  FAIL: should not emit after 2 steps")
        return
    print("  Step 1: buffered (count=2)")

    # Step 2: reward=3.0, not done → emit 3-step transition!
    var obs2 = InlineArray[Scalar[dtype], 2](fill=Scalar[dtype](0))
    obs2[0] = Scalar[dtype](5.0)
    var final_obs = InlineArray[Scalar[dtype], 2](fill=Scalar[dtype](0))
    final_obs[0] = Scalar[dtype](7.0)
    var r2 = buf.add(
        obs2, Scalar[dtype](0), Scalar[dtype](3.0), final_obs, False
    )
    if not r2.valid:
        print("  FAIL: should emit after 3 steps")
        return

    # Check: R_3 = 1.0 + 0.99*2.0 + 0.99²*3.0
    var expected = (
        Scalar[dtype](1.0)
        + Scalar[dtype](0.99) * Scalar[dtype](2.0)
        + Scalar[dtype](0.99 * 0.99) * Scalar[dtype](3.0)
    )
    var diff = r2.reward - expected
    if diff < 0:
        diff = -diff
    print("  R_3 =", r2.reward, " expected =", expected, " diff =", diff)

    # Check obs is from step 0
    if Float64(r2.obs[0]) != 1.0:
        print("  FAIL: obs should be from step 0, got", r2.obs[0])
        return

    # Check action is from step 0
    if Float64(r2.action) != 0.0:
        print("  FAIL: action should be 0, got", r2.action)
        return

    # Check next_obs is the final one
    if Float64(r2.next_obs[0]) != 7.0:
        print("  FAIL: next_obs should be final, got", r2.next_obs[0])
        return

    if diff < Scalar[dtype](1e-4):
        print("  PASS")
    else:
        print("  FAIL: return mismatch")


def test_episode_boundary() raises:
    """Test partial flush on done before N steps."""
    print("Test episode boundary flush...")
    var buf = NStepBuffer[3, 1](gamma=0.99)

    var obs = InlineArray[Scalar[dtype], 1](fill=Scalar[dtype](0))
    var nobs = InlineArray[Scalar[dtype], 1](fill=Scalar[dtype](0))

    # Step 0: reward=1.0
    _ = buf.add(obs, Scalar[dtype](0), Scalar[dtype](1.0), nobs, False)
    # Step 1: reward=2.0, done=True → flush 2-step return
    var r = buf.add(obs, Scalar[dtype](1), Scalar[dtype](2.0), nobs, True)

    if not r.valid:
        print("  FAIL: should emit on done")
        return

    # R_2 = 1.0 + 0.99*2.0 = 2.98
    var expected = Scalar[dtype](1.0) + Scalar[dtype](0.99) * Scalar[dtype](2.0)
    var diff = r.reward - expected
    if diff < 0:
        diff = -diff
    print("  R_2 =", r.reward, " expected =", expected)

    if not r.done:
        print("  FAIL: done should be True")
        return

    # After flush, buffer should be empty — next step should buffer
    var r2 = buf.add(obs, Scalar[dtype](0), Scalar[dtype](5.0), nobs, False)
    if r2.valid:
        print("  FAIL: should not emit after reset")
        return

    if diff < Scalar[dtype](1e-4):
        print("  PASS")
    else:
        print("  FAIL: return mismatch")


def test_overlapping_transitions() raises:
    """Test that transitions overlap: step 0-2, then 1-3, etc."""
    print("Test overlapping transitions...")
    var buf = NStepBuffer[3, 1](gamma=1.0)  # gamma=1 for easy verification

    var obs = InlineArray[Scalar[dtype], 1](fill=Scalar[dtype](0))
    var nobs = InlineArray[Scalar[dtype], 1](fill=Scalar[dtype](0))

    # Steps 0,1,2 → emit R = r0 + r1 + r2 = 1+2+3 = 6
    _ = buf.add(obs, Scalar[dtype](0), Scalar[dtype](1.0), nobs, False)
    _ = buf.add(obs, Scalar[dtype](0), Scalar[dtype](2.0), nobs, False)
    var r1 = buf.add(obs, Scalar[dtype](0), Scalar[dtype](3.0), nobs, False)

    if not r1.valid or Float64(r1.reward) != 6.0:
        print("  FAIL: first emission should be R=6, got", r1.reward)
        return
    print("  First: R =", r1.reward, "(expected 6)")

    # Step 3 → emit R = r1 + r2 + r3 = 2+3+4 = 9 (shifted)
    var r2 = buf.add(obs, Scalar[dtype](0), Scalar[dtype](4.0), nobs, False)

    if not r2.valid or Float64(r2.reward) != 9.0:
        print("  FAIL: second emission should be R=9, got", r2.reward)
        return
    print("  Second: R =", r2.reward, "(expected 9)")

    print("  PASS")


def main() raises:
    print("=== NStepBuffer CPU Tests ===")
    test_3step_basic()
    test_episode_boundary()
    test_overlapping_transitions()
    print("=== All Tests Complete ===")
