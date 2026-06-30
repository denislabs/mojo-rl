"""EZv2-Atari value-target + support verification (Stage 4) — CPU.

Pins the two parity facts behind Stage 4 (see docs/EZV2_ATARI_PARITY.md §C/§D):

1. **Atari value target = plain n-step bootstrap (td_steps=5), NOT TD(λ).**
   The official `prepare_reward_value` (`batch_worker.py:631-744`) sets
   `delta_td=0` for `value_target in {mixed,max}` (Atari uses 'mixed'), so
   td_steps is fixed at 5 and the target is `Σ_{i<5} γⁱ r_{k+i} + γ⁵ V(s_{k+5})`
   — the `td_lambda=0.95` in atari.yaml only feeds the GAE path, which Atari does
   NOT use (`model.value_target: bootstrapped`). Our `compute_nstep_value_targets`
   already implements exactly this. (Bootstrap-source caveat: EZ bootstraps the
   value HEAD at s_{k+5}; we bootstrap the search-root Q — a documented Stage-6
   reanalyze item.)

2. **601-atom [-300,300] support** (the Atari branch of `DiscreteSupport`):
   evenly-spaced integer bins, two-hot over `h(value)`. Our generic two-hot
   handles it directly; this checks sum-to-1, the defining expectation
   `Σ p_i·bin_i == h(value)`, and `h⁻¹` round-trip.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_targets.mojo
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.zero.nstep_targets import compute_nstep_value_targets
from mojo_rl.deep_agents.zero.twohot_targets import (
    mz_two_hot_target_one, mz_scalar_transform, mz_inverse_scalar_transform,
)


comptime BINS = 601
comptime V_MIN = Scalar[DT](-300.0)
comptime V_MAX = Scalar[DT](300.0)
comptime GAMMA = Scalar[DT](0.997)


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    # Raw scratch for the shared raw-pointer compute_nstep_value_targets.
    return alloc[Scalar[DT]](n).as_unsafe_any_origin()


def test_nstep_is_ez_atari_formula() raises:
    print("test_nstep (= EZ Atari n-step, td_steps=5) ...")
    comptime K = 2
    comptime N = 5
    var rewards = _a(K + N)
    var dones = _a(K + N)
    var roots = _a(K + N + 1)
    var to_play = _a(K + N + 1)
    var out = _a(K + 1)

    # A simple sparse-reward sequence, single-player (to_play all 0 → no flips).
    for i in range(K + N):
        rewards[i] = Scalar[DT](0.0)
        dones[i] = Scalar[DT](0.0)
    rewards[0] = Scalar[DT](1.0)
    rewards[3] = Scalar[DT](1.0)
    for i in range(K + N + 1):
        roots[i] = Scalar[DT](0.5)
        to_play[i] = Scalar[DT](0.0)

    compute_nstep_value_targets[K, N](
        rewards, dones, roots, to_play, GAMMA, out,
    )

    # Hand formula for k: Σ_{i<5} γⁱ r[k+i] + γ⁵ · roots[k+5]
    for k in range(K + 1):
        var refv = Scalar[DT](0.0)
        var gp = Scalar[DT](1.0)
        for i in range(N):
            refv += gp * rewards[k + i]
            gp *= GAMMA
        refv += gp * roots[k + N]
        print("   k=", k, " got", out[k], " ref", refv)
        assert_true(abs(out[k] - refv) < Scalar[DT](1e-6),
                    "n-step target matches EZ Atari formula")
    print("  ok")


def test_support_601_roundtrip() raises:
    print("test 601-atom [-300,300] two-hot ...")
    var probs = List[Scalar[DT]](length=BINS, fill=0)
    # bin_i = -300 + i  (601 integer atoms over [-300,300])
    var vals = _a(4)
    vals[0] = Scalar[DT](0.0)
    vals[1] = Scalar[DT](1.5)
    vals[2] = Scalar[DT](-3.0)
    vals[3] = Scalar[DT](21.0)
    for j in range(4):
        var raw = vals[j]
        mz_two_hot_target_one[BINS](raw, V_MIN, V_MAX, probs, 0)
        var s = Scalar[DT](0.0)
        var expect = Scalar[DT](0.0)
        for i in range(BINS):
            s += probs[i]
            expect += probs[i] * (V_MIN + Scalar[DT](i))   # Σ p_i · bin_i
        var ht = mz_scalar_transform(raw)
        var back = mz_inverse_scalar_transform(expect)
        print("   raw", raw, " Σp=", s, " Σp·bin=", expect, " h(raw)=", ht,
              " h⁻¹=", back)
        assert_true(abs(s - Scalar[DT](1.0)) < Scalar[DT](1e-5),
                    "two-hot sums to 1")
        assert_true(abs(expect - ht) < Scalar[DT](1e-4),
                    "Σ p_i·bin_i == h(raw)")
        assert_true(abs(back - raw) < Scalar[DT](1e-3),
                    "h⁻¹(Σ p_i·bin_i) round-trips raw")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("EZv2-Atari value-target + 601-support verification (Stage 4, CPU)")
    print("=" * 70)
    test_nstep_is_ez_atari_formula()
    test_support_601_roundtrip()
    print("=" * 70)
    print("PASSED")
    print("=" * 70)
