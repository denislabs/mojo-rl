"""At float32 the Newton solver must still settle where float64 settles.

    pixi run mojo run -I . tests/physics3d/test_newton_float32_tracks_float64.mojo

WHY THIS EXISTS. `NEWTON_TOL_GPU` is DTYPE-AWARE — 1e-8 at float64, 1e-6 at
float32 — and that is the only place in the solver where the two dtypes are
asked to do different things. The reason is a measurement, not a preference:

    scale * ||grad||   and   scale * improvement

are both compared against the tolerance, and at float32 NEITHER can reach 1e-8.
Each is a difference of same-magnitude terms, so its float32 rounding floor
sits orders of magnitude above the threshold; the exit test then never fires
and the solver runs its FULL 200-iteration budget on every step carrying a
single constraint row. Measured on SO-ARM100 (one shallow contact, 6 DOF):
1.04 ms/env step against 0.55 ms once the threshold clears the noise — half the
step spent iterating on rounding error. MuJoCo uses 1e-8 and is float64
throughout, so this deviation is ours to make, not theirs to match.

⚠⚠ NO EXISTING GATE COULD SEE ANY OF THIS. Every MuJoCo-parity test in the tree
runs at float64 — `test_elliptic_condim46_vs_mujoco`, `test_noslip_*`, both
SO-ARM suites, the dog and quadruped gates — so the float32 path that training
and the viewer actually use had no convergence coverage at all. Loosening the
float64 tolerance would be caught instantly; loosening float32's was invisible
in both directions.

WHAT THIS PINS. Not "the answer did not change" — comparing float32 at 1e-6
against float32 at 1e-8 is not worth gating, because the difference is 1.5e-8
of penetration, at or below float32's own distance from float64. The extra
iterations are noise; that is the whole finding. What is worth pinning is the
property a future edit to that constant would break: that float32 still lands
where float64 lands. Thresholds sit at ~3x the measured agreement, and the file
was checked RED at 1e-1 — a gate on a tolerance is worthless unless a wrong
tolerance actually fails it.

⚠ THE FIXTURE HAS TO BE LOADED, AND FINDING ONE TOOK TWO TRIES. A single sphere
dropped on a plane is the obvious choice and is USELESS here: its solve
converges through a different exit, and the settled depth came out
bit-identical from 1e-8 all the way to 1e0 — a gate that cannot go red. Walker2d
falling to rest puts five contacts under the body's full weight, and there the
tolerance is load-bearing: the manifold moves 1.6e-6 at 1e-6 and 1.2e-5 at 1e-1.

⚠ COMPARE THE MANIFOLD, NOT THE TRAJECTORY. Diffing qpos after 400 steps
measures butterfly divergence rather than convergence — two float32 runs 4e-4
apart were each ~3e-3 from float64, with the ordering flipping between runs.
The summed contact distance at the settled pose is what the contact solve
decides, and it reproduces run to run.
"""

from std.math import abs
from std.testing import assert_true, TestSuite

from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.walker2d import Walker2d
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE,
)

# Long enough for the walker to fall and come to rest, short enough to stay
# quick. The pose is settled well before this.
comptime STEPS: Int = 400
# Index of `dist` inside a contact record.
comptime C_DIST: Int = 8


def _settle[DTYPE: DType]() raises -> Tuple[Float64, Float64, Int]:
    """Drop the walker, then report (summed distance, deepest, ncon)."""
    var e = Walker2d[DTYPE]()
    _ = e.reset()
    var a = ContAction[e.ACTION_DIM]()
    for i in range(e.ACTION_DIM):
        a[i] = Float64(0.0)
    for _ in range(STEPS):
        _ = e.step(a)

    var n = Int(e.d.meta.data[META_IDX_NUM_CONTACTS])
    var total = Float64(0)
    var deepest = Float64(0)
    for c in range(n):
        var dist = Float64(e.d.contacts.data[c * CONTACT_SIZE + C_DIST])
        total += dist
        if dist < deepest:
            deepest = dist
    return (total, deepest, n)


def test_float32_settles_where_float64_settles() raises:
    """The settled contact manifold must agree across dtypes.

    ⚠ THIS IS THE ONLY FLOAT32 CONVERGENCE COVERAGE IN THE TREE. If it fails
    after a change to `NEWTON_TOL_GPU`, the float32 tolerance has been loosened
    past the point where the contact solve finishes — which shows up as the
    body resting DEEPER than it should, because the normal force never finishes
    building. Measured on this fixture: agreement is 1.6e-6 at the shipped
    1e-6, degrading to 1.2e-5 at 1e-1.
    """
    var r64 = _settle[DType.float64]()
    var r32 = _settle[DType.float32]()
    print("  f64  ncon", r64[2], " sum", r64[0], " deepest", r64[1])
    print("  f32  ncon", r32[2], " sum", r32[0], " deepest", r32[1])

    # ⚠ WITHOUT THIS THE WHOLE FILE IS VACUOUS. Zero contacts in both dtypes
    # makes every difference below 0.0, and it would pass at ANY tolerance.
    assert_true(
        r64[2] >= 3 and r32[2] == r64[2],
        "the walker must be resting on the floor in BOTH dtypes — got ncon "
        + String(r32[2]) + " at float32 against " + String(r64[2])
        + " at float64. Equal, nonzero counts are what makes the manifold"
        " comparison below mean anything",
    )

    var dsum = abs(r32[0] - r64[0])
    var ddeep = abs(r32[1] - r64[1])
    print("   |d sum|", dsum, "  |d deepest|", ddeep)
    assert_true(
        dsum < 5e-6,
        "the settled manifold sums to " + String(r32[0]) + " at float32 but "
        + String(r64[0]) + " at float64 (|d| = " + String(dsum) + ", limit"
        " 5e-6). The float32 Newton tolerance is too loose to finish the"
        " contact solve, so the body sinks into the floor",
    )
    assert_true(
        ddeep < 3e-6,
        "the deepest contact is " + String(r32[1]) + " at float32 but "
        + String(r64[1]) + " at float64 (|d| = " + String(ddeep) + ", limit"
        " 3e-6)",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
