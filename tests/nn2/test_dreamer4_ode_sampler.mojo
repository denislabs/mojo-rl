"""ODE sampler smoke + zero-init sanity (Phase 2.5).

`sample_one_timestep` runs the K-step flow-matching ODE on a fixed-T window.
At the zero-init flow head the prediction x̂1 is always 0, so the velocity is
b = (0 − z)/max(1e-4, 1−τ) and z integrates toward 0 — a clean closed-form
check that the integration loop (index bookkeeping + Euler update) is wired
correctly and produces finite output. (Quality is the lighthouse's job.)
"""

from std.memory import alloc
from std.math import sin
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.deep_agents2.dreamer4.dynamics import Dreamer4Dynamics
from mojo_rl.deep_agents2.dreamer4.ode_sampler import sample_one_timestep


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    print("=" * 70)
    print("Dreamer4 ODE sampler — smoke + zero-init sanity (Phase 2.5)")
    print("=" * 70)

    comptime DSP = 4
    comptime NSP = 4
    comptime D = 8
    comptime NH = 2
    comptime T = 3
    comptime NREG = 2
    comptime HID = 16
    comptime DEPTH = 2
    comptime KMAX = 4
    comptime K = 4
    comptime B = 2
    comptime ND = NSP * DSP

    var dyn = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX
    ].make[target="cpu", INIT=Xavier]()

    var ctx = _alloc(B * (T - 1) * ND)
    var z0 = _alloc(B * ND)
    var out = _alloc(B * ND)
    for i in range(B * (T - 1) * ND):
        ctx[i] = Scalar[DT](0.3 * sin(0.5 + 0.4 * Float64(i)))
    for i in range(B * ND):
        z0[i] = Scalar[DT](0.7 + 0.1 * Float64(i % 5))   # nonzero start

    sample_one_timestep[type_of(dyn), B, T, NSP, DSP, KMAX, K](
        dyn, ctx, z0, out
    )

    var maxabs: Float64 = 0.0
    var finite = True
    for i in range(B * ND):
        var v = Float64(out[i])
        if v != v:
            finite = False
        if abs(v) > maxabs:
            maxabs = v if v > 0 else -v
    print("  zero-init rollout max|z| =", maxabs, " (integrates toward 0)")
    assert_true(finite, "output finite")
    # With x̂1≡0, each Euler step shrinks z toward 0 (|z| strictly decreases).
    assert_true(maxabs < 0.7, "zero-flow-head ⇒ z shrinks below the 0.7 start")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
