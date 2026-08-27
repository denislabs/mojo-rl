"""`tolerance()` parity vs the dm_control reference `rewards.py`.

Diffs our Mojo port against the reference implementation over a dense grid of
(x, bounds, margin, value_at_margin) for all 8 sigmoid types.

The reference module is imported STRAIGHT FROM `references/dm_control-main`
rather than an installed dm_control: `rewards.py` only imports `warnings` and
`numpy`, so it needs none of dm_control's generated MuJoCo bindings (the
packaged dm-control 1.0.41 requires mujoco 3.11, which conda-forge does not
carry yet — see docs/DM_CONTROL_PORT.md, Stage 0).

Run with:
    pixi run mojo run -I . tests/dm_control/test_rewards_vs_dm_control.mojo
"""

from std.math import abs, inf
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.rewards import (
    tolerance,
    SIGMOID_GAUSSIAN,
    SIGMOID_HYPERBOLIC,
    SIGMOID_LONG_TAIL,
    SIGMOID_RECIPROCAL,
    SIGMOID_COSINE,
    SIGMOID_LINEAR,
    SIGMOID_QUADRATIC,
    SIGMOID_TANH_SQUARED,
)


# Agreement bound. The formulas are transcribed exactly, so any real error
# (wrong scale, wrong branch, swapped bound) shows up at O(1e-2)..O(1) — many
# orders above this. What is left is libm divergence: Mojo's exp/log/acosh vs
# numpy's differ by a few ULP, which the exponentiation in `gaussian` and
# `tanh_squared` amplifies to ~1e-10 relative. 1e-9 absorbs that and still
# fails loudly on any transcription mistake.
comptime TOL: Float64 = 1e-9

# Reference tree location (relative to the repo root, where tests are run).
comptime REF_PATH: StaticString = "references/dm_control-main"


def _load_reference() raises -> PythonObject:
    """A flat `(x, lo, hi, margin, sigmoid, v_at_m) -> float` view of the
    reference `tolerance`, so the Mojo side never has to build a Python
    tuple for `bounds=` or unwrap a numpy scalar."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var rw = Python.import_module("dm_control.utils.rewards")
    return Python.evaluate(
        "lambda rw: lambda x, lo, hi, m, s, v: float("
        "rw.tolerance(x, bounds=(lo, hi), margin=m, sigmoid=s,"
        " value_at_margin=v))"
    )(rw)


def _check[
    sigmoid: StaticString, value_at_margin: Float64
](
    refmod: PythonObject,
    mut n_checked: Int,
    mut max_diff: Float64,
    x: Float64,
    lower: Float64,
    upper: Float64,
    margin: Float64,
) raises:
    """One (x, bounds, margin) point for one sigmoid, ours vs reference."""
    var ours = tolerance[sigmoid, value_at_margin](x, lower, upper, margin)
    var theirs = Float64(
        py=refmod(x, lower, upper, margin, String(sigmoid), value_at_margin)
    )
    var diff = abs(ours - theirs)
    if diff > max_diff:
        max_diff = diff
    if diff > TOL:
        print(
            "MISMATCH sigmoid=", sigmoid,
            " v@m=", value_at_margin,
            " x=", x,
            " bounds=(", lower, ",", upper, ")",
            " margin=", margin,
            " ours=", ours,
            " ref=", theirs,
            " diff=", diff,
        )
    assert_true(diff <= TOL, "tolerance() mismatch vs dm_control reference")
    n_checked += 1


def _sweep[
    sigmoid: StaticString, value_at_margin: Float64
](refmod: PythonObject, mut n_checked: Int, mut max_diff: Float64) raises:
    """Sweep x across / outside several bound configurations."""
    # (lower, upper, margin) configurations exercised by the suite:
    #   exact target, symmetric interval, half-open interval, zero margin.
    var configs = [
        (0.0, 0.0, 1.0),        # exact target, unit margin
        (0.0, 0.0, 0.3),        # exact target, tight margin
        (-0.25, 0.25, 0.5),     # symmetric interval (cartpole cart range)
        (0.995, 1.0, 0.0),      # zero margin -> hard indicator
        (1.2, inf[DType.float64](), 0.6),  # half-open (walker stand height)
        (0.0, 0.05, 0.3),       # reach-style radius + margin
    ]
    for cfg in configs:
        var lower = cfg[0]
        var upper = cfg[1]
        var margin = cfg[2]
        # x grid: well inside, on the bounds, and far outside on both sides.
        var xs = [
            -3.0, -1.5, -1.0, -0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.25,
            0.5, 0.995, 1.0, 1.2, 1.5, 2.0, 3.0, 7.5,
        ]
        for x in xs:
            _check[sigmoid, value_at_margin](
                refmod, n_checked, max_diff, x, lower, upper, margin
            )


def test_tolerance_matches_dm_control() raises:
    var refmod = _load_reference()
    var n = 0
    var max_diff = 0.0

    # Default value_at_margin (0.1) for every sigmoid.
    _sweep[SIGMOID_GAUSSIAN, 0.1](refmod, n, max_diff)
    _sweep[SIGMOID_HYPERBOLIC, 0.1](refmod, n, max_diff)
    _sweep[SIGMOID_LONG_TAIL, 0.1](refmod, n, max_diff)
    _sweep[SIGMOID_RECIPROCAL, 0.1](refmod, n, max_diff)
    _sweep[SIGMOID_COSINE, 0.1](refmod, n, max_diff)
    _sweep[SIGMOID_LINEAR, 0.1](refmod, n, max_diff)
    _sweep[SIGMOID_QUADRATIC, 0.1](refmod, n, max_diff)
    _sweep[SIGMOID_TANH_SQUARED, 0.1](refmod, n, max_diff)

    # Non-default value_at_margin used by the suite:
    #   walker/humanoid move reward -> sigmoid='linear', value_at_margin=0.5
    #   point_mass small_control    -> sigmoid='quadratic', value_at_margin=0
    #   humanoid control cost       -> sigmoid='quadratic', value_at_margin=0
    _sweep[SIGMOID_LINEAR, 0.5](refmod, n, max_diff)
    _sweep[SIGMOID_QUADRATIC, 0.0](refmod, n, max_diff)
    _sweep[SIGMOID_COSINE, 0.0](refmod, n, max_diff)
    _sweep[SIGMOID_GAUSSIAN, 0.5](refmod, n, max_diff)
    _sweep[SIGMOID_LONG_TAIL, 0.01](refmod, n, max_diff)

    print(
        "tolerance(): ", n, " points vs dm_control — max |diff| = ",
        max_diff, " (bound ", TOL, ")",
    )


# =============================================================================
# Float32 instantiation parity (the GPU path's gate)
# =============================================================================
#
# `tolerance` grew a `DTYPE` parameter so the GPU reward hooks can call it —
# Metal has no `double`, so the float64 form cannot appear in a kernel at all
# (see the module docstring and physics3d/solver/newton_solve.mojo:1046).
#
# ⚠ Compiling is NOT the gate. A float32 `tolerance` that builds but disagrees
# with the float64 one is a silently different reward curve on every suite
# task, and no learning curve would reveal it. This diffs the two
# instantiations over the same grid the reference sweep uses.

# Bound for float32 vs float64. `tolerance` returns values in [0, 1], and the
# sigmoids are smooth there, so the error is a few float32 ULP (~1e-7) with the
# `exp`/`tanh` cases amplifying it a little. 5e-6 absorbs that and still fails
# loudly if a branch or a scale constant diverges between instantiations.
comptime TOL_F32: Float64 = 5e-6


def _check_f32[
    sigmoid: StaticString, value_at_margin: Float64
](
    mut n_checked: Int,
    mut max_diff: Float64,
    x: Float64,
    lower: Float64,
    upper: Float64,
    margin: Float64,
) raises:
    """Same point through the float64 and float32 instantiations."""
    var f64 = tolerance[sigmoid, value_at_margin](x, lower, upper, margin)
    var f32 = tolerance[sigmoid, value_at_margin, DType.float32](
        Float32(x), Float32(lower), Float32(upper), Float32(margin)
    )
    var diff = abs(f64 - Float64(f32))
    if diff > max_diff:
        max_diff = diff
    if diff > TOL_F32:
        print(
            "F32 MISMATCH sigmoid=", sigmoid,
            " v@m=", value_at_margin,
            " x=", x,
            " bounds=(", lower, ",", upper, ")",
            " margin=", margin,
            " f64=", f64,
            " f32=", f32,
            " diff=", diff,
        )
    assert_true(
        diff <= TOL_F32, "tolerance() float32 instantiation diverges from f64"
    )
    n_checked += 1


def _sweep_f32[
    sigmoid: StaticString, value_at_margin: Float64
](mut n_checked: Int, mut max_diff: Float64) raises:
    # `inf` is deliberately absent here: Float32(inf[f64]()) is still inf, but
    # the half-open case is already covered by the reference sweep and adding
    # it buys no discrimination between the two instantiations.
    var configs = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.3),
        (-0.25, 0.25, 0.5),
        (0.995, 1.0, 0.0),
        (0.0, 0.05, 0.3),
    ]
    for cfg in configs:
        var lower = cfg[0]
        var upper = cfg[1]
        var margin = cfg[2]
        var xs = [
            -3.0, -1.5, -1.0, -0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.25,
            0.5, 0.995, 1.0, 1.2, 1.5, 2.0, 3.0, 7.5,
        ]
        for x in xs:
            _check_f32[sigmoid, value_at_margin](
                n_checked, max_diff, x, lower, upper, margin
            )


def test_tolerance_float32_matches_float64() raises:
    var n = 0
    var max_diff = 0.0

    _sweep_f32[SIGMOID_GAUSSIAN, 0.1](n, max_diff)
    _sweep_f32[SIGMOID_HYPERBOLIC, 0.1](n, max_diff)
    _sweep_f32[SIGMOID_LONG_TAIL, 0.1](n, max_diff)
    _sweep_f32[SIGMOID_RECIPROCAL, 0.1](n, max_diff)
    _sweep_f32[SIGMOID_COSINE, 0.1](n, max_diff)
    _sweep_f32[SIGMOID_LINEAR, 0.1](n, max_diff)
    _sweep_f32[SIGMOID_QUADRATIC, 0.1](n, max_diff)
    _sweep_f32[SIGMOID_TANH_SQUARED, 0.1](n, max_diff)

    # The non-default value_at_margin cases the suite actually uses.
    _sweep_f32[SIGMOID_LINEAR, 0.5](n, max_diff)
    _sweep_f32[SIGMOID_QUADRATIC, 0.0](n, max_diff)
    _sweep_f32[SIGMOID_COSINE, 0.0](n, max_diff)

    print(
        "tolerance() f32 vs f64: ", n, " points — max |diff| = ",
        max_diff, " (bound ", TOL_F32, ")",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
