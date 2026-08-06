"""`z` sampler gate — the norm invariant, the mixture, and scale invariance.

`docs/BFM_ZERO_SHOT_RL.md` §11 ranks "z not renormalised" as the FIRST silent
failure of this project: nothing crashes, the loss still descends, and the
policy emits plausible noise. A gate is the only thing that can catch it, and it
has to check EVERY producer — the training sampler and the inference path both,
because the inference path is the one that gets refactored later by someone who
believes the sampler already handled it.

What is checked, and why each is not redundant:

  [1] every row of every producer has norm exactly sqrt(d);
  [2] the uniform draw is uniform ON THE SPHERE, not merely on it — a
      per-coordinate second moment of 1. A sampler that drew one fixed
      direction and normalised it would pass [1] perfectly;
  [3] the mixture actually mixes, checked at both endpoints;
  [4] `z_from_reward` is INVARIANT to the reward's scale. This is the property
      that says the projection is doing its job in inference: doubling every
      reward doubles the raw expectation, and only the projection makes the
      resulting policy the same one. It is also the check that would fail if
      someone "optimised away" the projection there on the grounds that the
      training sampler already normalises;
  [5] a degenerate (all-zero) input yields a finite z, not NaN.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_z_sampler.mojo
"""

from std.math import abs, sqrt
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.fb import (
    sample_z,
    sample_z_uniform,
    z_from_b,
    z_from_reward,
)


comptime D: Int = 16
comptime BATCH: Int = 512
comptime SEED: Int = 20260805
comptime NORM_TOL: Float64 = 1e-4


def _norm(ref z: List[Scalar[DT]], row: Int) -> Float64:
    var acc = Float64(0)
    for k in range(D):
        var v = Float64(z[row * D + k])
        acc += v * v
    return sqrt(acc)


def _assert_all_on_sphere(
    ref z: List[Scalar[DT]], rows: Int, label: String
) raises:
    var radius = sqrt(Float64(D))
    var worst = Float64(0)
    for r in range(rows):
        var e = abs(_norm(z, r) - radius)
        if e > worst:
            worst = e
    assert_true(
        worst < NORM_TOL,
        String(label) + ": worst |‖z‖ - sqrt(d)| = " + String(worst)
        + ". A z off the sphere trains to a policy that emits plausible"
        " garbage and reports no error — see this file's docstring.",
    )
    print("      ", label, " worst norm error", worst)


def _fake_b_rows(n: Int) -> List[Scalar[DT]]:
    """`B(s)` output with rows of DELIBERATELY varied magnitude.

    If every row already had norm sqrt(d), the projection inside `z_from_b`
    would be a no-op and [1] would pass whether or not it ran.
    """
    var b = List[Scalar[DT]](length=n * D, fill=Scalar[DT](0))
    for r in range(n):
        var scale = 0.01 + 3.0 * Float64(r % 7)
        for k in range(D):
            b[r * D + k] = Scalar[DT](
                scale * (0.3 * Float64(k + 1) - 0.11 * Float64(r + 1))
            )
    return b^


def test_uniform_on_sphere() raises:
    print("[1] sample_z_uniform: norm and isotropy ...")
    seed(SEED)
    var z = sample_z_uniform[D](BATCH)
    _assert_all_on_sphere(z, BATCH, "uniform")

    # Isotropy. ‖z‖² = d exactly, so E[z_k²] = 1 for every coordinate, and
    # E[z_k] = 0. A degenerate sampler (one fixed direction, renormalised)
    # passes the norm check above and fails both of these.
    var worst_mean = Float64(0)
    var worst_m2 = Float64(0)
    for k in range(D):
        var s = Float64(0)
        var s2 = Float64(0)
        for r in range(BATCH):
            var v = Float64(z[r * D + k])
            s += v
            s2 += v * v
        var mean = s / Float64(BATCH)
        var m2 = s2 / Float64(BATCH)
        if abs(mean) > worst_mean:
            worst_mean = abs(mean)
        if abs(m2 - 1.0) > worst_m2:
            worst_m2 = abs(m2 - 1.0)
    print("       worst |mean|", worst_mean, " worst |E[z^2]-1|", worst_m2)
    assert_true(worst_mean < 0.2, "z coordinates are not centred")
    assert_true(
        worst_m2 < 0.3,
        "z coordinate second moment is " + String(worst_m2) + " away from 1 —"
        " the draw is not isotropic on the sphere",
    )


def test_z_from_b_projects() raises:
    print("[2] z_from_b projects rows of wildly different magnitude ...")
    var b = _fake_b_rows(64)
    var z = z_from_b[D](b, 64)
    _assert_all_on_sphere(z, 64, "z_from_b")

    # The raw rows must NOT already be on the sphere, or the projection was
    # never exercised.
    var raw_spread = Float64(0)
    var radius = sqrt(Float64(D))
    for r in range(64):
        var e = abs(_norm(b, r) - radius)
        if e > raw_spread:
            raw_spread = e
    assert_true(
        raw_spread > 1.0,
        "the fake B rows are already near the sphere (worst deviation "
        + String(raw_spread) + ") — this gate never exercised the projection",
    )


def test_mixture_endpoints() raises:
    """`uniform_frac` 0 and 1 must produce visibly different populations."""
    print("[3] mixture endpoints ...")
    var b = _fake_b_rows(8)

    seed(SEED)
    var all_b = sample_z[D](256, b, 8, uniform_frac=0.0)
    _assert_all_on_sphere(all_b, 256, "frac=0")
    # With 8 source rows and no uniform component, every z must coincide with
    # one of the 8 projected B directions.
    var proj = z_from_b[D](b, 8)
    var unmatched = 0
    for r in range(256):
        var best = Float64(1e30)
        for q in range(8):
            var d2 = Float64(0)
            for k in range(D):
                var diff = Float64(all_b[r * D + k]) - Float64(proj[q * D + k])
                d2 += diff * diff
            if d2 < best:
                best = d2
        if best > 1e-6:
            unmatched += 1
    assert_true(
        unmatched == 0,
        String(unmatched) + " of 256 draws at uniform_frac=0 did not match any"
        " projected B row — the B branch is not being taken",
    )

    seed(SEED)
    var all_u = sample_z[D](256, b, 8, uniform_frac=1.0)
    _assert_all_on_sphere(all_u, 256, "frac=1")
    var matched = 0
    for r in range(256):
        var best = Float64(1e30)
        for q in range(8):
            var d2 = Float64(0)
            for k in range(D):
                var diff = Float64(all_u[r * D + k]) - Float64(proj[q * D + k])
                d2 += diff * diff
            if d2 < best:
                best = d2
        if best < 1e-6:
            matched += 1
    assert_true(
        matched == 0,
        String(matched) + " draws at uniform_frac=1 landed exactly on a B row"
        " — the uniform branch is not being taken",
    )
    print("       frac=0: all 256 from B;  frac=1: none from B  OK")


def test_z_from_reward_scale_invariant() raises:
    """Doubling every reward must not move `z`.

    The raw expectation `E[B(s)·r(s)]` scales linearly with the reward, so this
    holds ONLY because `z_from_reward` projects. Delete that projection and
    this test fails while every other test in the file still passes — which is
    exactly the point: the inference path needs its own gate.
    """
    print("[4] z_from_reward is invariant to reward scale ...")
    var b = _fake_b_rows(128)
    var r1 = List[Scalar[DT]](length=128, fill=Scalar[DT](0))
    var r2 = List[Scalar[DT]](length=128, fill=Scalar[DT](0))
    for i in range(128):
        var v = 0.37 * Float64(i % 11) - 1.2
        r1[i] = Scalar[DT](v)
        r2[i] = Scalar[DT](v * 17.0)

    var z1 = z_from_reward[D](b, r1, 128)
    var z2 = z_from_reward[D](b, r2, 128)
    _assert_all_on_sphere(z1, 1, "z_from_reward")

    var worst = Float64(0)
    for k in range(D):
        var e = abs(Float64(z1[k]) - Float64(z2[k]))
        if e > worst:
            worst = e
    print("       worst |z(r) - z(17r)| =", worst)
    assert_true(
        worst < 1e-3,
        "z moved by " + String(worst) + " when the reward was scaled by 17 —"
        " z_from_reward is not projecting, so zero-shot inference queries the"
        " policy family at a point training never reached",
    )

    # And it must actually DEPEND on the reward: a z that ignored r would be
    # trivially scale-invariant.
    var r3 = List[Scalar[DT]](length=128, fill=Scalar[DT](0))
    for i in range(128):
        r3[i] = Scalar[DT](-0.9 * Float64((i * 7) % 13) + 2.0)
    var z3 = z_from_reward[D](b, r3, 128)
    var diff = Float64(0)
    for k in range(D):
        diff += abs(Float64(z1[k]) - Float64(z3[k]))
    assert_true(
        diff > 1e-3,
        "a different reward produced the same z — z_from_reward is ignoring"
        " its reward argument, and the scale-invariance check above is vacuous",
    )


def test_degenerate_input_is_finite() raises:
    print("[5] all-zero B rows give a finite z ...")
    var b = List[Scalar[DT]](length=4 * D, fill=Scalar[DT](0))
    var z = z_from_b[D](b, 4)
    _assert_all_on_sphere(z, 4, "degenerate")
    for i in range(4 * D):
        var v = Float64(z[i])
        assert_true(v == v, "NaN in z from an all-zero B row")
    print("       finite, on the sphere  OK")


def main() raises:
    test_uniform_on_sphere()
    test_z_from_b_projects()
    test_mixture_endpoints()
    test_z_from_reward_scale_invariant()
    test_degenerate_input_is_finite()
    print("\n[PASS] FB z sampler gate")
