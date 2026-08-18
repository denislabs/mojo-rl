"""`ObsNorm` gate — the sidecar, and the train/eval agreement it enforces.

`fb/obs_norm.mojo` exists to prevent ONE failure: a run trains on standardised
observations and is then evaluated on raw ones. Nothing raises. The actor gets
inputs from a distribution it never saw, still emits actions in [-1, 1], and the
eval prints a plausible ratio — indistinguishable from "that arm of the sweep
did not help". `docs/BFM_ZERO_SHOT_RL.md` §16.3 makes obs normalization a sweep
arm, so this is live code, not a hypothetical.

The gate therefore checks the SEAM, not just the arithmetic:

  [1] `fit` recovers a known mean and std;
  [2] `apply_rows` standardises — mean ~0, std ~1 per dimension;
  [3] a CONSTANT column is left alone (`sd = 1`), not divided by its own float
      noise. That is what an unused or padded observation slot looks like, and
      dividing it would turn a dead input into the batch's largest signal;
  [4] ⭐ **`apply_row` (the eval rollout path) and `apply_rows` (the training
      path) agree bit-for-bit.** Two functions, one transform: the whole point
      of the module is that these cannot drift, and only a test says so;
  [5] a save/load round trip through the sidecar preserves the transform — the
      check that the number reaching eval is the number training used;
  [6] an ABSENT sidecar reads as `None`, so a raw-input run evaluates raw;
  [7] ⭐ a sidecar of the WRONG WIDTH **raises**. This is the negative control.
      Returning `None` there would be the original bug wearing a helmet: a
      d-mismatched file would be silently ignored and the run evaluated raw.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_obs_norm.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.fb.obs_norm import ObsNorm


comptime N: Int = 4
comptime R: Int = 64
comptime TMP: StaticString = "/tmp/mojo_rl_test_obs_norm.sidecar"
comptime TMP_BAD: StaticString = "/tmp/mojo_rl_test_obs_norm.bad"


def _make() raises -> Tensor:
    """`R` rows of `N` dims: three with deliberately different scales, and one
    held CONSTANT so [3] has something to catch."""
    var t = Tensor()
    t.ensure(R * N)
    for r in range(R):
        var x = Float64(r)
        t.data[r * N + 0] = Scalar[DT](x)                  # mean 31.5
        t.data[r * N + 1] = Scalar[DT](100.0 * x - 500.0)  # 100x the spread
        t.data[r * N + 2] = Scalar[DT](-0.01 * x)          # tiny, negative
        t.data[r * N + 3] = Scalar[DT](7.0)                # CONSTANT
    return t^


def main() raises:
    print("== ObsNorm gate ==")

    # ── [1] fit recovers the known moments ──────────────────────────────
    var data = _make()
    var n = ObsNorm[N].fit(data, R)
    # mean of 0..R-1 is (R-1)/2; population std of that ramp is
    # sqrt((R^2 - 1)/12).
    var want_mu0 = Float64(R - 1) / 2.0
    var want_sd0 = sqrt((Float64(R) * Float64(R) - 1.0) / 12.0)
    assert_true(
        abs(n.mu[0] - want_mu0) < 1e-9,
        "[1] mu[0] " + String(n.mu[0]) + " != " + String(want_mu0),
    )
    assert_true(
        abs(n.sd[0] - want_sd0) < 1e-6,
        "[1] sd[0] " + String(n.sd[0]) + " != " + String(want_sd0),
    )
    assert_true(
        abs(n.sd[1] - 100.0 * want_sd0) < 1e-3,
        "[1] sd[1] should be 100x sd[0], got " + String(n.sd[1]),
    )
    print("  [1] fit recovers mean/std           OK  (sd0", n.sd[0],
          ", sd1", n.sd[1], ")")

    # ── [3] the constant column is untouched ────────────────────────────
    assert_true(
        n.sd[3] == 1.0,
        "[3] constant column got sd " + String(n.sd[3])
        + " — it must be left at 1, not divided by its own noise",
    )
    print("  [3] constant column left at sd = 1  OK")

    # ── [2] applying it standardises ────────────────────────────────────
    var applied = _make()
    n.apply_rows(applied, R)
    for k in range(N):
        var m = Float64(0)
        for r in range(R):
            m += Float64(applied.data[r * N + k])
        m /= Float64(R)
        assert_true(
            abs(m) < 1e-4,
            "[2] dim " + String(k) + " mean after apply is " + String(m),
        )
        if k == 3:
            continue  # constant column maps to 0, std 0 by construction
        var v = Float64(0)
        for r in range(R):
            var d = Float64(applied.data[r * N + k]) - m
            v += d * d
        var s = sqrt(v / Float64(R))
        assert_true(
            abs(s - 1.0) < 1e-4,
            "[2] dim " + String(k) + " std after apply is " + String(s),
        )
    print("  [2] apply_rows -> mean 0, std 1     OK")

    # ── [4] the two application paths agree ─────────────────────────────
    # `apply_rows` is what training calls on the whole store; `apply_row` is
    # what the eval rollout calls once per env step. A divergence here is the
    # exact train/eval mismatch this module exists to prevent, and it would be
    # invisible in both scripts.
    var single = Tensor()
    single.ensure(N)
    var worst = Float64(0)
    for r in range(R):
        for k in range(N):
            single.data[k] = Scalar[DT](
                Float64(_row_val(r, k))
            )
        n.apply_row(single)
        for k in range(N):
            var d = abs(
                Float64(single.data[k]) - Float64(applied.data[r * N + k])
            )
            if d > worst:
                worst = d
    assert_true(
        worst == 0.0,
        "[4] apply_row and apply_rows disagree by " + String(worst)
        + " — the eval rollout and the training store would see different"
          " inputs from the SAME statistics",
    )
    print("  [4] apply_row == apply_rows         OK  (exact)")

    # ── [5] sidecar round trip ──────────────────────────────────────────
    n.save(String(TMP))
    var back_opt = ObsNorm[N].try_load(String(TMP))
    assert_true(Bool(back_opt), "[5] sidecar just written did not load")
    var back = back_opt.take()
    var rt = Tensor()
    rt.ensure(N)
    var worst_rt = Float64(0)
    for r in range(R):
        for k in range(N):
            rt.data[k] = Scalar[DT](Float64(_row_val(r, k)))
        back.apply_row(rt)
        for k in range(N):
            var d = abs(Float64(rt.data[k]) - Float64(applied.data[r * N + k]))
            if d > worst_rt:
                worst_rt = d
    assert_true(
        worst_rt < 1e-6,
        "[5] round-tripped sidecar transforms differently by "
        + String(worst_rt),
    )
    print("  [5] save -> load preserves it       OK  (max dev", worst_rt, ")")

    # ── [6] absent sidecar is None, not an error ────────────────────────
    var missing = ObsNorm[N].try_load(
        String("/tmp/mojo_rl_test_obs_norm.definitely_absent")
    )
    assert_true(
        not Bool(missing),
        "[6] an absent sidecar must read as None so a raw-input run evaluates"
        " raw",
    )
    print("  [6] absent sidecar -> None          OK")

    # ── [7] NEGATIVE CONTROL: wrong width must RAISE ────────────────────
    # A d-mismatched sidecar that returned None would be silently ignored and
    # the checkpoint evaluated on raw inputs — the original bug, one level
    # down. It has to be loud.
    with open(String(TMP_BAD), "w") as f:
        f.write(String(N + 1) + "\n0 1\n0 1\n0 1\n0 1\n0 1\n")
    var raised = False
    try:
        _ = ObsNorm[N].try_load(String(TMP_BAD))
    except:
        raised = True
    assert_true(
        raised,
        "[7] a sidecar declaring " + String(N + 1) + " dims loaded into an"
        " N=" + String(N) + " build WITHOUT raising",
    )
    print("  [7] wrong-width sidecar RAISES      OK")

    print("")
    print("all ObsNorm checks passed")


def _row_val(r: Int, k: Int) -> Float64:
    """The generator behind `_make`, so [4] and [5] rebuild rows without
    depending on the (already mutated) tensor."""
    var x = Float64(r)
    if k == 0:
        return x
    if k == 1:
        return 100.0 * x - 500.0
    if k == 2:
        return -0.01 * x
    return 7.0
