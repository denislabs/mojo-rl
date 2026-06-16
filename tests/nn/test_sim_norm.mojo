"""SimNorm[DIM, GROUPS] smoke + parity test (Phase 2, PORTING_PLAN.md).

Checks three properties of the per-group softmax:

  1. Each group sums to 1 along the GROUP_SIZE axis (forward correctness).
  2. The backward output matches an analytic per-group softmax Jacobian
     reference to ≤1e-6 in fp32.
  3. Sum-zero per-group invariant: Σ_k grad_x[g·G+k] · 1 cancels in the
     local softmax — equivalently, grad_x is orthogonal to the all-ones
     vector within each group when grad_y is constant within the group.
"""

from std.math import exp
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.sim_norm import SimNorm
from mojo_rl.nn.initializer import Zero


def test_forward_sums_to_one() raises:
    print("test_forward_sums_to_one ...")
    comptime BATCH = 2
    comptime DIM = 12
    comptime GROUPS = 3
    comptime GROUP_SIZE = DIM // GROUPS
    comptime N = BATCH * DIM
    var s = SimNorm[DIM, GROUPS].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-2.0 + 0.3 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    s.forward["cpu", BATCH](x_t, output=y_t)

    var max_dev: Scalar[DT] = 0.0
    for b in range(BATCH):
        for g in range(GROUPS):
            var base = g * GROUP_SIZE
            var total: Scalar[DT] = 0.0
            for k in range(GROUP_SIZE):
                var v = y[b * DIM + base + k]
                total += v
                assert_true(
                    v >= Scalar[DT](0.0),
                    "SimNorm forward produced negative probability",
                )
            var d = total - Scalar[DT](1.0)
            var ad = d if d >= Scalar[DT](0) else -d
            if ad > max_dev:
                max_dev = ad
    print("  max |Σ_k y_gk - 1| =", max_dev)
    assert_true(
        max_dev < Scalar[DT](1e-6),
        "SimNorm per-group sum should be 1",
    )
    print("  ok")


def test_backward_parity() raises:
    print("test_backward_parity ...")
    comptime BATCH = 1
    comptime DIM = 8
    comptime GROUPS = 2
    comptime GROUP_SIZE = DIM // GROUPS
    comptime N = BATCH * DIM
    var s = SimNorm[DIM, GROUPS].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-1.0 + 0.25 * Float64(i))
        go[i] = Scalar[DT](0.5 + 0.1 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    s.forward["cpu", BATCH](x_t, output=y_t)
    s.vjp["cpu", BATCH](go_t, gi_t)

    # Reference: recompute y from x then apply softmax Jacobian per group.
    var max_diff: Scalar[DT] = 0.0
    for b in range(BATCH):
        for g in range(GROUPS):
            var base = g * GROUP_SIZE
            var max_val: Scalar[DT] = x[b * DIM + base]
            for k in range(1, GROUP_SIZE):
                var v = x[b * DIM + base + k]
                if v > max_val:
                    max_val = v
            var sum_exp: Scalar[DT] = 0.0
            for k in range(GROUP_SIZE):
                sum_exp += exp(x[b * DIM + base + k] - max_val)
            var inv_sum = Scalar[DT](1.0) / sum_exp
            var dot: Scalar[DT] = 0.0
            for k in range(GROUP_SIZE):
                var yk = exp(x[b * DIM + base + k] - max_val) * inv_sum
                dot += go[b * DIM + base + k] * yk
            for k in range(GROUP_SIZE):
                var yk = exp(x[b * DIM + base + k] - max_val) * inv_sum
                var ref_g = yk * (go[b * DIM + base + k] - dot)
                var d = gi[b * DIM + base + k] - ref_g
                var ad = d if d >= Scalar[DT](0) else -d
                if ad > max_diff:
                    max_diff = ad
    print("  max |gi - ref| =", max_diff)
    assert_true(
        max_diff < Scalar[DT](1e-6),
        "SimNorm backward should match analytic softmax Jacobian within 1e-6",
    )
    print("  ok")


def test_grad_orthogonality() raises:
    """With constant grad_y within a group, Σ_k grad_x[g·G+k] must be 0.

    Reason: softmax outputs sum to 1, so the Jacobian's null space contains
    the all-ones vector — constant inputs in grad_y project away."""
    print("test_grad_orthogonality ...")
    comptime BATCH = 1
    comptime DIM = 6
    comptime GROUPS = 2
    comptime GROUP_SIZE = DIM // GROUPS
    comptime N = BATCH * DIM
    var s = SimNorm[DIM, GROUPS].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-0.7 + 0.4 * Float64(i))
        # Constant grad_y within each group.
        go[i] = Scalar[DT](0.5) if (i // GROUP_SIZE) % 2 == 0 else Scalar[DT](
            -1.3
        )

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    s.forward["cpu", BATCH](x_t, output=y_t)
    s.vjp["cpu", BATCH](go_t, gi_t)

    var max_sum: Scalar[DT] = 0.0
    for b in range(BATCH):
        for g in range(GROUPS):
            var base = g * GROUP_SIZE
            var total: Scalar[DT] = 0.0
            for k in range(GROUP_SIZE):
                total += gi[b * DIM + base + k]
            var at = total if total >= Scalar[DT](0) else -total
            if at > max_sum:
                max_sum = at
    print("  max |Σ_k gi_gk| =", max_sum)
    assert_true(
        max_sum < Scalar[DT](1e-6),
        "SimNorm grad_x should be orthogonal to 1 within each group",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("SimNorm[DIM, GROUPS] smoke (Phase 2, PORTING_PLAN.md)")
    print("=" * 70)
    test_forward_sums_to_one()
    test_backward_parity()
    test_grad_orthogonality()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
