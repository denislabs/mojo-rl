"""Dropout[DIM, p, SEED] smoke + statistical test (Phase 2, PORTING_PLAN.md).

Validates:
  1. **Eval mode** is identity (no mask, no counter advance).
  2. **Train mode** zeroes ~p fraction of lanes and scales survivors
     by 1/(1-p); mean over a large batch is preserved.
  3. **Backward** is the same elementwise mask multiplication —
     `grad_x[b,i] = grad_y[b,i] · mask[b,i]` for a constant grad_y the
     surviving lanes still equal grad_y / (1-p) times survival
     indicator scaled.
  4. **Successive forward calls** see different masks (call_counter
     bump works).
  5. **set_attr["training"]** flips eval mode in and out.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.dropout import Dropout
from mojo_rl.nn.initializer import Zero


def test_eval_is_identity() raises:
    print("test_eval_is_identity ...")
    comptime BATCH = 4
    comptime DIM = 16
    comptime N = BATCH * DIM
    var d = Dropout[DIM, 0.5, 12345].make[target="cpu", INIT=Zero]()
    d.training = False

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-1.0 + 0.05 * Float64(i))
        go[i] = Scalar[DT](0.3 + 0.03 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    d.forward["cpu", BATCH](x_t, output=y_t)
    d.vjp["cpu", BATCH](go_t, gi_t)

    var max_fwd: Scalar[DT] = 0.0
    var max_bwd: Scalar[DT] = 0.0
    for i in range(N):
        var df = y[i] - x[i]
        var adf = df if df >= Scalar[DT](0) else -df
        if adf > max_fwd:
            max_fwd = adf
        var db = gi[i] - go[i]
        var adb = db if db >= Scalar[DT](0) else -db
        if adb > max_bwd:
            max_bwd = adb
    print("  max |y - x| =", max_fwd, "  max |gi - go| =", max_bwd)
    assert_true(
        max_fwd == Scalar[DT](0.0) and max_bwd == Scalar[DT](0.0),
        "Dropout eval should be identity",
    )
    assert_true(
        d.call_counter == UInt64(0),
        "Eval mode must not bump call_counter",
    )
    print("  ok")


def test_train_mean_preserved() raises:
    """Inverted dropout preserves the expectation: E[y] = x.

    Sample a large slab and check that mean(y) ≈ mean(x) to within a
    Bernoulli sampling tolerance."""
    print("test_train_mean_preserved ...")
    comptime BATCH = 64
    comptime DIM = 128
    comptime N = BATCH * DIM
    var d = Dropout[DIM, 0.3, 99].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var mean_x: Scalar[DT] = 0.0
    for i in range(N):
        x[i] = Scalar[DT](1.0)  # constant input → mean check is sharp
        mean_x += x[i]
    mean_x = mean_x / Scalar[DT](Float64(N))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    d.forward["cpu", BATCH](x_t, output=y_t)

    var mean_y: Scalar[DT] = 0.0
    var n_zero: Int = 0
    for i in range(N):
        mean_y += y[i]
        if y[i] == Scalar[DT](0.0):
            n_zero += 1
    mean_y = mean_y / Scalar[DT](Float64(N))
    var frac_zero = Float64(n_zero) / Float64(N)

    print(
        "  mean_x =", mean_x,
        "  mean_y =", mean_y,
        "  frac_zero =", frac_zero, " (target p=0.3)",
    )
    var diff = mean_y - mean_x
    var ad = diff if diff >= Scalar[DT](0) else -diff
    # 3σ Bernoulli sampling tolerance for N=8192 lanes with p=0.3:
    # σ_mean ≈ sqrt(p / ((1-p)·N)) · scale ≈ 0.013 in inverted dropout.
    assert_true(
        ad < Scalar[DT](0.03),
        "Inverted dropout should preserve mean to within 3σ",
    )
    var df = frac_zero - 0.3
    var adf = df if df >= 0.0 else -df
    assert_true(
        adf < 0.03,
        "Drop fraction should match target p within sampling tolerance",
    )
    assert_true(
        d.call_counter == UInt64(1),
        "Train forward should bump call_counter",
    )
    print("  ok")


def test_backward_matches_mask() raises:
    print("test_backward_matches_mask ...")
    comptime BATCH = 8
    comptime DIM = 32
    comptime N = BATCH * DIM
    var d = Dropout[DIM, 0.5, 7].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](1.0)  # so y[i] equals the mask value
        go[i] = Scalar[DT](0.5 + 0.01 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    d.forward["cpu", BATCH](x_t, output=y_t)
    d.vjp["cpu", BATCH](go_t, gi_t)

    # y[i] IS the mask (since x[i]=1). Check gi[i] == go[i] * y[i].
    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var expected = go[i] * y[i]
        var diff = gi[i] - expected
        var ad = diff if diff >= Scalar[DT](0) else -diff
        if ad > max_diff:
            max_diff = ad
    print("  max |gi - go·mask| =", max_diff)
    assert_true(
        max_diff < Scalar[DT](1e-6),
        "Dropout backward must equal grad_y · mask elementwise",
    )
    print("  ok")


def test_successive_calls_differ() raises:
    """Two train-mode forwards on the same input must produce different
    masks (call_counter advance)."""
    print("test_successive_calls_differ ...")
    comptime BATCH = 4
    comptime DIM = 64
    comptime N = BATCH * DIM
    var d = Dropout[DIM, 0.5, 42].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](1.0)

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y1_t = TileTensor(y1, row_major[BATCH, DIM]())
    var y2_t = TileTensor(y2, row_major[BATCH, DIM]())
    d.forward["cpu", BATCH](x_t, output=y1_t)
    d.forward["cpu", BATCH](x_t, output=y2_t)

    var n_diff: Int = 0
    for i in range(N):
        if y1[i] != y2[i]:
            n_diff += 1
    print("  lanes that changed between calls =", n_diff, " / ", N)
    # With p=0.5, both calls are coin flips → ~50% of lanes change.
    assert_true(
        n_diff > N // 4,
        "Successive Dropout calls must produce different masks",
    )
    print("  ok")


def test_set_attr_training() raises:
    print("test_set_attr_training ...")
    comptime BATCH = 2
    comptime DIM = 8
    comptime N = BATCH * DIM
    var d = Dropout[DIM, 0.5, 1].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](1.0)
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())

    # Flip to eval via set_attr.
    d.set_attr["training"](Scalar[DT](0.0))
    assert_true(not d.training, "set_attr 0.0 must set training=False")
    d.forward["cpu", BATCH](x_t, output=y_t)
    for i in range(N):
        assert_true(
            y[i] == Scalar[DT](1.0),
            "Eval mode after set_attr must be identity",
        )

    # Flip back to train via set_attr.
    d.set_attr["training"](Scalar[DT](1.0))
    assert_true(d.training, "set_attr 1.0 must set training=True")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Dropout[DIM, p, SEED] smoke (Phase 2, PORTING_PLAN.md)")
    print("=" * 70)
    test_eval_is_identity()
    test_train_mean_preserved()
    test_backward_matches_mask()
    test_successive_calls_differ()
    test_set_attr_training()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
