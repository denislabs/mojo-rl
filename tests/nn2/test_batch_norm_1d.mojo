"""BatchNorm1D[DIM] smoke + parity tests (Phase 4, PORTING_PLAN.md).

Validates:
  1. **Train forward** produces per-feature zero-mean unit-var x̂, and
     `y = γ·x̂ + β` ≡ x̂ when γ=1, β=0.
  2. **Running stats** EMA-update under repeated train forwards with the
     same batch — converges to the batch's true (μ, σ²).
  3. **Eval forward** uses the running stats (not the batch stats) and
     does NOT bump them.
  4. **Backward** FD-gradchecks dgamma, dbeta, and grad_input.
  5. **Programming error**: calling vjp after an eval-only forward
     raises (cache_is_training=False).
"""

from std.math import sqrt
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.batch_norm_1d import BatchNorm1D
from mojo_rl.nn2.initializer import Zero


def _fill_linear(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, a: Float64, b: Float64) raises:
    for i in range(n):
        p[i] = Scalar[DT](a + b * Float64(i))


def test_train_normalizes_per_feature() raises:
    print("test_train_normalizes_per_feature ...")
    comptime BATCH = 32
    comptime DIM = 4
    comptime N = BATCH * DIM
    var bn = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    bn.training = True

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    # Per-feature distinct mean + spread.
    for b in range(BATCH):
        for f in range(DIM):
            x[b * DIM + f] = Scalar[DT](
                Float64(f) + 0.1 * Float64(b - BATCH // 2)
            )

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    bn.forward["cpu", BATCH](x_t, output=y_t)

    var max_mean_err: Scalar[DT] = 0.0
    var max_var_err: Scalar[DT] = 0.0
    for f in range(DIM):
        var mean: Scalar[DT] = 0.0
        for b in range(BATCH):
            mean += y[b * DIM + f]
        mean = mean / Scalar[DT](Float64(BATCH))
        var var_: Scalar[DT] = 0.0
        for b in range(BATCH):
            var d = y[b * DIM + f] - mean
            var_ += d * d
        var_ = var_ / Scalar[DT](Float64(BATCH))
        var am = mean if mean >= Scalar[DT](0) else -mean
        if am > max_mean_err:
            max_mean_err = am
        var dv = var_ - Scalar[DT](1.0)
        var adv = dv if dv >= Scalar[DT](0) else -dv
        # γ=1, β=0 → expect zero-mean unit-var per feature.
        if adv > max_var_err:
            max_var_err = adv
    print("  max |mean| =", max_mean_err, "  max |var-1| =", max_var_err)
    assert_true(
        max_mean_err < Scalar[DT](1e-5),
        "BN train output should be zero-mean per feature",
    )
    assert_true(
        max_var_err < Scalar[DT](1e-4),
        "BN train output should be unit-variance per feature",
    )
    print("  ok")


def test_running_stats_converge() raises:
    """After many train forwards with the same batch the EMA settles to
    the batch's true (μ, σ²)."""
    print("test_running_stats_converge ...")
    comptime BATCH = 16
    comptime DIM = 3
    comptime N = BATCH * DIM
    var bn = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    bn.training = True

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for b in range(BATCH):
        for f in range(DIM):
            # Feature 0: μ=2, feature 1: μ=-1, feature 2: μ=0; spread σ²≈1.
            var base = Float64(0)
            if f == 0:
                base = 2.0
            elif f == 1:
                base = -1.0
            x[b * DIM + f] = Scalar[DT](base + 0.1 * Float64(b - BATCH // 2))
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())

    for _ in range(200):
        bn.forward["cpu", BATCH](x_t, output=y_t)

    # Compute true (μ, σ²) per feature.
    for f in range(DIM):
        var true_mean: Scalar[DT] = 0.0
        for b in range(BATCH):
            true_mean += x[b * DIM + f]
        true_mean = true_mean / Scalar[DT](Float64(BATCH))
        var true_var: Scalar[DT] = 0.0
        for b in range(BATCH):
            var d = x[b * DIM + f] - true_mean
            true_var += d * d
        true_var = true_var / Scalar[DT](Float64(BATCH))
        var rm = bn.running_mean.val.cpu[f]
        var rv = bn.running_var.val.cpu[f]
        var dm = rm - true_mean
        var adm = dm if dm >= Scalar[DT](0) else -dm
        var dv = rv - true_var
        var adv = dv if dv >= Scalar[DT](0) else -dv
        print(
            "  feature ", f,
            ": μ run=", rm, " true=", true_mean,
            "  σ² run=", rv, " true=", true_var,
        )
        # After 200 EMA updates with momentum=0.1, (1-0.1)^200 ≈ 7e-10
        # weight remains on the initial estimate; settles tight to truth.
        assert_true(
            adm < Scalar[DT](1e-3),
            "Running mean should converge to batch mean",
        )
        assert_true(
            adv < Scalar[DT](1e-3),
            "Running var should converge to batch var",
        )
    print("  ok")


def test_eval_uses_running_stats() raises:
    print("test_eval_uses_running_stats ...")
    comptime BATCH = 8
    comptime DIM = 2
    comptime N = BATCH * DIM
    var bn = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    # Set running stats explicitly so the math is deterministic.
    bn.running_mean.val.cpu[0] = Scalar[DT](1.0)
    bn.running_mean.val.cpu[1] = Scalar[DT](-2.0)
    bn.running_var.val.cpu[0]  = Scalar[DT](4.0)
    bn.running_var.val.cpu[1]  = Scalar[DT](0.25)
    bn.training = False

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for b in range(BATCH):
        x[b * DIM + 0] = Scalar[DT](3.0)   # (3-1)/√(4+ε) ≈ 1.0
        x[b * DIM + 1] = Scalar[DT](-1.5)  # (-1.5+2)/√(0.25+ε) ≈ 1.0

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    bn.forward["cpu", BATCH](x_t, output=y_t)

    var max_err: Scalar[DT] = 0.0
    var inv0 = Scalar[DT](1.0) / sqrt(Scalar[DT](4.0 + 1e-5))
    var inv1 = Scalar[DT](1.0) / sqrt(Scalar[DT](0.25 + 1e-5))
    for b in range(BATCH):
        var e0 = (Scalar[DT](3.0) - Scalar[DT](1.0)) * inv0
        var e1 = (Scalar[DT](-1.5) - Scalar[DT](-2.0)) * inv1
        var d0 = y[b * DIM + 0] - e0
        var d1 = y[b * DIM + 1] - e1
        var ad0 = d0 if d0 >= Scalar[DT](0) else -d0
        var ad1 = d1 if d1 >= Scalar[DT](0) else -d1
        if ad0 > max_err:
            max_err = ad0
        if ad1 > max_err:
            max_err = ad1
    print("  max |y - expected| =", max_err)
    assert_true(
        max_err < Scalar[DT](1e-6),
        "Eval BN should use running stats exactly",
    )
    # Running stats should not have moved (eval mode).
    assert_true(
        bn.running_mean.val.cpu[0] == Scalar[DT](1.0),
        "Eval mode must not update running_mean",
    )
    print("  ok")


def test_backward_fd() raises:
    """FD gradcheck over a moderate-sized batch."""
    print("test_backward_fd ...")
    comptime BATCH = 8
    comptime DIM = 4
    comptime N = BATCH * DIM
    var eps = Scalar[DT](1e-2)
    var tol = Scalar[DT](2e-2)
    var bn = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    bn.training = True

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_pos: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_neg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    _fill_linear(x, N, -1.0, 0.07)
    _fill_linear(go, N, 0.4, 0.05)

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var xp_t = TileTensor(x_p, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var ypos_t = TileTensor(y_pos, row_major[BATCH, DIM]())
    var yneg_t = TileTensor(y_neg, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    bn.forward["cpu", BATCH](x_t, output=y_t)
    bn.zero_grad["cpu"]()
    bn.vjp["cpu", BATCH](go_t, gi_t)

    # FD per lane on a fresh BN per perturbation (otherwise EMA contaminates).
    var max_gi: Scalar[DT] = 0.0
    for i in range(N):
        var bn_pos = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
        var bn_neg = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
        bn_pos.training = True
        bn_neg.training = True
        for j in range(N):
            x_p[j] = x[j]
        x_p[i] = x[i] + eps
        bn_pos.forward["cpu", BATCH](xp_t, output=ypos_t)
        x_p[i] = x[i] - eps
        bn_neg.forward["cpu", BATCH](xp_t, output=yneg_t)
        var fd: Scalar[DT] = 0.0
        for k in range(N):
            fd += go[k] * (y_pos[k] - y_neg[k])
        fd = fd / (Scalar[DT](2.0) * eps)
        var d = gi[i] - fd
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_gi:
            max_gi = ad
    print("  max |gi - fd| =", max_gi, " (tol=", tol, ")")
    assert_true(
        max_gi < tol,
        "BN grad_input FD gradcheck failed",
    )

    # FD-check dgamma + dbeta via Σ go·y, perturbing γ/β.
    var bn2 = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    bn2.training = True
    bn2.forward["cpu", BATCH](x_t, output=y_t)
    bn2.zero_grad["cpu"]()
    bn2.vjp["cpu", BATCH](go_t, gi_t)

    var max_dg: Scalar[DT] = 0.0
    var max_db: Scalar[DT] = 0.0
    for f in range(DIM):
        # dgamma[f]
        var bn_p = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
        var bn_n = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
        bn_p.training = True
        bn_n.training = True
        bn_p.gamma.val.cpu[f] = Scalar[DT](1.0) + eps
        bn_n.gamma.val.cpu[f] = Scalar[DT](1.0) - eps
        bn_p.forward["cpu", BATCH](x_t, output=ypos_t)
        bn_n.forward["cpu", BATCH](x_t, output=yneg_t)
        var fd_dg: Scalar[DT] = 0.0
        for k in range(N):
            fd_dg += go[k] * (y_pos[k] - y_neg[k])
        fd_dg = fd_dg / (Scalar[DT](2.0) * eps)
        var d = bn2.gamma.grd.cpu[f] - fd_dg
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_dg:
            max_dg = ad

        # dbeta[f]
        var bn_p2 = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
        var bn_n2 = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
        bn_p2.training = True
        bn_n2.training = True
        bn_p2.beta.val.cpu[f] = eps
        bn_n2.beta.val.cpu[f] = -eps
        bn_p2.forward["cpu", BATCH](x_t, output=ypos_t)
        bn_n2.forward["cpu", BATCH](x_t, output=yneg_t)
        var fd_db: Scalar[DT] = 0.0
        for k in range(N):
            fd_db += go[k] * (y_pos[k] - y_neg[k])
        fd_db = fd_db / (Scalar[DT](2.0) * eps)
        var d2 = bn2.beta.grd.cpu[f] - fd_db
        var ad2 = d2 if d2 >= Scalar[DT](0) else -d2
        if ad2 > max_db:
            max_db = ad2
    print("  max |dgamma - fd| =", max_dg, "  max |dbeta - fd| =", max_db)
    assert_true(
        max_dg < tol,
        "BN dgamma FD gradcheck failed",
    )
    assert_true(
        max_db < tol,
        "BN dbeta FD gradcheck failed",
    )
    print("  ok")


def test_vjp_without_training_cache_raises() raises:
    print("test_vjp_without_training_cache_raises ...")
    comptime BATCH = 4
    comptime DIM = 2
    comptime N = BATCH * DIM
    var bn = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    bn.training = False

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](0.5)
        go[i] = Scalar[DT](1.0)
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())
    bn.forward["cpu", BATCH](x_t, output=y_t)
    var raised = False
    try:
        bn.vjp["cpu", BATCH](go_t, gi_t)
    except _:
        raised = True
    assert_true(
        raised,
        "vjp without a training-mode cache must raise",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("BatchNorm1D[DIM] smoke (Phase 4, PORTING_PLAN.md)")
    print("=" * 70)
    test_train_normalizes_per_feature()
    test_running_stats_converge()
    test_eval_uses_running_stats()
    test_backward_fd()
    test_vjp_without_training_cache_raises()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
