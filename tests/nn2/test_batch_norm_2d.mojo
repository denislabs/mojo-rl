"""BatchNorm2D[C, H, W] smoke + parity tests (Phase 5, PORTING_PLAN.md).

Mirrors `test_batch_norm_1d.mojo` for the spatial case. Stats are
reduced over BATCH and H·W per channel, so `N_eff = BATCH·H·W`.
"""

from std.math import sqrt
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.batch_norm_2d import BatchNorm2D
from mojo_rl.nn2.initializer import Zero


def test_train_normalizes_per_channel() raises:
    print("test_train_normalizes_per_channel ...")
    comptime BATCH = 8
    comptime C = 3
    comptime HH = 2
    comptime WW = 2
    comptime FLAT = C * HH * WW
    comptime N = BATCH * FLAT
    comptime N_PER_CH = BATCH * HH * WW
    var bn = BatchNorm2D[C, HH, WW].make[target="cpu", INIT=Zero]()
    bn.training = True

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    # Per-channel distinct means + per-position spread.
    for b in range(BATCH):
        for c in range(C):
            for s in range(HH * WW):
                var base = Float64(c) * 2.0  # channel-level mean offset
                var jitter = 0.1 * Float64(b * (HH * WW) + s)
                x[b * FLAT + c * (HH * WW) + s] = Scalar[DT](base + jitter)

    var x_t = TileTensor(x, row_major[BATCH, FLAT]())
    var y_t = TileTensor(y, row_major[BATCH, FLAT]())
    bn.forward["cpu", BATCH](x_t, output=y_t)

    var max_mean: Scalar[DT] = 0.0
    var max_var:  Scalar[DT] = 0.0
    for c in range(C):
        var mean: Scalar[DT] = 0.0
        for b in range(BATCH):
            for s in range(HH * WW):
                mean += y[b * FLAT + c * (HH * WW) + s]
        mean = mean / Scalar[DT](Float64(N_PER_CH))
        var var_: Scalar[DT] = 0.0
        for b in range(BATCH):
            for s in range(HH * WW):
                var d = y[b * FLAT + c * (HH * WW) + s] - mean
                var_ += d * d
        var_ = var_ / Scalar[DT](Float64(N_PER_CH))
        var am = mean if mean >= Scalar[DT](0) else -mean
        if am > max_mean:
            max_mean = am
        var dv = var_ - Scalar[DT](1.0)
        var adv = dv if dv >= Scalar[DT](0) else -dv
        if adv > max_var:
            max_var = adv
    print("  max |mean| =", max_mean, "  max |var-1| =", max_var)
    assert_true(
        max_mean < Scalar[DT](1e-5)
        and max_var < Scalar[DT](1e-4),
        "BN2D train output should be zero-mean unit-var per channel",
    )
    print("  ok")


def test_running_stats_converge() raises:
    print("test_running_stats_converge ...")
    comptime BATCH = 4
    comptime C = 2
    comptime HH = 2
    comptime WW = 2
    comptime FLAT = C * HH * WW
    comptime N = BATCH * FLAT
    var bn = BatchNorm2D[C, HH, WW].make[target="cpu", INIT=Zero]()
    bn.training = True

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for b in range(BATCH):
        for c in range(C):
            for s in range(HH * WW):
                var base = Scalar[DT](2.0) if c == 0 else Scalar[DT](-1.0)
                var jit = Scalar[DT](0.1 * Float64(b * (HH * WW) + s))
                x[b * FLAT + c * (HH * WW) + s] = base + jit

    var x_t = TileTensor(x, row_major[BATCH, FLAT]())
    var y_t = TileTensor(y, row_major[BATCH, FLAT]())
    for _ in range(200):
        bn.forward["cpu", BATCH](x_t, output=y_t)

    for c in range(C):
        var n_eff = Scalar[DT](Float64(BATCH * HH * WW))
        var true_mean: Scalar[DT] = 0.0
        for b in range(BATCH):
            for s in range(HH * WW):
                true_mean += x[b * FLAT + c * (HH * WW) + s]
        true_mean = true_mean / n_eff
        var true_var: Scalar[DT] = 0.0
        for b in range(BATCH):
            for s in range(HH * WW):
                var d = x[b * FLAT + c * (HH * WW) + s] - true_mean
                true_var += d * d
        true_var = true_var / n_eff
        var dm = bn.running_mean.val.cpu[c] - true_mean
        var adm = dm if dm >= Scalar[DT](0) else -dm
        var dv = bn.running_var.val.cpu[c] - true_var
        var adv = dv if dv >= Scalar[DT](0) else -dv
        print(
            "  ch ", c,
            ": μ run=", bn.running_mean.val.cpu[c], " true=", true_mean,
            "  σ² run=", bn.running_var.val.cpu[c], " true=", true_var,
        )
        assert_true(
            adm < Scalar[DT](1e-3),
            "BN2D running mean should converge",
        )
        assert_true(
            adv < Scalar[DT](1e-3),
            "BN2D running var should converge",
        )
    print("  ok")


def test_backward_fd() raises:
    print("test_backward_fd ...")
    comptime BATCH = 2
    comptime C = 2
    comptime HH = 2
    comptime WW = 2
    comptime FLAT = C * HH * WW
    comptime N = BATCH * FLAT
    var eps = Scalar[DT](1e-2)
    var tol = Scalar[DT](2e-2)

    var bn = BatchNorm2D[C, HH, WW].make[target="cpu", INIT=Zero]()
    bn.training = True
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_pos: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_neg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-0.5 + 0.13 * Float64(i))
        go[i] = Scalar[DT](0.3 + 0.07 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, FLAT]())
    var xp_t = TileTensor(x_p, row_major[BATCH, FLAT]())
    var y_t = TileTensor(y, row_major[BATCH, FLAT]())
    var ypos_t = TileTensor(y_pos, row_major[BATCH, FLAT]())
    var yneg_t = TileTensor(y_neg, row_major[BATCH, FLAT]())
    var go_t = TileTensor(go, row_major[BATCH, FLAT]())
    var gi_t = TileTensor(gi, row_major[BATCH, FLAT]())

    bn.forward["cpu", BATCH](x_t, output=y_t)
    bn.zero_grad["cpu"]()
    bn.vjp["cpu", BATCH](go_t, gi_t)

    var max_gi: Scalar[DT] = 0.0
    for i in range(N):
        var bn_p = BatchNorm2D[C, HH, WW].make[
            target="cpu", INIT=Zero,
        ]()
        var bn_n = BatchNorm2D[C, HH, WW].make[
            target="cpu", INIT=Zero,
        ]()
        bn_p.training = True
        bn_n.training = True
        for j in range(N):
            x_p[j] = x[j]
        x_p[i] = x[i] + eps
        bn_p.forward["cpu", BATCH](xp_t, output=ypos_t)
        x_p[i] = x[i] - eps
        bn_n.forward["cpu", BATCH](xp_t, output=yneg_t)
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
        "BN2D grad_input FD gradcheck failed",
    )

    # dgamma / dbeta per channel.
    var max_dg: Scalar[DT] = 0.0
    var max_db: Scalar[DT] = 0.0
    for c in range(C):
        var bn_p = BatchNorm2D[C, HH, WW].make[
            target="cpu", INIT=Zero,
        ]()
        var bn_n = BatchNorm2D[C, HH, WW].make[
            target="cpu", INIT=Zero,
        ]()
        bn_p.training = True
        bn_n.training = True
        bn_p.gamma.val.cpu[c] = Scalar[DT](1.0) + eps
        bn_n.gamma.val.cpu[c] = Scalar[DT](1.0) - eps
        bn_p.forward["cpu", BATCH](x_t, output=ypos_t)
        bn_n.forward["cpu", BATCH](x_t, output=yneg_t)
        var fd_dg: Scalar[DT] = 0.0
        for k in range(N):
            fd_dg += go[k] * (y_pos[k] - y_neg[k])
        fd_dg = fd_dg / (Scalar[DT](2.0) * eps)
        var d = bn.gamma.grd.cpu[c] - fd_dg
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_dg:
            max_dg = ad

        var bn_p2 = BatchNorm2D[C, HH, WW].make[
            target="cpu", INIT=Zero,
        ]()
        var bn_n2 = BatchNorm2D[C, HH, WW].make[
            target="cpu", INIT=Zero,
        ]()
        bn_p2.training = True
        bn_n2.training = True
        bn_p2.beta.val.cpu[c] = eps
        bn_n2.beta.val.cpu[c] = -eps
        bn_p2.forward["cpu", BATCH](x_t, output=ypos_t)
        bn_n2.forward["cpu", BATCH](x_t, output=yneg_t)
        var fd_db: Scalar[DT] = 0.0
        for k in range(N):
            fd_db += go[k] * (y_pos[k] - y_neg[k])
        fd_db = fd_db / (Scalar[DT](2.0) * eps)
        var d2 = bn.beta.grd.cpu[c] - fd_db
        var ad2 = d2 if d2 >= Scalar[DT](0) else -d2
        if ad2 > max_db:
            max_db = ad2
    print("  max |dgamma - fd| =", max_dg, "  max |dbeta - fd| =", max_db)
    assert_true(
        max_dg < tol and max_db < tol,
        "BN2D dgamma/dbeta FD gradcheck failed",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("BatchNorm2D[C, H, W] smoke (Phase 5, PORTING_PLAN.md)")
    print("=" * 70)
    test_train_normalizes_per_channel()
    test_running_stats_converge()
    test_backward_fd()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
