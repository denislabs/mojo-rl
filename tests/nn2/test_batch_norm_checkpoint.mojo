"""BatchNorm running-stats checkpoint round-trip (M1).

Before M1 the BN running mean/var were side-channel List/DeviceBuffer
fields that the `for_each_param`-based v2 checkpoint walk never saw — so
eval-mode inference after save/load silently used the default 0/1 stats.
M1 makes them decay-exempt zero-grad `Param`s, so they ride the existing
v2 envelope.

Gates:
  (1) Run BN forward in TRAINING mode several times so the EMA running
      stats drift away from their 0/1 init.
  (2) `save_state_v2` → fresh BN (stats at default 0/1) → `load_state_v2`.
  (3) The fresh BN's running_mean/var must MATCH the trained BN exactly
      (the core regression: a non-persisted stat would stay 0/1 here).
  (4) An EVAL-mode forward (which reads ONLY running stats) on the loaded
      BN must match the trained BN bit-for-bit. A pre-load fresh BN must
      NOT match — gate sanity that (4) isn't trivially true.

Run: `pixi run mojo run -I . tests/nn2/test_batch_norm_checkpoint.mojo`
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.batch_norm_1d import BatchNorm1D
from mojo_rl.nn2.core.checkpoint import save_state_v2, load_state_v2


comptime DIM = 4
comptime BATCH = 8


def _eval_forward(
    mut bn: BatchNorm1D[DIM],
    x: UnsafePointer[Scalar[DT], MutAnyOrigin],
    y: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    bn.training = False
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    bn.forward["cpu", BATCH](x_t, output=y_t)


def main() raises:
    print("=" * 70)
    print("BatchNorm running-stats checkpoint round-trip (M1)")
    print("=" * 70)
    seed(7)

    var bn = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    # Give gamma/beta non-trivial values so the eval forward depends on
    # them too (not just on the running stats).
    var g_ptr = bn.gamma.value_unsafe_ptr_cpu()
    var b_ptr = bn.beta.value_unsafe_ptr_cpu()
    for f in range(DIM):
        g_ptr[f] = Scalar[DT](0.5 + 0.1 * Float64(f))
        b_ptr[f] = Scalar[DT](-0.2 + 0.05 * Float64(f))

    # (1) Train-mode forwards to move the EMA running stats off 0/1.
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    bn.training = True
    for _step in range(20):
        for i in range(BATCH * DIM):
            x[i] = Scalar[DT](random_float64() * 3.0 - 1.0)
        var x_t = TileTensor(x, row_major[BATCH, DIM]())
        var y_t = TileTensor(y, row_major[BATCH, DIM]())
        bn.forward["cpu", BATCH](x_t, output=y_t)

    print(
        "  trained running_mean[0..1] =",
        bn.running_mean.t.cpu[0], bn.running_mean.t.cpu[1],
    )
    # Sanity: the EMA actually moved the stats off their 0/1 init.
    assert_true(
        bn.running_mean.t.cpu[0] != Scalar[DT](0.0)
        or bn.running_var.t.cpu[0] != Scalar[DT](1.0),
        "running stats should have drifted off 0/1 after training",
    )

    # Capture a fixed eval probe through the TRAINED net.
    var probe: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    for i in range(BATCH * DIM):
        probe[i] = Scalar[DT](0.3 * Float64(i % DIM) - 0.4)
    var y_orig: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    _eval_forward(bn, probe, y_orig)

    # (2) Save the trained BN.
    var path = String("/tmp/bn_ckpt.txt")
    save_state_v2(bn, path)

    # Fresh BN: running stats at the 0/1 default + default gamma/beta.
    var fresh = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    var fg = fresh.gamma.value_unsafe_ptr_cpu()
    var fb = fresh.beta.value_unsafe_ptr_cpu()
    for f in range(DIM):
        fg[f] = Scalar[DT](0.5 + 0.1 * Float64(f))
        fb[f] = Scalar[DT](-0.2 + 0.05 * Float64(f))
    var y_fresh_pre: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
        Scalar[DT]
    ](BATCH * DIM)
    _eval_forward(fresh, probe, y_fresh_pre)
    # Gate sanity: pre-load eval differs (default 0/1 stats ≠ trained).
    var pre_diff: Scalar[DT] = 0.0
    for i in range(BATCH * DIM):
        var d = y_fresh_pre[i] - y_orig[i]
        pre_diff += d if d >= Scalar[DT](0) else -d
    assert_true(
        pre_diff > Scalar[DT](1e-4),
        "pre-load fresh BN should differ from trained (gate sanity)",
    )

    # (3) Load → running stats must match the trained BN exactly.
    load_state_v2(fresh, path)
    var stat_diff: Scalar[DT] = 0.0
    for f in range(DIM):
        var dm = fresh.running_mean.t.cpu[f] - bn.running_mean.t.cpu[f]
        var dv = fresh.running_var.t.cpu[f] - bn.running_var.t.cpu[f]
        stat_diff += (dm if dm >= 0 else -dm) + (dv if dv >= 0 else -dv)
    print("  |loaded - trained| running stats =", stat_diff)
    assert_true(
        stat_diff == Scalar[DT](0.0),
        "loaded running stats must match the trained BN (M1 regression)",
    )

    # (4) Eval forward through the loaded BN must match the trained BN.
    var y_loaded: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    _eval_forward(fresh, probe, y_loaded)
    var max_err: Scalar[DT] = 0.0
    for i in range(BATCH * DIM):
        var d = y_loaded[i] - y_orig[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_err:
            max_err = ad
    print("  max |y_loaded - y_orig| (eval mode) =", max_err)
    assert_true(
        max_err == Scalar[DT](0.0),
        "eval-mode forward after load must match the trained BN bit-for-bit",
    )

    x.free(); y.free(); probe.free()
    y_orig.free(); y_fresh_pre.free(); y_loaded.free()
    print("=" * 70)
    print("PASS — BN running stats survive the v2 checkpoint round-trip")
    print("=" * 70)
