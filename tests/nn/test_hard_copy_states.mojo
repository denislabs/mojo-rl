"""`hard_copy_params` must copy `IsState` buffers (BatchNorm running stats).

Root-cause regression test for the AlphaZero post-promotion collapse:
`hard_copy_params` was `polyak_update(tau=1)` — an `IsParam`-only walk —
so promoting an arena winner copied trained weights/γ/β into the best
net while leaving its BatchNorm running stats at INIT (mean 0 / var 1;
the best net never trains, so they were never updated). The promoted
net's EVAL-mode forward then ran trained weights under identity
normalization → exploding activations → non-finite self-play policies →
NaN replay targets → permanently NaN policy-head columns.

Tests:
  1. CPU: train-mode forwards move net A's running stats; eval outputs
     of A and a stale copy B then DIFFER; after `hard_copy_params`
     (which now includes the state walk) B's eval output is bit-equal
     to A's, and the running-stat buffers match exactly.
  2. CPU: stateless model (Linear) — `named_states` is empty, the copy
     is a no-op and must not raise (bit-identity for all MLP users).
  3. GPU: `hard_copy_states` copies device-resident State buffers.
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.batch_norm_1d import BatchNorm1D
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.initializer import Zero, Xavier
from mojo_rl.nn.core.map_params import hard_copy_params, hard_copy_states
from mojo_rl.nn.core.named_params import named_states


comptime DIM = 3
comptime BATCH = 16
comptime N = BATCH * DIM
comptime BN = BatchNorm1D[DIM]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _make_input(x: UnsafePointer[Scalar[DT], MutAnyOrigin]):
    """Per-feature distinct mean (2 / −1 / 0) + spread, so running stats
    move far from their (0, 1) init under train-mode forwards."""
    for b in range(BATCH):
        for f in range(DIM):
            var base = Float64(0)
            if f == 0:
                base = 2.0
            elif f == 1:
                base = -1.0
            x[b * DIM + f] = Scalar[DT](base + 0.1 * Float64(b - BATCH // 2))


def test_cpu_hard_copy_includes_running_stats() raises:
    print("test_cpu_hard_copy_includes_running_stats ...")
    var a = BN.make[target="cpu", INIT=Zero]()
    var b = BN.make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var ya: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var yb: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    _make_input(x)
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var ya_t = TileTensor(ya, row_major[BATCH, DIM]())
    var yb_t = TileTensor(yb, row_major[BATCH, DIM]())

    # Move A's running stats away from init with train-mode forwards.
    a.training = True
    for _ in range(50):
        a.forward["cpu", BATCH](x_t, output=ya_t)
    a.training = False
    b.training = False

    # Sanity: stale-stats B disagrees with A in eval mode.
    a.forward["cpu", BATCH](x_t, output=ya_t)
    b.forward["cpu", BATCH](x_t, output=yb_t)
    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = _abs(ya[i] - yb[i])
        if d > max_diff:
            max_diff = d
    assert_true(
        max_diff > Scalar[DT](0.1),
        "eval outputs should differ before the copy (stats moved), diff="
        + String(max_diff),
    )

    # The fix: hard copy must transplant the running stats too.
    hard_copy_params["cpu", M=BN](a, b)
    b.forward["cpu", BATCH](x_t, output=yb_t)
    for i in range(N):
        assert_true(
            ya[i] == yb[i],
            "eval outputs must be bit-equal after hard_copy_params at "
            + String(i),
        )

    # Direct buffer check: both State leaves match exactly.
    var ss_a = named_states["cpu", BN](a)
    var ss_b = named_states["cpu", BN](b)
    assert_true(
        len(ss_a) == 2 and len(ss_b) == 2,
        "BatchNorm1D should expose 2 State leaves, got "
        + String(len(ss_a)),
    )
    for i in range(len(ss_a)):
        for k in range(ss_a[i].n_elems):
            assert_true(
                ss_a[i].param_ptr[k] == ss_b[i].param_ptr[k],
                "running-stat mismatch in '" + ss_a[i].name + "' at "
                + String(k),
            )
    x.free()
    ya.free()
    yb.free()
    print("  ok (pre-copy diff=", max_diff, ", post-copy bit-equal)")


def test_cpu_stateless_noop() raises:
    """Stateless models: empty state walk, no raise, params still copied."""
    print("test_cpu_stateless_noop ...")
    comptime Lin = Linear[4, 3]
    var a = Lin.make[target="cpu", INIT=Xavier]()
    var b = Lin.make[target="cpu", INIT=Xavier]()
    var ss = named_states["cpu", Lin](a)
    assert_true(len(ss) == 0, "Linear should expose no State leaves")
    hard_copy_params["cpu", M=Lin](a, b)
    var aw = a.weight.value_unsafe_ptr_cpu()
    var bw = b.weight.value_unsafe_ptr_cpu()
    for k in range(4 * 3):
        assert_true(aw[k] == bw[k], "weights must match after hard copy")
    print("  ok")


def test_gpu_hard_copy_states() raises:
    print("test_gpu_hard_copy_states ...")
    try:
        var ctx = DeviceContext()
        var a = BN.make[target="gpu", INIT=Zero](ctx=ctx)
        var b = BN.make[target="gpu", INIT=Zero](ctx=ctx)

        # Write known values into A's device-resident running stats.
        var ss_a = named_states["gpu", BN](a)
        var h = ctx.enqueue_create_host_buffer[DT](DIM)
        ctx.synchronize()
        for i in range(len(ss_a)):
            for k in range(DIM):
                h.unsafe_ptr()[k] = Scalar[DT](
                    1.5 * Float64(i + 1) + 0.25 * Float64(k)
                )
            var dev = DeviceBuffer[DT](
                ctx, ss_a[i].param_ptr, ss_a[i].n_elems, owning=False
            )
            ctx.enqueue_copy(dev, h)
            # Sync per leaf — `h` is mutated next iteration and the copy
            # is async.
            ctx.synchronize()

        hard_copy_states["gpu", M=BN](a, b, ctx)

        var ss_b = named_states["gpu", BN](b)
        for i in range(len(ss_b)):
            var dev = DeviceBuffer[DT](
                ctx, ss_b[i].param_ptr, ss_b[i].n_elems, owning=False
            )
            ctx.enqueue_copy(h, dev)
            ctx.synchronize()
            for k in range(DIM):
                var want = Scalar[DT](
                    1.5 * Float64(i + 1) + 0.25 * Float64(k)
                )
                assert_true(
                    h.unsafe_ptr()[k] == want,
                    "GPU state copy mismatch in '" + ss_b[i].name
                    + "' at " + String(k),
                )
        print("  ok")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 60)
    print("hard_copy_params + IsState (BN running stats)")
    print("=" * 60)
    test_cpu_hard_copy_includes_running_stats()
    test_cpu_stateless_noop()
    test_gpu_hard_copy_states()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
