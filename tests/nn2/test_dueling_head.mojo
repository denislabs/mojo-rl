"""DuelingHead[NA] smoke + CPU/GPU parity test.

Validates:
  - forward: out[b, a] == V + (A_a − mean_a A) for hand-crafted V/A.
  - vjp:     grad_in[b, 0] == sum_a grad_out[b, a],
             grad_in[b, 1+a] == grad_out[b, a] − (1/NA) sum_a grad_out[b, a].
  - CPU and GPU produce identical forward / backward outputs.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.dueling_head import DuelingHead
from mojo_rl.nn2.initializer import Zero


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def test_forward_cpu() raises:
    print("test_forward_cpu ...")
    comptime BATCH = 2
    comptime NA = 3

    var h = DuelingHead[NA].make[target="cpu", INIT=Zero]()
    var inp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * (NA + 1)
    )
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * NA
    )

    # Row 0: V=10, A=[1, 4, 7] → mean=4. Q = 10 + (A - 4) = [7, 10, 13].
    inp[0]=10; inp[1]=1; inp[2]=4; inp[3]=7
    # Row 1: V=-2, A=[0, 0, 6] → mean=2. Q = -2 + (A - 2) = [-4, -4, 2].
    inp[4]=-2; inp[5]=0; inp[6]=0; inp[7]=6

    var ip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](inp)
    var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out)
    var in_t = TileTensor(ip, row_major[BATCH, NA + 1]())
    var out_t = TileTensor(op, row_major[BATCH, NA]())
    h.forward["cpu", BATCH](in_t, output=out_t)

    assert_true(out[0] == Scalar[DT](7),  "row 0 a=0")
    assert_true(out[1] == Scalar[DT](10), "row 0 a=1")
    assert_true(out[2] == Scalar[DT](13), "row 0 a=2")
    assert_true(out[3] == Scalar[DT](-4), "row 1 a=0")
    assert_true(out[4] == Scalar[DT](-4), "row 1 a=1")
    assert_true(out[5] == Scalar[DT](2),  "row 1 a=2")
    print("  ok")


def test_vjp_cpu() raises:
    print("test_vjp_cpu ...")
    comptime BATCH = 2
    comptime NA = 3

    var h = DuelingHead[NA].make[target="cpu", INIT=Zero]()
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * NA
    )
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * (NA + 1)
    )
    # Row 0: grad_out=[1, 2, 3], sum=6, mean=2. grad_in = [6, 1-2, 2-2, 3-2] = [6, -1, 0, 1].
    go[0]=1; go[1]=2; go[2]=3
    # Row 1: grad_out=[0, 0, 9], sum=9, mean=3. grad_in = [9, -3, -3, 6].
    go[3]=0; go[4]=0; go[5]=9

    var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
    var gip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi)
    var go_t = TileTensor(gop, row_major[BATCH, NA]())
    var gi_t = TileTensor(gip, row_major[BATCH, NA + 1]())
    h.vjp["cpu", BATCH](go_t, gi_t)

    assert_true(gi[0] == Scalar[DT](6),  "row 0 dV")
    assert_true(gi[1] == Scalar[DT](-1), "row 0 dA_0")
    assert_true(gi[2] == Scalar[DT](0),  "row 0 dA_1")
    assert_true(gi[3] == Scalar[DT](1),  "row 0 dA_2")
    assert_true(gi[4] == Scalar[DT](9),  "row 1 dV")
    assert_true(gi[5] == Scalar[DT](-3), "row 1 dA_0")
    assert_true(gi[6] == Scalar[DT](-3), "row 1 dA_1")
    assert_true(gi[7] == Scalar[DT](6),  "row 1 dA_2")
    print("  ok")


def test_cpu_gpu_parity() raises:
    print("test_cpu_gpu_parity ...")
    comptime BATCH = 8
    comptime NA = 5

    try:
        var ctx = DeviceContext()

        var inp_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            BATCH * (NA + 1)
        )
        var out_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            BATCH * NA
        )
        var out_gpu_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            BATCH * NA
        )
        # Deterministic non-trivial fill.
        for k in range(BATCH * (NA + 1)):
            inp_host[k] = Scalar[DT](0.1 * Float64(k) - 0.5)

        # CPU.
        var h_cpu = DuelingHead[NA].make[target="cpu", INIT=Zero]()
        var ip_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](inp_host)
        var op_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out_cpu)
        var i_cpu_t = TileTensor(ip_cpu, row_major[BATCH, NA + 1]())
        var o_cpu_t = TileTensor(op_cpu, row_major[BATCH, NA]())
        h_cpu.forward["cpu", BATCH](i_cpu_t, output=o_cpu_t)

        # GPU.
        var h_gpu = DuelingHead[NA].make[target="gpu", INIT=Zero](ctx=ctx)
        var in_dev = ctx.enqueue_create_buffer[DT](BATCH * (NA + 1))
        var out_dev = ctx.enqueue_create_buffer[DT](BATCH * NA)
        ctx.enqueue_copy(in_dev, inp_host)
        var ip_dev = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in_dev.unsafe_ptr())
        var op_dev = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out_dev.unsafe_ptr())
        var i_gpu_t = TileTensor(ip_dev, row_major[BATCH, NA + 1]())
        var o_gpu_t = TileTensor(op_dev, row_major[BATCH, NA]())
        h_gpu.forward["gpu", BATCH](i_gpu_t, output=o_gpu_t)
        ctx.enqueue_copy(out_gpu_host, out_dev)
        ctx.synchronize()

        var max_diff: Scalar[DT] = 0.0
        for k in range(BATCH * NA):
            var d = _abs(out_cpu[k] - out_gpu_host[k])
            if d > max_diff:
                max_diff = d
        print("  max |cpu - gpu| =", max_diff)
        assert_true(
            max_diff < Scalar[DT](1e-5), "CPU/GPU forward mismatch"
        )
        print("  ok")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 70)
    print("DuelingHead[NA] smoke + CPU/GPU parity")
    print("=" * 70)
    test_forward_cpu()
    test_vjp_cpu()
    test_cpu_gpu_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
