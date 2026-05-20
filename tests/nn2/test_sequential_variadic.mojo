"""Variadic Sequential[*MODULES] tests (CPU + GPU)."""

from std.math import abs as fabs
from std.memory import alloc
from std.random import seed, random_float64
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential


def test_seq_cpu_n3() raises:
    """3-layer Sequential on CPU: Linear → ReLU → Linear."""
    comptime IN = 4
    comptime H = 6
    comptime OUT = 3
    comptime BATCH = 2

    var net = Sequential(
        Linear[IN, H].make["cpu", INIT=Zero](),
        ReLU[H].make["cpu", INIT=Zero](),
        Linear[H, OUT].make["cpu", INIT=Zero](),
    )

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)

    for k in range(BATCH * IN): in_buf[k] = Scalar[DT](0.1 * Float32(k))
    for k in range(BATCH * OUT): go_buf[k] = 1.0

    var input  = TileTensor(in_buf,  row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, OUT]())
    var go     = TileTensor(go_buf,  row_major[BATCH, OUT]())
    var gi     = TileTensor(gi_buf,  row_major[BATCH, IN]())

    net.forward["cpu", BATCH](input, output)
    net.backward["cpu", BATCH](go, gi)

    print("  test_seq_cpu_n3 PASSED  output[0,0]=", out_buf[0])

    in_buf.free(); out_buf.free(); go_buf.free(); gi_buf.free()


def test_seq_cpu_n5() raises:
    """5-layer Sequential on CPU: Linear → ReLU → Linear → ReLU → Linear."""
    comptime IN = 3
    comptime H1 = 5
    comptime H2 = 4
    comptime OUT = 2
    comptime BATCH = 2

    var net = Sequential(
        Linear[IN,  H1].make["cpu", INIT=Zero](),
        ReLU[H1].make["cpu", INIT=Zero](),
        Linear[H1, H2].make["cpu", INIT=Zero](),
        ReLU[H2].make["cpu", INIT=Zero](),
        Linear[H2, OUT].make["cpu", INIT=Zero](),
    )

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)

    for k in range(BATCH * IN): in_buf[k] = Scalar[DT](0.1 * Float32(k))
    for k in range(BATCH * OUT): go_buf[k] = 1.0

    var input  = TileTensor(in_buf,  row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, OUT]())
    var go     = TileTensor(go_buf,  row_major[BATCH, OUT]())
    var gi     = TileTensor(gi_buf,  row_major[BATCH, IN]())

    net.forward["cpu", BATCH](input, output)
    net.backward["cpu", BATCH](go, gi)

    print("  test_seq_cpu_n5 PASSED  output[0,0]=", out_buf[0])

    in_buf.free(); out_buf.free(); go_buf.free(); gi_buf.free()


def test_seq_gpu_n3() raises:
    """3-layer Sequential on GPU: Linear → ReLU → Linear."""
    comptime IN = 4
    comptime H = 6
    comptime OUT = 3
    comptime BATCH = 2

    var ctx = DeviceContext()
    var net = Sequential(
        Linear[IN, H].make["gpu", INIT=Zero](ctx),
        ReLU[H].make["gpu", INIT=Zero](ctx),
        Linear[H, OUT].make["gpu", INIT=Zero](ctx),
        ctx=ctx,
    )

    var in_dev  = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var go_dev  = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var gi_dev  = ctx.enqueue_create_buffer[DT](BATCH * IN)
    in_dev.enqueue_fill(0.5)
    go_dev.enqueue_fill(1.0)
    var input  = TileTensor(in_dev,  row_major[BATCH, IN]())
    var output = TileTensor(out_dev, row_major[BATCH, OUT]())
    var go     = TileTensor(go_dev,  row_major[BATCH, OUT]())
    var gi     = TileTensor(gi_dev,  row_major[BATCH, IN]())

    net.forward["gpu", BATCH](input, output)
    net.backward["gpu", BATCH](go, gi)
    ctx.synchronize()

    print("  test_seq_gpu_n3 PASSED")


def main() raises:
    seed(42)
    print("=" * 60)
    print("nn2 Variadic Sequential[*MODULES] tests")
    print("=" * 60)
    test_seq_cpu_n3()
    test_seq_cpu_n5()
    test_seq_gpu_n3()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
