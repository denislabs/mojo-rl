"""PR-1 DreamerV3 primitive GPU parity vs real jax.

Same `prims_fixture.txt` ground truth as the CPU test; validates the GPU
kernels for GELU/SiLU (Elementwise), RMSNorm, and BlockLinear via H2D /
D2H. Tolerance 1e-4 (jax f32 + GPU reduction-order noise).

Run: `pixi run -e apple mojo run -I . tests/nn2/test_dreamer_prims_gpu.mojo`
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.element_op import ElementOp
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.gelu_op import GELUOp
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.nn2.primitives.rms_norm import RMSNorm
from mojo_rl.nn2.primitives.block_linear import BlockLinear
from mojo_rl.nn2.initializer import Zero


comptime FIXTURE = "tests/nn2/dreamerv3/fixtures/prims_fixture.txt"


# ── Fixture parsing (shared shape with the CPU test) ────────────────────


def _split_lines(content: String) raises -> List[String]:
    var lines = List[String]()
    var current = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(current)
            current = String("")
        else:
            current += chr(Int(c))
    if current.byte_length() > 0:
        lines.append(current)
    return lines^


def _read_flat(lines: List[String], name: String) raises -> List[Scalar[DT]]:
    var pfx = name + "#size="
    for i in range(len(lines)):
        if lines[i].startswith(pfx):
            var n = atol(String(lines[i][byte=pfx.byte_length():]))
            var out = List[Scalar[DT]]()
            for k in range(n):
                out.append(Scalar[DT](atof(lines[i + 1 + k])))
            return out^
    raise Error("fixture: section not found: " + name)


@always_inline
def _p(buf: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    """Device buffer → MutAnyOrigin pointer (forward/vjp mut args require
    origin=MutAnyOrigin; a TileTensor built straight from a DeviceBuffer
    carries the buffer's own origin instead)."""
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr())


def _max_abs_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    var n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        var d = a[i] - b[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > m:
            m = ad
    return m


# ── H2D / D2H helpers ───────────────────────────────────────────────────


def _h2d(ctx: DeviceContext, dev: DeviceBuffer[DT], src: List[Scalar[DT]]) raises:
    var h = ctx.enqueue_create_host_buffer[DT](len(src))
    ctx.synchronize()
    for k in range(len(src)):
        h.unsafe_ptr()[k] = src[k]
    ctx.enqueue_copy(dev, h)
    ctx.synchronize()


def _d2h(ctx: DeviceContext, dev: DeviceBuffer[DT], n: Int) raises -> List[Scalar[DT]]:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    ctx.enqueue_copy(h, dev)
    ctx.synchronize()
    var out = List[Scalar[DT]]()
    for k in range(n):
        out.append(h.unsafe_ptr()[k])
    return out^


# ── Elementwise GPU (GELU, SiLU) ────────────────────────────────────────


def _check_elementwise_gpu[
    OP: ElementOp, DIM: Int
](ctx: DeviceContext, lines: List[String], name: String) raises:
    comptime BATCH = 1
    var xs = _read_flat(lines, name + ".x")
    var gos = _read_flat(lines, name + ".go")
    var y_ref = _read_flat(lines, name + ".y")
    var gx_ref = _read_flat(lines, name + ".gx")
    var n = len(xs)

    var op = Elementwise[DIM, OP].make[target="gpu", INIT=Zero](ctx=ctx)
    var in_dev = ctx.enqueue_create_buffer[DT](n)
    var out_dev = ctx.enqueue_create_buffer[DT](n)
    var go_dev = ctx.enqueue_create_buffer[DT](n)
    var gi_dev = ctx.enqueue_create_buffer[DT](n)
    _h2d(ctx, in_dev, xs)
    _h2d(ctx, go_dev, gos)
    var in_t = TileTensor(_p(in_dev), row_major[BATCH, DIM]())
    var out_t = TileTensor(_p(out_dev), row_major[BATCH, DIM]())
    var go_t = TileTensor(_p(go_dev), row_major[BATCH, DIM]())
    var gi_t = TileTensor(_p(gi_dev), row_major[BATCH, DIM]())

    op.forward["gpu", BATCH](in_t, output=out_t)
    op.vjp["gpu", BATCH](go_t, gi_t)

    var got_y = _d2h(ctx, out_dev, n)
    var got_gx = _d2h(ctx, gi_dev, n)
    var df = _max_abs_diff(got_y, y_ref)
    var db = _max_abs_diff(got_gx, gx_ref)
    print("  " + name + " gpu fwd diff =", df, " bwd diff =", db)
    assert_true(df < Scalar[DT](1e-4), name + " gpu forward parity")
    assert_true(db < Scalar[DT](1e-4), name + " gpu backward parity")
    _ = in_dev  # input-cache alias must outlive vjp


def test_gelu_silu_gpu() raises:
    print("test_gelu_silu_gpu ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    var ctx = DeviceContext()
    _check_elementwise_gpu[GELUOp, 32](ctx, lines, "gelu")
    _check_elementwise_gpu[SwishOp, 32](ctx, lines, "silu")
    print("  ok")


def test_rmsnorm_gpu() raises:
    print("test_rmsnorm_gpu ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    comptime BATCH = 3
    comptime DIM = 8
    comptime N = BATCH * DIM
    var xs = _read_flat(lines, "rms.x")
    var gamma_fix = _read_flat(lines, "rms.gamma")
    var gos = _read_flat(lines, "rms.go")
    var y_ref = _read_flat(lines, "rms.y")
    var gx_ref = _read_flat(lines, "rms.gx")
    var ggamma_ref = _read_flat(lines, "rms.ggamma")

    var ctx = DeviceContext()
    var rn = RMSNorm[DIM].make[target="gpu", INIT=Zero](ctx=ctx)
    _h2d(ctx, rn.gamma.val.dev.value(), gamma_fix)
    rn.zero_grad["gpu"]()

    var in_dev = ctx.enqueue_create_buffer[DT](N)
    var out_dev = ctx.enqueue_create_buffer[DT](N)
    var go_dev = ctx.enqueue_create_buffer[DT](N)
    var gi_dev = ctx.enqueue_create_buffer[DT](N)
    _h2d(ctx, in_dev, xs)
    _h2d(ctx, go_dev, gos)
    var in_t = TileTensor(_p(in_dev), row_major[BATCH, DIM]())
    var out_t = TileTensor(_p(out_dev), row_major[BATCH, DIM]())
    var go_t = TileTensor(_p(go_dev), row_major[BATCH, DIM]())
    var gi_t = TileTensor(_p(gi_dev), row_major[BATCH, DIM]())

    rn.forward["gpu", BATCH](in_t, output=out_t)
    rn.vjp["gpu", BATCH](go_t, gi_t)

    var got_y = _d2h(ctx, out_dev, N)
    var got_gx = _d2h(ctx, gi_dev, N)
    var got_gg = _d2h(ctx, rn.gamma.grd.dev.value(), DIM)
    var df = _max_abs_diff(got_y, y_ref)
    var dgx = _max_abs_diff(got_gx, gx_ref)
    var dgg = _max_abs_diff(got_gg, ggamma_ref)
    print("  rms gpu fwd =", df, " gx =", dgx, " ggamma =", dgg)
    assert_true(df < Scalar[DT](1e-4), "RMSNorm gpu forward parity")
    assert_true(dgx < Scalar[DT](1e-4), "RMSNorm gpu grad_x parity")
    assert_true(dgg < Scalar[DT](1e-4), "RMSNorm gpu grad_gamma parity")
    print("  ok")


def test_blocklinear_gpu() raises:
    print("test_blocklinear_gpu ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    comptime BATCH = 4
    comptime IN = 12
    comptime OUT = 16
    comptime BLK = 4
    var xs = _read_flat(lines, "bl.x")
    var kernel = _read_flat(lines, "bl.kernel")
    var bias_fix = _read_flat(lines, "bl.bias")
    var gos = _read_flat(lines, "bl.go")
    var y_ref = _read_flat(lines, "bl.y")
    var gx_ref = _read_flat(lines, "bl.gx")
    var gk_ref = _read_flat(lines, "bl.gkernel")
    var gb_ref = _read_flat(lines, "bl.gbias")

    var ctx = DeviceContext()
    var bl = BlockLinear[IN, OUT, BLK].make[target="gpu", INIT=Zero](ctx=ctx)
    _h2d(ctx, bl.weight.val.dev.value(), kernel)
    _h2d(ctx, bl.bias.val.dev.value(), bias_fix)
    bl.zero_grad["gpu"]()

    var in_dev = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var gi_dev = ctx.enqueue_create_buffer[DT](BATCH * IN)
    _h2d(ctx, in_dev, xs)
    _h2d(ctx, go_dev, gos)
    var in_t = TileTensor(_p(in_dev), row_major[BATCH, IN]())
    var out_t = TileTensor(_p(out_dev), row_major[BATCH, OUT]())
    var go_t = TileTensor(_p(go_dev), row_major[BATCH, OUT]())
    var gi_t = TileTensor(_p(gi_dev), row_major[BATCH, IN]())

    bl.forward["gpu", BATCH](in_t, output=out_t)
    bl.vjp["gpu", BATCH](go_t, gi_t)

    var got_y = _d2h(ctx, out_dev, BATCH * OUT)
    var got_gx = _d2h(ctx, gi_dev, BATCH * IN)
    var got_gk = _d2h(ctx, bl.weight.grd.dev.value(), len(kernel))
    var got_gb = _d2h(ctx, bl.bias.grd.dev.value(), OUT)
    var df = _max_abs_diff(got_y, y_ref)
    var dgx = _max_abs_diff(got_gx, gx_ref)
    var dgk = _max_abs_diff(got_gk, gk_ref)
    var dgb = _max_abs_diff(got_gb, gb_ref)
    print("  bl gpu fwd =", df, " gx =", dgx, " gkernel =", dgk, " gbias =", dgb)
    assert_true(df < Scalar[DT](1e-4), "BlockLinear gpu forward parity")
    assert_true(dgx < Scalar[DT](1e-4), "BlockLinear gpu grad_x parity")
    assert_true(dgk < Scalar[DT](1e-4), "BlockLinear gpu grad_kernel parity")
    assert_true(dgb < Scalar[DT](1e-4), "BlockLinear gpu grad_bias parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PR-1 DreamerV3 primitive GPU parity (vs jax)")
    print("=" * 70)
    test_gelu_silu_gpu()
    test_rmsnorm_gpu()
    test_blocklinear_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
