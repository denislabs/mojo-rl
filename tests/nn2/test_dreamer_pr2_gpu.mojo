"""PR-2 GPU parity: lambda_return on GPU vs the optax/jax fixture.

OneHotKL + PercentileNormalize are CPU-only at landing (Pendulum v1 trains
the world model on CPU); only `lambda_return` has a GPU path in PR 2.

Run: `pixi run -e apple mojo run -I . tests/nn2/test_dreamer_pr2_gpu.mojo`
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.lambda_return import lambda_return_gpu


comptime FIXTURE = "tests/nn2/dreamerv3/fixtures/pr2_fixture.txt"


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


def _get_scalar(lines: List[String], key: String) raises -> Float64:
    var pfx = key + "="
    for i in range(len(lines)):
        if lines[i].startswith(pfx):
            return atof(String(lines[i][byte=pfx.byte_length():]))
    raise Error("fixture: key not found: " + key)


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
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr())


def _h2d(ctx: DeviceContext, dev: DeviceBuffer[DT], src: List[Scalar[DT]]) raises:
    var h = ctx.enqueue_create_host_buffer[DT](len(src))
    ctx.synchronize()
    for k in range(len(src)):
        h.unsafe_ptr()[k] = src[k]
    ctx.enqueue_copy(dev, h)
    ctx.synchronize()


def test_lambda_return_gpu() raises:
    print("test_lambda_return_gpu ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    comptime B = 2
    comptime T = 6
    var disc = Scalar[DT](_get_scalar(lines, "lr.disc"))
    var lam = Scalar[DT](_get_scalar(lines, "lr.lam"))
    var ret_ref = _read_flat(lines, "lr.ret")

    var ctx = DeviceContext()
    var last_d = ctx.enqueue_create_buffer[DT](B * T)
    var term_d = ctx.enqueue_create_buffer[DT](B * T)
    var rew_d = ctx.enqueue_create_buffer[DT](B * T)
    var boot_d = ctx.enqueue_create_buffer[DT](B * T)
    var out_d = ctx.enqueue_create_buffer[DT](B * (T - 1))
    _h2d(ctx, last_d, _read_flat(lines, "lr.last"))
    _h2d(ctx, term_d, _read_flat(lines, "lr.term"))
    _h2d(ctx, rew_d, _read_flat(lines, "lr.rew"))
    _h2d(ctx, boot_d, _read_flat(lines, "lr.boot"))

    var last_t = TileTensor(_p(last_d), row_major[B, T]())
    var term_t = TileTensor(_p(term_d), row_major[B, T]())
    var rew_t = TileTensor(_p(rew_d), row_major[B, T]())
    var boot_t = TileTensor(_p(boot_d), row_major[B, T]())
    var out_t = TileTensor(_p(out_d), row_major[B, T - 1]())

    lambda_return_gpu[B, T](
        ctx, last_t, term_t, rew_t, boot_t, out_t, disc, lam
    )

    var h = ctx.enqueue_create_host_buffer[DT](B * (T - 1))
    ctx.synchronize()
    ctx.enqueue_copy(h, out_d)
    ctx.synchronize()
    var worst: Scalar[DT] = 0.0
    for i in range(B * (T - 1)):
        var d = h.unsafe_ptr()[i] - ret_ref[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > worst:
            worst = ad
    print("  lambda_return gpu diff =", worst)
    assert_true(worst < Scalar[DT](1e-5), "lambda_return gpu parity vs jax")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PR-2 GPU parity (lambda_return)")
    print("=" * 70)
    test_lambda_return_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
