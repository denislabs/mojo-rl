"""DreamerOpt GPU step parity vs the reference optax chain.

Same fixture as `test_dreamer_opt_step_parity.mojo` (the verbatim optax
chain under real jax). The GPU kernels (`_agc_scale_kernel` +
`_dreamer_update_kernel`) are validated against the same ground truth.

Setup mirrors `test_grad_clip_gpu.mojo`: stuff fixture params/grads into
a `Linear[3,4]`'s `value_dev` / `grad_dev` via host buffers + H2D, step,
then D2H the params and compare. The 1e-4 tolerance absorbs both optax's
f32 cancellation quirk (see CPU test) AND GPU-vs-CPU reduction-order
noise (block.sum vs SIMD reduce). A real algorithmic error is O(1).

Run: `pixi run -e apple mojo run -I . tests/nn2/test_dreamer_opt_gpu_parity.mojo`
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.dreamer_opt import DreamerOpt


comptime IN = 3
comptime OUT = 4
comptime W = IN * OUT          # 12 — leaf0 (weight)
comptime B = OUT               # 4  — leaf1 (bias)
comptime MODEL = Linear[IN, OUT]
comptime FIXTURE = "tests/nn2/dreamerv3/fixtures/dreamer_opt_fixture.txt"


# ── Fixture parsing (same helpers as the CPU test) ──────────────────────


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


# ── H2D / D2H helpers (Linear[3,4] = weight[12] + bias[4]) ──────────────


def _h2d_split(
    ctx: DeviceContext,
    dev_w: DeviceBuffer[DT],
    dev_b: DeviceBuffer[DT],
    flat: List[Scalar[DT]],
) raises:
    """Copy `flat` (leaf0[12] ++ leaf1[4]) into the two device buffers."""
    var w_h = ctx.enqueue_create_host_buffer[DT](W)
    var b_h = ctx.enqueue_create_host_buffer[DT](B)
    ctx.synchronize()
    for k in range(W):
        w_h.unsafe_ptr()[k] = flat[k]
    for k in range(B):
        b_h.unsafe_ptr()[k] = flat[W + k]
    ctx.enqueue_copy(dev_w, w_h)
    ctx.enqueue_copy(dev_b, b_h)
    ctx.synchronize()


def _d2h_split(
    ctx: DeviceContext,
    dev_w: DeviceBuffer[DT],
    dev_b: DeviceBuffer[DT],
) raises -> List[Scalar[DT]]:
    var w_h = ctx.enqueue_create_host_buffer[DT](W)
    var b_h = ctx.enqueue_create_host_buffer[DT](B)
    ctx.synchronize()
    ctx.enqueue_copy(w_h, dev_w)
    ctx.enqueue_copy(b_h, dev_b)
    ctx.synchronize()
    var out = List[Scalar[DT]]()
    for k in range(W):
        out.append(w_h.unsafe_ptr()[k])
    for k in range(B):
        out.append(b_h.unsafe_ptr()[k])
    return out^


def _max_abs_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    var n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        var d = a[i] - b[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > m:
            m = ad
    return m


def test_gpu_step_parity() raises:
    print("test_gpu_step_parity ...")
    var content: String
    with open(FIXTURE, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)

    var beta1 = Scalar[DT](_get_scalar(lines, "beta1"))
    var beta2 = Scalar[DT](_get_scalar(lines, "beta2"))
    var eps = Scalar[DT](_get_scalar(lines, "eps"))
    var lr = Scalar[DT](_get_scalar(lines, "lr"))
    var agc_clip = Scalar[DT](_get_scalar(lines, "agc_clip"))
    var agc_pmin = Scalar[DT](_get_scalar(lines, "agc_pmin"))
    var n_steps = Int(_get_scalar(lines, "n_steps"))

    var ctx = DeviceContext()
    var model = MODEL.make[target="gpu", INIT=Xavier](ctx=ctx)
    var opt = DreamerOpt.make[target="gpu", M=MODEL](model, ctx=ctx)
    opt.lr = lr
    opt.beta1 = beta1
    opt.beta2 = beta2
    opt.eps = eps
    opt.agc_clip = agc_clip
    opt.agc_pmin = agc_pmin

    # Initial params → device.
    var init = _read_flat(lines, "init")
    _h2d_split(
        ctx, model.weight.val.dev.value(), model.bias.val.dev.value(), init
    )

    var worst: Scalar[DT] = 0.0
    for t in range(n_steps):
        var grads = _read_flat(lines, "step" + String(t) + ".grad")
        _h2d_split(
            ctx,
            model.weight.grd.dev.value(),
            model.bias.grd.dev.value(),
            grads,
        )
        opt.step["gpu", M=MODEL](model)
        var got = _d2h_split(
            ctx, model.weight.val.dev.value(), model.bias.val.dev.value()
        )
        var expected = _read_flat(lines, "step" + String(t) + ".param")
        var d = _max_abs_diff(got, expected)
        print("  step", t, "max|param diff vs optax| =", d)
        if d > worst:
            worst = d

    print("  worst over", n_steps, "steps =", worst)
    assert_true(
        worst < Scalar[DT](1e-4),
        "DreamerOpt GPU step must match optax chain within f32 noise (1e-4)",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("DreamerOpt GPU parity (vs optax)")
    print("=" * 70)
    test_gpu_step_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
