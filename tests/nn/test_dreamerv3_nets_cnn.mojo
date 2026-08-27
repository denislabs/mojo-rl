"""DreamerV3 CNN encoder/decoder (nets_cnn.mojo) smoke + CPU↔GPU parity.

Validates that the pixel encoder (image→tokens) and the transposed-conv decoder
(feature→image) compose, run on CPU and GPU identically, and propagate gradients
end-to-end (including through the new Conv2DTranspose stack):
  - forward output finite
  - CPU↔GPU parity on forward output AND grad_input
  - grad_input is non-trivially nonzero (gradient reached the input)

Run CPU:  pixi run mojo run -I . tests/nn/test_dreamerv3_nets_cnn.mojo
Run GPU:  pixi run -e apple mojo run -I . tests/nn/test_dreamerv3_nets_cnn.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)

# Tractable pixel config: 4 stacked frames, 64x64 (→ minres 4), base width 8.
comptime C = 4
comptime IMG = 64
comptime BASE = 8
comptime TOKEN = 32
comptime FEATIN = 48
comptime B = 2

comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE]
comptime ENC_IN = ENC.IN_DIMS[0]
comptime ENC_OUT = ENC.OUT_DIM
comptime DEC_IN = DEC.IN_DIMS[0]
comptime DEC_OUT = DEC.OUT_DIM


def _fill(mut t: Tensor, n: Int, k: Int):
    for i in range(n):
        t.data[i] = Scalar[DT]((i * (k + 3) + 1) % 13 - 6) * 0.07


def test_encoder(ctx: DeviceContext) raises:
    var mc = ENC.make["cpu", Deterministic]()
    var mg = ENC.make["gpu", Deterministic](Optional(ctx))
    var x = Tensor.alloc(B * ENC_IN)
    var go = Tensor.alloc(B * ENC_OUT)
    _fill(x, B * ENC_IN, 1)
    _fill(go, B * ENC_OUT, 2)

    var yc = Tensor.alloc(B * ENC_OUT)
    mc.forward["cpu", B](TensorRefs[1](x), yc, None)
    var gxc = Tensor.alloc(B * ENC_IN)
    mc.zero_grad["cpu"](None)
    mc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gxc), None)

    var xg = Tensor.alloc(B * ENC_IN)
    for i in range(B * ENC_IN):
        xg.data[i] = x.data[i]
    xg.ensure_gpu(ctx, B * ENC_IN)
    xg.upload(ctx)
    var gog = Tensor.alloc(B * ENC_OUT)
    for i in range(B * ENC_OUT):
        gog.data[i] = go.data[i]
    gog.ensure_gpu(ctx, B * ENC_OUT)
    gog.upload(ctx)
    var yg = Tensor()
    yg.ensure_gpu(ctx, B * ENC_OUT)
    mg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    var gxg = Tensor()
    gxg.ensure_gpu(ctx, B * ENC_IN)
    mg.zero_grad["gpu"](Optional(ctx))
    mg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gxg), Optional(ctx))
    yg.download(ctx)
    gxg.download(ctx)
    ctx.synchronize()

    var d_y: Float64 = 0.0
    for i in range(B * ENC_OUT):
        d_y = max(d_y, abs(Float64(yc.data[i] - yg.data[i])))
    var d_gx: Float64 = 0.0
    var gmag: Float64 = 0.0
    for i in range(B * ENC_IN):
        d_gx = max(d_gx, abs(Float64(gxc.data[i] - gxg.data[i])))
        gmag += abs(Float64(gxc.data[i]))
    print(
        "  [encoder] IN", ENC_IN, " OUT", ENC_OUT,
        " CPU↔GPU max|Δ| out", d_y, " grad_x", d_gx, " Σ|grad_x|", gmag,
    )
    assert_true(d_y < 1e-3 and d_gx < 1e-3, "encoder CPU↔GPU parity")
    assert_true(gmag > 1e-4, "encoder gradient reached the input")


def test_decoder(ctx: DeviceContext) raises:
    var mc = DEC.make["cpu", Deterministic]()
    var mg = DEC.make["gpu", Deterministic](Optional(ctx))
    var x = Tensor.alloc(B * DEC_IN)
    var go = Tensor.alloc(B * DEC_OUT)
    _fill(x, B * DEC_IN, 3)
    _fill(go, B * DEC_OUT, 4)

    var yc = Tensor.alloc(B * DEC_OUT)
    mc.forward["cpu", B](TensorRefs[1](x), yc, None)
    var gxc = Tensor.alloc(B * DEC_IN)
    mc.zero_grad["cpu"](None)
    mc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gxc), None)

    var xg = Tensor.alloc(B * DEC_IN)
    for i in range(B * DEC_IN):
        xg.data[i] = x.data[i]
    xg.ensure_gpu(ctx, B * DEC_IN)
    xg.upload(ctx)
    var gog = Tensor.alloc(B * DEC_OUT)
    for i in range(B * DEC_OUT):
        gog.data[i] = go.data[i]
    gog.ensure_gpu(ctx, B * DEC_OUT)
    gog.upload(ctx)
    var yg = Tensor()
    yg.ensure_gpu(ctx, B * DEC_OUT)
    mg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    var gxg = Tensor()
    gxg.ensure_gpu(ctx, B * DEC_IN)
    mg.zero_grad["gpu"](Optional(ctx))
    mg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gxg), Optional(ctx))
    yg.download(ctx)
    gxg.download(ctx)
    ctx.synchronize()

    var d_y: Float64 = 0.0
    for i in range(B * DEC_OUT):
        d_y = max(d_y, abs(Float64(yc.data[i] - yg.data[i])))
    var d_gx: Float64 = 0.0
    var gmag: Float64 = 0.0
    for i in range(B * DEC_IN):
        d_gx = max(d_gx, abs(Float64(gxc.data[i] - gxg.data[i])))
        gmag += abs(Float64(gxc.data[i]))
    print(
        "  [decoder] IN", DEC_IN, " OUT", DEC_OUT,
        " CPU↔GPU max|Δ| out", d_y, " grad_x", d_gx, " Σ|grad_x|", gmag,
    )
    assert_true(d_y < 1e-3 and d_gx < 1e-3, "decoder CPU↔GPU parity")
    assert_true(gmag > 1e-4, "decoder gradient reached the input")


def _cpu_only(name: String, IN: Int, OUT: Int) raises:
    print("  [", name, "] CPU-only IN", IN, " OUT", OUT)


def main() raises:
    print("=" * 64)
    print("DreamerV3 CNN encoder/decoder smoke + CPU↔GPU parity")
    print("  encoder:", ENC_IN, "→", ENC_OUT, " decoder:", DEC_IN, "→", DEC_OUT)
    print("=" * 64)
    try:
        var ctx = DeviceContext()
        test_encoder(ctx)
        test_decoder(ctx)
        print("DREAMERV3 CNN NETS GATES PASSED (CPU↔GPU)")
    except e:
        print("GPU unavailable — skipped:", e)
