"""LeWMEncoderCLS — CLS-token encoder variant composes + runs (toy, storage).

Validates the [CLS]-token encoder (prepend LearnedTokens → transformer over
N_PATCHES+1 → Slice token 0 → projector): image (B, IN_CH·IMG·IMG) → (B, EMB),
forward + vjp finite, on CPU and GPU.

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm/lewm/test_encoder_cls.mojo
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn import Tensor, TensorRefs, Kaiming
from mojo_rl.experimental.lewm.encoder import LeWMEncoderCLS


comptime IN_CH = 3
comptime IMG = 8
comptime PATCH = 4
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)   # 4
comptime HIDDEN = 8
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 2
comptime EMB = 8
comptime PROJ_H = 16
comptime FF_MULT = 2
comptime B = 4
comptime IMG_DIM = IN_CH * IMG * IMG                   # 192

comptime Enc = LeWMEncoderCLS[
    IN_CH, IMG, PATCH, N_PATCHES, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
    PROJ_H, FF_MULT,
]


def _det(i: Int) -> Scalar[DT]:
    return Scalar[DT]((Float64((i * 2654435761) % 1000) / 500.0) - 1.0)


def _finite(x: Scalar[DT]) -> Bool:
    return x == x and (x - x) == Scalar[DT](0.0)


def test_cpu() raises:
    print("encoder_cls cpu ...")
    comptime assert Enc.OUT_DIM == EMB, "CLS encoder OUT_DIM must be EMB"
    var enc = Enc.make["cpu", Kaiming]()
    var x = Tensor.alloc(B * IMG_DIM)
    for k in range(B * IMG_DIM):
        x.data[k] = _det(k + 1)
    var y = Tensor.alloc(B * EMB)
    enc.forward["cpu", B](TensorRefs[1](x), y, None)
    var fin = True
    for k in range(B * EMB):
        if not _finite(y.data[k]):
            fin = False
    assert_true(fin, "forward finite (cpu)")

    var w = Tensor.alloc(B * EMB)
    for k in range(B * EMB):
        w.data[k] = _det(k + 5)
    var gx = Tensor.alloc(B * IMG_DIM)
    enc.vjp["cpu", B](TensorRefs[1](x), w, TensorRefs[1](gx), None)
    var gfin = True
    for k in range(B * IMG_DIM):
        if not _finite(gx.data[k]):
            gfin = False
    assert_true(gfin, "vjp grad finite (cpu)")
    _ = enc^
    print("  ok")


def test_gpu() raises:
    print("encoder_cls gpu ...")
    var ctx = DeviceContext()
    var enc = Enc.make["gpu", Kaiming](Optional(ctx))
    var x = Tensor.alloc(B * IMG_DIM)
    for k in range(B * IMG_DIM):
        x.data[k] = _det(k + 1)
    x.upload(ctx)
    var y = Tensor.alloc_gpu(ctx, B * EMB)
    enc.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))
    y.download(ctx)
    var fin = True
    for k in range(B * EMB):
        if not _finite(y.data[k]):
            fin = False
    assert_true(fin, "forward finite (gpu)")
    _ = enc^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWMEncoderCLS — CLS-token encoder variant (storage)")
    print("=" * 70)
    test_cpu()
    test_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
