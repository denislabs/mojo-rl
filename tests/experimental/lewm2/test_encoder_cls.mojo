"""LeWMEncoderCLS — CLS-token encoder variant composes + runs (toy).

Validates the [CLS]-token encoder (prepend LearnedTokens → transformer over
N_PATCHES+1 → Slice token 0 → projector): image (B, IN_CH·IMG·IMG) → (B, EMB),
forward + vjp finite, on CPU and GPU. De-risks the novel piece of the
CLS-retrain prep; the full retrain is the NVIDIA run.

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm2/test_encoder_cls.mojo
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.experimental.lewm2.encoder import LeWMEncoderCLS


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


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int) -> Scalar[DT]:
    return Scalar[DT]((Float64((i * 2654435761) % 1000) / 500.0) - 1.0)


def _finite(x: Scalar[DT]) -> Bool:
    return x == x and (x - x) == Scalar[DT](0.0)


def test_cpu() raises:
    print("encoder_cls cpu ...")
    comptime assert Enc.OUT_DIM == EMB, "CLS encoder OUT_DIM must be EMB"
    var enc = Enc.make[target="cpu", INIT=Kaiming]()
    var x = _a(B * IMG_DIM); var y = _a(B * EMB)
    for k in range(B * IMG_DIM):
        x[k] = _det(k + 1)
    var x_t = TileTensor(x, row_major[B, IMG_DIM]())
    var y_t = TileTensor(y, row_major[B, EMB]())
    enc.forward["cpu", B](TensorPack[1].of(x_t), output=y_t)
    var fin = True
    for k in range(B * EMB):
        if not _finite(y[k]):
            fin = False
    assert_true(fin, "forward finite (cpu)")

    var w = _a(B * EMB); var gx = _a(B * IMG_DIM)
    for k in range(B * EMB):
        w[k] = _det(k + 5)
    var w_t = TileTensor(w, row_major[B, EMB]())
    var gx_t = TileTensor(gx, row_major[B, IMG_DIM]())
    enc.vjp["cpu", B](w_t, TensorPack[1].of(gx_t))
    var gfin = True
    for k in range(B * IMG_DIM):
        if not _finite(gx[k]):
            gfin = False
    assert_true(gfin, "vjp grad finite (cpu)")
    x.free(); y.free(); w.free(); gx.free()
    _ = enc^
    print("  ok")


def test_gpu() raises:
    print("encoder_cls gpu ...")
    var ctx = DeviceContext()
    var enc = Enc.make[target="gpu", INIT=Kaiming](ctx)
    var xd = ctx.enqueue_create_buffer[DT](B * IMG_DIM)
    var yd = ctx.enqueue_create_buffer[DT](B * EMB)
    var xh = ctx.enqueue_create_host_buffer[DT](B * IMG_DIM)
    var yh = ctx.enqueue_create_host_buffer[DT](B * EMB)
    ctx.synchronize()
    for k in range(B * IMG_DIM):
        xh.unsafe_ptr()[k] = _det(k + 1)
    ctx.enqueue_copy(xd, xh); ctx.synchronize()
    var x_t = TileTensor(_p(xd), row_major[B, IMG_DIM]())
    var y_t = TileTensor(_p(yd), row_major[B, EMB]())
    enc.forward["gpu", B](TensorPack[1].of(x_t), output=y_t)
    ctx.enqueue_copy(yh, yd); ctx.synchronize()
    var fin = True
    for k in range(B * EMB):
        if not _finite(yh.unsafe_ptr()[k]):
            fin = False
    assert_true(fin, "forward finite (gpu)")
    _ = enc^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWMEncoderCLS — CLS-token encoder variant")
    print("=" * 70)
    test_cpu()
    test_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
