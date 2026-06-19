"""Flatten (identity self-check) + Embedding (legacy↔storage parity), CPU+GPU.

Run: pixi run -e apple mojo run -I . tests/nn/test_flatten_embedding_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.embedding import Embedding as LegacyEmbedding
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.flatten import Flatten
from mojo_rl.nn.storage.primitives.embedding import Embedding


def test_flatten() raises:
    print("test_flatten (identity, CPU+GPU) ...")
    comptime DIM = 7
    comptime B = 4
    comptime M = B * DIM
    comptime TOL = Scalar[DT](1e-7)
    var c = DeviceContext()
    var ok = True
    # CPU
    var fc = Flatten[DIM].make["cpu", Deterministic]()
    var x = Tensor.alloc(M)
    var go = Tensor.alloc(M)
    for i in range(M):
        x.data[i] = Scalar[DT](i) * 0.31 - 1.0
        go.data[i] = Scalar[DT]((i % 5) - 2) * 0.4
    var out = Tensor.alloc(M)
    var gi = Tensor.alloc(M)
    fc.forward["cpu", B](TensorRefs[1](x), out, None)
    fc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    for i in range(M):
        if abs(out.data[i] - x.data[i]) > TOL: ok = False
        if abs(gi.data[i] - go.data[i]) > TOL: ok = False
    # GPU
    var fg = Flatten[DIM].make["gpu", Deterministic](Optional(c))
    var gx = Tensor.alloc(M)
    var ggo = Tensor.alloc(M)
    for i in range(M):
        gx.data[i] = x.data[i]
        ggo.data[i] = go.data[i]
    gx.upload(c); ggo.upload(c)
    var gout = Tensor.alloc(M)
    var ggi = Tensor.alloc(M)
    fg.forward["gpu", B](TensorRefs[1](gx), gout, Optional(c))
    fg.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](ggi), Optional(c))
    gout.download(c); ggi.download(c)
    for i in range(M):
        if abs(gout.data[i] - x.data[i]) > TOL: ok = False
        if abs(ggi.data[i] - go.data[i]) > TOL: ok = False
    print("   ", "OK" if ok else "FAIL")
    assert_true(ok, "Flatten identity")


def _emb_check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime VOCAB = 6
    comptime EMBED = 4
    comptime B = 5
    comptime TOL = Scalar[DT](1e-5) if target == "cpu" else Scalar[DT](2e-5)

    var leg = LegacyEmbedding[VOCAB, EMBED].make[target="cpu", INIT=Zero]()
    var lw = leg.weight.value_unsafe_ptr_cpu()
    for k in range(VOCAB * EMBED):
        lw[k] = Scalar[DT]((k % 9) - 4) * 0.06
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * VOCAB)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * EMBED)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * EMBED)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * VOCAB)
    for i in range(B * VOCAB):
        x[i] = Scalar[DT]((i % 4)) * 0.5  # general (not strictly one-hot) input
    for i in range(B * EMBED):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var x_t = TileTensor(x, row_major[B, VOCAB]())
    var y_t = TileTensor(y, row_major[B, EMBED]())
    var go_t = TileTensor(go, row_major[B, EMBED]())
    var gi_t = TileTensor(gi, row_major[B, VOCAB]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)

    var st = Embedding[VOCAB, EMBED].make[target, Deterministic](ctx)
    for k in range(VOCAB * EMBED):
        st.weight.val.data[k] = lw[k]
    var sx = Tensor.alloc(B * VOCAB)
    var sgo = Tensor.alloc(B * EMBED)
    var sout = Tensor.alloc(B * EMBED)
    var sgi = Tensor.alloc(B * VOCAB)
    for i in range(B * VOCAB):
        sx.data[i] = x[i]
    for i in range(B * EMBED):
        sgo.data[i] = go[i]
    comptime if target == "cpu":
        st.forward["cpu", B](TensorRefs[1](sx), sout, None)
        st.zero_grad["cpu"](None)
        st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)
    else:
        var c = ctx.value()
        st.weight.val.upload(c)
        sx.upload(c); sgo.upload(c)
        st.forward["gpu", B](TensorRefs[1](sx), sout, ctx)
        st.zero_grad["gpu"](ctx)
        st.vjp["gpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), ctx)
        sout.download(c); sgi.download(c); st.weight.grd.download(c)

    var ok = True
    for i in range(B * EMBED):
        if abs(sout.data[i] - y[i]) > TOL: ok = False
    for i in range(B * VOCAB):
        if abs(sgi.data[i] - gi[i]) > TOL: ok = False
    for k in range(VOCAB * EMBED):
        if abs(st.weight.grd.data[k] - leg.weight.grd.cpu[k]) > TOL: ok = False
    return ok


def main() raises:
    print("=" * 70)
    print("Flatten + Embedding storage parity")
    print("=" * 70)
    test_flatten()
    var oc = _emb_check["cpu"](None)
    print("Embedding CPU (legacy↔storage):", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _emb_check["gpu"](Optional(c))
    print("Embedding GPU (vs legacy CPU):", "OK" if og else "FAIL")
    assert_true(oc and og, "Embedding parity")
    print("ALL PASSED")
