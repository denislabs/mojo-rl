"""Embedding parity: forward + vjp (grad_in, grad_weight) vs an INDEPENDENT
dense-matmul reference. CPU + GPU. Validates the max_matmul rewrite (A4).

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spikes/spike_embedding_parity.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.embedding import Embedding
from mojo_rl.nn.storage.core.initializer import Deterministic


def _run[
    target: StaticString, VOCAB: Int, ED: Int, B: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4)
    var e = Embedding[VOCAB, ED].make[target, Deterministic](ctx)

    # Deterministic dense input + grad_output (dense exercises the GEMM).
    var x = Tensor.alloc(B * VOCAB)
    for i in range(B * VOCAB):
        x.data[i] = Scalar[DT]((i % 9) - 4) * 0.13
    var go = Tensor.alloc(B * ED)
    for i in range(B * ED):
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.21

    # Snapshot weights (make() initialised them) for the oracle.
    var w = List[Scalar[DT]](length=VOCAB * ED, fill=Scalar[DT](0))
    for i in range(VOCAB * ED):
        w[i] = e.weight.val.data[i]

    # ---- independent dense-matmul oracle ----
    var r_out = List[Scalar[DT]](length=B * ED, fill=Scalar[DT](0))
    var r_gi = List[Scalar[DT]](length=B * VOCAB, fill=Scalar[DT](0))
    var r_gw = List[Scalar[DT]](length=VOCAB * ED, fill=Scalar[DT](0))
    for b in range(B):
        for j in range(ED):
            var acc = Scalar[DT](0)
            for v in range(VOCAB):
                acc += x.data[b * VOCAB + v] * w[v * ED + j]
            r_out[b * ED + j] = acc
    for b in range(B):
        for v in range(VOCAB):
            var acc = Scalar[DT](0)
            for j in range(ED):
                acc += go.data[b * ED + j] * w[v * ED + j]
            r_gi[b * VOCAB + v] = acc
    for v in range(VOCAB):
        for j in range(ED):
            var acc = Scalar[DT](0)
            for b in range(B):
                acc += x.data[b * VOCAB + v] * go.data[b * ED + j]
            r_gw[v * ED + j] = acc

    var out = Tensor.alloc(B * ED)
    var gi = Tensor.alloc(B * VOCAB)
    comptime if target == "cpu":
        e.forward["cpu", B](TensorRefs[1](x), out, None)
        e.zero_grad["cpu"](None)
        e.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    else:
        var c = ctx.value()
        x.upload(c)
        go.upload(c)
        e.forward["gpu", B](TensorRefs[1](x), out, ctx)
        e.zero_grad["gpu"](ctx)
        e.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        out.download(c)
        gi.download(c)
        e.weight.grd.download(c)

    var d_out = Scalar[DT](0)
    for i in range(B * ED):
        d_out = max(d_out, abs(out.data[i] - r_out[i]))
    var d_gi = Scalar[DT](0)
    for i in range(B * VOCAB):
        d_gi = max(d_gi, abs(gi.data[i] - r_gi[i]))
    var d_gw = Scalar[DT](0)
    for i in range(VOCAB * ED):
        d_gw = max(d_gw, abs(e.weight.grd.data[i] - r_gw[i]))
    print(
        "  EMB[V=", VOCAB, " ED=", ED, "]", target,
        " d_out=", d_out, " d_gi=", d_gi, " d_gw=", d_gw,
    )
    return d_out <= TOL and d_gi <= TOL and d_gw <= TOL


def main() raises:
    var c = DeviceContext()
    var ok = True
    print("Embedding parity (dense-matmul oracle):")
    ok = _run["cpu", 5, 4, 3](None) and ok
    ok = _run["gpu", 5, 4, 3](Optional(c)) and ok
    ok = _run["cpu", 64, 32, 16](None) and ok
    ok = _run["gpu", 64, 32, 16](Optional(c)) and ok
    print("EMBEDDING PARITY", "OK" if ok else "FAIL")
