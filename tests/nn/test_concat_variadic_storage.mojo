"""Variadic Concat[*DIMS] storage gate — N=3 forward + backward, CPU & GPU.

Proves the storage concat is no longer binary-only: a 3-input `Concat[D0,D1,D2]`
column-stacks three separate inputs and its vjp slice-splits grad_output back to
the three grad-inputs. Inputs live in one `TensorPack` (shared origin — the §B0
constraint the `TensorRefs` pack requires), mirroring how ComputeGraph feeds a
node from its pool.

Run:
  pixi run mojo run -I . tests/nn/test_concat_variadic_storage.mojo
  pixi run -e apple mojo run -I . tests/nn/test_concat_variadic_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.concat import Concat


comptime D0 = 3
comptime D1 = 2
comptime D2 = 4
comptime OUT = D0 + D1 + D2  # 9
comptime B = 5


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-6)
    var op = Concat[D0, D1, D2].make[target, Deterministic](ctx)
    comptime assert op.ARITY == 3, "Concat[3,2,4] must have ARITY 3"
    comptime assert op.OUT_DIM == OUT, "OUT_DIM must be 9"

    var ins = TensorPack[3]()
    ins[0].ensure(B * D0)
    ins[1].ensure(B * D1)
    ins[2].ensure(B * D2)
    for i in range(B * D0):
        ins[0].data[i] = Scalar[DT]((i % 7) - 3) * 0.5
    for i in range(B * D1):
        ins[1].data[i] = Scalar[DT](((i + 11) % 7) - 3) * 0.5
    for i in range(B * D2):
        ins[2].data[i] = Scalar[DT](((i + 23) % 7) - 3) * 0.5
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](((i + 5) % 9) - 4) * 0.3

    var out = Tensor.alloc(B * OUT)
    var g = TensorPack[3]()
    comptime if target == "cpu":
        op.forward["cpu", B](TensorRefs[3](ins[0], ins[1], ins[2]), out, None)
        op.vjp["cpu", B](
            TensorRefs[3](ins[0], ins[1], ins[2]), go,
            TensorRefs[3](g[0], g[1], g[2]), None,
        )
    else:
        var c = ctx.value()
        ins[0].upload(c); ins[1].upload(c); ins[2].upload(c); go.upload(c)
        op.forward["gpu", B](TensorRefs[3](ins[0], ins[1], ins[2]), out, ctx)
        op.vjp["gpu", B](
            TensorRefs[3](ins[0], ins[1], ins[2]), go,
            TensorRefs[3](g[0], g[1], g[2]), ctx,
        )
        out.download(c); g[0].download(c); g[1].download(c); g[2].download(c)

    var ok = True
    # forward: out is column-stack [in0 | in1 | in2]
    for bi in range(B):
        for d in range(D0):
            if abs(out.data[bi * OUT + d] - ins[0].data[bi * D0 + d]) > TOL:
                ok = False
        for d in range(D1):
            if abs(out.data[bi * OUT + D0 + d] - ins[1].data[bi * D1 + d]) > TOL:
                ok = False
        for d in range(D2):
            if (
                abs(out.data[bi * OUT + D0 + D1 + d] - ins[2].data[bi * D2 + d])
                > TOL
            ):
                ok = False
    # backward: grad-inputs are the matching column slices of grad_output
    for bi in range(B):
        for d in range(D0):
            if abs(g[0].data[bi * D0 + d] - go.data[bi * OUT + d]) > TOL:
                ok = False
        for d in range(D1):
            if abs(g[1].data[bi * D1 + d] - go.data[bi * OUT + D0 + d]) > TOL:
                ok = False
        for d in range(D2):
            if (
                abs(g[2].data[bi * D2 + d] - go.data[bi * OUT + D0 + D1 + d])
                > TOL
            ):
                ok = False
    return ok


def main() raises:
    print("=" * 60)
    print("Variadic Concat[3,2,4] storage gate (N=3, CPU + GPU)")
    print("=" * 60)
    var cpu_ok = _check["cpu"](None)
    print("  CPU forward+backward:", "OK" if cpu_ok else "FAIL")
    with DeviceContext() as ctx:
        var gpu_ok = _check["gpu"](Optional(ctx))
        print("  GPU forward+backward:", "OK" if gpu_ok else "FAIL")
        assert_true(cpu_ok and gpu_ok, "variadic Concat[3,2,4] CPU+GPU parity")
    print("CONCAT VARIADIC OK")
