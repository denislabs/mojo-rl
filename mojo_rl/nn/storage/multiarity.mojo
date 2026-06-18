"""Multi-arity: AddS (binary) conforms ModuleS, driven via TensorRefs[2].

The §B0 constraint made concrete: a homogeneous `TensorRefs[2, o]` needs both
inputs to share ONE origin, so they're sourced from one owner — a `TensorPack`
pool (`pool[0]`/`pool[1]` share its origin). A real binary leaf (Residual,
Concat) gets its inputs from the orchestrator's node pool the same way.

Run: pixi run mojo run -I . mojo_rl/nn/storage/multiarity.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.tensor import Tensor
from mojo_rl.nn.storage.tensor_refs import TensorRefs
from mojo_rl.nn.storage.tensor_pack import TensorPack
from mojo_rl.nn.storage.leaves import AddS


def main() raises:
    comptime B = 2
    comptime DIM = 3

    var add = AddS[DIM].make_cpu()

    # Two inputs from ONE pool → shared origin (the §B0 requirement).
    var inp = TensorPack[2]()
    for k in range(2):
        inp[k].ensure(B * DIM)
    for i in range(B * DIM):
        inp[0].data[i] = Scalar[DT](i + 1)     # a
        inp[1].data[i] = Scalar[DT](10 + i)    # b
    var out = Tensor.alloc(B * DIM)
    add.forward["cpu", B](TensorRefs[2].of2(inp[0], inp[1]), out, None)
    print("a+b:", out.data[0], out.data[1], out.data[5])  # 11, 13, 21

    var go = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        go.data[i] = Scalar[DT](1)
    var grad = TensorPack[2]()
    add.vjp["cpu", B](
        TensorRefs[2].of2(inp[0], inp[1]),
        go,
        TensorRefs[2].of2(grad[0], grad[1]),
        None,
    )
    print("grad_a[0], grad_b[0]:", grad[0].data[0], grad[1].data[0])  # 1, 1

    var total = out.data[0] + out.data[5] + grad[0].data[0] + grad[1].data[0]
    if total == Scalar[DT](34):  # 11 + 21 + 1 + 1
        print("MULTI-ARITY OK — AddS conforms ModuleS via TensorRefs[2]")
    else:
        print("MULTI-ARITY FAIL (total", total, ")")
