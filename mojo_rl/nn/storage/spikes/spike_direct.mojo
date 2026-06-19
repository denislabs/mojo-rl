"""Direct Linear forward+vjp smoke (storage, CPU). New N-ary target signatures."""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.core.initializer import Deterministic


def main() raises:
    comptime B = 2
    comptime IN = 3
    comptime OUT = 2

    var lin = Linear[IN, OUT].make["cpu", Deterministic]()
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](i + 1)
    var out = Tensor.alloc(B * OUT)
    lin.forward["cpu", B](TensorRefs[1](x), out, None)
    print("forward:", out.data[0], out.data[1], out.data[2], out.data[3])

    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](1)
    var gi = Tensor.alloc(B * IN)
    lin.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    print("grad_input:", gi.data[0], gi.data[1], gi.data[2])
    print(
        "grad_w[0], grad_b[0]:",
        lin.weight.grd.data[0],
        lin.bias.grd.data[0],
    )
    print("STORAGE DIRECT OK")
