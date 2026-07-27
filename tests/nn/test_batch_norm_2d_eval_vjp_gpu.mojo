"""BatchNorm2D eval-mode (running-stat) vjp — CPU↔GPU parity gate.

The frozen perceptual backbone backprops through BN in EVAL mode on GPU. The
eval-mode input gradient is gi = γ·inv_std_running·dy (no batch reductions);
this gate checks the GPU eval-vjp kernel matches the (finite-difference-validated,
see test_batch_norm_2d_eval_vjp) CPU eval path.

Populates the running stats with one train-mode forward on identical input, then
switches both to eval (`set_attr["training"](0)`) and compares gi.

Run: pixi run -e apple mojo run -I . tests/nn/test_batch_norm_2d_eval_vjp_gpu.mojo
"""

from std.math import abs
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D


def main() raises:
    print("BatchNorm2D eval-mode vjp CPU↔GPU parity gate")
    comptime C = 4
    comptime HH = 4
    comptime WW = 4
    comptime B = 6
    comptime FLAT = C * HH * WW
    comptime TOL = Scalar[DT](3e-5)

    var c = DeviceContext()
    var cpu = BatchNorm2D[C, HH, WW].make["cpu", Deterministic]()
    var gpu = BatchNorm2D[C, HH, WW].make["gpu", Deterministic](Optional(c))
    for k in range(C):
        cpu.gamma.val.data[k] = Scalar[DT](0.7 + 0.1 * Float64(k))
        cpu.beta.val.data[k] = Scalar[DT](-0.3 + 0.05 * Float64(k))
        gpu.gamma.val.data[k] = cpu.gamma.val.data[k]
        gpu.beta.val.data[k] = cpu.beta.val.data[k]
    gpu.gamma.val.upload(c)
    gpu.beta.val.upload(c)

    var sx = Tensor.alloc(B * FLAT)
    var sgo = Tensor.alloc(B * FLAT)
    for i in range(B * FLAT):
        sx.data[i] = Scalar[DT]((i % 17) - 8) * 0.13
        sgo.data[i] = Scalar[DT]((i % 9) - 4) * 0.25
    var gx = Tensor.alloc(B * FLAT)
    var ggo = Tensor.alloc(B * FLAT)
    for i in range(B * FLAT):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)

    var c_out = Tensor.alloc(B * FLAT)
    var g_out = Tensor.alloc(B * FLAT)
    # train-mode forward → populate running stats (identically on both)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))

    # switch to EVAL
    cpu.set_attr["training"](Scalar[DT](0.0))
    gpu.set_attr["training"](Scalar[DT](0.0))

    # eval forward (marks cache_is_training=False) + eval vjp
    var c_gi = Tensor.alloc(B * FLAT)
    var g_gi = Tensor.alloc(B * FLAT)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_gi.download(c)

    var mgi: Scalar[DT] = 0
    for i in range(B * FLAT):
        var d = abs(g_gi.data[i] - c_gi.data[i])
        if d > mgi:
            mgi = d
    print("  eval-mode gi max Δ (CPU↔GPU) =", mgi)
    assert_true(mgi < TOL, "BatchNorm2D eval-mode vjp CPU/GPU parity")
    print("BN2D EVAL VJP GPU PARITY OK")
