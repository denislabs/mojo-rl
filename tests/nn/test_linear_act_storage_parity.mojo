"""LinearAct legacy ↔ storage parity (CPU) + storage GPU vs CPU.

Tests the GENERAL fused leaf `LinearAct[IN, OUT, OP]` on the storage surface,
parametric over the activation op. CPU parity vs the legacy LinearAct (max|Δ| <
1e-6 on out + grad_input + weight.grd + bias.grd) for TWO activations (Tanh =
output-cache, Mish = input-cache). Storage GPU-vs-CPU at a looser matmul TOL.
Also checks the LinearAct[…, ReLUOp] instantiation matches storage LinearReLU.

  pixi run mojo run -I . tests/nn/test_linear_act_storage_parity.mojo
  pixi run -e apple mojo run -I . tests/nn/test_linear_act_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.element_op import ElementOp

# legacy general fused leaf
from mojo_rl.nn.primitives.linear_act import LinearAct as LegacyLinearAct
from mojo_rl.nn.primitives.ops.tanh_op import TanhOp
from mojo_rl.nn.primitives.ops.mish_op import MishOp
from mojo_rl.nn.primitives.ops.relu_op import ReLUOp
from mojo_rl.nn.initializer import Zero as LegacyZero

# storage
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.linear_act import LinearAct
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU


comptime IN = 6
comptime OUT = 5
comptime B = 4
comptime N_X = B * IN
comptime N_Y = B * OUT


def _cpu_parity[OP: ElementOp](name: String) raises:
    print("test_linact_cpu_parity[", name, "] (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    # legacy leaf (Deterministic-equivalent weights set by hand below)
    var leg = LegacyLinearAct[IN, OUT, OP].make[target="cpu", INIT=LegacyZero]()
    var lw = leg.weight.value_unsafe_ptr_cpu()
    var lb = leg.bias.value_unsafe_ptr_cpu()
    for k in range(IN * OUT):
        lw[k] = Scalar[DT](0.05 + 0.013 * Float64(k % 17))
    for k in range(OUT):
        lb[k] = Scalar[DT](-0.1 + 0.04 * Float64(k))

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    for i in range(N_X):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(N_Y):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[B, IN]())
    var y_t = TileTensor(y, row_major[B, OUT]())
    var go_t = TileTensor(go, row_major[B, OUT]())
    var gi_t = TileTensor(gi, row_major[B, IN]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)

    # storage leaf — copy the SAME weights + bias + input
    var st = LinearAct[IN, OUT, OP].make["cpu", Deterministic]()
    for k in range(IN * OUT):
        st.weight.val.data[k] = lw[k]
    for k in range(OUT):
        st.bias.val.data[k] = lb[k]
    var sx = Tensor.alloc(N_X)
    var sgo = Tensor.alloc(N_Y)
    var sout = Tensor.alloc(N_Y)
    var sgi = Tensor.alloc(N_X)
    # NOTE: legacy `vjp` rewrote `go` in place (gated to grad_z), so reseed the
    # storage grad_output from the ORIGINAL formula, not the consumed `go`.
    for i in range(N_X):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(N_Y):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(N_Y):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    for i in range(N_X):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    var gw_leg = leg.weight.grad_unsafe_ptr_cpu()
    var gb_leg = leg.bias.grad_unsafe_ptr_cpu()
    var mdw: Scalar[DT] = 0
    var mdb: Scalar[DT] = 0
    for k in range(IN * OUT):
        if abs(st.weight.grd.data[k] - gw_leg[k]) > mdw:
            mdw = abs(st.weight.grd.data[k] - gw_leg[k])
    for k in range(OUT):
        if abs(st.bias.grd.data[k] - gb_leg[k]) > mdb:
            mdb = abs(st.bias.grd.data[k] - gb_leg[k])
    print("  max Δ: out", mo, " gi", mgi, " dw", mdw, " db", mdb)
    assert_true(
        mo < TOL and mgi < TOL and mdw < TOL and mdb < TOL,
        "LinearAct CPU parity",
    )
    print("  ok")


def _gpu_parity[OP: ElementOp](name: String) raises:
    print("test_linact_gpu_parity[", name, "] (storage GPU vs CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = LinearAct[IN, OUT, OP].make["cpu", Deterministic]()
    var gpu = LinearAct[IN, OUT, OP].make["gpu", Deterministic](Optional(c))
    for k in range(IN * OUT):
        cpu.weight.val.data[k] = Scalar[DT](0.05 + 0.013 * Float64(k % 17))
        gpu.weight.val.data[k] = cpu.weight.val.data[k]
    for k in range(OUT):
        cpu.bias.val.data[k] = Scalar[DT](-0.1 + 0.04 * Float64(k))
        gpu.bias.val.data[k] = cpu.bias.val.data[k]
    gpu.weight.val.upload(c)
    gpu.bias.val.upload(c)

    var sx = Tensor.alloc(N_X)
    var sgo = Tensor.alloc(N_Y)
    for i in range(N_X):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(N_Y):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(N_Y)
    var c_gi = Tensor.alloc(N_X)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    # NOTE: the CPU `vjp` above rewrote `sgo` in place (gated to grad_z), so
    # reseed the GPU grad_output from the ORIGINAL formula, not consumed `sgo`.
    var gx = Tensor.alloc(N_X)
    var ggo = Tensor.alloc(N_Y)
    for i in range(N_X):
        gx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(N_Y):
        ggo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(N_Y)
    var g_gi = Tensor.alloc(N_X)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    gpu.weight.grd.download(c)
    gpu.bias.grd.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(N_Y):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    for i in range(N_X):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mdw: Scalar[DT] = 0
    var mdb: Scalar[DT] = 0
    for k in range(IN * OUT):
        if abs(gpu.weight.grd.data[k] - cpu.weight.grd.data[k]) > mdw:
            mdw = abs(gpu.weight.grd.data[k] - cpu.weight.grd.data[k])
    for k in range(OUT):
        if abs(gpu.bias.grd.data[k] - cpu.bias.grd.data[k]) > mdb:
            mdb = abs(gpu.bias.grd.data[k] - cpu.bias.grd.data[k])
    print("  max Δ: out", mo, " gi", mgi, " dw", mdw, " db", mdb)
    assert_true(
        mo < TOL and mgi < TOL and mdw < TOL and mdb < TOL,
        "LinearAct GPU vs CPU",
    )
    print("  ok")


def test_relu_alias_matches_linear_relu() raises:
    """LinearAct[…, ReLUOp] must match the existing storage LinearReLU (CPU)."""
    print("test_relu_alias_matches_linear_relu (CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    var act = LinearAct[IN, OUT, ReLUOp].make["cpu", Deterministic]()
    var relu = LinearReLU[IN, OUT].make["cpu", Deterministic]()
    for k in range(IN * OUT):
        var w = Scalar[DT](0.05 + 0.013 * Float64(k % 17))
        act.weight.val.data[k] = w
        relu.weight.val.data[k] = w
    for k in range(OUT):
        var bb = Scalar[DT](-0.1 + 0.04 * Float64(k))
        act.bias.val.data[k] = bb
        relu.bias.val.data[k] = bb

    var sx = Tensor.alloc(N_X)
    var sgo = Tensor.alloc(N_Y)
    for i in range(N_X):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(N_Y):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var a_out = Tensor.alloc(N_Y)
    var a_gi = Tensor.alloc(N_X)
    act.forward["cpu", B](TensorRefs[1](sx), a_out, None)
    act.zero_grad["cpu"](None)
    act.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](a_gi), None)

    var sgo2 = Tensor.alloc(N_Y)
    for i in range(N_Y):
        sgo2.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var r_out = Tensor.alloc(N_Y)
    var r_gi = Tensor.alloc(N_X)
    relu.forward["cpu", B](TensorRefs[1](sx), r_out, None)
    relu.zero_grad["cpu"](None)
    relu.vjp["cpu", B](TensorRefs[1](sx), sgo2, TensorRefs[1](r_gi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    var mdw: Scalar[DT] = 0
    var mdb: Scalar[DT] = 0
    for i in range(N_Y):
        if abs(a_out.data[i] - r_out.data[i]) > mo: mo = abs(a_out.data[i] - r_out.data[i])
    for i in range(N_X):
        if abs(a_gi.data[i] - r_gi.data[i]) > mgi: mgi = abs(a_gi.data[i] - r_gi.data[i])
    for k in range(IN * OUT):
        if abs(act.weight.grd.data[k] - relu.weight.grd.data[k]) > mdw:
            mdw = abs(act.weight.grd.data[k] - relu.weight.grd.data[k])
    for k in range(OUT):
        if abs(act.bias.grd.data[k] - relu.bias.grd.data[k]) > mdb:
            mdb = abs(act.bias.grd.data[k] - relu.bias.grd.data[k])
    print("  max Δ: out", mo, " gi", mgi, " dw", mdw, " db", mdb)
    assert_true(
        mo < TOL and mgi < TOL and mdw < TOL and mdb < TOL,
        "LinearAct[ReLUOp] == LinearReLU",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LinearAct legacy ↔ storage parity")
    print("=" * 70)
    _cpu_parity[TanhOp]("Tanh")
    _cpu_parity[MishOp]("Mish")
    test_relu_alias_matches_linear_relu()
    _gpu_parity[TanhOp]("Tanh")
    _gpu_parity[MishOp]("Mish")
    print("ALL PASSED")
