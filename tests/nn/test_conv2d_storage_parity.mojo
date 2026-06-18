"""Conv2D legacy ↔ storage parity (CPU).

The gold-standard gate for the storage migration: run the LEGACY
`nn.primitives.Conv2D` and the storage `nn.storage.ConvS` with identical
weights/bias/input/grad_output, compare forward + grad_input + grad_weight +
grad_bias. If the storage leaf matches the real legacy leaf, no kernel / cblas /
math feature was dropped in the port (stronger than a hand-written reference).

Run: pixi run mojo run -I . tests/nn/test_conv2d_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.tensor import Tensor
from mojo_rl.nn.storage.tensor_refs import TensorRefs
from mojo_rl.nn.storage.conv2d import ConvS


def test_conv_parity() raises:
    print("test_conv_parity (legacy Conv2D vs storage ConvS, CPU) ...")
    comptime IC = 2
    comptime OC = 3
    comptime K = 3
    comptime S = 2
    comptime P = 1
    comptime H = 5
    comptime W = 5
    comptime B = 2
    comptime OH = (H + 2 * P - K) // S + 1
    comptime OW = (W + 2 * P - K) // S + 1
    comptime IN_FLAT = IC * H * W
    comptime OUT_FLAT = OC * OH * OW
    comptime W_SIZE = OC * IC * K * K
    comptime TOL = Scalar[DT](1e-5)

    # ---- legacy leaf ----
    var leg = Conv2D[IC, OC, K, S, P, H, W].make[target="cpu", INIT=Zero]()
    var lw = leg.weight.value_unsafe_ptr_cpu()
    var lb = leg.bias.value_unsafe_ptr_cpu()
    for k in range(W_SIZE):
        lw[k] = Scalar[DT]((k % 9) - 4) * 0.05
    for k in range(OC):
        lb[k] = Scalar[DT](k + 1) * 0.1

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN_FLAT)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT_FLAT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT_FLAT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN_FLAT)
    for i in range(B * IN_FLAT):
        x[i] = Scalar[DT]((i % 11) - 5) * 0.17
    for i in range(B * OUT_FLAT):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.2

    var x_t = TileTensor(x, row_major[B, IN_FLAT]())
    var y_t = TileTensor(y, row_major[B, OUT_FLAT]())
    var go_t = TileTensor(go, row_major[B, OUT_FLAT]())
    var gi_t = TileTensor(gi, row_major[B, IN_FLAT]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)

    # ---- storage leaf (identical weights) ----
    var st = ConvS[IC, OC, K, S, P, H, W].make_cpu()
    for k in range(W_SIZE):
        st.weight.val.data[k] = lw[k]
    for k in range(OC):
        st.bias.val.data[k] = lb[k]
    var sx = Tensor.alloc(B * IN_FLAT)
    var sgo = Tensor.alloc(B * OUT_FLAT)
    var sout = Tensor.alloc(B * OUT_FLAT)
    var sgi = Tensor.alloc(B * IN_FLAT)
    for i in range(B * IN_FLAT):
        sx.data[i] = x[i]
    for i in range(B * OUT_FLAT):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1].of1(sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1].of1(sx), sgo, TensorRefs[1].of1(sgi), None)

    # ---- compare ----
    var max_out: Scalar[DT] = 0
    for i in range(B * OUT_FLAT):
        var d = abs(sout.data[i] - y[i])
        if d > max_out:
            max_out = d
    var max_gi: Scalar[DT] = 0
    for i in range(B * IN_FLAT):
        var d = abs(sgi.data[i] - gi[i])
        if d > max_gi:
            max_gi = d
    var max_dw: Scalar[DT] = 0
    for k in range(W_SIZE):
        var d = abs(st.weight.grd.data[k] - leg.weight.grd.cpu[k])
        if d > max_dw:
            max_dw = d
    var max_db: Scalar[DT] = 0
    for k in range(OC):
        var d = abs(st.bias.grd.data[k] - leg.bias.grd.cpu[k])
        if d > max_db:
            max_db = d

    print("  max |Δout| =", max_out)
    print("  max |Δgrad_input| =", max_gi)
    print("  max |Δgrad_weight| =", max_dw)
    print("  max |Δgrad_bias| =", max_db)
    assert_true(max_out < TOL, "Conv2D forward parity")
    assert_true(max_gi < TOL, "Conv2D grad_input parity")
    assert_true(max_dw < TOL, "Conv2D grad_weight parity")
    assert_true(max_db < TOL, "Conv2D grad_bias parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Conv2D legacy ↔ storage parity")
    print("=" * 70)
    test_conv_parity()
    print("ALL PASSED")
