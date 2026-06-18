"""BatchNorm1D legacy ↔ storage parity (CPU).

Gold-standard gate: run legacy `nn.primitives.BatchNorm1D` and storage
`BatchNorm1DS` with identical γ/β/input/grad_output, compare across BOTH modes:
  - training forward: output + running_mean + running_var (EMA) + the cache,
  - backward: grad_input + grad_gamma + grad_beta,
  - eval forward (after training updated the running stats): output.
Bit-identical ⇒ the State (running stats), train/eval split, and backward math
all survived the port.

Run: pixi run mojo run -I . tests/nn/test_batch_norm_1d_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.batch_norm_1d import BatchNorm1D
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.tensor import Tensor
from mojo_rl.nn.storage.tensor_refs import TensorRefs
from mojo_rl.nn.storage.batch_norm_1d import BatchNorm1DS


def test_bn1d_parity() raises:
    print("test_bn1d_parity (legacy BatchNorm1D vs storage BatchNorm1DS, CPU) ...")
    comptime DIM = 5
    comptime B = 8
    comptime TOL = Scalar[DT](1e-6)

    # ---- legacy ----
    var leg = BatchNorm1D[DIM].make[target="cpu", INIT=Zero]()
    var lg = leg.gamma.value_unsafe_ptr_cpu()
    var lb = leg.beta.value_unsafe_ptr_cpu()
    for k in range(DIM):
        lg[k] = Scalar[DT](0.5 + 0.1 * Float64(k))
        lb[k] = Scalar[DT](-0.2 + 0.05 * Float64(k))

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var ye: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    for i in range(B * DIM):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.21
        go[i] = Scalar[DT]((i % 7) - 3) * 0.3

    var x_t = TileTensor(x, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, DIM]())
    var go_t = TileTensor(go, row_major[B, DIM]())
    var gi_t = TileTensor(gi, row_major[B, DIM]())
    var ye_t = TileTensor(ye, row_major[B, DIM]())
    leg.forward["cpu", B](x_t, output=y_t)          # training
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)
    leg.set_attr["training"](Scalar[DT](0.0))        # eval
    leg.forward["cpu", B](x_t, output=ye_t)

    # ---- storage ----
    var st = BatchNorm1DS[DIM].make_cpu()
    for k in range(DIM):
        st.gamma.val.data[k] = lg[k]
        st.beta.val.data[k] = lb[k]
    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    var sout = Tensor.alloc(B * DIM)
    var sgi = Tensor.alloc(B * DIM)
    var soute = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1].of1(sx), sout, None)   # training
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1].of1(sx), sgo, TensorRefs[1].of1(sgi), None)
    st.set_training(False)                                    # eval
    st.forward["cpu", B](TensorRefs[1].of1(sx), soute, None)

    # ---- compare ----
    var mo: Scalar[DT] = 0
    var me: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
        if abs(soute.data[i] - ye[i]) > me: me = abs(soute.data[i] - ye[i])
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    var mrm: Scalar[DT] = 0
    var mrv: Scalar[DT] = 0
    var mdg: Scalar[DT] = 0
    var mdb: Scalar[DT] = 0
    for f in range(DIM):
        if abs(st.running_mean.data[f] - leg.running_mean.t.cpu[f]) > mrm:
            mrm = abs(st.running_mean.data[f] - leg.running_mean.t.cpu[f])
        if abs(st.running_var.data[f] - leg.running_var.t.cpu[f]) > mrv:
            mrv = abs(st.running_var.data[f] - leg.running_var.t.cpu[f])
        if abs(st.gamma.grd.data[f] - leg.gamma.grd.cpu[f]) > mdg:
            mdg = abs(st.gamma.grd.data[f] - leg.gamma.grd.cpu[f])
        if abs(st.beta.grd.data[f] - leg.beta.grd.cpu[f]) > mdb:
            mdb = abs(st.beta.grd.data[f] - leg.beta.grd.cpu[f])

    print("  max |Δout(train)| =", mo)
    print("  max |Δrunning_mean| =", mrm, "  max |Δrunning_var| =", mrv)
    print("  max |Δgrad_input| =", mgi)
    print("  max |Δgrad_gamma| =", mdg, "  max |Δgrad_beta| =", mdb)
    print("  max |Δout(eval)| =", me)
    assert_true(mo < TOL, "BN1D train forward parity")
    assert_true(mrm < TOL, "BN1D running_mean parity")
    assert_true(mrv < TOL, "BN1D running_var parity")
    assert_true(mgi < TOL, "BN1D grad_input parity")
    assert_true(mdg < TOL, "BN1D grad_gamma parity")
    assert_true(mdb < TOL, "BN1D grad_beta parity")
    assert_true(me < TOL, "BN1D eval forward parity")
    print("  ok")


def test_bn1d_gpu_parity() raises:
    """Storage GPU vs storage CPU (== legacy CPU). Confirms the verbatim
    GPU kernels match. TOL loosened for fp32 reduction-order differences."""
    print("test_bn1d_gpu_parity (storage GPU vs storage CPU) ...")
    comptime DIM = 5
    comptime B = 8
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()

    # CPU reference (storage).
    var cpu = BatchNorm1DS[DIM].make_cpu()
    var gpu = BatchNorm1DS[DIM].make_gpu(c)
    for k in range(DIM):
        cpu.gamma.val.data[k] = Scalar[DT](0.5 + 0.1 * Float64(k))
        cpu.beta.val.data[k] = Scalar[DT](-0.2 + 0.05 * Float64(k))
        gpu.gamma.val.data[k] = cpu.gamma.val.data[k]
        gpu.beta.val.data[k] = cpu.beta.val.data[k]
    gpu.gamma.val.upload(c)
    gpu.beta.val.upload(c)

    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.21
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.3
    var c_out = Tensor.alloc(B * DIM)
    var c_gi = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1].of1(sx), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1].of1(sx), sgo, TensorRefs[1].of1(c_gi), None)

    var gx = Tensor.alloc(B * DIM)
    var ggo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * DIM)
    var g_gi = Tensor.alloc(B * DIM)
    gpu.forward["gpu", B](TensorRefs[1].of1(gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1].of1(gx), ggo, TensorRefs[1].of1(g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    gpu.running_mean.download(c)
    gpu.running_var.download(c)
    gpu.gamma.grd.download(c)
    gpu.beta.grd.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mrm: Scalar[DT] = 0
    var mdg: Scalar[DT] = 0
    for f in range(DIM):
        if abs(gpu.running_mean.data[f] - cpu.running_mean.data[f]) > mrm:
            mrm = abs(gpu.running_mean.data[f] - cpu.running_mean.data[f])
        if abs(gpu.gamma.grd.data[f] - cpu.gamma.grd.data[f]) > mdg:
            mdg = abs(gpu.gamma.grd.data[f] - cpu.gamma.grd.data[f])
    print("  max |Δout| =", mo, "  |Δgrad_input| =", mgi,
          "  |Δrunning_mean| =", mrm, "  |Δgrad_gamma| =", mdg)
    assert_true(mo < TOL and mgi < TOL and mrm < TOL and mdg < TOL,
                "BN1D GPU vs CPU parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("BatchNorm1D legacy ↔ storage parity")
    print("=" * 70)
    test_bn1d_parity()
    test_bn1d_gpu_parity()
    print("ALL PASSED")
