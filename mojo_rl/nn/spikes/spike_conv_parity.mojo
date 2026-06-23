"""Conv2D parity harness: forward + vjp (d_input, d_weight, d_bias) vs an
INDEPENDENT direct-convolution reference (naive nested loops — a different
algorithm than im2col+GEMM, so this is a true oracle, not a circular check).
CPU + GPU.

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spike_conv_parity.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.core.initializer import Deterministic


# Compile-time conv geometry for the test instance.
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
comptime COL = IC * K * K


def _w_idx(oc: Int, ic: Int, kh: Int, kw: Int) -> Int:
    return ((oc * IC + ic) * K + kh) * K + kw


def main() raises:
    # Deterministic input / weight / bias / grad_output.
    var x = Tensor.alloc(B * IN_FLAT)
    for i in range(B * IN_FLAT):
        x.data[i] = Scalar[DT]((i % 11) - 5) * 0.17
    var w = List[Scalar[DT]](length=OC * IC * K * K, fill=Scalar[DT](0))
    for i in range(len(w)):
        w[i] = Scalar[DT]((i % 9) - 4) * 0.05
    var bias = List[Scalar[DT]](length=OC, fill=Scalar[DT](0))
    for i in range(OC):
        bias[i] = Scalar[DT](i + 1) * 0.1
    var go = Tensor.alloc(B * OUT_FLAT)
    for i in range(B * OUT_FLAT):
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.2

    # ---- direct-conv reference (forward + grads) ----
    var ref_out = List[Scalar[DT]](length=B * OUT_FLAT, fill=Scalar[DT](0))
    var ref_gi = List[Scalar[DT]](length=B * IN_FLAT, fill=Scalar[DT](0))
    var ref_dw = List[Scalar[DT]](length=OC * IC * K * K, fill=Scalar[DT](0))
    var ref_db = List[Scalar[DT]](length=OC, fill=Scalar[DT](0))
    for b in range(B):
        for oc in range(OC):
            for oh in range(OH):
                for ow in range(OW):
                    var acc = bias[oc]
                    for ic in range(IC):
                        for kh in range(K):
                            var ih = oh * S + kh - P
                            if ih < 0 or ih >= H:
                                continue
                            for kw in range(K):
                                var iw = ow * S + kw - P
                                if iw < 0 or iw >= W:
                                    continue
                                acc += (
                                    w[_w_idx(oc, ic, kh, kw)]
                                    * x.data[
                                        b * IN_FLAT + ic * H * W + ih * W + iw
                                    ]
                                )
                    ref_out[b * OUT_FLAT + oc * OH * OW + oh * OW + ow] = acc
    for b in range(B):
        for oc in range(OC):
            for oh in range(OH):
                for ow in range(OW):
                    var g = go.data[b * OUT_FLAT + oc * OH * OW + oh * OW + ow]
                    ref_db[oc] += g
                    for ic in range(IC):
                        for kh in range(K):
                            var ih = oh * S + kh - P
                            if ih < 0 or ih >= H:
                                continue
                            for kw in range(K):
                                var iw = ow * S + kw - P
                                if iw < 0 or iw >= W:
                                    continue
                                var xv = x.data[
                                    b * IN_FLAT + ic * H * W + ih * W + iw
                                ]
                                ref_dw[_w_idx(oc, ic, kh, kw)] += g * xv
                                ref_gi[
                                    b * IN_FLAT + ic * H * W + ih * W + iw
                                ] += (w[_w_idx(oc, ic, kh, kw)] * g)

    var ok_cpu = _run["cpu"](
        x, w, bias, go, ref_out, ref_gi, ref_dw, ref_db, None
    )
    print("Conv2D parity CPU:", "OK" if ok_cpu else "FAIL")
    var c = DeviceContext()
    var ok_gpu = _run["gpu"](
        x, w, bias, go, ref_out, ref_gi, ref_dw, ref_db, Optional(c)
    )
    print("Conv2D parity GPU:", "OK" if ok_gpu else "FAIL")
    if ok_cpu and ok_gpu:
        print("CONV PARITY OK")
    else:
        print("CONV PARITY FAIL")


def _run[
    target: StaticString
](
    ref x: Tensor,
    ref w: List[Scalar[DT]],
    ref bias: List[Scalar[DT]],
    ref go_in: Tensor,
    ref ref_out: List[Scalar[DT]],
    ref ref_gi: List[Scalar[DT]],
    ref ref_dw: List[Scalar[DT]],
    ref ref_db: List[Scalar[DT]],
    ctx: Optional[DeviceContext],
) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4)
    var conv = Conv2D[IC, OC, K, S, P, H, W].make[target, Deterministic](ctx)
    # overwrite weights/bias deterministically
    for i in range(len(w)):
        conv.weight.val.data[i] = w[i]
    for i in range(OC):
        conv.bias.val.data[i] = bias[i]

    var xin = Tensor.alloc(B * IN_FLAT)
    for i in range(B * IN_FLAT):
        xin.data[i] = x.data[i]
    var go = Tensor.alloc(B * OUT_FLAT)
    for i in range(B * OUT_FLAT):
        go.data[i] = go_in.data[i]
    var out = Tensor.alloc(B * OUT_FLAT)
    var gi = Tensor.alloc(B * IN_FLAT)

    comptime if target == "cpu":
        conv.forward["cpu", B](TensorRefs[1](xin), out, None)
        conv.zero_grad["cpu"](None)
        conv.vjp["cpu", B](TensorRefs[1](xin), go, TensorRefs[1](gi), None)
    else:
        var c = ctx.value()
        conv.weight.val.upload(c)
        conv.bias.val.upload(c)
        xin.upload(c)
        go.upload(c)
        conv.forward["gpu", B](TensorRefs[1](xin), out, ctx)
        conv.zero_grad["gpu"](ctx)
        conv.vjp["gpu", B](TensorRefs[1](xin), go, TensorRefs[1](gi), ctx)
        out.download(c)
        gi.download(c)
        conv.weight.grd.download(c)
        conv.bias.grd.download(c)

    var ok = True
    for i in range(B * OUT_FLAT):
        if abs(out.data[i] - ref_out[i]) > TOL:
            ok = False
    for i in range(B * IN_FLAT):
        if abs(gi.data[i] - ref_gi[i]) > TOL:
            ok = False
    for i in range(len(w)):
        if abs(conv.weight.grd.data[i] - ref_dw[i]) > TOL:
            ok = False
    for i in range(OC):
        if abs(conv.bias.grd.data[i] - ref_db[i]) > TOL:
            ok = False
    return ok
