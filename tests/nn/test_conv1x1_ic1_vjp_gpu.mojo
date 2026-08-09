"""Does the GPU Conv2D vjp compute CORRECT grads for IC=1, K=1 (COL=1)?

This is the action-embedding conv in MuZero's dynamics (Conv2D[1, REDC, 1,1,0]).
COL = IC*K*K = 1 — the degenerate matmul width. The CPU vjp crashes on NVIDIA
(max_matmul N=1); this checks whether the GPU vjp (the real-run path) silently
corrupts the gradient, which would break the action→latent learning and stall
loss_reward. Compares the GPU vjp to a hand-computed reference for a 1x1/IC=1
conv: out[b,oc,p] = w[oc]*x[b,p] + bias[oc].

  grads match reference  → GPU COL=1 conv vjp OK; reward bug is elsewhere
  grads WRONG            → GPU COL=1 conv vjp broken = the reward/dynamics bug
"""

from max.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.conv2d import Conv2D


def main() raises:
    comptime IC = 1
    comptime OC = 4
    comptime K = 1
    comptime H = 2
    comptime W = 2
    comptime HW = H * W          # 4
    comptime B = 2
    comptime Conv = Conv2D[IC, OC, K, 1, 0, H, W]   # COL = 1
    var ctx = DeviceContext()
    var conv = Conv.make["gpu", INIT=Kaiming](Optional(ctx))

    # set known weights/bias (w[oc], bias[oc]) and upload
    for oc in range(OC):
        conv.weight.val.data[oc] = Scalar[DT](0.5) * Scalar[DT](oc + 1)   # 0.5,1,1.5,2
        conv.bias.val.data[oc] = Scalar[DT](0.1) * Scalar[DT](oc)
    conv.weight.val.upload(ctx)
    conv.bias.val.upload(ctx)

    # input x[b, p]  (IC=1 so IN_FLAT = HW)
    var x = Tensor.alloc(B * HW)
    for b in range(B):
        for p in range(HW):
            x.data[b * HW + p] = Scalar[DT](0.3) * Scalar[DT]((b * HW + p) % 5 - 2)
    x.ensure_gpu(ctx, B * HW); x.upload(ctx)

    var out = Tensor(); out.ensure_gpu(ctx, B * OC * HW)
    conv.forward["gpu", B](TensorRefs[1](x), out, Optional(ctx))

    # grad_output go[b, oc, p]
    var go = Tensor.alloc(B * OC * HW)
    for i in range(B * OC * HW):
        go.data[i] = Scalar[DT](0.2) * Scalar[DT](i % 7 - 3)
    go.ensure_gpu(ctx, B * OC * HW); go.upload(ctx)

    conv.zero_grad["gpu"](Optional(ctx))
    var gx = Tensor(); gx.ensure_gpu(ctx, B * HW)
    conv.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gx), Optional(ctx))
    conv.weight.grd.download(ctx); conv.bias.grd.download(ctx); gx.download(ctx)
    ctx.synchronize()

    # ── references ──
    # grad_w[oc] = Σ_b Σ_p go[b,oc,p]*x[b,p] ; grad_bias[oc]=Σ go ;
    # grad_x[b,p] = Σ_oc go[b,oc,p]*w[oc]
    var werr = Scalar[DT](0); var berr = Scalar[DT](0); var xerr = Scalar[DT](0)
    for oc in range(OC):
        var gw = Scalar[DT](0); var gb = Scalar[DT](0)
        for b in range(B):
            for p in range(HW):
                var g = go.data[b * OC * HW + oc * HW + p]
                gw += g * x.data[b * HW + p]
                gb += g
        werr += abs(gw - conv.weight.grd.data[oc])
        berr += abs(gb - conv.bias.grd.data[oc])
    for b in range(B):
        for p in range(HW):
            var rx = Scalar[DT](0)
            for oc in range(OC):
                rx += go.data[b * OC * HW + oc * HW + p] * conv.weight.val.data[oc]
            xerr += abs(rx - gx.data[b * HW + p])

    print("grad_weight err =", werr)
    print("grad_bias   err =", berr)
    print("grad_input  err =", xerr)
    if werr + berr + xerr > Scalar[DT](1e-3):
        print(">>> GPU COL=1 CONV VJP IS WRONG — the action-embedding gradient is corrupted <<<")
    else:
        print(">>> GPU COL=1 conv vjp correct — reward bug is elsewhere <<<")
