"""Conv2D Model wrappers: plain and fused with activations.

Conv2DLayer wraps a plain Conv2D as a Model.
Conv2DReLU/Tanh/Sigmoid/Mish use FusedConv2DActivation — the activation
is applied inside the conv kernel, eliminating a full output read+write.

Parameters:
    ic: input channels
    oc: output channels
    k:  kernel size
    s:  stride
    p:  padding
    h:  input height
    w:  input width

Output spatial dims: out_h = (h + 2*p - k) // s + 1, out_w = (w + 2*p - k) // s + 1
"""

from ..autodiff import AutoFused, Conv2D
from ..autodiff.fused import (
    FusedConv2DActivation,
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
    MishActivation,
)

comptime Conv2DLayer[
    ic: Int, oc: Int, k: Int, s: Int, p: Int, h: Int, w: Int
] = AutoFused[Conv2D[ic, oc, k, s, p, h, w]]

comptime Conv2DReLU[
    ic: Int, oc: Int, k: Int, s: Int, p: Int, h: Int, w: Int
] = AutoFused[FusedConv2DActivation[ic, oc, k, s, p, h, w, ReLUActivation]]

comptime Conv2DTanh[
    ic: Int, oc: Int, k: Int, s: Int, p: Int, h: Int, w: Int
] = AutoFused[FusedConv2DActivation[ic, oc, k, s, p, h, w, TanhActivation]]

comptime Conv2DSigmoid[
    ic: Int, oc: Int, k: Int, s: Int, p: Int, h: Int, w: Int
] = AutoFused[FusedConv2DActivation[ic, oc, k, s, p, h, w, SigmoidActivation]]

comptime Conv2DMish[
    ic: Int, oc: Int, k: Int, s: Int, p: Int, h: Int, w: Int
] = AutoFused[FusedConv2DActivation[ic, oc, k, s, p, h, w, MishActivation]]
