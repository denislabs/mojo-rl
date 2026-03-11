"""Pooling Model wrappers.

Aliases wrapping MaxPool2D and AvgPool2D DiffOp primitives as Models.

Parameters:
    ch: channels
    h:  input height
    w:  input width
    ps: pool size (square pooling window)

Output spatial dims: out_h = h // ps, out_w = w // ps
"""

from ..autodiff import AutoFused, MaxPool2D, AvgPool2D

comptime MaxPoolLayer[ch: Int, h: Int, w: Int, ps: Int] = AutoFused[
    MaxPool2D[ch, h, w, ps]
]

comptime AvgPoolLayer[ch: Int, h: Int, w: Int, ps: Int] = AutoFused[
    AvgPool2D[ch, h, w, ps]
]
