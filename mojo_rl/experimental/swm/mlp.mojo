"""A small two-layer MLP with hand-written gradients and Adam. CPU, float64.

**Why not `mojo_rl.nn`.** `nn` is float32 (`nn/constants.mojo`: `DT = float32`)
and GPU-oriented. Every SWM gate so far reads `det H` and residuals at the
1e-12 level, and Phase 3 has to distinguish "the frame channel collapsed to
rank 1" from "the obstruction is genuinely absent" — a distinction float32
noise would blur. The networks here are tiny (16 -> 32 -> 10), the gradients are
textbook, and `experimental/` explicitly warns that its own APIs churn, so
coupling this research line to `nn`'s evolving surface buys nothing. If SWM
graduates, porting to `nn` is the graduation work.

tanh hidden, linear output, single flat parameter slab so Adam is one loop.
"""

from std.math import sqrt, tanh

from .rng import Rng


struct Mlp[
    IN: Int, HID: Int, OUT: Int, dtype: DType = DType.float64
](Copyable, Movable):
    comptime W1: Int = 0
    comptime B1: Int = Self.IN * Self.HID
    comptime W2: Int = Self.B1 + Self.HID
    comptime B2: Int = Self.W2 + Self.HID * Self.OUT
    comptime PSIZE: Int = Self.B2 + Self.OUT

    var p: List[Scalar[Self.dtype]]
    var g: List[Scalar[Self.dtype]]
    var m: List[Scalar[Self.dtype]]
    var v: List[Scalar[Self.dtype]]
    var t: Int
    var b1t: Float64
    var b2t: Float64

    def __init__(out self, mut rng: Rng):
        """Xavier-ish init; biases at zero."""
        self.p = List[Scalar[Self.dtype]](length=Self.PSIZE, fill=0)
        self.g = List[Scalar[Self.dtype]](length=Self.PSIZE, fill=0)
        self.m = List[Scalar[Self.dtype]](length=Self.PSIZE, fill=0)
        self.v = List[Scalar[Self.dtype]](length=Self.PSIZE, fill=0)
        self.t = 0
        self.b1t = 1.0
        self.b2t = 1.0
        var s1 = sqrt(2.0 / Float64(Self.IN + Self.HID))
        for i in range(Self.IN * Self.HID):
            self.p[Self.W1 + i] = Scalar[Self.dtype](rng.normal() * s1)
        var s2 = sqrt(2.0 / Float64(Self.HID + Self.OUT))
        for i in range(Self.HID * Self.OUT):
            self.p[Self.W2 + i] = Scalar[Self.dtype](rng.normal() * s2)

    def __init__(out self, *, copy: Self):
        self.p = copy.p.copy()
        self.g = copy.g.copy()
        self.m = copy.m.copy()
        self.v = copy.v.copy()
        self.t = copy.t
        self.b1t = copy.b1t
        self.b2t = copy.b2t

    def __init__(out self, *, deinit move: Self):
        self.p = move.p^
        self.g = move.g^
        self.m = move.m^
        self.v = move.v^
        self.t = move.t
        self.b1t = move.b1t
        self.b2t = move.b2t

    def zero_grad(mut self):
        for i in range(Self.PSIZE):
            self.g[i] = 0

    def forward(
        self,
        x: List[Scalar[Self.dtype]],
        mut hid: List[Scalar[Self.dtype]],
        mut out: List[Scalar[Self.dtype]],
    ):
        """`hid = tanh(x W1 + b1)`, `out = hid W2 + b2`. `hid` is kept for backward."""
        for j in range(Self.HID):
            var s = self.p[Self.B1 + j]
            for i in range(Self.IN):
                s += x[i] * self.p[Self.W1 + i * Self.HID + j]
            hid[j] = Scalar[Self.dtype](tanh(Float64(s)))
        for k in range(Self.OUT):
            var s = self.p[Self.B2 + k]
            for j in range(Self.HID):
                s += hid[j] * self.p[Self.W2 + j * Self.OUT + k]
            out[k] = s

    def backward(
        mut self,
        x: List[Scalar[Self.dtype]],
        hid: List[Scalar[Self.dtype]],
        d_out: List[Scalar[Self.dtype]],
        mut d_x: List[Scalar[Self.dtype]],
    ):
        """Accumulate parameter grads; write `dL/dx` into `d_x` (for chaining)."""
        var d_hid = List[Scalar[Self.dtype]](length=Self.HID, fill=0)
        for k in range(Self.OUT):
            var dk = d_out[k]
            if dk == 0:
                continue
            self.g[Self.B2 + k] += dk
            for j in range(Self.HID):
                self.g[Self.W2 + j * Self.OUT + k] += hid[j] * dk
                d_hid[j] += self.p[Self.W2 + j * Self.OUT + k] * dk
        for i in range(Self.IN):
            d_x[i] = 0
        for j in range(Self.HID):
            # tanh' = 1 - tanh^2
            var dz = d_hid[j] * (Scalar[Self.dtype](1) - hid[j] * hid[j])
            if dz == 0:
                continue
            self.g[Self.B1 + j] += dz
            for i in range(Self.IN):
                self.g[Self.W1 + i * Self.HID + j] += x[i] * dz
                d_x[i] += self.p[Self.W1 + i * Self.HID + j] * dz

    def adam_step(
        mut self,
        lr: Float64,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        eps: Float64 = 1e-8,
    ):
        self.t += 1
        # Bias correction: 1 - beta^t, accumulated rather than pow()'d. The
        # accumulators are carried on the struct so this stays O(1) per step
        # instead of O(t).
        self.b1t *= beta1
        self.b2t *= beta2
        var bc1 = 1.0 - self.b1t
        var bc2 = 1.0 - self.b2t
        if bc1 <= 1e-16:
            bc1 = 1e-16
        if bc2 <= 1e-16:
            bc2 = 1e-16
        for i in range(Self.PSIZE):
            var gi = Float64(self.g[i])
            var mi = beta1 * Float64(self.m[i]) + (1.0 - beta1) * gi
            var vi = beta2 * Float64(self.v[i]) + (1.0 - beta2) * gi * gi
            self.m[i] = Scalar[Self.dtype](mi)
            self.v[i] = Scalar[Self.dtype](vi)
            var step = lr * (mi / bc1) / (sqrt(vi / bc2) + eps)
            self.p[i] = self.p[i] - Scalar[Self.dtype](step)
