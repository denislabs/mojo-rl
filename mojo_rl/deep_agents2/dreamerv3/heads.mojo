"""Reward (twohot) + Cont (binary) MLP heads — forward + loss + vjp.

Ports `embodied/jax/heads.py:MLPHead` for the reward/cont heads of
`agent.py` (`rewhead: symexp_twohot, bins:255`; `conhead: binary`). Both are
a 1-layer MLP (`[Linear, RMSNorm, act]`) + a head Linear:

  RewardHead: feat → gelu(rms(lin0(feat))) → logits[B,BINS];
              loss = twohot_ce(logits, symlog? no — bins are symexp, squash
              identity); pred = Σ softmax·bins.
  ContHead:   feat → gelu(rms(lin0(feat))) → logit[B,1];
              loss = −logp_bernoulli(target); grad = sigmoid(logit) − target.

The reward `feat2tensor` is `concat([deter, stoch_flat])` (FEAT = DETER+S·C).
Forward + recompute-in-vjp (same pattern as encoder/decoder). Activation is
GELU here to match the rest of the PR4/5b port (the config's SiLU switch is a
PR5c-wide change). Validated ≤1e-4 vs jax.vjp (`extract_pr5b2.py`).
"""

from std.memory import alloc
from std.math import exp, log
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.rms_norm import RMSNorm
from mojo_rl.nn2.initializer import Zero
from .rssm import _gelu_buf, _gelu_grad, lin_fwd, rms_fwd, lin_vjp, rms_vjp
from .twohot import twohot_pred, twohot_loss, twohot_loss_backward


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


struct RewardHead[FEAT: Int, UNITS: Int, BINS: Int](
    Movable & ImplicitlyDestructible
):
    """1-layer MLP + symexp-twohot head."""

    var lin0: Linear[Self.FEAT, Self.UNITS]
    var n0: RMSNorm[Self.UNITS]
    var logits: Linear[Self.UNITS, Self.BINS]

    def __init__(out self):
        self.lin0 = Linear[Self.FEAT, Self.UNITS]()
        self.n0 = RMSNorm[Self.UNITS]()
        self.logits = Linear[Self.UNITS, Self.BINS]()

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", "RewardHead: PR5b CPU-only"
        var m = Self()
        m.lin0 = Linear[Self.FEAT, Self.UNITS].make[target, Zero]()
        m.n0 = RMSNorm[Self.UNITS].make[target, Zero]()
        m.logits = Linear[Self.UNITS, Self.BINS].make[target, Zero]()
        return m^

    def forward[
        B: Int
    ](
        mut self,
        feat: UnsafePointer[Scalar[DT], MutAnyOrigin],         # [B, FEAT]
        out_logits: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, BINS]
    ) raises:
        comptime U = Self.UNITS
        var a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var n: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        lin_fwd[Self.FEAT, U, B](self.lin0, feat, a0)
        rms_fwd[U, B](self.n0, a0, n)
        _gelu_buf(n, B * U)
        lin_fwd[U, Self.BINS, B](self.logits, n, out_logits)
        a0.free(); n.free()

    def loss[
        B: Int
    ](
        mut self,
        feat: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, FEAT]
        bins: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [BINS]
        target: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B]
        out_loss: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B]
    ) raises:
        var lg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.BINS
        )
        self.forward[B](feat, lg)
        for b in range(B):
            out_loss[b] = twohot_loss[Self.BINS](lg, b * Self.BINS, bins,
                                                 target[b])
        lg.free()

    def loss_vjp[
        B: Int
    ](
        mut self,
        feat: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, FEAT]
        bins: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [BINS]
        target: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B]
        cot: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [B] loss cotangent
        grad_feat: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, FEAT]
    ) raises:
        comptime U = Self.UNITS
        # recompute forward retaining caches
        var a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var npre: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var lg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.BINS
        )
        lin_fwd[Self.FEAT, U, B](self.lin0, feat, a0)
        rms_fwd[U, B](self.n0, a0, npre)
        for i in range(B * U):
            g[i] = npre[i]
        _gelu_buf(g, B * U)
        lin_fwd[U, Self.BINS, B](self.logits, g, lg)
        # grad on logits = cot·(softmax − twohot(target))
        var d_lg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.BINS
        )
        for i in range(B * Self.BINS):
            d_lg[i] = 0.0
        for b in range(B):
            twohot_loss_backward[Self.BINS](lg, b * Self.BINS, bins, target[b],
                                            cot[b], d_lg)
        # backward through head + MLP
        var d_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        lin_vjp[U, Self.BINS, B](self.logits, d_lg, d_g)
        for i in range(B * U):
            d_g[i] = d_g[i] * _gelu_grad(npre[i])
        var d_a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        rms_vjp[U, B](self.n0, d_g, d_a0)
        lin_vjp[Self.FEAT, U, B](self.lin0, d_a0, grad_feat)
        a0.free(); npre.free(); g.free(); lg.free(); d_lg.free(); d_g.free()
        d_a0.free()


struct ContHead[FEAT: Int, UNITS: Int](Movable & ImplicitlyDestructible):
    """1-layer MLP + binary head. logit is a single unit per row."""

    var lin0: Linear[Self.FEAT, Self.UNITS]
    var n0: RMSNorm[Self.UNITS]
    var logit: Linear[Self.UNITS, 1]

    def __init__(out self):
        self.lin0 = Linear[Self.FEAT, Self.UNITS]()
        self.n0 = RMSNorm[Self.UNITS]()
        self.logit = Linear[Self.UNITS, 1]()

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", "ContHead: PR5b CPU-only"
        var m = Self()
        m.lin0 = Linear[Self.FEAT, Self.UNITS].make[target, Zero]()
        m.n0 = RMSNorm[Self.UNITS].make[target, Zero]()
        m.logit = Linear[Self.UNITS, 1].make[target, Zero]()
        return m^

    def forward[
        B: Int
    ](
        mut self,
        feat: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [B, FEAT]
        out_logit: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, 1]
    ) raises:
        comptime U = Self.UNITS
        var a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var n: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        lin_fwd[Self.FEAT, U, B](self.lin0, feat, a0)
        rms_fwd[U, B](self.n0, a0, n)
        _gelu_buf(n, B * U)
        lin_fwd[U, 1, B](self.logit, n, out_logit)
        a0.free(); n.free()

    def loss[
        B: Int
    ](
        mut self,
        feat: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, FEAT]
        target: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B] in {0,1}
        out_loss: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B]
    ) raises:
        var lo: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
        self.forward[B](feat, lo)
        # loss = −[t·log σ(x) + (1−t)·log σ(−x)] = softplus(x) − t·x
        for b in range(B):
            var x = lo[b]
            var sp = log(Scalar[DT](1.0) + exp(-(x if x >= Scalar[DT](0) else -x)))
            sp += (x if x >= Scalar[DT](0) else Scalar[DT](0.0))  # softplus stable
            out_loss[b] = sp - target[b] * x
        lo.free()

    def loss_vjp[
        B: Int
    ](
        mut self,
        feat: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, FEAT]
        target: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B]
        cot: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [B]
        grad_feat: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, FEAT]
    ) raises:
        comptime U = Self.UNITS
        var a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var npre: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var lo: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
        lin_fwd[Self.FEAT, U, B](self.lin0, feat, a0)
        rms_fwd[U, B](self.n0, a0, npre)
        for i in range(B * U):
            g[i] = npre[i]
        _gelu_buf(g, B * U)
        lin_fwd[U, 1, B](self.logit, g, lo)
        # d loss/d logit = sigmoid(logit) − target
        var d_lo: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
        for b in range(B):
            d_lo[b] = cot[b] * (_sigmoid(lo[b]) - target[b])
        var d_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        lin_vjp[U, 1, B](self.logit, d_lo, d_g)
        for i in range(B * U):
            d_g[i] = d_g[i] * _gelu_grad(npre[i])
        var d_a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        rms_vjp[U, B](self.n0, d_g, d_a0)
        lin_vjp[Self.FEAT, U, B](self.lin0, d_a0, grad_feat)
        a0.free(); npre.free(); g.free(); lo.free(); d_lo.free(); d_g.free()
        d_a0.free()
