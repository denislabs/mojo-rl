"""Decoder — DreamerV3 vector (MLP) observation decoder, symlog_mse head.

Port of `references/dreamerv3-main/dreamerv3/rssm.py:Decoder` vec path +
`embodied/jax/heads.py:Head.symlog_mse`:

    inp  = concat([stoch_flat, deter])          # stoch FIRST, then deter
    for i in LAYERS:  x = gelu(rms(linear_i(x)))
    pred = pred_linear(x)                        # [B, OBS], symlog-space
    # reconstruction in obs space = symexp(pred); pred() returns `pred`
    recon_loss = Σ_o (pred - symlog(target))²    # MSE(squash=symlog), Agg sum

LAYERS pinned to 2 (PR4 gate unit). Image (CNN/transpose-conv) decoder is
PR6. Validated to ≤1e-4 vs the actual reference (extract_pr4.py).
"""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.rms_norm import RMSNorm
from mojo_rl.nn2.initializer import Zero

from .rssm import (
    _gelu_buf, _gelu_grad, _symlog_scalar, lin_fwd, rms_fwd, lin_vjp, rms_vjp,
)


struct Decoder[FEATIN: Int, OBS: Int, UNITS: Int](
    Movable & ImplicitlyDestructible
):
    """2-layer MLP decoder with a symlog_mse vec head."""

    var lin0: Linear[Self.FEATIN, Self.UNITS]
    var n0: RMSNorm[Self.UNITS]
    var lin1: Linear[Self.UNITS, Self.UNITS]
    var n1: RMSNorm[Self.UNITS]
    var pred: Linear[Self.UNITS, Self.OBS]

    def __init__(out self):
        self.lin0 = Linear[Self.FEATIN, Self.UNITS]()
        self.n0 = RMSNorm[Self.UNITS]()
        self.lin1 = Linear[Self.UNITS, Self.UNITS]()
        self.n1 = RMSNorm[Self.UNITS]()
        self.pred = Linear[Self.UNITS, Self.OBS]()

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", "Decoder: PR4 is CPU-only (forward)"
        var m = Self()
        m.lin0 = Linear[Self.FEATIN, Self.UNITS].make[target, Zero]()
        m.n0 = RMSNorm[Self.UNITS].make[target, Zero]()
        m.lin1 = Linear[Self.UNITS, Self.UNITS].make[target, Zero]()
        m.n1 = RMSNorm[Self.UNITS].make[target, Zero]()
        m.pred = Linear[Self.UNITS, Self.OBS].make[target, Zero]()
        return m^

    def forward[
        B: Int
    ](
        mut self,
        stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, FEATIN-DETER]
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, DETER]
        stoch_dim: Int,
        deter_dim: Int,
        out_pred: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, OBS]
    ) raises:
        comptime U = Self.UNITS
        # inp = concat([stoch_flat, deter])  — stoch first.
        var inp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.FEATIN
        )
        for b in range(B):
            for k in range(stoch_dim):
                inp[b * Self.FEATIN + k] = stoch[b * stoch_dim + k]
            for k in range(deter_dim):
                inp[b * Self.FEATIN + stoch_dim + k] = deter[b * deter_dim + k]
        var a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var nb0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var it0 = TileTensor(inp, row_major[B, Self.FEATIN]())
        var ot0 = TileTensor(a0, row_major[B, U]())
        self.lin0.forward["cpu", B](it0, output=ot0)
        var ri0 = TileTensor(a0, row_major[B, U]())
        var ro0 = TileTensor(nb0, row_major[B, U]())
        self.n0.forward["cpu", B](ri0, output=ro0)
        _gelu_buf(nb0, B * U)

        var a1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var nb1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var it1 = TileTensor(nb0, row_major[B, U]())
        var ot1 = TileTensor(a1, row_major[B, U]())
        self.lin1.forward["cpu", B](it1, output=ot1)
        var ri1 = TileTensor(a1, row_major[B, U]())
        var ro1 = TileTensor(nb1, row_major[B, U]())
        self.n1.forward["cpu", B](ri1, output=ro1)
        _gelu_buf(nb1, B * U)

        var itp = TileTensor(nb1, row_major[B, U]())
        var otp = TileTensor(out_pred, row_major[B, Self.OBS]())
        self.pred.forward["cpu", B](itp, output=otp)
        inp.free(); a0.free(); nb0.free(); a1.free(); nb1.free()

    def vjp[
        B: Int
    ](
        mut self,
        stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [B, stoch_dim]
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [B, deter_dim]
        stoch_dim: Int,
        deter_dim: Int,
        grad_pred: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, OBS]
        grad_stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, stoch_dim]
        grad_deter: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, deter_dim]
    ) raises:
        """Backward via recompute. `grad_pred` is the upstream on the head
        output (= 2·(pred − symlog(target)) for the symlog_mse recon loss).
        Param grads accumulate in the leaves (zeroed here); grads to the
        feat split back into stoch/deter."""
        comptime U = Self.UNITS
        var inp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.FEATIN
        )
        for b in range(B):
            for k in range(stoch_dim):
                inp[b * Self.FEATIN + k] = stoch[b * stoch_dim + k]
            for k in range(deter_dim):
                inp[b * Self.FEATIN + stoch_dim + k] = deter[b * deter_dim + k]
        var a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var nb0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var g0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var a1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var nb1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var g1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        lin_fwd[Self.FEATIN, U, B](self.lin0, inp, a0)
        rms_fwd[U, B](self.n0, a0, nb0)
        for i in range(B * U):
            g0[i] = nb0[i]
        _gelu_buf(g0, B * U)
        lin_fwd[U, U, B](self.lin1, g0, a1)
        rms_fwd[U, B](self.n1, a1, nb1)
        for i in range(B * U):
            g1[i] = nb1[i]
        _gelu_buf(g1, B * U)
        # re-run the head forward so its input cache points at this `g1`
        # (the forward() call's g1 was freed); needed for pred grad_w.
        var pscratch: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.OBS
        )
        lin_fwd[U, Self.OBS, B](self.pred, g1, pscratch)

        # backward
        var d_g1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        lin_vjp[U, Self.OBS, B](self.pred, grad_pred, d_g1)
        for i in range(B * U):
            d_g1[i] = d_g1[i] * _gelu_grad(nb1[i])
        var d_a1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        rms_vjp[U, B](self.n1, d_g1, d_a1)
        var d_g0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        lin_vjp[U, U, B](self.lin1, d_a1, d_g0)
        for i in range(B * U):
            d_g0[i] = d_g0[i] * _gelu_grad(nb0[i])
        var d_a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        rms_vjp[U, B](self.n0, d_g0, d_a0)
        var d_inp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.FEATIN
        )
        lin_vjp[Self.FEATIN, U, B](self.lin0, d_a0, d_inp)
        for b in range(B):
            for k in range(stoch_dim):
                grad_stoch[b * stoch_dim + k] = d_inp[b * Self.FEATIN + k]
            for k in range(deter_dim):
                grad_deter[b * deter_dim + k] = d_inp[
                    b * Self.FEATIN + stoch_dim + k
                ]
        inp.free(); a0.free(); nb0.free(); g0.free(); a1.free(); nb1.free()
        g1.free(); pscratch.free(); d_g1.free(); d_a1.free(); d_g0.free()
        d_a0.free(); d_inp.free()

    @staticmethod
    def recon_loss[
        B: Int
    ](
        pred: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, OBS]
        target: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, OBS]
        out_loss: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B]
    ):
        """Σ_o (pred - symlog(target))²  per row (MSE in symlog space)."""
        for b in range(B):
            var s: Scalar[DT] = 0.0
            for o in range(Self.OBS):
                var d = pred[b * Self.OBS + o] - _symlog_scalar(
                    target[b * Self.OBS + o]
                )
                s += d * d
            out_loss[b] = s
