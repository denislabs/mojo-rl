"""Encoder — DreamerV3 vector (MLP) observation encoder.

Port of `references/dreamerv3-main/dreamerv3/rssm.py:Encoder` vec path:

    x = symlog(obs)                       # squish (symlog=True default)
    for i in LAYERS:  x = gelu(rms(linear_i(x)))
    tokens = x                            # [B, UNITS]

Image (CNN) path is PR6 (needs GPU Conv2D). LAYERS is pinned to 2 here —
the validated unit for the PR4 gate; more layers extend by adding fields
(a comptime InlineArray of Linear[UNITS,UNITS] is the PR5 generalization).
Validated to ≤1e-4 vs the actual reference (extract_pr4.py).
"""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.rms_norm import RMSNorm
from mojo_rl.nn2.initializer import Zero

from .rssm import (
    _gelu_buf, _gelu_grad, _symlog_scalar, _symlog_grad,
    lin_fwd, rms_fwd, lin_vjp, rms_vjp,
)


struct Encoder[OBS: Int, UNITS: Int](Movable & ImplicitlyDestructible):
    """2-layer MLP encoder. symlog → [Linear,RMSNorm,GELU]×2 → tokens."""

    var lin0: Linear[Self.OBS, Self.UNITS]
    var n0: RMSNorm[Self.UNITS]
    var lin1: Linear[Self.UNITS, Self.UNITS]
    var n1: RMSNorm[Self.UNITS]

    def __init__(out self):
        self.lin0 = Linear[Self.OBS, Self.UNITS]()
        self.n0 = RMSNorm[Self.UNITS]()
        self.lin1 = Linear[Self.UNITS, Self.UNITS]()
        self.n1 = RMSNorm[Self.UNITS]()

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", "Encoder: PR4 is CPU-only (forward)"
        var m = Self()
        m.lin0 = Linear[Self.OBS, Self.UNITS].make[target, Zero]()
        m.n0 = RMSNorm[Self.UNITS].make[target, Zero]()
        m.lin1 = Linear[Self.UNITS, Self.UNITS].make[target, Zero]()
        m.n1 = RMSNorm[Self.UNITS].make[target, Zero]()
        return m^

    def forward[
        B: Int
    ](
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],        # [B, OBS]
        tokens: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, UNITS]
    ) raises:
        comptime U = Self.UNITS
        # symlog squish of the input.
        var sx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.OBS
        )
        for i in range(B * Self.OBS):
            sx[i] = _symlog_scalar(obs[i])
        var a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var nb0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var it0 = TileTensor(sx, row_major[B, Self.OBS]())
        var ot0 = TileTensor(a0, row_major[B, U]())
        self.lin0.forward["cpu", B](it0, output=ot0)
        var ri0 = TileTensor(a0, row_major[B, U]())
        var ro0 = TileTensor(nb0, row_major[B, U]())
        self.n0.forward["cpu", B](ri0, output=ro0)
        _gelu_buf(nb0, B * U)

        var a1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var it1 = TileTensor(nb0, row_major[B, U]())
        var ot1 = TileTensor(a1, row_major[B, U]())
        self.lin1.forward["cpu", B](it1, output=ot1)
        var to = TileTensor(tokens, row_major[B, U]())
        var ri1 = TileTensor(a1, row_major[B, U]())
        self.n1.forward["cpu", B](ri1, output=to)
        _gelu_buf(tokens, B * U)
        sx.free(); a0.free(); nb0.free(); a1.free()

    def vjp[
        B: Int
    ](
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],          # [B, OBS]
        grad_tokens: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, UNITS]
        grad_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, OBS]
    ) raises:
        """Backward via recompute: re-run forward retaining pre-gelu caches,
        then chain the primitive vjps. Param grads accumulate in the leaves
        (zeroed here). `grad_obs` is produced for completeness (obs has no
        upstream consumer in the WM)."""
        comptime U = Self.UNITS
        # ── recompute forward, retaining buffers ─────────────────────────
        var sx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.OBS
        )
        for i in range(B * Self.OBS):
            sx[i] = _symlog_scalar(obs[i])
        var a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var nb0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var g0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var a1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        var tk: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        lin_fwd[Self.OBS, U, B](self.lin0, sx, a0)
        rms_fwd[U, B](self.n0, a0, nb0)
        for i in range(B * U):
            g0[i] = nb0[i]
        _gelu_buf(g0, B * U)
        lin_fwd[U, U, B](self.lin1, g0, a1)
        rms_fwd[U, B](self.n1, a1, tk)

        # ── backward ─────────────────────────────────────────────────────
        var d_tk: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        for i in range(B * U):
            d_tk[i] = grad_tokens[i] * _gelu_grad(tk[i])
        var d_a1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        rms_vjp[U, B](self.n1, d_tk, d_a1)
        var d_g0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        lin_vjp[U, U, B](self.lin1, d_a1, d_g0)
        for i in range(B * U):
            d_g0[i] = d_g0[i] * _gelu_grad(nb0[i])
        var d_a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * U)
        rms_vjp[U, B](self.n0, d_g0, d_a0)
        var d_sx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.OBS
        )
        lin_vjp[Self.OBS, U, B](self.lin0, d_a0, d_sx)
        for i in range(B * Self.OBS):
            grad_obs[i] = d_sx[i] * _symlog_grad(obs[i])

        sx.free(); a0.free(); nb0.free(); g0.free(); a1.free(); tk.free()
        d_tk.free(); d_a1.free(); d_g0.free(); d_a0.free(); d_sx.free()
