"""Spike: SIMD-on-trait dispatch for `ElementOp`.

Front-loaded risk for Phase 1.3 (`Elementwise[DIM, OP: ElementOp]`).
Verifies that:

  1. Mojo nightly accepts `@staticmethod fn forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]`
     declared on a trait.
  2. A conforming struct (`TanhOp`) can implement it.
  3. A templated dispatcher `apply_simd[OP: ElementOp, W: Int](x)` correctly
     resolves OP and dispatches `OP.forward_simd[W](x)` for multiple W.
  4. The same trait also supports a non-SIMD static method
     (`forward_scalar`) declared the same way.

If this PASSES: `Elementwise[DIM, OP]` ships as planned in §8.4.2.
If this FAILS: fall back to per-leaf modules sharing a shared free-
function body (~600 LOC dedup still possible, slightly less elegant).
"""

from std.math import tanh

from mojo_rl.nn.constants import DT


# ──────────────────────────────────────────────────────────────────────
# Trait declaration: static methods, both scalar and SIMD-templated.
# `fn` is removed in Mojo nightly (see `feedback_mojo_nightly_self_required`).
# Use `def` without `raises` for pure math.
# ──────────────────────────────────────────────────────────────────────


trait ElementOp(Movable & Deinitable):
    """Per-element forward + backward op for `Elementwise[DIM, OP]`."""

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]: ...

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]: ...


# ──────────────────────────────────────────────────────────────────────
# Conformer: TanhOp. Both static methods just call std.math.tanh,
# which natively dispatches scalar / SIMD.
# ──────────────────────────────────────────────────────────────────────


struct TanhOp(ElementOp):
    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return tanh(x)

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        return tanh(x)


# ──────────────────────────────────────────────────────────────────────
# Templated dispatcher. This is the shape `Elementwise.forward` will use.
# ──────────────────────────────────────────────────────────────────────


def apply_simd[OP: ElementOp, W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
    return OP.forward_simd[W](x)


def apply_scalar[OP: ElementOp](x: Scalar[DT]) -> Scalar[DT]:
    return OP.forward_scalar(x)


def main() raises:
    print("=" * 60)
    print("Spike: ElementOp SIMD-on-trait dispatch")
    print("=" * 60)

    # Scalar dispatch.
    var s = Scalar[DT](0.5)
    var s_out = apply_scalar[TanhOp](s)
    var s_ref = tanh(s)
    print("scalar: tanh(0.5) =", s_out, " (ref:", s_ref, ")")
    if (s_out - s_ref).__abs__() > Scalar[DT](1e-6):
        raise Error("scalar dispatch mismatch")

    # SIMD W=4.
    var v4 = SIMD[DT, 4](0.1, 0.5, 1.0, 2.0)
    var out4 = apply_simd[TanhOp, 4](v4)
    var ref4 = tanh(v4)
    print("simd[4]: tanh([0.1, 0.5, 1.0, 2.0]) =", out4)
    print("      ref =", ref4)
    for k in range(4):
        if (out4[k] - ref4[k]).__abs__() > Scalar[DT](1e-6):
            raise Error("simd[4] dispatch mismatch at " + String(k))

    # SIMD W=8 — different specialization from W=4.
    var v8 = SIMD[DT, 8](0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    var out8 = apply_simd[TanhOp, 8](v8)
    var ref8 = tanh(v8)
    print("simd[8]: tanh(...) =", out8)
    print("      ref =", ref8)
    for k in range(8):
        if (out8[k] - ref8[k]).__abs__() > Scalar[DT](1e-6):
            raise Error("simd[8] dispatch mismatch at " + String(k))

    # SIMD W=16 — third specialization.
    var v16 = SIMD[DT, 16](
        -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
         2.0,  2.5,  3.0,  3.5, 4.0, 4.5, 5.0, 5.5,
    )
    var out16 = apply_simd[TanhOp, 16](v16)
    var ref16 = tanh(v16)
    for k in range(16):
        if (out16[k] - ref16[k]).__abs__() > Scalar[DT](1e-6):
            raise Error("simd[16] dispatch mismatch at " + String(k))
    print("simd[16]: all 16 lanes match tanh(v) reference")

    print("=" * 60)
    print("SPIKE PASSED")
    print("=" * 60)
    print(
        "Elementwise[DIM, OP: ElementOp] design viable — proceed with §8.4.2."
    )
