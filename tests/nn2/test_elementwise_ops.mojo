"""CPU tests for the Phase 8.4 elementwise op Modules.

Covers Scale[DIM], Sum[DIM], Mean[DIM]:
  - forward correctness against hand-computed values
  - backward correctness via FD gradcheck (`virtual loss = Σ go·output`)

The packed `Sub[DIM]` and `ElemMin[DIM]` are gone (Cleanup 11) — see
`tests/nn2/test_binary_ops.mojo` for their two-input replacements
`BinarySub[DIM]` / `BinaryElemMin[DIM]`.

All ops are CPU-only in Phase 8.4 — the GPU paths raise.
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_almost_equal, assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Module
from mojo_rl.nn2.primitives.scale import Scale
from mojo_rl.nn2.primitives.reduce import Sum, Mean
from mojo_rl.nn2.initializer import Zero


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────
# Scale
# ──────────────────────────────────────────────────────────────────────────


def test_scale_forward_backward() raises:
    comptime DIM = 3
    comptime BATCH = 4
    var in_buf = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf = alloc[Scalar[DT]](BATCH * DIM)
    var go_buf = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf = alloc[Scalar[DT]](BATCH * DIM)
    for i in range(BATCH * DIM):
        in_buf[i] = Scalar[DT](Float32(i) * 0.13 - 0.5)
        go_buf[i] = Scalar[DT](Float32(i) * 0.07 - 0.4)

    var s = Scale[DIM].make[target="cpu", INIT=Zero]()
    s.multiplier = Scalar[DT](2.5)

    var in_t = TileTensor(in_buf, row_major[BATCH, DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, DIM]())
    s.forward["cpu", BATCH](in_t, out_t)
    for i in range(BATCH * DIM):
        assert_almost_equal(
            out_buf[i], in_buf[i] * Scalar[DT](2.5), atol=1e-6
        )

    var go_t = TileTensor(go_buf, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi_buf, row_major[BATCH, DIM]())
    s.backward["cpu", BATCH](go_t, gi_t)
    for i in range(BATCH * DIM):
        assert_almost_equal(
            gi_buf[i], go_buf[i] * Scalar[DT](2.5), atol=1e-6
        )

    # FD gradcheck: virtual loss = Σ go·output, d/d_in[k] = go[k]·multiplier
    var eps: Scalar[DT] = 1e-3
    var max_rel: Scalar[DT] = 0.0
    for idx in range(BATCH * DIM):
        var orig = in_buf[idx]
        in_buf[idx] = orig + eps
        s.forward["cpu", BATCH](in_t, out_t)
        var L_plus: Scalar[DT] = 0.0
        for k in range(BATCH * DIM):
            L_plus += go_buf[k] * out_buf[k]
        in_buf[idx] = orig - eps
        s.forward["cpu", BATCH](in_t, out_t)
        var L_minus: Scalar[DT] = 0.0
        for k in range(BATCH * DIM):
            L_minus += go_buf[k] * out_buf[k]
        in_buf[idx] = orig
        var num = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
        var ana = gi_buf[idx]
        var ae = fabs(num - ana)
        var denom = fabs(num) + fabs(ana) + Scalar[DT](1e-6)
        var rel = ae / denom
        if rel > max_rel:
            max_rel = rel
    print("  Scale FD gradcheck max_rel=", max_rel)
    assert_true(max_rel < Scalar[DT](1e-3), "Scale FD too loose")

    in_buf.free(); out_buf.free(); go_buf.free(); gi_buf.free()
    print("  test_scale PASSED")


# ──────────────────────────────────────────────────────────────────────────
# Sum
# ──────────────────────────────────────────────────────────────────────────


def test_sum_forward_backward() raises:
    comptime DIM = 4
    comptime BATCH = 3
    var in_buf = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf = alloc[Scalar[DT]](BATCH * 1)
    var go_buf = alloc[Scalar[DT]](BATCH * 1)
    var gi_buf = alloc[Scalar[DT]](BATCH * DIM)

    for i in range(BATCH * DIM):
        in_buf[i] = Scalar[DT](Float32(i) * 0.11 - 0.3)
    for b in range(BATCH):
        go_buf[b] = Scalar[DT](Float32(b) * 0.5 + 0.25)

    var sm = Sum[DIM].make[target="cpu", INIT=Zero]()
    var in_t = TileTensor(in_buf, row_major[BATCH, DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, 1]())
    sm.forward["cpu", BATCH](in_t, out_t)
    for b in range(BATCH):
        var expected: Scalar[DT] = 0.0
        for d in range(DIM):
            expected += in_buf[b * DIM + d]
        assert_almost_equal(out_buf[b], expected, atol=1e-6)

    var go_t = TileTensor(go_buf, row_major[BATCH, 1]())
    var gi_t = TileTensor(gi_buf, row_major[BATCH, DIM]())
    sm.backward["cpu", BATCH](go_t, gi_t)
    for b in range(BATCH):
        for d in range(DIM):
            assert_almost_equal(gi_buf[b * DIM + d], go_buf[b], atol=1e-7)

    print("  test_sum PASSED")
    in_buf.free(); out_buf.free(); go_buf.free(); gi_buf.free()


# ──────────────────────────────────────────────────────────────────────────
# Mean
# ──────────────────────────────────────────────────────────────────────────


def test_mean_forward_backward() raises:
    comptime DIM = 4
    comptime BATCH = 3
    var in_buf = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf = alloc[Scalar[DT]](BATCH * 1)
    var go_buf = alloc[Scalar[DT]](BATCH * 1)
    var gi_buf = alloc[Scalar[DT]](BATCH * DIM)

    for i in range(BATCH * DIM):
        in_buf[i] = Scalar[DT](Float32(i) * 0.11 - 0.3)
    for b in range(BATCH):
        go_buf[b] = Scalar[DT](Float32(b) * 0.5 + 0.25)

    var mn = Mean[DIM].make[target="cpu", INIT=Zero]()
    var in_t = TileTensor(in_buf, row_major[BATCH, DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, 1]())
    mn.forward["cpu", BATCH](in_t, out_t)
    var inv_dim = Scalar[DT](1.0) / Scalar[DT](DIM)
    for b in range(BATCH):
        var expected: Scalar[DT] = 0.0
        for d in range(DIM):
            expected += in_buf[b * DIM + d]
        assert_almost_equal(out_buf[b], expected * inv_dim, atol=1e-6)

    var go_t = TileTensor(go_buf, row_major[BATCH, 1]())
    var gi_t = TileTensor(gi_buf, row_major[BATCH, DIM]())
    mn.backward["cpu", BATCH](go_t, gi_t)
    for b in range(BATCH):
        for d in range(DIM):
            assert_almost_equal(
                gi_buf[b * DIM + d], go_buf[b] * inv_dim, atol=1e-7
            )

    print("  test_mean PASSED")
    in_buf.free(); out_buf.free(); go_buf.free(); gi_buf.free()


def main() raises:
    print("=" * 70)
    print("nn2 Phase 8.4 — elementwise op Modules CPU tests")
    print("=" * 70)
    test_scale_forward_backward()
    test_sum_forward_backward()
    test_mean_forward_backward()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
