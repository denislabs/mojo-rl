"""Spike: autograd-lite codegen for chains of `ElementOp`.

Phase H.1. Front-loads the risk of synthesising a forward + VJP body
directly from the comptime-known list of per-op `forward_scalar` /
`backward_scalar` methods on `ElementOp` conformers — *without* spelling
out an Elementwise module per slot, and without the inter-module slabs +
per-op kernel launches that `Sequential[Elementwise[OP0], ...]` incurs.

Why this matters (per the audit §H.1):
  - MBPO's dynamics-loss + reward-prediction head + synthetic-rollout
    chain pile up ~25 hand-written VJPs.  Each one is silent-ε surface
    area.
  - Sequential composition is correct but expensive: N mid-slabs
    (BATCH*DIM each) + N kernel launches + N global-memory round-trips.
  - This spike proves we can collapse all N forward ops into a single
    register-resident per-element pass (CPU) and that the symmetric
    reverse pass produces bit-identical gradients.

Scope (deliberately narrow — this is a spike, not a production rollout):
  - Unary chain only (single input → single output, all ARITY-1 ops).
    The audit's literal example (Slice + Tanh + Scale + BinarySub +
    Sum) crosses 4 op families — not the right shape for a "does the
    codegen pattern work?" risk-burn.  The dominant pattern in MBPO's
    actual loss bodies is unary activation chains; this is where the
    fused-kernel win is biggest.  Binary / reduce extension is a
    mechanical follow-up once the unary pattern is proven.
  - CPU only.  GPU kernel codegen on a variadic-OP fused chain follows
    once the CPU dispatch path is bit-identity-validated.
  - Scalar inner loop only.  SIMD specialisation drops in by replacing
    `OPS[i].forward_scalar` with `OPS[i].forward_simd[W]` — same comptime
    iteration shape.

Test chain (5 ops, mixing both `owns_cache` modes):

    AutoFusedUnary[DIM,
        TanhOp,    # owns_cache=True  → cache stores y
        ReLUOp,    # owns_cache=False → cache stores x (= prev y)
        SymlogOp,  # owns_cache=False → cache stores x
        TanhOp,    # owns_cache=True
        TanhOp,    # owns_cache=True (consecutive owns_cache=True ops)
    ]

Validation (two layers):
  1. **Manual scalar oracle** on a 1-row, 4-element input — hand-traced
     forward chain matches; hand-traced gradient via chain rule matches.
     Catches off-by-one / wrong-cache-source bugs that bit-identity
     against Sequential would miss (since both could agree on the
     same wrong answer).
  2. **Sequential bit-identity** — build the same chain as
     `Sequential[Elementwise[DIM, TanhOp], Elementwise[DIM, ReLUOp], ...]`,
     run forward + vjp on the same input.  Outputs and grad-inputs must
     match to bit-identity (single-precision exact) on multi-row input.
"""

from std.memory import alloc
from std.random import seed, random_float64
from std.testing import assert_true
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.element_op import ElementOp
from mojo_rl.nn.primitives.ops.tanh_op import TanhOp
from mojo_rl.nn.primitives.ops.relu_op import ReLUOp
from mojo_rl.nn.primitives.ops.symlog_op import SymlogOp

# Reference path (Sequential of Elementwise leaves).
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.initializer import Kaiming


# ──────────────────────────────────────────────────────────────────────
# AutoFusedUnary — N ElementOps fused into a single per-element scalar
# pass per direction. No per-op module; no inter-op slabs.
#
# Owns: one contiguous CPU cache buffer of size N * BATCH_MAX * DIM,
# indexed as cache[i * BATCH * DIM + r * DIM + c].  Grown lazily on
# first forward.
#
# Forward (per (row, col), comptime-unrolled over N ops):
#     prev = input[r, c]
#     for i in 0..N:
#         next = OPS[i].forward_scalar(prev)
#         if OPS[i].owns_cache:  cache[i][r, c] = next
#         else:                  cache[i][r, c] = prev
#         prev = next
#     output[r, c] = prev
#
# Backward (per (row, col), comptime-unrolled in reverse):
#     g = grad_output[r, c]
#     for i in N-1..0:
#         g = OPS[i].backward_scalar(cache[i][r, c], g)
#     grad_input[r, c] = g
# ──────────────────────────────────────────────────────────────────────


struct AutoFusedUnary[DIM: Int, *OPS: ElementOp]:
    comptime N: Int = Self.OPS.size
    comptime IN_DIM: Int = Self.DIM
    comptime OUT_DIM: Int = Self.DIM

    var cache: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var cache_cap: Int  # N * BATCH * DIM that `cache` currently sized for.

    def __init__(out self):
        comptime assert Self.N >= 1, "AutoFusedUnary needs at least one op"
        self.cache = alloc[Scalar[DT]](1)
        self.cache_cap = 0

    def __del__(deinit self):
        self.cache.free()

    def _ensure_cache(mut self, batch: Int):
        var needed = Self.N * batch * Self.DIM
        if self.cache_cap < needed:
            self.cache.free()
            self.cache = alloc[Scalar[DT]](needed)
            self.cache_cap = needed

    def forward[BATCH: Int](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        self._ensure_cache(BATCH)
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
        var cache_p = self.cache
        comptime BD = BATCH * Self.DIM

        for k in range(BD):
            var prev = in_p[k]
            # Comptime-unrolled apply of OPS[0..N).  The dispatch
            # `Self.OPS[i].forward_scalar(prev)` resolves to a direct
            # static call at codegen time — no vtable, no closure.
            comptime for i in range(Self.N):
                var nxt = Self.OPS[i].forward_scalar(prev)
                # Per-op cache write at offset i*BD + k.
                comptime if Self.OPS[i].owns_cache:
                    cache_p[i * BD + k] = nxt
                else:
                    cache_p[i * BD + k] = prev
                prev = nxt
            out_p[k] = prev

    def vjp[BATCH: Int](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut grad_input: TileTensor[
            mut=True, dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
        var cache_p = self.cache
        comptime BD = BATCH * Self.DIM

        for k in range(BD):
            var g = go_p[k]
            # Reverse comptime-unrolled apply of OPS[N-1..0].
            comptime for j in range(Self.N):
                comptime i = Self.N - 1 - j
                var c = cache_p[i * BD + k]
                g = Self.OPS[i].backward_scalar(c, g)
            gi_p[k] = g


# ──────────────────────────────────────────────────────────────────────
# Test 1 — Manual scalar oracle on a 1×4 input.  Hand-traces forward
# AND backward through the 5-op chain so we catch wrong-cache-source
# bugs that wouldn't surface in Test 2 (bit-identity against Sequential
# would agree on the wrong answer if both shared the bug).
# ──────────────────────────────────────────────────────────────────────


from std.math import tanh as math_tanh, log as math_log


def _expected_forward(x: Scalar[DT]) -> Scalar[DT]:
    """Reference Python-style chain: tanh → relu → symlog → tanh → tanh."""
    var y0 = math_tanh(x)
    var y1 = y0 if y0 > Scalar[DT](0) else Scalar[DT](0)
    var ax = y1 if y1 >= Scalar[DT](0) else -y1
    var sgn = Scalar[DT](1) if y1 >= Scalar[DT](0) else Scalar[DT](-1)
    var y2 = sgn * math_log(Scalar[DT](1) + ax)
    var y3 = math_tanh(y2)
    var y4 = math_tanh(y3)
    return y4


def _expected_backward(x: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
    """Reference chain rule through all 5 ops, computed lane-wise."""
    # Forward to recover every intermediate.
    var y0 = math_tanh(x)
    var y1 = y0 if y0 > Scalar[DT](0) else Scalar[DT](0)
    var ax = y1 if y1 >= Scalar[DT](0) else -y1
    var sgn = Scalar[DT](1) if y1 >= Scalar[DT](0) else Scalar[DT](-1)
    var y2 = sgn * math_log(Scalar[DT](1) + ax)
    var y3 = math_tanh(y2)
    var y4 = math_tanh(y3)

    # Backward: g flows from output back to input.
    var g = go
    # tanh(y3) → tanh': 1 - y4²
    g = g * (Scalar[DT](1) - y4 * y4)
    # tanh(y2) → tanh': 1 - y3²
    g = g * (Scalar[DT](1) - y3 * y3)
    # symlog(y1) → 1 / (1 + |y1|)
    g = g / (Scalar[DT](1) + ax)
    # relu(y0) → 1 if y0 > 0 else 0
    if y0 <= Scalar[DT](0):
        g = Scalar[DT](0)
    # tanh(x) → tanh': 1 - y0²
    g = g * (Scalar[DT](1) - y0 * y0)
    return g


def test_scalar_oracle() raises:
    print("test_scalar_oracle ...")
    comptime DIM = 4
    comptime BATCH = 1

    var chain = AutoFusedUnary[DIM, TanhOp, ReLUOp, SymlogOp, TanhOp, TanhOp]()

    var xs = [
        Scalar[DT](-1.5),
        Scalar[DT](0.25),
        Scalar[DT](2.0),
        Scalar[DT](-0.1),
    ]
    var go_s = [
        Scalar[DT](1.0),
        Scalar[DT](0.5),
        Scalar[DT](-0.7),
        Scalar[DT](1.3),
    ]

    var x_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var y_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for c in range(DIM):
        x_buf[c] = xs[c]
        go_buf[c] = go_s[c]

    var x_tt = TileTensor(x_buf, row_major[BATCH, DIM]())
    var y_tt = TileTensor(y_buf, row_major[BATCH, DIM]())
    var go_tt = TileTensor(go_buf, row_major[BATCH, DIM]())
    var gi_tt = TileTensor(gi_buf, row_major[BATCH, DIM]())

    chain.forward[BATCH](x_tt, output=y_tt)
    chain.vjp[BATCH](go_tt, grad_input=gi_tt)

    for c in range(DIM):
        var got_y = y_buf[c]
        var ref_y = _expected_forward(xs[c])
        var dy = got_y - ref_y
        var ady = dy if dy >= Scalar[DT](0) else -dy
        print("  fwd lane", c, ": got=", got_y, "ref=", ref_y, "|d|=", ady)
        assert_true(
            ady < Scalar[DT](1e-6),
            "forward oracle mismatch at lane " + String(c),
        )

        var got_g = gi_buf[c]
        var ref_g = _expected_backward(xs[c], go_s[c])
        var dg = got_g - ref_g
        var adg = dg if dg >= Scalar[DT](0) else -dg
        print("  vjp lane", c, ": got=", got_g, "ref=", ref_g, "|d|=", adg)
        assert_true(
            adg < Scalar[DT](1e-6),
            "vjp oracle mismatch at lane " + String(c),
        )

    x_buf.free()
    y_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  ok (5-op chain, scalar oracle matches in fwd + vjp)")


# ──────────────────────────────────────────────────────────────────────
# Test 2 — Bit-identity against Sequential[*Elementwise[DIM, OP]].
# Same input, same chain, same RNG → outputs must be exactly equal.
# Runs on a multi-row (BATCH=8) random input to exercise the per-row
# loop and catch any row-stride / cache-index bugs.
# ──────────────────────────────────────────────────────────────────────


def test_bit_identity_vs_sequential() raises:
    print("test_bit_identity_vs_sequential ...")
    comptime DIM = 6
    comptime BATCH = 8
    comptime N_ELEMS = BATCH * DIM

    seed(42)
    var x_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_ELEMS)
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_ELEMS)
    for k in range(N_ELEMS):
        x_buf[k] = Scalar[DT](random_float64() * 4.0 - 2.0)
        go_buf[k] = Scalar[DT](random_float64() * 2.0 - 1.0)

    # Fused chain.
    var fused = AutoFusedUnary[DIM, TanhOp, ReLUOp, SymlogOp, TanhOp, TanhOp]()
    var y_fused: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_ELEMS)
    var gi_fused: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_ELEMS)
    var x_tt = TileTensor(x_buf, row_major[BATCH, DIM]())
    var y_fused_tt = TileTensor(y_fused, row_major[BATCH, DIM]())
    var go_tt = TileTensor(go_buf, row_major[BATCH, DIM]())
    var gi_fused_tt = TileTensor(gi_fused, row_major[BATCH, DIM]())
    fused.forward[BATCH](x_tt, output=y_fused_tt)
    fused.vjp[BATCH](go_tt, grad_input=gi_fused_tt)

    # Reference: Sequential[Elementwise[DIM, OP], ...].  Each Elementwise
    # leaf is the production unary module; Sequential threads them with
    # per-pair mid slabs.  This is the "expensive but correct" oracle
    # the fused chain must match bit-for-bit.
    comptime RefSeq = Sequential[
        Elementwise[DIM, TanhOp],
        Elementwise[DIM, ReLUOp],
        Elementwise[DIM, SymlogOp],
        Elementwise[DIM, TanhOp],
        Elementwise[DIM, TanhOp],
    ]
    var seq_ref = RefSeq.make[target="cpu", INIT=Kaiming]()

    var y_ref: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_ELEMS)
    var gi_ref: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_ELEMS)
    var y_ref_tt = TileTensor(y_ref, row_major[BATCH, DIM]())
    var gi_ref_tt = TileTensor(gi_ref, row_major[BATCH, DIM]())
    seq_ref.forward["cpu", BATCH](x_tt, output=y_ref_tt)
    seq_ref.vjp["cpu", BATCH](go_tt, gi_ref_tt)

    var max_fwd_diff = Scalar[DT](0.0)
    var max_bwd_diff = Scalar[DT](0.0)
    for k in range(N_ELEMS):
        var df = y_fused[k] - y_ref[k]
        var adf = df if df >= Scalar[DT](0) else -df
        if adf > max_fwd_diff:
            max_fwd_diff = adf
        var db = gi_fused[k] - gi_ref[k]
        var adb = db if db >= Scalar[DT](0) else -db
        if adb > max_bwd_diff:
            max_bwd_diff = adb
    print("  max |y_fused - y_ref| =", max_fwd_diff)
    print("  max |gi_fused - gi_ref| =", max_bwd_diff)
    assert_true(
        max_fwd_diff == Scalar[DT](0),
        "forward must be bit-identical vs Sequential",
    )
    assert_true(
        max_bwd_diff == Scalar[DT](0),
        "vjp must be bit-identical vs Sequential",
    )

    x_buf.free()
    go_buf.free()
    y_fused.free()
    gi_fused.free()
    y_ref.free()
    gi_ref.free()
    print("  ok (BATCH=", BATCH, "DIM=", DIM, "bit-identical fused vs Sequential)")


def main() raises:
    print("=" * 70)
    print("Spike H.1: autograd-lite codegen for ElementOp chains")
    print("=" * 70)
    test_scalar_oracle()
    test_bit_identity_vs_sequential()
    print("=" * 70)
    print("SPIKE PASSED")
    print("=" * 70)
    print(
        "AutoFusedUnary[DIM, *OPS: ElementOp] codegen viable —"
        " forward + vjp synthesised from per-op scalar methods are"
        " bit-identical to the hand-composed Sequential reference."
    )
    print(
        "Next: extend to BinaryElementOp + ReduceOp slots, then add"
        " SIMD + GPU kernel codegen behind the same comptime-for"
        " unroll pattern."
    )
