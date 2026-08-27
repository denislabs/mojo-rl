"""Probe — is `Self.IN_DIMS[k]` comptime-foldable for comptime k?

Two questions:
  Q1. Heterogeneous InlineArray construction at comptime (per-index assign).
  Q2. Whether IN_DIMS[k] for comptime k produces a comptime value usable
      in row_major[BATCH, dim]() construction. If yes, the variadic-Module
      approach can replace the IN/IN1/IN2/IN3_DIM ladder entirely.
"""

from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major
from mojo_rl.nn.constants import DT


# ──────────────────────────────────────────────────────────────────
# Q1 — heterogeneous IN_DIMS construction at comptime
# ──────────────────────────────────────────────────────────────────


struct HeteroDimsProbe[ACT_: Int](Movable & Deinitable):
    comptime ARITY = 4
    comptime IN_DIMS: InlineArray[Int, 4] = Self._build()

    @staticmethod
    def _build() -> InlineArray[Int, 4]:
        var d = InlineArray[Int, 4](fill=0)
        d[0] = 2 * Self.ACT_
        d[1] = Self.ACT_
        d[2] = 1
        d[3] = 1
        return d

    def __init__(out self):
        pass


def test_q1_heterogeneous() raises:
    print("Q1: heterogeneous IN_DIMS construction ...")
    comptime probe = HeteroDimsProbe[3]()
    print("  IN_DIMS =")
    print("    [0] =", probe.IN_DIMS[0], "(want 6 = 2*ACT)")
    print("    [1] =", probe.IN_DIMS[1], "(want 3 = ACT)")
    print("    [2] =", probe.IN_DIMS[2], "(want 1)")
    print("    [3] =", probe.IN_DIMS[3], "(want 1)")
    print("  Q1: PASS" if probe.IN_DIMS[0] == 6 else "  Q1: FAIL")


# ──────────────────────────────────────────────────────────────────
# Q2 — does Self.IN_DIMS[k] resolve at comptime for use in Layout?
# ──────────────────────────────────────────────────────────────────


struct ComptimeIndexProbe[*DIMS: Int](Movable & Deinitable):
    comptime ARITY = Self.DIMS.size

    def __init__(out self):
        pass

    # Test A: per-index access via DIMS variadic (known-comptime).
    @staticmethod
    def test_via_variadic[BATCH: Int]() raises:
        comptime for k in range(Self.ARITY):
            comptime dim_k = Self.DIMS[k]
            var stub_ptr = Pointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=Int(0))
            var t = TileTensor(stub_ptr, row_major[BATCH, dim_k]())
            print("  variadic[", k, "] dim =", dim_k)


struct InlineArrayIndexProbe(Movable & Deinitable):
    """Does IN_DIMS[k] for comptime k produce a comptime value?"""
    comptime IN_DIMS: InlineArray[Int, 3] = Self._build()

    @staticmethod
    def _build() -> InlineArray[Int, 3]:
        var d = InlineArray[Int, 3](fill=0)
        d[0] = 5
        d[1] = 7
        d[2] = 11
        return d

    def __init__(out self):
        pass

    # Attempt: use IN_DIMS[k] for comptime k as a Layout dim.
    @staticmethod
    def test_inline_array_index[BATCH: Int]() raises:
        comptime for k in range(3):
            comptime dim_k = Self.IN_DIMS[k]   # ← THE QUESTION
            var stub_ptr = Pointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=Int(0))
            var t = TileTensor(stub_ptr, row_major[BATCH, dim_k]())
            print("  IN_DIMS[", k, "] dim =", dim_k)


def test_q2_comptime_indexing() raises:
    print("Q2: comptime indexing into IN_DIMS ...")
    print(" -- baseline: comptime index into struct variadic *DIMS --")
    ComptimeIndexProbe[5, 7, 11].test_via_variadic[BATCH=8]()
    print(" -- main test: comptime index into InlineArray IN_DIMS --")
    InlineArrayIndexProbe.test_inline_array_index[BATCH=8]()
    print("  Q2: PASS (if both branches printed without error)")


def main() raises:
    print("=" * 70)
    print("InlineArray IN_DIMS comptime-feasibility probe")
    print("=" * 70)
    test_q1_heterogeneous()
    print()
    test_q2_comptime_indexing()
    print("=" * 70)
