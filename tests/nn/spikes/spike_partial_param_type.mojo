"""Spike DR.8 — `TileTensor[named=..., ...]` at FUNCTION PARAMETER position.

DR.7 tested `TileTensor[DT, ...]` at storage positions (variable, List,
Variant) — all rejected as non-concrete. But the user pointed out MAX
kernels `linalg/matmul/__init__.mojo` uses this very pattern at FUNCTION
PARAMETER positions:

    def matmul[...](
        c: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
        a: TileTensor[address_space=AddressSpace.GENERIC, ...],
        b: TileTensor[address_space=AddressSpace.GENERIC, ...],
        ...
    ) raises:

The function takes three TileTensors with partial specification — named
params constrained, others inferred per call site. If this pattern works,
multi-input Modules can use it instead of per-arg `L: TensorLayout, O:
MutOrigin` generic boilerplate.

Probes:

A. Single param with `[address_space=..., ...]` — does it compile + accept
   any concrete TileTensor at the call site?

B. Three params (matmul shape) with partial specs — does the function
   accept three TileTensors of distinct layouts + origins?

C. Inside the function body — can we index `input[b, j]` and access
   `.dim[k]()`, or does the partial type lose the methods?

D. As a TRAIT method parameter (the actual question for multi-input
   Modules). Function params and trait method params may behave
   differently in Mojo nightly.
"""

from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT


# ──────────────────────────────────────────────────────────────────────
# Probe A — single param, partial spec at function position.
# ──────────────────────────────────────────────────────────────────────


def probe_a_single(
    input: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, ...],
) raises:
    comptime assert input.flat_rank == 2, "input rank-2"
    print("  Probe A inner: input[0, 0] =", input[0, 0])


def smoke_a() raises:
    print("--- Probe A: single param `TileTensor[address_space=..., ...]` ---")
    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](7.0))
    var ta = TileTensor(a.unsafe_ptr(), row_major[2, 2]())
    probe_a_single(ta)


# ──────────────────────────────────────────────────────────────────────
# Probe B — three params with partial specs (matmul-style).
# ──────────────────────────────────────────────────────────────────────


def probe_b_three(
    in0: TileTensor[address_space=AddressSpace.GENERIC, ...],
    in1: TileTensor[address_space=AddressSpace.GENERIC, ...],
    in2: TileTensor[address_space=AddressSpace.GENERIC, ...],
    mut output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC, ...
    ],
) raises:
    """Sum three inputs into output. Inputs have distinct origins (per
    source buffer) and possibly distinct layouts."""
    comptime assert in0.flat_rank == 2
    comptime assert in1.flat_rank == 2
    comptime assert in2.flat_rank == 2
    comptime assert output.flat_rank == 2
    # Index using runtime dims since static_shape may not be resolved.
    var rows = Int(in0.dim[0]())
    var cols = Int(in0.dim[1]())
    for b in range(rows):
        for d in range(cols):
            output[b, d] = in0[b, d] + in1[b, d] + in2[b, d]


def smoke_b() raises:
    print("--- Probe B: three params, distinct origins ---")
    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](1.0))
    var b = List[Scalar[DT]](length=4, fill=Scalar[DT](2.0))
    var c = List[Scalar[DT]](length=4, fill=Scalar[DT](3.0))
    var out = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))
    var ta = TileTensor(a.unsafe_ptr(), row_major[2, 2]())
    var tb = TileTensor(b.unsafe_ptr(), row_major[2, 2]())
    var tc = TileTensor(c.unsafe_ptr(), row_major[2, 2]())
    var to = TileTensor(out.unsafe_ptr(), row_major[2, 2]())
    probe_b_three(ta, tb, tc, to)
    print("  out=[", out[0], out[1], out[2], out[3], "]  expected 6.0 ×4")
    var ok = out[0] == 6.0 and out[1] == 6.0 and out[2] == 6.0 and out[3] == 6.0
    if ok:
        print("  Probe B: PASSED — 3-arg partial-spec works, no per-arg L/O")
    else:
        print("  Probe B: FAILED")


# ──────────────────────────────────────────────────────────────────────
# Probe C — distinct layouts (different shapes per input).
# ──────────────────────────────────────────────────────────────────────


def probe_c_diff_layouts(
    in0: TileTensor[address_space=AddressSpace.GENERIC, ...],
    in1: TileTensor[address_space=AddressSpace.GENERIC, ...],
    mut output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC, ...
    ],
) raises:
    """Inputs have DIFFERENT shapes. Concatenate in0 along feature axis
    with in1 into output. This is the SAC `[s | action]` case."""
    comptime assert in0.flat_rank == 2
    comptime assert in1.flat_rank == 2
    comptime assert output.flat_rank == 2
    var rows = Int(in0.dim[0]())
    var c0 = Int(in0.dim[1]())
    var c1 = Int(in1.dim[1]())
    for b in range(rows):
        for d in range(c0):
            output[b, d] = in0[b, d]
        for d in range(c1):
            output[b, c0 + d] = in1[b, d]


def smoke_c() raises:
    print("--- Probe C: distinct layouts per input (concat use case) ---")
    var a = List[Scalar[DT]](length=6, fill=Scalar[DT](1.0))    # [2, 3]
    var b = List[Scalar[DT]](length=4, fill=Scalar[DT](2.0))    # [2, 2]
    var out = List[Scalar[DT]](length=10, fill=Scalar[DT](0.0)) # [2, 5]
    var ta = TileTensor(a.unsafe_ptr(), row_major[2, 3]())
    var tb = TileTensor(b.unsafe_ptr(), row_major[2, 2]())
    var to = TileTensor(out.unsafe_ptr(), row_major[2, 5]())
    probe_c_diff_layouts(ta, tb, to)
    print("  out row 0 =[", out[0], out[1], out[2], out[3], out[4], "]  expected 1 1 1 2 2")
    print("  out row 1 =[", out[5], out[6], out[7], out[8], out[9], "]  expected 1 1 1 2 2")
    var ok = (
        out[0] == 1.0 and out[1] == 1.0 and out[2] == 1.0
        and out[3] == 2.0 and out[4] == 2.0
        and out[5] == 1.0 and out[6] == 1.0 and out[7] == 1.0
        and out[8] == 2.0 and out[9] == 2.0
    )
    if ok:
        print("  Probe C: PASSED — distinct-layout multi-input works")
    else:
        print("  Probe C: FAILED")


# ──────────────────────────────────────────────────────────────────────
# Probe D — as a TRAIT method parameter.
# ──────────────────────────────────────────────────────────────────────


trait BinaryOp(Defaultable & Movable & ImplicitlyDestructible):
    def forward(
        mut self,
        in0: TileTensor[address_space=AddressSpace.GENERIC, ...],
        in1: TileTensor[address_space=AddressSpace.GENERIC, ...],
        mut output: TileTensor[
            mut=True, address_space=AddressSpace.GENERIC, ...
        ],
    ) raises:
        ...


@fieldwise_init
struct SubOp(BinaryOp):
    def forward(
        mut self,
        in0: TileTensor[address_space=AddressSpace.GENERIC, ...],
        in1: TileTensor[address_space=AddressSpace.GENERIC, ...],
        mut output: TileTensor[
            mut=True, address_space=AddressSpace.GENERIC, ...
        ],
    ) raises:
        comptime assert in0.flat_rank == 2
        comptime assert in1.flat_rank == 2
        comptime assert output.flat_rank == 2
        var rows = Int(in0.dim[0]())
        var cols = Int(in0.dim[1]())
        for b in range(rows):
            for d in range(cols):
                output[b, d] = in0[b, d] - in1[b, d]


def smoke_d() raises:
    print("--- Probe D: BinaryOp trait with partial-spec method params ---")
    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](10.0))
    var b = List[Scalar[DT]](length=4, fill=Scalar[DT](3.0))
    var out = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))
    var ta = TileTensor(a.unsafe_ptr(), row_major[2, 2]())
    var tb = TileTensor(b.unsafe_ptr(), row_major[2, 2]())
    var to = TileTensor(out.unsafe_ptr(), row_major[2, 2]())
    var op = SubOp()
    op.forward(ta, tb, to)
    print("  out=[", out[0], out[1], out[2], out[3], "]  expected 7.0 ×4")
    var ok = out[0] == 7.0 and out[1] == 7.0 and out[2] == 7.0 and out[3] == 7.0
    if ok:
        print("  Probe D: PASSED — partial-spec works on trait methods")
    else:
        print("  Probe D: FAILED")


def main() raises:
    print("=" * 70)
    print("DR.8 — `...` at FUNCTION PARAMETER position (MAX kernels style)")
    print("=" * 70)
    smoke_a()
    smoke_b()
    smoke_c()
    smoke_d()
    print("=" * 70)
