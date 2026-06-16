"""Final partial-spec spike: method-level dtype generic + `TileTensor[DTP, ...]`.

The previous probes showed that `TileTensor[dtype=DT, address_space=..., ...]`
as a function param introduces per-arg comptime dtypes that DON'T unify
across params (`in0.dtype` ≠ `in1.dtype` even when both call sites have
the same DT).

This spike tries one more form: a *method-level* dtype generic, passed
into each TileTensor's positional dtype slot:

    def forward[DTP: DType](
        in0: TileTensor[DTP, ...],
        in1: TileTensor[DTP, ...],
        mut output: TileTensor[mut=True, dtype=DTP, ...],
    )

If `DTP` is method-scope (not param-scope), all three TileTensor params
should bind their dtype slot to the same `DTP` and arithmetic should
work across them.
"""

from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT


def binary_sub[DTP: DType, ELEMS: Int](
    in0: TileTensor[DTP, address_space=AddressSpace.GENERIC, element_size=ELEMS, ...],
    in1: TileTensor[DTP, address_space=AddressSpace.GENERIC, element_size=ELEMS, ...],
    mut output: TileTensor[
        mut=True, dtype=DTP, address_space=AddressSpace.GENERIC, element_size=ELEMS, ...
    ],
) raises:
    """in0 and in1 are read-only TileTensors of dtype DTP; output is mut.
    Body should be able to do `output[b, d] = in0[b, d] - in1[b, d]`."""
    comptime assert in0.flat_rank == 2
    comptime assert in1.flat_rank == 2
    comptime assert output.flat_rank == 2
    var rows = Int(in0.dim[0]())
    var cols = Int(in0.dim[1]())
    for b in range(rows):
        for d in range(cols):
            output[b, d] = in0[b, d] - in1[b, d]


def main() raises:
    print("--- Method-level dtype generic + partial spec ---")
    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](10.0))
    var b = List[Scalar[DT]](length=4, fill=Scalar[DT](3.0))
    var out = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))
    var ta = TileTensor(a.unsafe_ptr(), row_major[2, 2]())
    var tb = TileTensor(b.unsafe_ptr(), row_major[2, 2]())
    var to = TileTensor(out.unsafe_ptr(), row_major[2, 2]())
    binary_sub[DT, 1](ta, tb, to)
    print("  out=[", out[0], out[1], out[2], out[3], "]  expected 7.0 ×4")
    var ok = out[0] == 7.0 and out[1] == 7.0 and out[2] == 7.0 and out[3] == 7.0
    if ok:
        print("  PASSED — method-level dtype generic + partial spec works")
    else:
        print("  FAILED")
