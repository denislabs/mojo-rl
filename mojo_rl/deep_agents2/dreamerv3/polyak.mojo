"""Polyak slow-value sync for a Module (CPU; GPU = PR5c Step 5).

`slowvalue ← (1-rate)·slowvalue + rate·value`. src and dst share the SAME
module type → `for_each_param` visits params in identical order, so an
index-keyed collect-then-mix is exact (no name matching).
"""

from layout import TileTensor
from std.gpu.memory import AddressSpace

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.core.module import Module


@fieldwise_init
struct _PolyakCollect(ParamVisitor):
    var ptrs: UnsafePointer[
        List[UnsafePointer[Scalar[DT], MutAnyOrigin]], MutAnyOrigin
    ]

    def visit(
        mut self, name: String,
        param: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        grad: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        self.ptrs[].append(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        )


@fieldwise_init
struct _PolyakMix(ParamVisitor):
    var ptrs: UnsafePointer[
        List[UnsafePointer[Scalar[DT], MutAnyOrigin]], MutAnyOrigin
    ]
    var rate: Scalar[DT]
    var idx: Int

    def visit(
        mut self, name: String,
        param: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        grad: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        var dp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var sp = self.ptrs[][self.idx]
        var keep = Scalar[DT](1.0) - self.rate
        for k in range(n_elems):
            dp[k] = keep * dp[k] + self.rate * sp[k]
        self.idx += 1


def polyak_module[
    target: StaticString, V: Module
](mut src: V, mut dst: V, rate: Scalar[DT]) raises:
    comptime assert target == "cpu", "polyak_module: GPU lands in PR5c Step 5"
    var sp = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
    var c = _PolyakCollect(ptrs=UnsafePointer(to=sp))
    src.for_each_param[target, _PolyakCollect](String(""), c)
    var m = _PolyakMix(ptrs=UnsafePointer(to=sp), rate=rate, idx=0)
    dst.for_each_param[target, _PolyakMix](String(""), m)
    _ = sp^
