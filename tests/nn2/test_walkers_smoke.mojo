"""Smoke test: for_each_param_auto walks Param-typed fields of a struct."""

from layout import TileTensor
from std.gpu.memory import AddressSpace

from mojo_rl.nn2.core.param import Param
from mojo_rl.nn2.core.param_visitor import ParamVisitor
from mojo_rl.nn2.core.walkers import for_each_param_auto, zero_grad_auto
from mojo_rl.nn2.constants import DT


@fieldwise_init
struct _RecordVisitor(ParamVisitor):
    var names_ptr: UnsafePointer[List[String], MutAnyOrigin]
    var sizes_ptr: UnsafePointer[List[Int],   MutAnyOrigin]
    var decays_ptr: UnsafePointer[List[Bool], MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        self.names_ptr[].append(name)
        self.sizes_ptr[].append(n_elems)
        self.decays_ptr[].append(apply_decay)


# Mimics a Linear[4, 3] leaf — two Params + some non-Param fields that
# the walker should skip.
struct _MiniLinear(Movable & ImplicitlyDestructible):
    var weight: Param["weight", True,  12]
    var bias:   Param["bias",   False,  3]
    var cache:  List[Scalar[DT]]   # NOT a Param — must be skipped
    var tag:    Int8               # NOT a Param

    def __init__(out self):
        self.weight = Param["weight", True,  12]()
        self.bias   = Param["bias",   False,  3]()
        self.cache  = List[Scalar[DT]]()
        self.tag    = Int8(0)

    @staticmethod
    def make() raises -> Self:
        var l = Self()
        l.weight = Param["weight", True,  12].make_cpu()
        l.bias   = Param["bias",   False,  3].make_cpu()
        l.tag = Int8(1)
        return l^


def main() raises:
    var lin = _MiniLinear.make()

    # Seed grads so we can prove zero_grad_auto runs.
    var gw_ptr = lin.weight.grad_unsafe_ptr_cpu()
    var gb_ptr = lin.bias.grad_unsafe_ptr_cpu()
    for k in range(12):
        gw_ptr[k] = Scalar[DT](42.0)
    for k in range(3):
        gb_ptr[k] = Scalar[DT](42.0)

    var names  = List[String]()
    var sizes  = List[Int]()
    var decays = List[Bool]()
    var v = _RecordVisitor(
        names_ptr=UnsafePointer(to=names),
        sizes_ptr=UnsafePointer(to=sizes),
        decays_ptr=UnsafePointer(to=decays),
    )

    for_each_param_auto[_MiniLinear, _RecordVisitor, target="cpu"](
        lin, String("layer0"), v,
    )

    print("Auto-discovered params:")
    for i in range(len(names)):
        print(
            "  name=", names[i],
            "  n_elems=", sizes[i],
            "  apply_decay=", decays[i],
        )

    var ok_walk = (
        len(names) == 2
        and names[0] == "layer0.weight"  and sizes[0] == 12 and decays[0]
        and names[1] == "layer0.bias"    and sizes[1] == 3  and not decays[1]
    )

    zero_grad_auto[_MiniLinear, target="cpu"](lin)
    var ok_zero = True
    for k in range(12):
        if gw_ptr[k] != Scalar[DT](0.0):
            ok_zero = False
    for k in range(3):
        if gb_ptr[k] != Scalar[DT](0.0):
            ok_zero = False

    if ok_walk and ok_zero:
        print()
        print("PASS — for_each_param_auto + zero_grad_auto work.")
    else:
        print("FAIL — ok_walk=", ok_walk, " ok_zero=", ok_zero)
        raise Error("walkers smoke failed")
