"""Smoke test: Param[NAME, DECAY, SIZE] compiles + basic CPU helpers work."""

from layout import TileTensor, row_major

from mojo_rl.nn2.core.param import Param, IsParam
from mojo_rl.nn2.core.param_visitor import ParamVisitor
from mojo_rl.nn2.constants import DT
from std.gpu.memory import AddressSpace


@fieldwise_init
struct _ProbeVisitor(ParamVisitor):
    var names_ptr: UnsafePointer[List[String], MutAnyOrigin]
    var sizes_ptr: UnsafePointer[List[Int],   MutAnyOrigin]
    var decays_ptr: UnsafePointer[List[Bool],  MutAnyOrigin]

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


def main() raises:
    var w = Param["weight", True,  12].make_cpu()
    var b = Param["bias",   False,  3].make_cpu()

    # Pointer accessors.
    var wp = w.value_unsafe_ptr_cpu()
    var bp = b.value_unsafe_ptr_cpu()
    for k in range(12):
        wp[k] = Scalar[DT](k) + Scalar[DT](1.0)
    for k in range(3):
        bp[k] = Scalar[DT](k) + Scalar[DT](100.0)

    var names  = List[String]()
    var sizes  = List[Int]()
    var decays = List[Bool]()
    var v = _ProbeVisitor(
        names_ptr=UnsafePointer(to=names),
        sizes_ptr=UnsafePointer(to=sizes),
        decays_ptr=UnsafePointer(to=decays),
    )

    w.visit_with[_ProbeVisitor, target="cpu"](String("layer.weight"), v)
    b.visit_with[_ProbeVisitor, target="cpu"](String("layer.bias"),   v)

    var ok = (
        len(names) == 2
        and names[0] == "layer.weight" and sizes[0] == 12 and decays[0]
        and names[1] == "layer.bias"   and sizes[1] == 3  and not decays[1]
    )

    # zero_grad_with smoke.
    var gp = w.grad_unsafe_ptr_cpu()
    for k in range(12):
        gp[k] = Scalar[DT](42.0)
    w.zero_grad_with[target="cpu"]()
    var zero_ok = True
    for k in range(12):
        if gp[k] != Scalar[DT](0.0):
            zero_ok = False

    if ok and zero_ok:
        print("PASS — Param[NAME, DECAY, SIZE] basic CPU helpers work.")
    else:
        print("FAIL — visit_with / zero_grad_with did not produce expected state.")
        print("       names=", names, " sizes=", sizes, " decays=", decays)
        print("       zero_ok=", zero_ok)
        raise Error("param_smoke failed")

    # param_name / param_decay
    if not (w.param_name() == StaticString("weight") and w.param_decay()):
        raise Error("Param.param_name / param_decay broken")
    if not (b.param_name() == StaticString("bias") and not b.param_decay()):
        raise Error("Param.param_name / param_decay broken")
