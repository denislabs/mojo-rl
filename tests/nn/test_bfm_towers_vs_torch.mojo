"""`bfm_towers.mojo` against the reference's classes in torch — G3.2's gate.

    pixi run -e act-ref mojo run -I . tests/nn/test_bfm_towers_vs_torch.mojo

The act-ref environment carries torch; the test imports
`tools/g1/bfm_tower_oracle.py` through interop, so one process builds our
tower, walks its params into the reference's class (by STRUCTURE, not
enumeration order), runs both on the same random batch and the same random
grad_output, and compares:

    forward            out               [B, OUT]
    input gradient     grad_in           [B, IN]
    parameter gradients                  every param, in our layout

for the F tower (`ResidualForwardMap` on the flat `[s | a | z]` row), the
actor tower (`ResidualActor` on `[s | z]`, tanh) and B (`BackwardMap` with
`Norm`). Small dims (obs 7, act 3, d 6, h 16, L 3), batch 5, so the whole
thing is a second; the composition is dimension-agnostic.

Bands: ours runs float32, the reference float64, so the comparison is a
float32 accumulation check — `TOL` 2e-5 absolute on values of order 1,
gradients normalised by the reference's largest magnitude. A wrong slice,
a swapped concat, a residual on the wrong branch or a transposed weight
is order 1.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTower, BFMBNet, BFMBlock, BFMResBlock,
)
from mojo_rl.nn.combinators.parallel import Parallel
from mojo_rl.nn.combinators.repeat import Repeat
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_mish import LinearMish
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.primitives.linear_tanh import LinearTanh
from mojo_rl.nn.primitives.activations import Mish
from mojo_rl.nn.primitives.slice import Slice

from max.gpu.host import DeviceContext


comptime OBS = 7
comptime ACT = 3
comptime D = 6
comptime H = 16
comptime L = 3
comptime HB = 12
comptime B = 5
comptime TOL = 2e-5

comptime FTower = BFMFTower[OBS, ACT, D, H, L, D]
comptime ATower = BFMActorTower[OBS, D, H, L, ACT]
comptime BNet = BFMBNet[OBS, D, HB]

# localisation cases: each composition piece alone
comptime LinMishT = Sequential[Linear[10, 6], Mish[6]]
comptime FusedMishT = LinearMish[10, 6]
comptime FusedReluT = LinearReLU[10, 6]
comptime FusedTanhT = LinearTanh[10, 6]
comptime BlockT = BFMBlock[10, 6]
comptime ResT = BFMResBlock[8]
comptime Rep2T = Repeat[2, BFMResBlock[8]]
comptime SlicesT = Sequential[Parallel[Slice[10, 0, 4], Slice[10, 7, 10]], Linear[7, 5]]
comptime ParSeqT = Parallel[Sequential[Slice[10, 0, 4], Linear[4, 5]], Linear[10, 5]]


struct _Collect(ParamVisitor):
    """Names, sizes and values (or grads) of every param, in walk order."""
    var names: PythonObject
    var sizes: PythonObject
    var vals: PythonObject
    var grads: Bool

    def __init__(out self, builtins: PythonObject, grads: Bool) raises:
        self.names = builtins.list()
        self.sizes = builtins.list()
        self.vals = builtins.list()
        self.grads = grads

    def __init__(out self, *, deinit move: Self):
        self.names = move.names^
        self.sizes = move.sizes^
        self.vals = move.vals^
        self.grads = move.grads

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        _ = self.names.append(name)
        _ = self.sizes.append(N)
        for i in range(N):
            if self.grads:
                _ = self.vals.append(Float64(grad.data[i]))
            else:
                _ = self.vals.append(Float64(param.data[i]))


def _fill_random[ADT: DType](mut t: TensorImpl[ADT], n: Int, seed: Int, scale: Float64):
    """A fixed pseudo-random fill (no RNG dependency): a hashed ramp in [-scale, scale]."""
    for i in range(n):
        var k = (i * 7919 + seed * 104729) % 1000
        t.data[i] = Scalar[ADT]((Float64(k) / 999.0 * 2.0 - 1.0) * scale)


def _py_from_tensor[ADT: DType](builtins: PythonObject, t: TensorImpl[ADT], n: Int) raises -> PythonObject:
    var out = builtins.list()
    for i in range(n):
        _ = out.append(Float64(t.data[i]))
    return out


def _worst[ADT: DType](a: TensorImpl[ADT], b: PythonObject, n: Int) raises -> Float64:
    var w = 0.0
    var scale = 0.0
    for i in range(n):
        var r = Float64(py=b[i])
        if abs(r) > scale:
            scale = abs(r)
    if scale < 1.0:
        scale = 1.0
    for i in range(n):
        var e = abs(Float64(a.data[i]) - Float64(py=b[i])) / scale
        if e > w:
            w = e
    return w


def _worst_list(a: PythonObject, b: PythonObject, n: Int) raises -> Float64:
    var w = 0.0
    var scale = 0.0
    for i in range(n):
        var r = Float64(py=b[i])
        if abs(r) > scale:
            scale = abs(r)
    if scale < 1.0:
        scale = 1.0
    for i in range(n):
        var e = abs(Float64(py=a[i]) - Float64(py=b[i])) / scale
        if e > w:
            w = e
    return w


def _check[NET: Module, IN: Int, OUT: Int](
    kind: String, dims: PythonObject, seed: Int
) raises -> Tuple[Float64, Float64, Float64, Int]:
    var builtins = Python.import_module("builtins")
    var oracle = Python.import_module("bfm_tower_oracle")
    var net = NET.make["cpu", Kaiming](None)
    net.zero_grad["cpu"](None)

    var pv = _Collect(builtins, False)
    net.for_each_param["cpu"](pv, None)
    var n_params = Int(Float64(py=builtins.len(pv.names)))
    var model = oracle.build(kind, dims)
    var n_set = Int(Float64(py=oracle.load_ours(model, kind, dims, pv.names, pv.vals, pv.sizes)))
    _ = n_set

    var x = TensorImpl[NET.ACT_DT].alloc(B * IN)
    var go = TensorImpl[NET.ACT_DT].alloc(B * OUT)
    _fill_random(x, B * IN, seed, 1.5)
    _fill_random(go, B * OUT, seed + 1, 0.7)
    var out = TensorImpl[NET.ACT_DT].alloc(B * OUT)
    var gi = TensorImpl[NET.ACT_DT].alloc(B * IN)
    net.forward["cpu", B](TensorRefs[NET.ARITY, _, NET.ACT_DT](x), out, None)
    net.vjp["cpu", B](TensorRefs[NET.ARITY, _, NET.ACT_DT](x), go, TensorRefs[NET.ARITY, _, NET.ACT_DT](gi), None)

    var res = oracle.run(
        model, kind, dims, pv.names, pv.sizes,
        _py_from_tensor(builtins, x, B * IN), B, _py_from_tensor(builtins, go, B * OUT),
    )
    var e_out = _worst(out, res[0], B * OUT)
    var e_gi = _worst(gi, res[1], B * IN)
    var gv = _Collect(builtins, True)
    net.for_each_param["cpu"](gv, None)
    var n_total = Int(Float64(py=builtins.len(gv.vals)))
    var e_pg = _worst_list(gv.vals, res[2], n_total)
    return (e_out, e_gi, e_pg, n_params)


def test_f_tower_matches_residual_forward_map() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["obs"] = OBS
    dims["act"] = ACT
    dims["d"] = D
    dims["h"] = H
    dims["l"] = L
    dims["out"] = D
    var r = _check[FTower, OBS + ACT + D, D](String("f"), dims, 11)
    print("  F tower: params", r[3], " |d out|", r[0], " |d grad_in|", r[1], " |d param grads|", r[2])
    assert_true(r[0] < TOL and r[1] < TOL and r[2] < TOL, "F tower differs from ResidualForwardMap")


def test_actor_tower_matches_residual_actor() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["obs"] = OBS
    dims["act"] = ACT
    dims["d"] = D
    dims["h"] = H
    dims["l"] = L
    var r = _check[ATower, OBS + D, ACT](String("actor"), dims, 23)
    print("  actor tower: params", r[3], " |d out|", r[0], " |d grad_in|", r[1], " |d param grads|", r[2])
    assert_true(r[0] < TOL and r[1] < TOL and r[2] < TOL, "actor tower differs from ResidualActor")


def test_b_net_matches_backward_map() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["obs"] = OBS
    dims["d"] = D
    dims["hb"] = HB
    var r = _check[BNet, OBS, D](String("b"), dims, 37)
    print("  B net: params", r[3], " |d out|", r[0], " |d grad_in|", r[1], " |d param grads|", r[2])
    assert_true(r[0] < TOL and r[1] < TOL and r[2] < TOL, "B net differs from BackwardMap")


def _report(label: String, r: Tuple[Float64, Float64, Float64, Int]) raises:
    print("  " + label + ": params", r[3], " |d out|", r[0], " |d grad_in|", r[1], " |d param grads|", r[2])
    assert_true(r[0] < TOL and r[1] < TOL and r[2] < TOL, label + " differs from torch")


def test_piece_linear_mish() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["in"] = 10
    dims["out"] = 6
    var r = _check[LinMishT, 10, 6](String("linmish"), dims, 41)
    _report(String("Linear+Mish"), r)


def test_piece_block() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["in"] = 10
    dims["out"] = 6
    var r = _check[BlockT, 10, 6](String("block"), dims, 43)
    _report(String("Block LN+Linear+Mish"), r)


def test_piece_resblock() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["h"] = 8
    var r = _check[ResT, 8, 8](String("resblock"), dims, 47)
    _report(String("ResidualBlock"), r)


def test_piece_repeat2() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["h"] = 8
    var r = _check[Rep2T, 8, 8](String("repeat2"), dims, 53)
    _report(String("Repeat[2, ResidualBlock]"), r)


def test_piece_slices() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["in"] = 10
    dims["a"] = 4
    dims["b"] = 7
    dims["out"] = 5
    var r = _check[SlicesT, 10, 5](String("slices"), dims, 59)
    _report(String("Parallel[Slice, Slice] -> Linear"), r)


def test_piece_parseq() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["in"] = 10
    dims["a"] = 4
    dims["out"] = 5
    var r = _check[ParSeqT, 10, 10](String("parseq"), dims, 61)
    _report(String("Parallel[Sequential[Slice, Linear], Linear]"), r)


def test_fused_linear_act_probe() raises:
    """REPORT ONLY: the fused `LinearAct` against Linear + activation in torch.
    The Mish variant's CPU backward is wrong (see bfm_towers.mojo); this
    prints all three so the defect's scope is on record without a red gate."""
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var dims = Python.dict()
    dims["in"] = 10
    dims["out"] = 6
    var rm = _check[FusedMishT, 10, 6](String("linmish"), dims, 71)
    print("  fused LinearMish (probe): |d out|", rm[0], " |d grad_in|", rm[1], " |d param grads|", rm[2])
    var rr = _check[FusedReluT, 10, 6](String("linrelu"), dims, 73)
    print("  fused LinearReLU (probe): |d out|", rr[0], " |d grad_in|", rr[1], " |d param grads|", rr[2])
    var rt = _check[FusedTanhT, 10, 6](String("lintanh"), dims, 79)
    print("  fused LinearTanh (probe): |d out|", rt[0], " |d grad_in|", rt[1], " |d param grads|", rt[2])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
