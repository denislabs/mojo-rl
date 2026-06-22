"""Integration spike: TD-MPC2 storage WM graph forward+vjp (CPU + GPU).

Constructs the migrated `TDMPC2WMGraph` (storage ComputeGraph, arity-4
`Concat[1,1,1,1]` loss vector) plus its 8 external nets (Dynamics, Reward,
5×QNet, Termination — all storage Sequential trunks), seeds the 6 input slots,
runs forward + vjp, and checks the output [B, 8+LATENT] and the carry-in grad
`grad_input["z"]` are finite. This is the make-or-break that the migrated
nets + losses + arity-generic graph compose end to end.

Run:
  pixi run mojo run -I . tests/nn/spike_tdmpc2_wm_graph_storage.mojo
  pixi run -e apple mojo run -I . tests/nn/spike_tdmpc2_wm_graph_storage.mojo
"""

from std.math import isnan, isinf
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.deep_agents.tdmpc2.nets import (
    TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Termination,
)
from mojo_rl.deep_agents.tdmpc2.wm_graph import TDMPC2WMGraph


comptime LATENT = 8
comptime ACT = 2
comptime MLP = 16
comptime BINS = 5
comptime SN = 4
comptime VMIN = -5
comptime VMAX = 5
comptime B = 3
comptime OUT = 8 + LATENT

comptime Graph = TDMPC2WMGraph[LATENT, ACT, MLP, BINS, SN, VMIN, VMAX]
comptime Dyn = TDMPC2Dynamics[LATENT, ACT, MLP, SN]
comptime Rew = TDMPC2Reward[LATENT, ACT, MLP, BINS]
comptime QNet = TDMPC2QNet[LATENT, ACT, MLP, BINS]
comptime Term = TDMPC2Termination[LATENT, ACT, MLP]


def _seed(mut g: Graph, ctx: Optional[DeviceContext]) raises:
    var z = Tensor.alloc(B * LATENT)
    var a = Tensor.alloc(B * ACT)
    var zn = Tensor.alloc(B * LATENT)
    var r = Tensor.alloc(B)
    var td = Tensor.alloc(B)
    var dn = Tensor.alloc(B)
    for i in range(B * LATENT):
        z.data[i] = Scalar[DT]((i % 7) - 3) * 0.1
        zn.data[i] = Scalar[DT]((i % 5) - 2) * 0.1
    for i in range(B * ACT):
        a.data[i] = Scalar[DT]((i % 3) - 1) * 0.2
    for b in range(B):
        r.data[b] = Scalar[DT]((b % 3) - 1) * 0.5
        td.data[b] = Scalar[DT]((b % 4) - 1) * 0.7
        dn.data[b] = Scalar[DT](0.0)
    if ctx:
        var c = ctx.value()
        z.upload(c); a.upload(c); zn.upload(c); r.upload(c); td.upload(c); dn.upload(c)
    g.set_input["z", B](z, ctx)
    g.set_input["a", B](a, ctx)
    g.set_input["z_enc_next", B](zn, ctx)
    g.set_input["r", B](r, ctx)
    g.set_input["td", B](td, ctx)
    g.set_input["done", B](dn, ctx)


def _run[target: StaticString](ctx: Optional[DeviceContext]) raises:
    var g = Graph.make[target, Kaiming](ctx)
    var dyn = Dyn.make[target, Kaiming](ctx)
    var rew = Rew.make[target, Kaiming](ctx)
    var q0 = QNet.make[target, Kaiming](ctx)
    var q1 = QNet.make[target, Kaiming](ctx)
    var q2 = QNet.make[target, Kaiming](ctx)
    var q3 = QNet.make[target, Kaiming](ctx)
    var q4 = QNet.make[target, Kaiming](ctx)
    var term = Term.make[target, Kaiming](ctx)

    _seed(g, ctx)
    var out = Tensor.alloc(B * OUT)
    if ctx:
        out.upload(ctx.value())
    g.forward[B, target](out, ctx, dyn, rew, q0, q1, q2, q3, q4, term)

    var seed = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        seed.data[i] = Scalar[DT](1.0)
    if ctx:
        seed.upload(ctx.value())
    g.vjp[B, target](seed, ctx, dyn, rew, q0, q1, q2, q3, q4, term)

    if ctx:
        out.download(ctx.value())
        g.grad_input["z"]().download(ctx.value())
    var n_bad = 0
    for i in range(B * OUT):
        if isnan(Float64(out.data[i])) or isinf(Float64(out.data[i])):
            n_bad += 1
    ref gz = g.grad_input["z"]()
    var n_bad_g = 0
    for i in range(B * LATENT):
        if isnan(Float64(gz.data[i])) or isinf(Float64(gz.data[i])):
            n_bad_g += 1
    print("  out[0..3]:", out.data[0], out.data[1], out.data[2], out.data[3])
    print("  grad_z[0..3]:", gz.data[0], gz.data[1], gz.data[2], gz.data[3])
    assert_true(n_bad == 0, "output finite")
    assert_true(n_bad_g == 0, "grad_z finite")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("TD-MPC2 storage WM graph integration spike")
    print("=" * 60)
    print("CPU ...")
    _run["cpu"](None)
    print("GPU ...")
    var c = DeviceContext()
    _run["gpu"](Optional(c))
    print("ALL PASSED")
