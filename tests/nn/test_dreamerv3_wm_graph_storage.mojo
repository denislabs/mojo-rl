"""DreamerV3 WM-graph storage gate — the keystone integration test.

Builds `WMLossGraph` (the full single-step WM loss + carry passthrough) on the
storage ComputeGraph and runs forward + vjp. This one graph wires EVERY migrated
DreamerV3 leaf op — ActionSquash / BlockGroupAssemble / GRUGate /
StraightThroughSample (rssm_ops), OneHotKLLoss (onehot_kl), SymlogMSELoss /
TwoHotLoss / BinaryLoss (wm_loss_ops) — plus the storage nets (decoder / reward /
cont heads) and the arity-generic multi-way Concat loss-vector assembly. Confirms
the graph-owns-params model runs end-to-end (make → set_input → forward → vjp),
forward+grad are finite, and CPU/GPU agree.

Run: pixi run -e apple mojo run -I . tests/nn/test_dreamerv3_wm_graph_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.dreamerv3.wm import WMLossGraph


comptime DETER = 8
comptime H = 8
comptime STOCH = 3
comptime CLASSES = 4
comptime SC = STOCH * CLASSES
comptime BLOCKS = 2
comptime ACT = 2
comptime TOKEN = 5
comptime OBS = 6
comptime DEC_U = 8
comptime HU = 8
comptime BINS = 7
comptime B = 4
comptime OUTW = 5 + DETER + SC

comptime GRAPH = WMLossGraph[
    DETER, H, STOCH, CLASSES, BLOCKS, ACT, TOKEN, OBS, DEC_U, HU, BINS,
]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _fill(
    mut deter: Tensor, mut stoch: Tensor, mut action: Tensor, mut tokens: Tensor,
    mut rtgt: Tensor, mut rew_t: Tensor, mut con_t: Tensor, mut go: Tensor,
) raises:
    for i in range(B * DETER):
        deter.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    for i in range(B * SC):
        stoch.data[i] = Scalar[DT]((i % 5) - 2) * 0.15
    for i in range(B * ACT):
        action.data[i] = Scalar[DT]((i % 3) - 1) * 0.4
    for i in range(B * TOKEN):
        tokens.data[i] = Scalar[DT]((i % 5) - 2) * 0.2
    for i in range(B * OBS):
        rtgt.data[i] = Scalar[DT]((i % 7) - 3) * 0.1
    for i in range(B):
        rew_t.data[i] = Scalar[DT]((i % 3) - 1) * 0.5
        con_t.data[i] = Scalar[DT](1.0) if (i % 2 == 0) else Scalar[DT](0.0)
    for i in range(B * OUTW):
        go.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.1


def _run[
    target: StaticString
](ctx: Optional[DeviceContext], mut out_rec: List[Scalar[DT]]) raises -> Bool:
    var g = GRAPH.make[target, Deterministic](ctx)
    var deter = Tensor.alloc(B * DETER)
    var stoch = Tensor.alloc(B * SC)
    var action = Tensor.alloc(B * ACT)
    var tokens = Tensor.alloc(B * TOKEN)
    var rtgt = Tensor.alloc(B * OBS)
    var rew_t = Tensor.alloc(B)
    var con_t = Tensor.alloc(B)
    var go = Tensor.alloc(B * OUTW)
    _fill(deter, stoch, action, tokens, rtgt, rew_t, con_t, go)
    var out = Tensor.alloc(B * OUTW)
    comptime if target == "gpu":
        var c = ctx.value()
        deter.upload(c); stoch.upload(c); action.upload(c); tokens.upload(c)
        rtgt.upload(c); rew_t.upload(c); con_t.upload(c); go.upload(c)

    g.set_input["deter", B](deter, ctx)
    g.set_input["stoch", B](stoch, ctx)
    g.set_input["action", B](action, ctx)
    g.set_input["tokens", B](tokens, ctx)
    g.set_input["recon_target", B](rtgt, ctx)
    g.set_input["rew_target", B](rew_t, ctx)
    g.set_input["con_target", B](con_t, ctx)

    g.forward[B, target](out, ctx)
    g.vjp[B, target](go, ctx)

    comptime if target == "gpu":
        out.download(ctx.value())
        g.grad_input["deter"]().download(ctx.value())

    var ok = True
    for i in range(B * OUTW):
        if not (out.data[i] == out.data[i]):  # NaN
            ok = False
        out_rec.append(out.data[i])
    # grad to "deter" input must be finite (BPTT carry path).
    ref gd = g.grad_input["deter"]()
    for i in range(B * DETER):
        if not (gd.data[i] == gd.data[i]):
            ok = False
    return ok


def main() raises:
    print("DreamerV3 WMLossGraph storage gate (full WM loss + BPTT carry)")
    var c = DeviceContext()
    var oc = List[Scalar[DT]]()
    var og = List[Scalar[DT]]()
    var rc = _run["cpu"](None, oc)
    print("  CPU forward+vjp finite:", "OK" if rc else "FAIL")
    var rg = _run["gpu"](Optional(c), og)
    print("  GPU forward+vjp finite:", "OK" if rg else "FAIL")
    var maxd = Scalar[DT](0)
    for i in range(len(oc)):
        var d = _abs(oc[i] - og[i])
        if d > maxd:
            maxd = d
    var parity = maxd < Scalar[DT](2e-3)
    print("  CPU/GPU output parity (max Δ", maxd, "):", "OK" if parity else "FAIL")
    assert_true(rc and rg and parity, "DreamerV3 WMLossGraph storage")
    print("DREAMERV3 WM GRAPH OK")
