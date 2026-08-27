"""DreamerV3 nets storage gate: encoder/decoder/head aliases build + run on the
storage framework (Sequential of storage Symlog/Linear/RMSNorm/Elementwise),
CPU+GPU finite + parity. Confirms the nets.mojo import swap.

Run: pixi run -e apple mojo run -I . tests/nn/test_dreamerv3_nets_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.dreamerv3.nets import (
    DreamerEncoder, DreamerDecoder, DreamerRewardMLP,
)


comptime OBS = 5
comptime U = 16
comptime FEATIN = 12
comptime BINS = 7
comptime B = 4
comptime EncT = DreamerEncoder[OBS, U]
comptime DecT = DreamerDecoder[FEATIN, OBS, U]
comptime RewT = DreamerRewardMLP[FEATIN, U, BINS]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _run[
    target: StaticString
](ctx: Optional[DeviceContext], mut enc_out: List[Scalar[DT]]) raises -> Bool:
    var enc = EncT.make[target, Deterministic](ctx)
    var dec = DecT.make[target, Deterministic](ctx)
    var rew = RewT.make[target, Deterministic](ctx)

    var x = Tensor.alloc(B * OBS)
    for i in range(B * OBS):
        x.data[i] = Scalar[DT]((i % 7) - 3) * 0.25
    var f = Tensor.alloc(B * FEATIN)
    for i in range(B * FEATIN):
        f.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    var eo = Tensor.alloc(B * U)
    var dout = Tensor.alloc(B * OBS)
    var rout = Tensor.alloc(B * BINS)
    comptime if target == "gpu":
        x.upload(ctx.value()); f.upload(ctx.value())

    enc.forward[target, B](TensorRefs[1](x), eo, ctx)
    dec.forward[target, B](TensorRefs[1](f), dout, ctx)
    rew.forward[target, B](TensorRefs[1](f), rout, ctx)
    comptime if target == "gpu":
        eo.download(ctx.value()); dout.download(ctx.value()); rout.download(ctx.value())

    var ok = True
    for i in range(B * U):
        if not (eo.data[i] == eo.data[i]):  # NaN check
            ok = False
        enc_out.append(eo.data[i])
    for i in range(B * OBS):
        if not (dout.data[i] == dout.data[i]):
            ok = False
    for i in range(B * BINS):
        if not (rout.data[i] == rout.data[i]):
            ok = False
    return ok


def main() raises:
    print("DreamerV3 nets storage gate (encoder/decoder/reward head)")
    var c = DeviceContext()
    var oc = List[Scalar[DT]]()
    var og = List[Scalar[DT]]()
    var rc = _run["cpu"](None, oc)
    var rg = _run["gpu"](Optional(c), og)
    print("  forward finite  CPU:", "OK" if rc else "FAIL",
          " GPU:", "OK" if rg else "FAIL")
    var parity = True
    for i in range(len(oc)):
        if _abs(oc[i] - og[i]) > Scalar[DT](1e-4):
            parity = False
    print("  encoder CPU/GPU parity:", "OK" if parity else "FAIL")
    assert_true(rc and rg and parity, "DreamerV3 nets storage")
    print("DREAMERV3 NETS OK")
