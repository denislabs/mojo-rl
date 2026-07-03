"""DreamerV3 trainer save_state/load_state round-trip gate (v3 binary).

Regression for the size200m checkpoint corruption: the v2 TEXT format was
~5 GB per save and silently truncated at the 2 GiB single-write(2) cap.
save_state now writes v3 binary; this gate proves the full-envelope
round-trip: train a tiny CPU trainer, save, load into a FRESH trainer, and
assert bit-equal params across all eight modules — plus that the loaded
trainer still trains (imagine mirror re-synced by load_state).

Run: pixi run mojo run -I . tests/nn/test_dreamerv3_ckpt_roundtrip.mojo
"""

from std.math import isfinite
from std.random import random_float64
from std.memory import alloc
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer


comptime OBS = 3
comptime ACT = 1
comptime DETER = 8
comptime H = 8
comptime STOCH = 3
comptime CLASSES = 4
comptime BLOCKS = 2
comptime TOKEN = 6
comptime DEC_U = 8
comptime HU = 8
comptime VU = 8
comptime PU = 8
comptime BINS = 7
comptime B = 4
comptime T = 3
comptime T_IMAG = 3
comptime CAP = 2000

comptime TrainerT = DreamerV3Trainer[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU, PU,
    BINS, B, T, T_IMAG, CAP, False,
]

comptime PATH = "/tmp/dreamerv3_ckpt_roundtrip_v3.ckpt"


struct _Capture(ParamVisitor):
    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            self.vals.append(param.data[i])


def _capture_all(mut tr: TrainerT) raises -> List[Scalar[DT]]:
    var c = _Capture()
    tr.enc.for_each_param["cpu"](c, None)
    tr.core.for_each_param["cpu"](c, None)
    tr.dec.for_each_param["cpu"](c, None)
    tr.rew.for_each_param["cpu"](c, None)
    tr.con.for_each_param["cpu"](c, None)
    tr.value.for_each_param["cpu"](c, None)
    tr.slowvalue.for_each_param["cpu"](c, None)
    tr.policy.for_each_param["cpu"](c, None)
    var out = c.vals.copy()
    return out^


def main() raises:
    print("DreamerV3 save_state/load_state round-trip (v3 binary)")
    var tr = TrainerT.make(learning_starts=8)
    var obs = alloc[Scalar[DT]](OBS)
    var act = alloc[Scalar[DT]](ACT)
    for s in range(64):
        for i in range(OBS):
            obs[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        for j in range(ACT):
            act[j] = Scalar[DT](random_float64() * 2.0 - 1.0)
        tr.record(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](obs),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](act),
            Scalar[DT](random_float64() - 0.5),
            Scalar[DT](1.0) if (s % 17 == 16) else Scalar[DT](0.0),
        )
    for _ in range(3):
        _ = tr.train_step()

    tr.save_state(String(PATH))
    var fresh = TrainerT.make(learning_starts=8)
    fresh.load_state(String(PATH))

    var va = _capture_all(tr)
    var vb = _capture_all(fresh)
    if len(va) != len(vb) or len(va) == 0:
        raise Error("param capture length mismatch")
    for i in range(len(va)):
        if va[i] != vb[i]:  # v3 is raw bytes → BIT-exact, not approx
            raise Error("param mismatch at flat index " + String(i))
    print("  params bit-equal across all 8 modules:", len(va), "values")

    # The loaded trainer must still train (imagine mirror synced on load).
    for s in range(64):
        for i in range(OBS):
            obs[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        for j in range(ACT):
            act[j] = Scalar[DT](random_float64() * 2.0 - 1.0)
        fresh.record(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](obs),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](act),
            Scalar[DT](random_float64() - 0.5),
            Scalar[DT](0.0),
        )
    if not fresh.train_step():
        raise Error("loaded trainer failed to train")
    if not isfinite(Float64(fresh.last_wm_loss())):
        raise Error("loaded trainer wm_loss not finite")
    print("  loaded trainer trains, wm_loss finite")

    obs.free()
    act.free()
    print("DREAMERV3 CKPT ROUNDTRIP OK")
