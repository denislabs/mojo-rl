"""DreamerV3 ↔ real CarRacing pixel env wiring smoke (P4 of the pixel arc).

Drives a tiny continuous DreamerV3 agent (CNN enc/dec) on the REAL
CarRacingMB[f32, PIXEL_OBS=True, PIX_RES=96] env (4×96×96 = 36864 obs, 3-D
continuous action), CPU, single env. Records a few frames, runs a couple WM+AC
train steps, and calls select_action — asserting finite/nonzero losses and a
valid action in [-1,1]. This gates the env→agent→WM path end-to-end on the
actual environment (test_dreamerv3_pixel_smoke uses synthetic obs).

Run: pixi run mojo run -I . tests/nn/test_dreamerv3_carracing_pixel_smoke.mojo
"""

from std.math import isfinite
from std.memory import alloc

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)
from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB

comptime C = 4          # 4-frame grayscale stack = conv input channels
comptime IMG = 96
comptime BASE = 4       # tiny conv width for a fast smoke
comptime OBS = C * IMG * IMG   # 36864
comptime ACT = 3        # steer, gas, brake
comptime DETER = 16
comptime H = 16
comptime STOCH = 3
comptime CLASSES = 4
comptime BLOCKS = 2
comptime TOKEN = 16
comptime DEC_U = 8
comptime HU = 8
comptime VU = 8
comptime PU = 8
comptime BINS = 7
comptime B = 4
comptime T = 3
comptime T_IMAG = 3
comptime CAP = 400

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN, SwishOp]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE, SwishOp]

comptime Ag = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU, PU,
    BINS, B, T, T_IMAG, CAP, False, ENC, DEC,
]
comptime Env = CarRacingMB[DT, True, IMG]


def main() raises:
    print("DreamerV3 ↔ CarRacing pixel wiring smoke — OBS =", OBS,
          "(", C, "x", IMG, "x", IMG, "), continuous action")
    var ag = Ag.make(learning_starts=8, action_scale=Scalar[DT](1.0))
    var env = Env()
    if env.obs_dim() != OBS:
        raise Error("env obs_dim != agent OBS")

    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

    var obs = env.reset_obs_list()
    var s: UInt64 = 7
    for step in range(48):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        # warmup: random normalized action in [-1,1]^3
        var a = List[Scalar[DT]]()
        for j in range(ACT):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var u = Float64((s >> 33)) / Float64(UInt64(1) << 31) - 1.0
            actbuf[j] = Scalar[DT](u)
            a.append(Scalar[DT](u))
        var r = env.step_continuous_vec[DT](a)
        ag.record(
            obsbuf, actbuf, r[1],
            Scalar[DT](1.0) if r[2] else Scalar[DT](0.0),
        )
        obs = r[0].copy()
        if r[2]:
            obs = env.reset_obs_list()
            ag.reset_belief()

    if not ag.trainer.can_train():
        raise Error("agent cannot train after 48 recorded frames")

    var saw_wm = False
    for _ in range(3):
        var did = ag.train_step()
        if not did:
            raise Error("train_step returned False after can_train()")
        var wl = ag.last_wm_loss()
        if not isfinite(Float64(wl)):
            raise Error("wm_loss not finite")
        if wl != Scalar[DT](0.0):
            saw_wm = True
    if not saw_wm:
        raise Error("wm_loss stayed exactly zero")
    print("  WM trains on real CarRacing pixels — last WM =", ag.last_wm_loss(),
          " AC =", ag.last_ac_loss())

    # select_action on a real frame → valid continuous action.
    for i in range(OBS):
        obsbuf[i] = obs[i]
    for j in range(ACT):
        actbuf[j] = Scalar[DT](0.0)
    ag.select_action(obsbuf, actbuf, explore=True)
    for j in range(ACT):
        if not isfinite(Float64(actbuf[j])):
            raise Error("select_action produced a non-finite action")
        if actbuf[j] < Scalar[DT](-1.0) or actbuf[j] > Scalar[DT](1.0):
            raise Error("action out of [-1,1]")
    print("  select_action OK: [", actbuf[0], actbuf[1], actbuf[2], "]")

    obsbuf.free()
    actbuf.free()
    print("DREAMERV3 CARRACING PIXEL WIRING SMOKE PASSED")
