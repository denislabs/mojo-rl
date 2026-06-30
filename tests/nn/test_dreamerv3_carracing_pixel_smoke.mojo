"""DreamerV3 continuous facade (`train_continuous`) ↔ real CarRacing pixel env
smoke (CPU).

Drives the single-env continuous facade on the REAL CarRacingMB[f32,
PIXEL_OBS=True, PIX_RES=96] env (4×96×96 = 36864 obs, 3-D continuous action) for
a few steps with short episodes, then asserts the world model trained (finite/
nonzero WM loss) and the replay is sampleable. Exercises the whole facade path:
warmup → select_action → step_continuous_vec → record (+record_terminal on done)
→ train_step → greedy eval — the continuous counterpart of train_single.

Run: pixi run mojo run -I . tests/nn/test_dreamerv3_carracing_pixel_smoke.mojo
"""

from std.math import isfinite

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)
from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB

comptime C = 4
comptime IMG = 96
comptime BASE = 4       # tiny conv width for a fast smoke
comptime OBS = C * IMG * IMG   # 36864
comptime ACT = 3
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
    print("DreamerV3 train_continuous facade ↔ CarRacing pixel smoke — OBS =",
          OBS, "(", C, "x", IMG, "x", IMG, ")")
    var ag = Ag.make(learning_starts=8, action_scale=Scalar[DT](1.0))
    var env = Env(max_steps=20)  # short episodes → quick record_terminal + train

    var final_ret = ag.train_continuous[Env](
        env, 48, learn_start=8, train_every=4, eval_every=24,
        eval_episodes=1, ep_len=20, print_every=24, verbose=True,
    )
    print("  facade returned final greedy eval =", final_ret)

    if not ag.can_train():
        raise Error("replay not trainable after facade run")
    var wl = ag.last_wm_loss()
    if not isfinite(Float64(wl)):
        raise Error("WM loss not finite after facade run")
    if wl == Scalar[DT](0.0):
        raise Error("WM loss stayed exactly zero — no training happened")
    print("  WM trained via train_continuous facade — last WM =", wl,
          " AC =", ag.last_ac_loss())
    print("DREAMERV3 CONTINUOUS FACADE SMOKE PASSED")
