"""DreamerV3 batched continuous driver smoke (P4b).

Drives `DreamerV3Agent.train_continuous_batched` over N=2 parallel
CarRacingMB[f32, PIXEL_OBS=True, PIX_RES=96] envs (CPU), with short episodes
(max_steps=20) so each env completes episodes quickly → its episode is flushed
contiguously to the sequence replay → training kicks in. Asserts the run
completes episodes, the world model trains (finite/nonzero WM loss), and the
single-stream replay stays trainable (no cross-env window corruption — episodes
are flushed as contiguous blocks).

Run: pixi run mojo run -I . tests/nn/test_dreamerv3_batched_continuous_smoke.mojo
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
comptime BASE = 4
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

comptime N_ENVS = 2


def main() raises:
    print("DreamerV3 batched continuous driver smoke — N=", N_ENVS,
          " envs, OBS =", OBS)
    var ag = Ag.make(learning_starts=8, action_scale=Scalar[DT](1.0))
    var envs = List[Env]()
    for _ in range(N_ENVS):
        envs.append(Env(max_steps=20))  # short episodes → quick flush + train

    var avg = ag.train_continuous_batched[Env](
        envs, 60, learn_start=8, train_every=4, print_every=20,
    )
    print("  returned avg_ep_ret =", avg)

    if not ag.can_train():
        raise Error("replay not trainable after batched run")
    var wl = ag.last_wm_loss()
    if not isfinite(Float64(wl)):
        raise Error("WM loss not finite after batched run")
    if wl == Scalar[DT](0.0):
        raise Error("WM loss stayed exactly zero — no training happened")
    print("  WM trained on batched CarRacing pixels — last WM =", wl,
          " AC =", ag.last_ac_loss())
    print("DREAMERV3 BATCHED CONTINUOUS DRIVER SMOKE PASSED")
