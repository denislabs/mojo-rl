"""DreamerV3Trainer Pendulum env-loop smoke.

Real PendulumV2 (CPU) env loop: collect transitions with random actions
(v1 — running-carry policy action is a follow-up), record to the sequence
replay, train every `train_ratio` steps after warmup. Gate: no NaN, WM +
AC losses finite, WM loss decreases across the run.

Run: `pixi run mojo run -I . tests/nn2/test_dreamerv3_pendulum_smoke.mojo`
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.trainer import DreamerV3Trainer
from mojo_rl.envs.pendulum import PendulumV2

comptime OBS = 3
comptime ACT = 1
comptime DETER = 64
comptime H = 16
comptime STOCH = 8
comptime CLASSES = 4
comptime BLOCKS = 4
comptime TOKEN = 16
comptime DEC_U = 16
comptime HU = 16
comptime VU = 16
comptime PU = 16
comptime BINS = 51
comptime B = 4
comptime T = 8
comptime T_IMAG = 6
comptime CAP = 4096

comptime Tr = DreamerV3Trainer[
    OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU, PU,
    BINS, B, T, T_IMAG, CAP,
]


def main() raises:
    print("=" * 70)
    print("DreamerV3Trainer Pendulum env-loop smoke")
    print("=" * 70)
    seed(42)
    var env = PendulumV2[DT]()
    var tr = Tr.make(lr=Scalar[DT](1e-3), learning_starts=64)

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

    comptime TOTAL = 400
    comptime LEARN_START = 64
    comptime TRAIN_EVERY = 6
    var first_wm: Scalar[DT] = 0.0
    var last_wm: Scalar[DT] = 0.0
    var n_train = 0
    for step in range(TOTAL):
        var a = Scalar[DT](random_float64() * 4.0 - 2.0)   # raw torque [-2,2]
        actbuf[0] = a
        var act_list = List[Scalar[DT]]()
        act_list.append(a)
        for i in range(OBS):
            obsbuf[i] = obs[i]
        var res = env.step_continuous_vec[DT](act_list)
        var reward = res[1]
        var done = res[2]
        tr.record(obsbuf, actbuf, reward, Scalar[DT](1.0) if done else Scalar[DT](0.0))
        obs = res[0].copy()
        if done:
            obs = env.reset_obs_list()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            var ok = tr.train_step()
            if ok:
                assert_true(tr.last_wm_loss == tr.last_wm_loss, "WM finite")
                assert_true(tr.last_ac_loss == tr.last_ac_loss, "AC finite")
                if n_train == 0:
                    first_wm = tr.last_wm_loss
                    print("  first train: WM =", tr.last_wm_loss,
                          " AC =", tr.last_ac_loss)
                last_wm = tr.last_wm_loss
                n_train += 1

    print("  trained", n_train, "steps; WM:", first_wm, "->", last_wm,
          " (train_steps=", tr.train_steps, ")")
    assert_true(n_train > 0, "should have trained")
    assert_true(last_wm < first_wm, "WM loss should decrease over the run")
    print("=" * 70)
    print("SMOKE PASSED — DreamerV3 trains on Pendulum env loop, no NaN")
    print("=" * 70)
    obsbuf.free(); actbuf.free()
