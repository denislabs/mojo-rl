"""CPU smoke for the multi-task TD-MPC2 agent on GPU (storage).

Constructs `TDMPC2MultiTaskAgent` for 2 tasks (Pendulum-like MAX_OBS=3 +
a padded second task), records random transitions for both task ids, runs a
few hundred train_steps, and asserts the WM + policy losses are finite and
train_step returns True. Mirrors the single-task `test_tdmpc2_pendulum_cpu_smoke`.
"""

from std.random import random_float64, seed
from std.math import isnan, isinf
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.config_mt import TDMPC2MultiTask


def main() raises:
    seed(0)
    comptime MAX_OBS = 3
    comptime MAX_ACT = 1
    comptime NUM_TASKS = 2
    comptime TASK_EMB = 4
    comptime B = 8
    comptime CAP = 4096

    var c = DeviceContext()
    var agent = TDMPC2MultiTask[
        "gpu", MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC=16, LATENT=16, MLP=16, BINS=21, SN=4, H=3,
    ](learning_starts=64, ctx=Optional(c))

    # Record random transitions for both task ids.
    var obs = List[Scalar[DT]](length=MAX_OBS, fill=0)
    var act = List[Scalar[DT]](length=MAX_ACT, fill=0)
    for ep in range(40):
        var tsk = ep % NUM_TASKS
        agent.set_task(tsk)
        for _ in range(20):
            for i in range(MAX_OBS):
                obs[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
            agent.select_action(obs, act, explore=True)
            var r = Scalar[DT](random_float64() - 0.5)
            var d = Scalar[DT](0.0)
            agent.record(obs, act, r, d)

    var trained = 0
    var any_true = False
    for _ in range(300):
        if agent.train_step():
            any_true = True
            trained += 1

    var wm = agent.last_wm_loss()
    var pi = agent.last_pi_loss()
    print("trained_steps =", trained)
    print("wm_loss =", wm, " pi_loss =", pi)

    if not any_true:
        raise Error("FAILED: train_step never returned True")
    if isnan(Float64(wm)) or isinf(Float64(wm)):
        raise Error("FAILED: wm_loss not finite")
    if isnan(Float64(pi)) or isinf(Float64(pi)):
        raise Error("FAILED: pi_loss not finite")
    print("ALL PASSED")
