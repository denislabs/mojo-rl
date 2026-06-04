"""TD-MPC2 Pendulum convergence on GPU (NVIDIA / Apple) — fast end-to-end gate.

GPU twin of test_tdmpc2_pendulum_convergence (same config + seed). On CUDA
this runs in minutes and is the decisive check that the GPU agent CONVERGES
(the CPU path solves to ~-126) before committing to the long HalfCheetah run.
If this rots while CPU solves, suspect TF32 accumulation (see
docs/TDMPC2_NVIDIA_GRADIENT_INVESTIGATION).

Run: `pixi run -e nvidia mojo run -I . tests/deep_agents/test_tdmpc2_pendulum_gpu_convergence.mojo`
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.tdmpc2.agent import TDMPC2Agent
from mojo_rl.envs.pendulum import PendulumV2

comptime OBS = 3
comptime ENC = 64
comptime ACT = 1
comptime LATENT = 64
comptime MLP = 64
comptime BINS = 51
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 32
comptime H = 3
comptime CAP = 50000

comptime LR = 3e-4
comptime ACTION_SCALE = 2.0
comptime LEARN_START = 500
comptime TRAIN_EVERY = 1
comptime TOTAL = 12000
comptime EVAL_EVERY = 1000
comptime EVAL_EPS = 4

comptime Ag = TDMPC2Agent[
    "gpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP
]


def _greedy_eval(mut ag: Ag, mut env: PendulumV2[DT]) raises -> Scalar[DT]:
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var total: Scalar[DT] = 0.0
    for _ep in range(EVAL_EPS):
        var obs = env.reset_obs_list()
        for _s in range(200):
            for i in range(OBS):
                obsbuf[i] = obs[i]
            ag.select_greedy_action(obsbuf, actbuf)
            var al = List[Scalar[DT]]()
            al.append(actbuf[0])
            var r = env.step_continuous_vec[DT](al)
            total += r[1]
            obs = r[0].copy()
            if r[2]:
                break
    obsbuf.free(); actbuf.free()
    return total / Scalar[DT](EVAL_EPS)


def main() raises:
    print("=" * 70)
    print("TD-MPC2 Pendulum convergence (GPU, MPC-off)")
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()
    var env = PendulumV2[DT]()
    var ag = Ag.make(
        lr=Scalar[DT](LR), action_scale=Scalar[DT](ACTION_SCALE),
        learning_starts=LEARN_START, ctx=ctx,
    )

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var best: Scalar[DT] = -1.0e9

    for step in range(TOTAL):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            actbuf[0] = Scalar[DT](random_float64() * 4.0 - 2.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var al = List[Scalar[DT]]()
        al.append(actbuf[0])
        var res = env.step_continuous_vec[DT](al)
        ag.record(obsbuf, actbuf, res[1], Scalar[DT](0.0))
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            _ = ag.train_step()
        if step > 0 and step % EVAL_EVERY == 0:
            var ret = _greedy_eval(ag, env)
            if ret > best:
                best = ret
            print(
                "  step", step, " eval_return=", ret, " best=", best,
                " wm=", ag.last_wm_loss(), " pi=", ag.last_pi_loss(),
            )

    print("  FINAL best eval return =", best)
    assert_true(best > Scalar[DT](-600.0), "TD-MPC2 GPU must converge on Pendulum")
    obsbuf.free(); actbuf.free()
