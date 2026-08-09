"""TD-MPC2 agent GPU smoke (Apple Metal, MPC-off) — PendulumV2 end-to-end.

Validates the GPU training stack wired together: sequence replay (host) →
TD-target (gpu) → WM BPTT (gpu) → policy update (gpu) → Polyak (gpu), with
MPC-off policy acting on device. Gate: actions in range, WM loss finite +
decreases, greedy eval finite. (Full GPU convergence is a separate run.)

Run: `pixi run -e apple mojo run -I . tests/deep_agents/test_tdmpc2_agent_gpu_smoke.mojo`
"""

from std.random import random_float64, seed
from std.math import isfinite
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.agent import TDMPC2Agent
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
comptime B = 4
comptime H = 3
comptime CAP = 4096

comptime Ag = TDMPC2Agent[
    "gpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP
]


def main() raises:
    print("=" * 70)
    print("TD-MPC2 agent GPU smoke (Apple, MPC-off) — Pendulum")
    print("=" * 70)
    seed(7)
    var ctx = DeviceContext()
    var env = PendulumV2[DT]()
    var ag = Ag.make(
        lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=64,
        ctx=ctx,
    )

    var obs = env.reset_obs_list()
    var obsbuf = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var actbuf = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))

    comptime TOTAL = 300
    comptime LEARN_START = 64
    comptime TRAIN_EVERY = 6
    var first_wm: Scalar[DT] = 0.0
    var last_wm: Scalar[DT] = 0.0
    var n_train = 0
    var max_abs_action: Scalar[DT] = 0.0

    for step in range(TOTAL):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            actbuf[0] = Scalar[DT](random_float64() * 4.0 - 2.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var aa = actbuf[0] if actbuf[0] >= 0 else -actbuf[0]
        if aa > max_abs_action:
            max_abs_action = aa
        var al = List[Scalar[DT]]()
        al.append(actbuf[0])
        var res = env.step_continuous_vec[DT](al)
        ag.record(obsbuf, actbuf, res[1], Scalar[DT](0.0))
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            if ag.train_step():
                var wm = ag.last_wm_loss()
                assert_true(isfinite(wm), "WM finite")
                assert_true(isfinite(ag.last_pi_loss()), "pi finite")
                if n_train == 0:
                    first_wm = wm
                last_wm = wm
                n_train += 1

    print("  trained", n_train, "steps; WM:", first_wm, "->", last_wm)
    print("  max|action| =", max_abs_action)
    assert_true(max_abs_action <= Scalar[DT](2.0001), "action in range")
    assert_true(n_train > 0, "should have trained")
    assert_true(last_wm < first_wm, "WM loss should decrease on GPU")

    var ev_obs = env.reset_obs_list()
    var ret: Scalar[DT] = 0.0
    for _s in range(200):
        for i in range(OBS):
            obsbuf[i] = ev_obs[i]
        ag.select_greedy_action(obsbuf, actbuf)
        var al2 = List[Scalar[DT]]()
        al2.append(actbuf[0])
        var r = env.step_continuous_vec[DT](al2)
        ret += r[1]
        ev_obs = r[0].copy()
        if r[2]:
            break
    print("  greedy eval return (1 ep) =", ret)
    assert_true(isfinite(ret), "eval finite")
    print("=" * 70)
    print("SMOKE PASSED — TD-MPC2 agent trains end-to-end on GPU")
    print("=" * 70)
