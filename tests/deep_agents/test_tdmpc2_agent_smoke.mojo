"""TD-MPC2 agent smoke (CPU, MPC-off) — PendulumV2 end-to-end.

Validates the whole learning stack wired together: sequence replay →
TD-target → WM BPTT → policy update → Polyak, with MPC-off policy acting.
Gate: actions in range, WM loss finite + decreases, greedy eval finite.
(Convergence tuning is a separate effort; this proves the pipeline runs
and the world model learns.)

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_agent_smoke.mojo`
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.math import isfinite
from std.testing import assert_true

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
    "cpu",
    OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP
]


def main() raises:
    print("=" * 70)
    print("TD-MPC2 agent smoke (CPU, MPC-off) — Pendulum")
    print("=" * 70)
    seed(7)
    var env = PendulumV2[DT]()
    var ag = Ag.make(
        lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=64,
    )

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

    comptime TOTAL = 400
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
        var act_list = List[Scalar[DT]]()
        act_list.append(actbuf[0])
        var res = env.step_continuous_vec[DT](act_list)
        var reward = res[1]
        var done = res[2]
        ag.record(
            obsbuf, actbuf, reward,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if done:
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
    print("  max|action| =", max_abs_action, "(should be <= 2.0)")
    assert_true(max_abs_action <= Scalar[DT](2.0001), "action in range")
    assert_true(n_train > 0, "should have trained")
    assert_true(last_wm < first_wm, "WM loss should decrease")

    # short greedy eval
    var ev_obs = env.reset_obs_list()
    var ret: Scalar[DT] = 0.0
    for _s in range(200):
        for i in range(OBS):
            obsbuf[i] = ev_obs[i]
        ag.select_greedy_action(obsbuf, actbuf)
        var al = List[Scalar[DT]]()
        al.append(actbuf[0])
        var r = env.step_continuous_vec[DT](al)
        ret += r[1]
        ev_obs = r[0].copy()
        if r[2]:
            break
    print("  greedy eval return (1 ep) =", ret)
    assert_true(isfinite(ret), "eval return finite")
    print("=" * 70)
    print("SMOKE PASSED — TD-MPC2 agent (MPC-off) trains end-to-end")
    print("=" * 70)
    obsbuf.free(); actbuf.free()
