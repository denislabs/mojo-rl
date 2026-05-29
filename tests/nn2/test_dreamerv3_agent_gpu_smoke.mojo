"""DreamerV3Agent GPU smoke — on-policy select_action on Metal.

GPU analog of `test_dreamerv3_agent_smoke`: instantiates the agent with
`train_target="gpu"`, runs the real PendulumV2 loop (warmup random → agent
select_action with running belief carry → train every N), then a short
greedy eval. Exercises the GPU `select_action` device-forward inference path
(H2D obs+belief → enc/core/policy device forward → D2H → host sample) AND
the GPU train_step. Gate: actions in range, WM/AC finite, WM decreases,
greedy eval finite.

Run: `pixi run -e apple mojo run -I . tests/nn2/test_dreamerv3_agent_gpu_smoke.mojo`
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.agent import DreamerV3Agent
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

comptime Ag = DreamerV3Agent[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP,
]


def main() raises:
    print("=" * 70)
    print("DreamerV3Agent GPU smoke — on-policy select_action (Pendulum, Metal)")
    print("=" * 70)
    seed(7)
    var ctx = DeviceContext()
    var env = PendulumV2[DT]()
    var ag = Ag.make(
        ctx=ctx, lr=Scalar[DT](1e-3), learning_starts=64,
        action_scale=Scalar[DT](2.0), warmup_steps=0,
    )

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

    comptime TOTAL = 220
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
        ag.record(obsbuf, actbuf, reward, Scalar[DT](1.0) if done else Scalar[DT](0.0))
        obs = res[0].copy()
        if done:
            obs = env.reset_obs_list()
            ag.reset_belief()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            if ag.train_step():
                var wm = ag.last_wm_loss()
                var ac = ag.last_ac_loss()
                assert_true(wm == wm, "WM finite")
                assert_true(ac == ac, "AC finite")
                if n_train == 0:
                    first_wm = wm
                last_wm = wm
                n_train += 1

    print("  trained", n_train, "steps; WM:", first_wm, "->", last_wm)
    print("  max|action| =", max_abs_action, "(should be <= 2.0)")
    assert_true(max_abs_action <= Scalar[DT](2.0001), "action in range")
    assert_true(n_train > 0, "should have trained")
    assert_true(last_wm < first_wm, "WM loss should decrease")

    ag.reset_belief()
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
    assert_true(ret == ret, "eval return finite")
    print("=" * 70)
    print("GPU SMOKE PASSED — DreamerV3Agent GPU select_action works")
    print("=" * 70)
    obsbuf.free(); actbuf.free()
