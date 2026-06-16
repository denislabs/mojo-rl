"""DreamerV3 (nn) — Pendulum CPU training run (realistic, not the 1M lighthouse).

Smaller-than-size1m config tuned to make CPU training observable: collect
on-policy with `DreamerV3Agent.select_action`, train every few env steps,
greedy-eval periodically and print the running return so we can SEE whether
the world-model + actor-critic loop learns on Pendulum.

Random policy on Pendulum-v1 returns ≈ −1200..−1600; a good policy ≈ −150.
Watch `mean_ret` trend downward in magnitude.

Run:
  pixi run mojo run -I . examples/pendulum/pendulum_dreamerv3_nn_cpu.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.envs.pendulum import PendulumV2

# ── CPU-realistic config (smaller than size1m DETER=512/B=16/T=64) ──────
comptime OBS = 3
comptime ACT = 1
comptime DETER = 128
comptime H = 64
comptime STOCH = 16
comptime CLASSES = 4
comptime BLOCKS = 8
comptime TOKEN = 64
comptime DEC_U = 64
comptime HU = 64
comptime VU = 64
comptime PU = 64
comptime BINS = 255
comptime B = 16
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 200_000

comptime Ag = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP,
]

comptime TOTAL = 10_000
comptime LEARN_START = 1000
comptime GRAD_STEPS_PER = 3       # grad steps per env step after warmup (ratio≈3)
comptime EVAL_EVERY = 1000
comptime EVAL_EPISODES = 5
comptime EP_LEN = 200


def _eval(mut ag: Ag) raises -> Scalar[DT]:
    var eenv = PendulumV2[DT]()
    var ob = alloc[Scalar[DT]](OBS)
    var ac = alloc[Scalar[DT]](ACT)
    var total: Scalar[DT] = 0.0
    for _e in range(EVAL_EPISODES):
        ag.reset_belief()
        var o = eenv.reset_obs_list()
        for _s in range(EP_LEN):
            for i in range(OBS):
                ob[i] = o[i]
            ag.select_greedy_action(ob, ac)
            var al = List[Scalar[DT]]()
            al.append(ac[0])
            var r = eenv.step_continuous_vec[DT](al)
            total += r[1]
            o = r[0].copy()
            if r[2]:
                break
    ob.free(); ac.free()
    return total / Scalar[DT](EVAL_EPISODES)


def main() raises:
    print("DreamerV3 CPU Pendulum run | DETER=", DETER, " B=", B, " T=", T,
          " BINS=", BINS, " total=", TOTAL)
    seed(42)
    var env = PendulumV2[DT]()
    var ag = Ag.make(
        lr=Scalar[DT](4e-5), learning_starts=LEARN_START,
        action_scale=Scalar[DT](2.0),
    )
    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var ep_ret: Scalar[DT] = 0.0

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
        ep_ret += res[1]
        ag.record(
            obsbuf, actbuf, res[1],
            Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
            ag.reset_belief()
            ep_ret = 0.0
        if step >= LEARN_START:
            for _g in range(GRAD_STEPS_PER):
                _ = ag.train_step()
        if step >= LEARN_START and step % EVAL_EVERY == 0:
            var mret = _eval(ag)
            print(
                "step", step, " eval_ret(", EVAL_EPISODES, ")=", mret,
                " WM=", ag.last_wm_loss(), " AC=", ag.last_ac_loss(),
                " train_steps=", ag.trainer.train_steps,
            )
            obs = env.reset_obs_list()
            ag.reset_belief()

    print("FINAL eval_ret =", _eval(ag))
    obsbuf.free(); actbuf.free()
