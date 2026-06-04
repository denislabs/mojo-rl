"""Multi-task TD-MPC2 (deep_agents2) — Pendulum + InvertedPendulum (CPU).

Item C lighthouse (§14.3): ONE task-conditioned TD-MPC2 agent trained over two
envs with different obs dims (Pendulum obs3/act1, InvertedPendulum obs4/act1),
padded to MAX_OBS=4 / MAX_ACT=1, conditioned on a learned per-task embedding.
Episodes round-robin the two tasks; we report each task's greedy return so you
can see the single agent learning distinct behaviors.

Acting is MPC-off (`a = π(encode([obs|task_emb]))`). This CPU script is the
correctness/convergence lighthouse; `..._gpu.mojo` is the GPU variant.

Run: `pixi run mojo run -I . examples/multitask/tdmpc2_pendulum_invpendulum_cpu.mojo`
"""

from std.memory import alloc
from std.random import random_float64, seed

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.tdmpc2.config_mt import TDMPC2MultiTask
from mojo_rl.deep_agents2.tdmpc2.agent_mt import TDMPC2MultiTaskAgent
from mojo_rl.envs.multitask_pendulum import MultiTaskEnv

comptime TARGET = "cpu"
comptime MAX_OBS = 4
comptime MAX_ACT = 1
comptime NUM_TASKS = 2
comptime TASK_EMB = 16
comptime B = 256
comptime CAP = 200_000
comptime ENC = 128
comptime LATENT = 128
comptime MLP = 128
comptime BINS = 101

comptime LR = 3e-4
comptime LEARN_START = 2_000
comptime TRAIN_EVERY = 1
comptime TOTAL = 60_000
comptime EVAL_EVERY = 10_000
comptime EP_LEN = 200
comptime EVAL_EPS_PER_TASK = 2


comptime TDMPC2MultiTaskAgentT = TDMPC2MultiTaskAgent[
    TARGET, MAX_OBS, ENC, MAX_ACT, LATENT, MLP, BINS, 8, -10, 10, B, 3, CAP,
    NUM_TASKS, TASK_EMB,
]


def _greedy_eval(mut ag: TDMPC2MultiTaskAgentT, mut env: MultiTaskEnv) raises:
    var obsbuf = alloc[Scalar[DT]](MAX_OBS)
    var actbuf = alloc[Scalar[DT]](MAX_ACT)
    var ret0: Scalar[DT] = 0.0
    var ret1: Scalar[DT] = 0.0
    for _ep in range(EVAL_EPS_PER_TASK * NUM_TASKS):
        var obs = env.reset()
        var task = env.task_id()
        ag.set_task(task)
        var ep_ret: Scalar[DT] = 0.0
        for _s in range(EP_LEN):
            for i in range(MAX_OBS):
                obsbuf[i] = obs[i]
            ag.select_greedy_action(obsbuf, actbuf)
            var al = List[Scalar[DT]]()
            al.append(actbuf[0])
            var r = env.step(al)
            ep_ret += r[1]
            obs = r[0].copy()
            if r[2]:
                break
        if task == 0:
            ret0 += ep_ret
        else:
            ret1 += ep_ret
    var inv = Scalar[DT](1.0) / Scalar[DT](EVAL_EPS_PER_TASK)
    print("    eval  pendulum=", ret0 * inv, " inverted_pendulum=", ret1 * inv)
    obsbuf.free(); actbuf.free()


def main() raises:
    print("=" * 70)
    print("Multi-task TD-MPC2 (CPU) — Pendulum + InvertedPendulum")
    print("  MAX_OBS=", MAX_OBS, " MAX_ACT=", MAX_ACT, " NUM_TASKS=", NUM_TASKS,
          " TASK_EMB=", TASK_EMB)
    print("=" * 70)
    seed(0)
    var ag = TDMPC2MultiTask[
        TARGET, MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS,
    ](lr=Scalar[DT](LR), learning_starts=LEARN_START)
    var env = MultiTaskEnv()

    var obs = env.reset()
    ag.set_task(env.task_id())
    var obsbuf = alloc[Scalar[DT]](MAX_OBS)
    var actbuf = alloc[Scalar[DT]](MAX_ACT)
    var ep_step = 0

    for step in range(TOTAL):
        for i in range(MAX_OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            actbuf[0] = Scalar[DT](random_float64() * 2.0 - 1.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var al = List[Scalar[DT]]()
        al.append(actbuf[0])
        var res = env.step(al)
        var term = Scalar[DT](1.0) if env.was_terminated() else Scalar[DT](0.0)
        ag.record(obsbuf, actbuf, res[1], term)
        obs = res[0].copy()
        ep_step += 1
        if res[2] or ep_step >= EP_LEN:
            obs = env.reset()          # round-robin to the next task
            ag.set_task(env.task_id())
            ep_step = 0
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            _ = ag.train_step()
        if step > 0 and step % EVAL_EVERY == 0:
            print("  step", step, " wm=", ag.last_wm_loss())
            _greedy_eval(ag, env)
            # restore a fresh episode for the actor after eval roll-outs
            obs = env.reset()
            ag.set_task(env.task_id())
            ep_step = 0

    print("=" * 70)
    print("  FINAL")
    _greedy_eval(ag, env)
    print("=" * 70)
    obsbuf.free(); actbuf.free()
