"""DreamerV3 (nn2) — CartPole lighthouse driver (discrete actor, CPU).

PR5d. End-to-end DreamerV3 world-model + actor-critic training on CartPole
using the block-composed `DreamerV3Agent[..., DISCRETE=True]`: the unimix
categorical actor outputs a one-hot action, the driver argmaxes it to the
env action index (0=left, 1=right), and the one-hot is recorded as the WM's
action input (the WM's `ActionSquash` is a no-op on {0,1}).

CartPole is the canonical "does the agent learn at all" lighthouse: dense
+1/step reward, bounded return (≤500), near-linear dynamics, short horizon —
it de-risks the WM+imagination pipeline far more cleanly than Pendulum's
swing-up. Pass = mean_ret(10) ≥ 475 (near-solved); early signal: mean_ret
climbing well above the ~20 of a random policy.

Discrete support: the categorical actor (logp / entropy / straight REINFORCE
gradient) is validated in `tests/nn2/test_dreamerv3_discrete_dist.mojo` and
`test_dreamerv3_imag_loss_discrete.mojo`; the continuous path is unchanged
(comptime-if elision → bit-identical). GPU discrete AC is a follow-up; this
driver is CPU (the trainer guards `train_target='gpu'` + DISCRETE).

v1 simplification: episode-end is treated as truncation in the repval value
target (term=0, same as the Pendulum driver) — the imagination `con` head
still learns done→0, so the imagined-return weighting handles termination.

Run (CPU):
  pixi run mojo run -I . examples/cartpole/cartpole_dreamerv3_nn2.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.agent import DreamerV3Agent
from mojo_rl.envs.cartpole import CartPoleEnv

# ── config (CPU-light; CartPole is easy → small RSSM + aggressive hypers) ──
# NOTE: lr / TRAIN_EVERY are tuned for FAST small-env CPU convergence, NOT the
# size1m/1M-step Atari recipe (lr=4e-5, train_ratio≈1024). CartPole is dense
# (+1/step) and low-dim, so a small RSSM + lr~3e-4 + frequent updates learns
# in a feasible CPU budget. Scale these toward the reference for harder envs.
comptime OBS = 4
comptime ACT = 2          # one-hot action dim = #actions (left/right)
comptime DETER = 128
comptime H = 32
comptime STOCH = 16
comptime CLASSES = 4
comptime BLOCKS = 4
comptime TOKEN = 32
comptime DEC_U = 32
comptime HU = 32
comptime VU = 32
comptime PU = 32
comptime BINS = 51
comptime B = 16
comptime T = 16
comptime T_IMAG = 10
comptime CAP = 200_000

comptime Ag = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True
]

comptime TOTAL_STEPS = 150_000
comptime LEARN_START = 1024
comptime TRAIN_EVERY = 4
comptime EVAL_EVERY = 2500
comptime EVAL_EPISODES = 10
comptime EP_LEN = 500


def _argmax(a: UnsafePointer[Scalar[DT], MutAnyOrigin]) -> Int:
    var k = 0
    var best = a[0]
    for i in range(1, ACT):
        if a[i] > best:
            best = a[i]
            k = i
    return k


def _greedy_eval(
    mut ag: Ag, mut env: CartPoleEnv[DT]
) raises -> Scalar[DT]:
    """Mean return over EVAL_EPISODES with the greedy (argmax) policy."""
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var total: Scalar[DT] = 0.0
    for _e in range(EVAL_EPISODES):
        ag.reset_belief()
        var o = env.reset_obs_list()
        for _s in range(EP_LEN):
            for i in range(OBS):
                obsbuf[i] = o[i]
            ag.select_greedy_action(obsbuf, actbuf)
            var r = env.step_obs(_argmax(actbuf))
            total += r[1]
            o = r[0].copy()
            if r[2]:
                break
    obsbuf.free(); actbuf.free()
    return total / Scalar[DT](EVAL_EPISODES)


def main() raises:
    print("=" * 70)
    print("DreamerV3 (nn2) CartPole lighthouse [discrete] —", TOTAL_STEPS, "steps")
    print("=" * 70)
    seed(42)
    var env = CartPoleEnv[DT]()
    var ag = Ag.make(
        lr=Scalar[DT](3e-4), learning_starts=LEARN_START, warmup_steps=500,
    )

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

    for step in range(TOTAL_STEPS):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        var idx: Int
        if step < LEARN_START:
            # warmup: uniform random action → one-hot recorded
            idx = Int(random_float64() * 2.0)
            if idx >= ACT:
                idx = ACT - 1
            for a in range(ACT):
                actbuf[a] = Scalar[DT](1.0) if a == idx else Scalar[DT](0.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
            idx = _argmax(actbuf)
        var res = env.step_obs(idx)
        ag.record(
            obsbuf, actbuf, res[1],
            Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
            ag.reset_belief()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            _ = ag.train_step()

        if step > 0 and step % EVAL_EVERY == 0:
            var eval_env = CartPoleEnv[DT]()
            var ev = _greedy_eval(ag, eval_env)
            print(
                "  step", step, " ret=", ev,
                " real_rew=", ag.dbg_real_rew(), " rew_pred=", ag.dbg_rew_pred(),
                " ret_m=", ag.dbg_ret_mean(), " ret_sd=", ag.dbg_ret_std(),
                " plogit_abs=", ag.dbg_pmean_abs(),
                " WM=", ag.last_wm_loss(), " AC=", ag.last_ac_loss(),
            )
            obs = env.reset_obs_list()
            ag.reset_belief()

    var fe = _greedy_eval(ag, env)
    print("=" * 70)
    print("FINAL mean_ret(", EVAL_EPISODES, ") =", fe,
          "  (lighthouse pass: >= 475)")
    print("=" * 70)
    obsbuf.free(); actbuf.free()
