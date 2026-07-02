"""DreamerV3 (nn) — CartPole lighthouse driver (discrete actor, CPU).

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
gradient) is validated in `tests/nn/test_dreamerv3_discrete_dist.mojo` and
`test_dreamerv3_imag_loss_discrete.mojo`; the continuous path is unchanged
(comptime-if elision → bit-identical). GPU discrete AC is a follow-up; this
driver is CPU (the trainer guards `train_target='gpu'` + DISCRETE).

v1 simplification: episode-end is treated as truncation in the repval value
target (term=0, same as the Pendulum driver) — the imagination `con` head
still learns done→0, so the imagined-return weighting handles termination.

Run (CPU):
  pixi run mojo run -I . examples/cartpole/cartpole_dreamerv3_nn.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
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
comptime CLASSES = 4      # tried 16 (SC 64→256) + H=64: DESTABILIZED value est (ret_m went negative, ret_sd~160, ret stuck at 8). Finer latent fed the SAME small heads (HU/VU=32) → value-noise. Reverted; capacity bump needs balanced (latent+heads) growth, which explodes CPU cost. CartPole instability is the model-exploitation gap, not latent resolution.
comptime BLOCKS = 4
comptime TOKEN = 32
comptime DEC_U = 32
comptime HU = 32
comptime VU = 32
comptime PU = 32
comptime BINS = 51
comptime B = 16
comptime T = 16
comptime T_IMAG = 15      # horizon 15 (was 10): the actor must SEE the slow cart
                          # drift that ends episodes; 10 was too short to credit it
                          # → a stability ceiling at ~100. See header note.
comptime CAP = 200_000

comptime Ag = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True
    OUT_INIT=Kaiming,  # full reward/critic output init (positive-reward optimism)
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
    var obsbuf = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var actbuf = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
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
    print("DreamerV3 (nn) CartPole lighthouse [discrete] —", TOTAL_STEPS, "steps")
    print("=" * 70)
    seed(42)
    var env = CartPoleEnv[DT]()
    # SOLVES (mean_ret(10)=500, sustained from ~35k). Two findings got it here:
    #   1. TERMINATION DATA (the real fix): the driver now stores the fallen obs
    #      via `record_terminal` so the WM continue head learns `latent(fall)→0`.
    #      Before, the fallen obs was overwritten by the reset → the cont head
    #      never fired → imagination over-survived → value collapsed to a constant
    #      → zero actor advantage → stuck ~25-43. (Localized with the open-loop
    #      diagnostic; see examples/cartpole/cartpole_dreamerv3_openloop_diag.mojo.)
    #   2. STABILITY (the secondary plateau): with termination fixed it learned but
    #      plateaued ~100 — a stability ceiling. lr=3e-4 + T_IMAG=10 sit on a
    #      stability boundary (T_IMAG=10 is too short to credit the slow cart drift
    #      that ends episodes; raising it at lr=3e-4 makes the value run away). The
    #      reference recipe — longer horizon (T_IMAG=15) + lower lr (1.5e-4) — is
    #      stable AND sees the drift → full solve.
    #   OUT_INIT=Kaiming (full reward/critic output init): the early optimism
    #   drives exploration on this dense POSITIVE-reward task; zero-init (good for
    #   negative-reward Pendulum) over-damps it. actent stays at the 3e-4 default
    #   (the unimix categorical already explores).
    var ag = Ag.make(
        lr=Scalar[DT](1.5e-4), learning_starts=LEARN_START, warmup_steps=500,
    )

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var actbuf = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()

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
            # store the terminal (fallen) obs as its own frame so the WM
            # continue head can learn `latent(terminal)→0` (else imagination
            # over-survives → value collapse → no actor signal).
            for i in range(OBS):
                obsbuf[i] = res[0][i]
            ag.record_terminal(obsbuf)
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
                " val_m=", ag.dbg_val_mean(), " val_sd=", ag.dbg_val_std(),
                " feat_sd=", ag.dbg_feat_std(),
                " con_m=", ag.dbg_con_mean(), " con_min=", ag.dbg_con_min(),
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
