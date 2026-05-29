"""DreamerV3 (nn2) — Pendulum lighthouse driver, GPU / size1m config.

PR5c Step 7 (GPU). Same end-to-end loop as the CPU driver
(`pendulum_dreamerv3_nn2_agent.mojo`) but instantiates the agent with
`train_target="gpu"` and the full **size1m** RSSM config (DETER=512, BLOCKS=8,
units=64, BINS=255, B=16, T=64) at the reference replay ratio. Run this for
the real ~1.1M-step convergence run once an NVIDIA GPU is available.

Run (NVIDIA):
  pixi run -e nvidia mojo run -I . examples/pendulum/pendulum_dreamerv3_nn2_gpu.mojo
Run (Apple Metal — parity/smoke only, slow at this scale):
  pixi run -e apple   mojo run -I . examples/pendulum/pendulum_dreamerv3_nn2_gpu.mojo

────────────────────────────────────────────────────────────────────────────
✅ GPU path is built & validated on Apple Metal (CPU↔GPU bit-match 3.8e-6)
────────────────────────────────────────────────────────────────────────────
Both previously-missing GPU pieces are done (2026-05-29):
  1. `DreamerV3Agent.select_action` GPU inference path (B=1 device-forward
     hybrid) — `tests/nn2/test_dreamerv3_agent_gpu_smoke.mojo`.
  2. `_ac_gpu` brought to `_ac_cpu` parity (NS=T·B starts, mean cotangents,
     repval) — `tests/nn2/test_dreamerv3_ac_parity.mojo` matches the CPU
     train step to float32 ULP (max |ΔWM|=|ΔAC|=3.8e-6 over 12 steps).
So `DreamerV3Agent["gpu", …]` compiles and trains; this size1m config runs on
Metal (slow) or NVIDIA. Convergence is still the open question (below).

────────────────────────────────────────────────────────────────────────────
⚠️ CONVERGENCE IS NOT SOLVED — this is a scaffold, not a passing lighthouse
────────────────────────────────────────────────────────────────────────────
After all six fixes the CPU run is still flat (~−1500, well above the −200
bar). The diagnosed root is **off-distribution imagined-reward optimism**: the
policy outruns the WM, the reward head reads imagined latents optimistically,
the value bootstraps positive, the advantage degenerates. The levers that
actually move WM quality (all GPU-phase) are, in order of expected impact:
  1. SCALE — this size1m config + ~1M steps + train_ratio≈1024 (a genuinely
     well-trained WM keeps imagined latents in-distribution).
  2. WM-only warmup — freeze actor/critic, train the WM K steps before AC
     starts (NOT YET BUILT: needs trainer.train_wm_only() + wm_warmup_steps).
  3. free_nats / KL balance — tighten prior↔posterior so prior-sampled
     (imagined) latents match posterior (the direct root lever).
Watch the `dbg_*` columns: success = `rew_pred` tracks `real_rew`, `ret_m`
goes NEGATIVE (Pendulum reward is always ≤0), `ret_sd` stays > 0.
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.agent import DreamerV3Agent
from mojo_rl.envs.pendulum import PendulumV2

# ── size1m config (runbook) ───────────────────────────────────────────────
comptime OBS = 3
comptime ACT = 1
comptime DETER = 512
comptime H = 64
comptime STOCH = 32
comptime CLASSES = 4
comptime BLOCKS = 8
comptime TOKEN = 64
comptime DEC_U = 64
comptime HU = 64
comptime VU = 64
comptime PU = 64
comptime BINS = 255
comptime B = 16
comptime T = 64
comptime T_IMAG = 15
comptime CAP = 1_000_000

comptime Ag = DreamerV3Agent[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP,
]

comptime TOTAL_STEPS = 1_100_000
comptime LEARN_START = 1024
# reference pendulum1m train_ratio=1024 (grad steps per env step ≈ B·T/ratio).
# TRAIN_EVERY=1 with B=16,T=64 ≈ 1024 replayed frames per env step.
comptime TRAIN_EVERY = 1
comptime EVAL_EVERY = 10000
comptime EVAL_EPISODES = 10
comptime EP_LEN = 200
comptime ACTION_SCALE = Scalar[DT](2.0)


def _greedy_eval(
    mut ag: Ag, mut env: PendulumV2[DT]
) raises -> Tuple[Scalar[DT], Scalar[DT]]:
    """Returns (mean_return, mean_|greedy_action|) over EVAL_EPISODES."""
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var total: Scalar[DT] = 0.0
    var act_abs: Scalar[DT] = 0.0
    var nsteps: Int = 0
    for _e in range(EVAL_EPISODES):
        ag.reset_belief()
        var o = env.reset_obs_list()
        for _s in range(EP_LEN):
            for i in range(OBS):
                obsbuf[i] = o[i]
            ag.select_greedy_action(obsbuf, actbuf)
            var aa = actbuf[0] if actbuf[0] >= 0 else -actbuf[0]
            act_abs += aa
            nsteps += 1
            var al = List[Scalar[DT]]()
            al.append(ACTION_SCALE * actbuf[0])   # normalized → env range
            var r = env.step_continuous_vec[DT](al)
            total += r[1]
            o = r[0].copy()
            if r[2]:
                break
    obsbuf.free(); actbuf.free()
    return Tuple(
        total / Scalar[DT](EVAL_EPISODES),
        act_abs / Scalar[DT](nsteps),
    )


def main() raises:
    print("=" * 70)
    print("DreamerV3 (nn2) Pendulum lighthouse [GPU/size1m] —", TOTAL_STEPS, "steps")
    print("=" * 70)
    seed(42)
    var ctx = DeviceContext()
    var env = PendulumV2[DT]()
    var ag = Ag.make(
        ctx=ctx, lr=Scalar[DT](4e-5), learning_starts=LEARN_START,
        action_scale=Scalar[DT](2.0),
    )

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

    for step in range(TOTAL_STEPS):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            # warmup explores the NORMALIZED [-1,1] space (recorded as-is)
            actbuf[0] = Scalar[DT](random_float64() * 2.0 - 1.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var al = List[Scalar[DT]]()
        al.append(ACTION_SCALE * actbuf[0])   # normalized → env range
        var res = env.step_continuous_vec[DT](al)
        ag.record(
            obsbuf, actbuf, res[1],            # record the NORMALIZED action
            Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
            ag.reset_belief()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            _ = ag.train_step()

        if step > 0 and step % EVAL_EVERY == 0:
            var eval_env = PendulumV2[DT]()
            var ev = _greedy_eval(ag, eval_env)
            print(
                "  step", step, " ret=", ev[0], " |act|=", ev[1],
                " real_rew=", ag.dbg_real_rew(), " rew_pred=", ag.dbg_rew_pred(),
                " ret_m=", ag.dbg_ret_mean(), " ret_sd=", ag.dbg_ret_std(),
                " pmean=", ag.dbg_pmean_abs(),
                " WM=", ag.last_wm_loss(), " AC=", ag.last_ac_loss(),
            )
            obs = env.reset_obs_list()
            ag.reset_belief()

    var fe = _greedy_eval(ag, env)
    print("=" * 70)
    print("FINAL mean_ret(", EVAL_EPISODES, ") =", fe[0],
          "  (lighthouse pass: >= -200)")
    print("=" * 70)
    obsbuf.free(); actbuf.free()
