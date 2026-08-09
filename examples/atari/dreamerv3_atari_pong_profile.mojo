"""DreamerV3 Atari Pong — wall-clock PROFILE of the training loop (GPU).

Answers "where do the ~820 ms/env-step go?" for the reference-aligned Pong
config (same comptime dims as `dreamerv3_atari_pong_training.mojo`). Times,
with device syncs around each section:

  per TRAIN step (eager — the run is compute-bound, capture ≈ 0 gain, so
  eager ≈ captured steady state):
    prologue   replay draw (device Philox) + imagination noise
    wm         WM-BPTT: encoder → RSSM(T=64, sequential) → decoder/rew/con
               losses → backward → 5 optimizers
    sync       core → imagine mirror param copy
    ac         imagination rollout (T_IMAG=15) + value/policy losses + opts
  per ENV step:
    env        CPU 6502 emulator step (4 ROM frames + gray-96 obs)
    act        select_action (B=1 encoder + RSSM posterior step + policy, H2D)
    record     replay insert (H2D of one obs row)

Then extrapolates s/1000 env steps (TRAIN_EVERY=4 → 250 train steps) and
compares against the observed pace. Run on NVIDIA (Apple = build check only):

    pixi run -e nvidia mojo run -I . examples/atari/dreamerv3_atari_pong_profile.mojo
"""

from max.gpu.host import DeviceContext
from std.memory import alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Zero, TruncNormalIn
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNNPool,
    DreamerDecoderCNNPool,
)
from mojo_rl.envs.atari import AtariEnv
from mojo_rl.envs.atari.games.registry import AtariGame

# ── identical arch to the training run ──
comptime C = 1
comptime IMG = 96
comptime TIER = "50m"  # MUST match the checkpoint's training TIER ("200m" | "50m")
comptime BASE = 64 if TIER == "200m" else 32
comptime OBS = C * IMG * IMG
comptime ACT = 6
comptime DETER = 8192 if TIER == "200m" else 4096
comptime H = 1024 if TIER == "200m" else 512
comptime STOCH = 32
comptime CLASSES = 64 if TIER == "200m" else 32
comptime BLOCKS = 8
comptime TOKEN = 4 * BASE * (IMG // 16) * (IMG // 16)
comptime UNITS = H  # decoder bspace-stem MLP width (= hidden, per tier)
comptime DEC_U = H
comptime HU = H
comptime VU = H
comptime PU = H
comptime BINS = 255
comptime B = 16
comptime T = 64
comptime T_IMAG = 15
comptime CAP = 4096  # profile only needs a small ring

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNNPool[C, IMG, IMG, BASE, SwishOp]
comptime DEC = DreamerDecoderCNNPool[
    FEATIN, DETER, C, IMG, IMG, BASE, UNITS, SwishOp
]
comptime Ag = DreamerV3Agent[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U,
    HU, VU, PU, BINS, B, T, T_IMAG, CAP,
    True, ENC, DEC, True, Zero, TruncNormalIn,
]
comptime Env = AtariEnv[3, DT]

comptime LEARN_START = 1024
comptime FILL_STEPS = LEARN_START + T + 8
comptime WARMUP_TRAIN = 10  # past warmup_steps=8 → steady lr
comptime PROF_TRAIN = 10
comptime PROF_ACT = 50
comptime TRAIN_EVERY = 4


def main() raises:
    seed(42)
    print("=" * 70)
    print(
        "DreamerV3 Pong PROFILE — TIER", TIER, "pool geometry (B", B,
        "T", T, "deter", DETER, "base", BASE, ")",
    )
    print("=" * 70)
    with DeviceContext() as ctx:
        var agent = Ag.make(
            ctx=ctx,
            lr=Scalar[DT](4e-5),
            learning_starts=LEARN_START,
            warmup_steps=8,  # lr constant almost immediately (profile realism)
            actent=Scalar[DT](3e-4),
            slowtar=False,
            ac_start=0,
            online=True,
        )
        var env = Env(AtariGame.PONG, sticky_prob=0.0, noop_max=30)
        var obsbuf = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
        var actbuf = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()

        # ── fill replay with random actions; time env vs record ──
        print("filling replay (", FILL_STEPS, "random env steps )...")
        agent.reset_belief()
        var obs = env.reset_obs_list()
        var env_ns: UInt = 0
        var rec_ns: UInt = 0
        for _s in range(FILL_STEPS):
            for i in range(OBS):
                obsbuf[i] = obs[i]
            var idx = Int(random_float64() * Float64(ACT))
            if idx >= ACT:
                idx = ACT - 1
            for a in range(ACT):
                actbuf[a] = Scalar[DT](1.0) if a == idx else Scalar[DT](0.0)
            var e0 = perf_counter_ns()
            var res = env.step_obs(idx)
            var e1 = perf_counter_ns()
            agent.record(
                obsbuf, actbuf, res[1].cast[DT](),
                Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
            )
            var e2 = perf_counter_ns()
            env_ns += e1 - e0
            rec_ns += e2 - e1
            obs = res[0].copy()
            if res[2]:
                for i in range(OBS):
                    obsbuf[i] = res[0][i]
                agent.record_terminal(obsbuf)
                obs = env.reset_obs_list()
                agent.reset_belief()
        var env_ms = Float64(env_ns) / 1e6 / Float64(FILL_STEPS)
        var rec_ms = Float64(rec_ns) / 1e6 / Float64(FILL_STEPS)
        print("  env.step_obs   ", env_ms, "ms/env step")
        print("  record (H2D)   ", rec_ms, "ms/env step")

        # ── select_action (the acting path: B=1 enc + posterior + policy) ──
        for i in range(OBS):
            obsbuf[i] = obs[i]
        agent.select_action(obsbuf, actbuf, explore=True)  # warm kernels
        ctx.synchronize()
        var a0 = perf_counter_ns()
        for _r in range(PROF_ACT):
            agent.select_action(obsbuf, actbuf, explore=True)
        ctx.synchronize()
        var act_ms = Float64(perf_counter_ns() - a0) / 1e6 / Float64(PROF_ACT)
        print("  select_action  ", act_ms, "ms/env step (eager, B=1)")

        # ── train-step warmup (compile+alloc+lr ramp), then profiled steps ──
        print("warmup:", WARMUP_TRAIN, "eager train steps...")
        for _w in range(WARMUP_TRAIN):
            _ = agent.train_step(want_diag=False)
        var acc = InlineArray[Float64, 5](fill=0.0)
        print("profiling", PROF_TRAIN, "train steps (synced sections)...")
        for _p in range(PROF_TRAIN):
            var s = agent.trainer.profile_sections()
            for k in range(5):
                acc[k] += s[k]
        for k in range(5):
            acc[k] /= Float64(PROF_TRAIN)

        print("-" * 70)
        print("TRAIN STEP breakdown (ms, mean of", PROF_TRAIN, "):")
        print("  prologue (draw+noise) ", acc[0], " (", 100.0 * acc[0] / acc[4], "% )")
        print("  WM-BPTT               ", acc[1], " (", 100.0 * acc[1] / acc[4], "% )")
        print("  core→imagine sync     ", acc[2], " (", 100.0 * acc[2] / acc[4], "% )")
        print("  imagination AC        ", acc[3], " (", 100.0 * acc[3] / acc[4], "% )")
        print("  TOTAL                 ", acc[4])
        var per_env = acc[4] / Float64(TRAIN_EVERY) + act_ms + env_ms + rec_ms
        print("-" * 70)
        print("projected per ENV step:", per_env, "ms  →",
              per_env, "s / 1000 env steps")
        print("  (train", acc[4] / Float64(TRAIN_EVERY), "+ act", act_ms,
              "+ env", env_ms, "+ record", rec_ms, ")")
        print("observed run pace ≈ 820 s / 1000 env steps — the gap, if any,")
        print("is logger/eval/diag overhead outside these sections.")
        print("=" * 70)
        obsbuf.free()
        actbuf.free()
        _ = env^
        _ = agent^
