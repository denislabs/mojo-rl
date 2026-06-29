"""DreamerV3 on CarRacing PIXEL observations — continuous control (P4).

End-to-end DreamerV3 world-model + actor-critic training on the faithful
multi-body `CarRacingMB` env with 96×96 pixel observations and CONTINUOUS
actions, via the CNN encoder / transposed-conv decoder (nets_cnn.mojo).

Path: **hybrid CPU-env / GPU-agent** at a single env — the CarRacingMB env is
stepped on the host (faithful closed-loop track → transfer-safe, non-cheatable),
while the DreamerV3 agent (encoder/decoder/RSSM/AC) runs on the GPU and trains
on B-sized minibatches drawn from the sequence replay. (#envs=1; a batched-CPU
driver for more throughput is a follow-up — P4b.)

Observation: 4×96×96 grayscale frame stack (OBS = 36864), values in [0,1].
Action: 3-D [steering, gas, brake]; the agent acts in normalized [-1,1] and the
env remaps gas/brake [-1,1]→[0,1] (Gymnasium convention) inside
`step_continuous_vec`, so ACTION_SCALE = 1.0 (no driver scaling).

⚠️ SCAFFOLD — convergence/tuning is P5 (open). Likely levers, in order:
  - Reward/value support: CarRacing rewards (≈ +1000/N per tile, −0.1/frame,
    −100 off-field) are larger/peakier than Pendulum; BINS / twohot grid and the
    return normalizer may need tuning so tile rewards are representable.
  - Pixel recon loss: currently SymlogMSE on [0,1] pixels (monotonic, works) —
    a plain-MSE / [-0.5,0.5] pixel decoder is more standard (P5).
  - Per-layer norm in the conv stack (DreamerV3 uses LayerNorm; v1 omits it).
  - Replay memory: pixel obs are big — fp32 replay is CAP×36864×4 B. Keep CAP
    modest or add a uint8 replay (follow-up). CAP below = GPU-affordable-ish.
  - Use NVIDIA for real runs (Apple Metal is for smoke/iteration).

Run (NVIDIA):
  pixi run -e nvidia mojo run -I . examples/car_racing/dreamerv3_car_racing_pixel_training.mojo
Smoke (Apple): use tests/nn/test_dreamerv3_carracing_pixel_smoke.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)
from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB

# ── config ──────────────────────────────────────────────────────────────
comptime C = 4            # 4-frame grayscale stack = conv input channels
comptime IMG = 96         # 96×96 (16-divisible → conv minres 6)
comptime BASE = 48        # conv base width (channels BASE·{1,2,4,8})
comptime OBS = C * IMG * IMG  # 36864
comptime ACT = 3          # steering, gas, brake
comptime DETER = 512
comptime H = 256
comptime STOCH = 32
comptime CLASSES = 32
comptime BLOCKS = 8
comptime TOKEN = 1024     # encoder output (flattened conv → Linear → tokens)
comptime DEC_U = 1024     # unused by the CNN decoder (BASE drives it)
comptime HU = 256
comptime VU = 256
comptime PU = 256
comptime BINS = 255
comptime B = 16
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 50_000     # pixel replay: CAP×36864×4 B ≈ 7.4 GB — tune to HW

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN, SwishOp]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE, SwishOp]

comptime Ag = DreamerV3Agent[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU, PU,
    BINS, B, T, T_IMAG, CAP, False, ENC, DEC,
]
comptime Env = CarRacingMB[DT, True, IMG]

comptime TOTAL_STEPS = 1_000_000
comptime LEARN_START = 1024
comptime TRAIN_EVERY = 4
comptime EVAL_EVERY = 5_000
comptime PRINT_EVERY = 200       # frequent heartbeat (step, WM/AC, throughput)
comptime WARMUP_PRINT = 200      # heartbeat during the no-train warmup phase
comptime EVAL_EPISODES = 3
comptime EP_LEN = 1000     # CarRacing max_steps
comptime ACTION_SCALE = Scalar[DT](1.0)  # env remaps gas/brake internally


def _greedy_eval(mut ag: Ag) raises -> Scalar[DT]:
    var env = Env()
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
            var a = List[Scalar[DT]]()
            for j in range(ACT):
                a.append(ACTION_SCALE * actbuf[j])
            var r = env.step_continuous_vec[DT](a)
            total += r[1]
            o = r[0].copy()
            if r[2]:
                break
    obsbuf.free()
    actbuf.free()
    return total / Scalar[DT](EVAL_EPISODES)


def main() raises:
    print("=" * 70)
    print("DreamerV3 CarRacing PIXEL (continuous) —", TOTAL_STEPS, "steps")
    print("  OBS =", OBS, "(", C, "x", IMG, "x", IMG, ")  ACT =", ACT)
    print("=" * 70)
    seed(42)
    var ctx = DeviceContext()
    var env = Env()
    var ag = Ag.make(
        ctx=ctx, lr=Scalar[DT](4e-5), learning_starts=LEARN_START,
        action_scale=ACTION_SCALE, actent=Scalar[DT](3e-4), slowtar=True,
    )

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

    print("  warming up", LEARN_START, "steps (CPU render + GPU record, no",
          "train) — heavy config, first train_steps are slow...")
    var t_mark = perf_counter_ns()
    var step_mark = 0
    for step in range(TOTAL_STEPS):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            for j in range(ACT):
                actbuf[j] = Scalar[DT](random_float64() * 2.0 - 1.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var a = List[Scalar[DT]]()
        for j in range(ACT):
            a.append(ACTION_SCALE * actbuf[j])
        var res = env.step_continuous_vec[DT](a)
        ag.record(
            obsbuf, actbuf, res[1],
            Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
            ag.reset_belief()

        # heartbeat during the no-train warmup so it's visibly alive
        if step < LEARN_START and step > 0 and step % WARMUP_PRINT == 0:
            var dt = Float64(perf_counter_ns() - t_mark) / 1e9
            var rate = Float64(step - step_mark) / dt if dt > 0 else 0.0
            print("  [warmup]", step, "/", LEARN_START, " (", rate,
                  "env-steps/s)")
            t_mark = perf_counter_ns()
            step_mark = step
        if step == LEARN_START:
            print("  warmup done — training starts (train_step every",
                  TRAIN_EVERY, "steps); each is a", T, "-step CNN BPTT, slow.")
            t_mark = perf_counter_ns()
            step_mark = step

        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            # diagnostics (WM/AC loss readout) only on heartbeat/eval steps —
            # want_diag=True every step adds a per-step D2H sync (slower).
            var wd = (step % PRINT_EVERY == 0) or (step % EVAL_EVERY == 0)
            _ = ag.train_step(want_diag=wd)

        # frequent heartbeat: confirms progress + measures train throughput
        if step >= LEARN_START and step % PRINT_EVERY == 0:
            var dt = Float64(perf_counter_ns() - t_mark) / 1e9
            var rate = Float64(step - step_mark) / dt if dt > 0 else 0.0
            # WM/AC + policy-learning probes: real_rew vs rew_pred (reward head
            # fit), ret_m/val_m (critic tracking the imagined return), pstd
            # (actor exploration). If the policy is learning, rew_pred tracks
            # real_rew and val_m tracks ret_m as training proceeds.
            print(
                "  step", step, " WM=", ag.last_wm_loss(), " AC=",
                ag.last_ac_loss(), " | real_rew=", ag.dbg_real_rew(),
                " rew_pred=", ag.dbg_rew_pred(), " ret_m=", ag.dbg_ret_mean(),
                " val_m=", ag.dbg_val_mean(), " pstd=", ag.dbg_pstd(),
                " (", rate, "steps/s)",
            )
            t_mark = perf_counter_ns()
            step_mark = step

        if step > 0 and step % EVAL_EVERY == 0:
            var ret = _greedy_eval(ag)
            print("  step", step, " eval_ret=", ret)
            obs = env.reset_obs_list()
            ag.reset_belief()
            t_mark = perf_counter_ns()
            step_mark = step

    var fe = _greedy_eval(ag)
    print("=" * 70)
    print("FINAL mean_ret(", EVAL_EPISODES, ") =", fe)
    print("=" * 70)
    obsbuf.free()
    actbuf.free()
