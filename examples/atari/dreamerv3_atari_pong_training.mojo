"""DreamerV3 on Atari Pong (GPU) via the `DreamerV3Agent` facade — discrete,
pixel observations.

The discrete-pixel corner of the DreamerV3 matrix (CartPole = discrete+vector,
CarRacing = continuous+pixel, Pong = discrete+pixel) and the canonical DreamerV3
Atari benchmark. The world model + actor-critic train on-device
(`train_target="gpu"`, the discrete device-resident AC `_ac_gpu_disc`); the CPU
6502 emulator (`AtariEnv`) steps the ROM and obs are marshalled H2D inside
`select_action`. The env is NOT the bottleneck — DreamerV3 is sample-efficient
(replays each frame ~train_ratio×), so the GPU train_step dominates.

Observation: **4×96×96 grayscale frame STACK** (OBS = 36864, values in [0,1]) —
`AtariEnv[OBS_MODE=4]`. NOTE: the DreamerV3 reference uses a single frame (the
RSSM is meant to carry motion), but a WM-overfit diagnostic showed our prior
can't *generalize* the fast 1-2px ball's velocity from single frames across
varied Pong (imagination collapsed in ~2 open-loop steps, even though it
overfit a single episode fine → machinery correct, signal too weak). Stacking 4
frames puts velocity directly in the obs, matching what worked for CarRacing.
The CNN encoder centers to [-0.5, 0.5] (CenterHalfOp) and reconstructs with
sigmoid + plain MSE (RECON_SIGMOID=True). Action: Pong minimal set (6 discrete).
Machado protocol:
sticky actions (0.25) + random no-op starts (30); frame-skip=4 + max-pool is done
inside the env, so no agent-side action repeat. Reward NOT clipped (symlog/twohot
handle the ±1 scale).

The facade owns the whole loop (warmup → select_action → step → record
(+record_terminal on done) → train_step → periodic greedy eval, which SAMPLES the
actor) and logs the SAME KNOWN_GROUPS metrics as CartPole/CarRacing.

⚠️ Requires `roms/pong.bin` (`pixi run setup-roms`). Convergence/tuning is P2 —
run on NVIDIA.

Run:
    pixi run -e apple  mojo run -I . examples/atari/dreamerv3_atari_pong_training.mojo
    pixi run -e nvidia mojo run -I . examples/atari/dreamerv3_atari_pong_training.mojo
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)
from mojo_rl.envs.atari import AtariEnv
from mojo_rl.envs.atari.games.registry import AtariGame

# =============================================================================
# Architecture
# =============================================================================
comptime C = 4  # 4-frame grayscale stack (velocity in the obs; see docstring)
comptime IMG = 96  # 96×96 (16-divisible → conv minres 6)
comptime BASE = 48  # conv base width (channels BASE·{1,2,4,8})
comptime OBS = C * IMG * IMG  # 36864
comptime ACT = 6  # Pong minimal action set (NOOP/FIRE/RIGHT/LEFT/RIGHTFIRE/LEFTFIRE)
comptime DETER = 2048
comptime H = 256
comptime STOCH = 32
comptime CLASSES = 32
comptime BLOCKS = 8
comptime TOKEN = 1024  # encoder output (flattened conv → Linear → tokens)
comptime DEC_U = 1024  # unused by the CNN decoder (BASE drives it)
comptime HU = 256
comptime VU = 256
comptime PU = 256
comptime BINS = 255
comptime B = 16
comptime T = 32  # training-sequence length (BPTT horizon). Reference uses 64;
# 16 was the first-run value — 32 doubles the deter-chain credit horizon at 2×
# WM-step cost. Raise to 64 if step time allows.
comptime T_IMAG = 15
# GPU-resident pixel replay: CAP×OBS×4 B of VRAM (≈7.4 GB either way below).
# Bigger CAP protects rare-state coverage (paddle pinned at an edge, ball-at-
# paddle events) from circular eviction — the reference keeps 5M transitions
# uniform. At C=1 the same VRAM budget affords 4× the horizon.
comptime CAP = 200_000 if C == 1 else 50_000

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN, SwishOp]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE, SwishOp]

comptime Ag = DreamerV3Agent[
    "gpu",
    OBS,
    ACT,
    DETER,
    H,
    STOCH,
    CLASSES,
    BLOCKS,
    TOKEN,
    DEC_U,
    HU,
    VU,
    PU,
    BINS,
    B,
    T,
    T_IMAG,
    CAP,
    True,  # DISCRETE=True (categorical actor)
    ENC,
    DEC,
    True,  # RECON_SIGMOID — reference pixel recon (sigmoid + plain MSE on [0,1])
]
comptime Env = AtariEnv[4, DT]  # OBS_MODE=4 (gray-96 4-frame stack)

comptime NUM_STEPS = 500_000  # agent decisions (each = 4 ROM frames = 2M frames)
comptime LEARN_START = 1024
# Replay ratio = B·T / TRAIN_EVERY replayed frames per env step. The reference
# trains at ratio 32; with T=32 the old TRAIN_EVERY=4 was ratio 128 — 4× the
# reference, which OVERFITS the continue/value heads on scarce data (Pong shows
# ~5 episode terminals in the first 20k steps). The observed failure chain at
# ratio 128: con head hallucinates terminals on imagined (OOD) states
# (imag_con_min → ~0.001) → a fake terminal truncates the (legitimately
# negative, γ=0.997 fixed point ≈ rew/(1−γ)) bootstrap → big positive
# advantage for whichever action reaches those dream states (adv_gap 0.002 →
# 0.1) → policy collapses (entropy 1.79 → 0.08 by 20k). TRAIN_EVERY=16
# restores the reference ratio 32.
comptime TRAIN_EVERY = 16
comptime LOG_EVERY = 1000  # WM/AC loss curves (cheap; no greedy eval) — frequent
comptime EVAL_EVERY = 5000  # greedy eval + episode returns (expensive, ~3 min)
comptime EVAL_EPISODES = 5
comptime EP_LEN = 2000  # eval-episode cap (agent decisions)
comptime CHECKPOINT_EVERY = 50_000
comptime CHECKPOINT_PATH = "dreamerv3_atari_pong_gpu.ckpt"


def main() raises:
    seed(42)
    print("=" * 70)
    print("DreamerV3 (facade) — Atari Pong PIXEL GPU (discrete)")
    print("=" * 70)
    print("  OBS / ACT          =", OBS, "(", C, "x", IMG, "x", IMG, ") /", ACT)
    print("  DETER/STOCH/CLASSES=", DETER, "/", STOCH, "/", CLASSES)
    print("  BASE / T / T_IMAG  =", BASE, "/", T, "/", T_IMAG)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Logger (remote; same KNOWN_GROUPS metrics as the other envs) ──
        var env_vars = load_dotenv()
        var logger = RemoteLogger(
            server_url=env_vars.get("RL_MONITOR_URL", ""),
            run_name="DreamerV3 Atari Pong PIXEL (GPU, discrete)",
            buffer_size=200,
            api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
        )
        logger.set_config("algorithm", "DreamerV3")
        logger.set_config("env", "AtariPong")
        logger.set_config("target", "gpu")
        logger.set_config("t_imag", String(T_IMAG))
        var logger_ptr = UnsafePointer(to=logger).as_unsafe_any_origin()

        # ─── Agent (GPU) + env (CPU 6502; obs marshalled H2D in select_action) ──
        var agent = Ag.make(
            ctx=ctx,
            lr=Scalar[DT](4e-5),
            learning_starts=LEARN_START,
            warmup_steps=1000,
            actent=Scalar[DT](3e-4),
            slowtar=True,
        )
        # Machado protocol: sticky actions 0.25 + random no-op starts 30; reward
        # unclipped (symlog/twohot). Loads roms/pong.bin.
        var env = Env(AtariGame.PONG, sticky_prob=0.25, noop_max=30)

        # ─── Single train() call — auto-eval + auto-log + auto-checkpoint ──
        print("Starting GPU training (heavy pixel config; warmup is slow)...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        var final_ret = agent.train_single[
            Env, L=RemoteLogger, USE_TRAIN_CUDA_GRAPH=True
        ](
            env,
            NUM_STEPS,
            learn_start=LEARN_START,
            train_every=TRAIN_EVERY,
            eval_every=EVAL_EVERY,
            eval_episodes=EVAL_EPISODES,
            ep_len=EP_LEN,
            print_every=LOG_EVERY,
            log_every=LOG_EVERY,
            verbose=True,
            logger=logger_ptr,
            checkpoint_path=CHECKPOINT_PATH,
            checkpoint_every=CHECKPOINT_EVERY,
        )
        var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
        logger.close()
        _ = logger  # lifetime extender for logger_ptr

        # ─── Summary ─────────────────────────────────────────────────────
        print("-" * 70)
        print("=" * 70)
        print("Training complete")
        print("  total env_steps   =", NUM_STEPS)
        print("  elapsed           =", elapsed_s, "s")
        print("  FINAL mean_ret(", EVAL_EPISODES, ")  =", final_ret)
        print("  remote points sent=", logger.total_logged())
        print("  (Pong: -21 = shutout loss, +21 = shutout win)")
        print("=" * 70)
