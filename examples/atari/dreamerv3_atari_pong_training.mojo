"""DreamerV3 on Atari Pong (GPU) via the `DreamerV3Agent` facade — discrete,
pixel observations.

The discrete-pixel corner of the DreamerV3 matrix (CartPole = discrete+vector,
CarRacing = continuous+pixel, Pong = discrete+pixel) and the canonical DreamerV3
Atari benchmark. The world model + actor-critic train on-device
(`train_target="gpu"`, the discrete device-resident AC `_ac_gpu_disc`); the CPU
6502 emulator (`AtariEnv`) steps the ROM and obs are marshalled H2D inside
`select_action`. The env is NOT the bottleneck — DreamerV3 is sample-efficient
(replays each frame ~train_ratio×), so the GPU train_step dominates.

Observation: **single 96×96 grayscale frame** (OBS = 9216, values in [0,1]) —
`AtariEnv[OBS_MODE=3]`, the DreamerV3 Atari preprocessing (NO frame stacking; the
RSSM carries motion). The CNN encoder centers it to [-0.5, 0.5] (CenterHalfOp)
and reconstructs with sigmoid + plain MSE (RECON_SIGMOID=True), both
reference-faithful. Action: Pong minimal set (6 discrete). Machado protocol:
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
comptime C = 1  # single grayscale frame (DreamerV3 does NOT frame-stack)
comptime IMG = 96  # 96×96 (16-divisible → conv minres 6)
comptime BASE = 48  # conv base width (channels BASE·{1,2,4,8})
comptime OBS = C * IMG * IMG  # 9216
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
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 100_000  # pixel replay: CAP×9216×4 B ≈ 3.7 GB — tune to HW

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
comptime Env = AtariEnv[3, DT]  # OBS_MODE=3 (gray-96 single frame)

comptime NUM_STEPS = 500_000  # agent decisions (each = 4 ROM frames = 2M frames)
comptime LEARN_START = 1024
comptime TRAIN_EVERY = 4
comptime EVAL_EVERY = 5000
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
            print_every=EVAL_EVERY,
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
