"""DreamerV3 on CarRacing PIXEL observations (GPU) via the `DreamerV3Agent`
facade — continuous control.

Continuous counterpart of `examples/cartpole/cartpole_dreamerv3_training_gpu.mojo`
(which is discrete via `train_single`). The world model + actor-critic train
on-device (`train_target="gpu"`, the device-resident continuous AC `_ac_gpu_cont`);
the faithful multi-body `CarRacingMB` env steps on CPU (transfer-safe, non-
cheatable closed-loop track) and obs are marshalled H2D inside `select_action`.

Observation: 4×96×96 grayscale frame stack (OBS = 36864), values in [0,1], via
the CNN encoder + transposed-conv decoder (nets_cnn.mojo). Action: 3-D
[steering, gas, brake]; the agent acts in normalized [-1,1] and the env remaps
gas/brake [-1,1]→[0,1] (Gymnasium), so `action_scale=1.0` (no driver scaling).

The facade owns the whole loop (warmup → select_action → step → record
(+record_terminal on done) → train_step → periodic greedy eval), and logs the
SAME KNOWN_GROUPS metrics as the discrete `train_single` (wm/obs/reward/continue
losses, dyn/rep KL, value/policy loss, imagined returns, episode rewards) so the
curves overlay for parity checks across optimizations.

⚠️ Convergence/tuning is P5 (open): reward/value support, pixel recon loss,
conv norm, uint8 replay. Use NVIDIA for real runs.

Run:
    pixi run -e apple  mojo run -I . examples/car_racing/dreamerv3_car_racing_pixel_training.mojo
    pixi run -e nvidia mojo run -I . examples/car_racing/dreamerv3_car_racing_pixel_training.mojo
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
from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB

# =============================================================================
# Architecture
# =============================================================================
comptime C = 4  # 4-frame grayscale stack = conv input channels
comptime IMG = 96  # 96×96 (16-divisible → conv minres 6)
comptime BASE = 48  # conv base width (channels BASE·{1,2,4,8})
comptime OBS = C * IMG * IMG  # 36864
comptime ACT = 3  # steering, gas, brake
comptime DETER = 2048  # size25m preset (was 512); bigger recurrent dynamics capacity to drive dyn_kl down + sharpen open-loop imagination
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
comptime CAP = 50_000  # pixel replay: CAP×36864×4 B ≈ 7.4 GB — tune to HW

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
    False,
    ENC,
    DEC,  # DISCRETE=False (continuous)
    True,  # RECON_SIGMOID — reference pixel recon (sigmoid + plain MSE on [0,1])
]
comptime Env = CarRacingMB[DT, True, IMG]  # PIXEL_OBS=True, PIX_RES=96

comptime NUM_STEPS = 1_000_000
comptime LEARN_START = 1024
comptime TRAIN_EVERY = 4
# Action repeat: hold each agent decision for FRAME_REPEAT env frames (reward
# summed) — the DreamerV3 reference ActionRepeat wrapper (4 for all its pixel
# suites: atari/atari100k/dmlab). For CarRacing this 2-4× extends imagination's
# real-time reach + steadies steering through turns. 2 = finer steering control;
# bump to 4 to match the reference pixel value. NUM_STEPS counts AGENT decisions
# (each = FRAME_REPEAT env frames).
comptime FRAME_REPEAT = 2
comptime LOG_EVERY = 1000  # WM/AC loss curves (cheap; no greedy eval) — frequent
# early data points (~every few min at this heavy cfg)
comptime EVAL_EVERY = 5000  # greedy eval + episode returns (expensive)
comptime EVAL_EPISODES = 3
comptime EP_LEN = 1000  # CarRacing max_steps
comptime CHECKPOINT_EVERY = 50_000
comptime CHECKPOINT_PATH = "dreamerv3_carracing_pixel_gpu.ckpt"


def main() raises:
    seed(42)
    print("=" * 70)
    print("DreamerV3 (facade) — CarRacing PIXEL GPU (continuous)")
    print("=" * 70)
    print("  OBS / ACT          =", OBS, "(", C, "x", IMG, "x", IMG, ") /", ACT)
    print("  DETER/STOCH/CLASSES=", DETER, "/", STOCH, "/", CLASSES)
    print("  BASE / T / T_IMAG  =", BASE, "/", T, "/", T_IMAG)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Logger (remote; same KNOWN_GROUPS metrics as the discrete path) ──
        var env_vars = load_dotenv()
        var logger = RemoteLogger(
            server_url=env_vars.get("RL_MONITOR_URL", ""),
            run_name="DreamerV3 CarRacing PIXEL (GPU, continuous)",
            buffer_size=200,
            api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
        )
        logger.set_config("algorithm", "DreamerV3")
        logger.set_config("env", "CarRacingPixel")
        logger.set_config("target", "gpu")
        logger.set_config("t_imag", String(T_IMAG))
        var logger_ptr = UnsafePointer(to=logger).as_unsafe_any_origin()

        # ─── Agent (GPU) + env (CPU; obs marshalled H2D in select_action) ──
        var agent = Ag.make(
            ctx=ctx,
            lr=Scalar[DT](4e-5),
            learning_starts=LEARN_START,
            warmup_steps=500,
            action_scale=Scalar[DT](1.0),  # env remaps gas/brake internally
            actent=Scalar[DT](3e-4),
            slowtar=True,
        )
        var env = Env()

        # ─── Single train() call — auto-eval + auto-log + auto-checkpoint ──
        print("Starting GPU training (heavy pixel config; warmup is slow)...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        var final_ret = agent.train_continuous[
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
            frame_repeat=FRAME_REPEAT,
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
        print("=" * 70)
