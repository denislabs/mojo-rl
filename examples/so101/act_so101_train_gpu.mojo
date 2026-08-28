# +--------------------------------------------------------------------------+ #
# | ACT on the SO-ARM101 — GPU training
# +--------------------------------------------------------------------------+ #
"""Train ACT on the LeRobot v3 dataset, on GPU.

    pixi run python tools/act/lerobot_v3_to_store.py \\
        --repo DenisLabs/record-test_20260828_092736 --height 240 --width 320

    export ACT_STORE=~/.cache/mojo_rl/act_so101/\\
DenisLabs__record-test_20260828_092736_240x320.h5

    pixi run -e nvidia mojo build -I . -o /tmp/act_train_gpu \\
        examples/so101/act_so101_train_gpu.mojo && /tmp/act_train_gpu

⚠ **Run it from the project root.** `mojo_rl/io/hdf5` resolves libhdf5 through
a path relative to the working directory (`.pixi/envs/<env>/lib/`), so the
binary aborts with `symbol not found: H5PLprepend` anywhere else. It also reads
`.env` from the working directory.

`ACT_STEPS` overrides the step count WITHOUT a rebuild — the graph type takes
minutes to compile, so "run it a bit longer" must not mean "build it again":

    ACT_STEPS=200000 /tmp/act_train_gpu

On Apple add `-Xlinker -ld_classic`: the fully-expanded graph type mangles to a
symbol longer than the new linker accepts. Healthy source, toolchain limit;
`mojo run` JITs and never invokes ld. NVIDIA needs no flag.

## What this configuration is, and what it is not

| | here | paper |
|---|---|---|
| chunk `K` | 60 | 100 |
| `hidden_dim` | 256 | 512 |
| `dim_feedforward` | 1024 | 3200 |
| encoder layers | 4 | 4 |
| decoder layers | 1 | 7 (output-equivalent — see `config.mojo`) |
| heads / latent | 8 / 32 | 8 / 32 |
| batch | 16 | 8 |
| images | 2 x 240x320 | 4 x 480x640 |

**`K = 60` is the paper's horizon, not a reduction.** ALOHA runs at 50 Hz and
chunks 100 steps — 2.0 seconds. This data is 30 fps, so the same two seconds is
60 frames. Copying the number 100 would have copied the wrong quantity.

`enc_layers` matches the paper. `hidden_dim` and `dim_feedforward` do not, and
that is the honest gap: the graph type expands with every one of these and
COMPILE time, not step time, is what bounds this file — the ACT GPU gate exists
in its stub-backbone form because a full-backbone build never finished on CUDA.
Raise them if a build you have measured says you can.

`batch = 16` rather than the paper's 8: 15,447 frames at 2 cameras is 32 images
per step, which is still small for a 5090, and halving the step count per epoch
is free wall-clock. It is a lever, not a finding.

## Two things this run does that the CPU example does not

**Validation is DETERMINISTIC.** The sampler's RNG is pinned to a fixed seed
around every validation pass, so each one draws the SAME batches. Otherwise
`best_val` selects on a lucky draw — over hundreds of validations from only 10
held-out episodes, the minimum of a noisy estimate is mostly noise, and the
checkpoint it keeps is not the best model.

**A `last` checkpoint is written at every validation**, beside the `best` one.
A multi-hour run that is killed, or an OOM at hour three, otherwise leaves
nothing to resume from or evaluate.

## Metrics

Streams to the monitoring server named by `RL_MONITOR_URL` / `RL_MONITOR_API_KEY`
in `.env` (`mojo_rl/core/dotenv.mojo`), the same path
`examples/half_cheetah/sac_half_cheetah_training_gpu.mojo` uses. With neither
set the logger is inert and the run is unaffected — training must not depend on
a monitoring server being reachable. `ACT_NO_MONITOR=1` forces it inert with
the keys present, which is what a smoke run should use so it does not land in
the dashboard beside a real one.

    train/l1  train/kl  train/loss  train/grad_norm  train/epoch   every 50 steps
    val/l1  val/kl  best/val_l1  perf/s_per_step                   every 1000

Training metrics are the MEAN over their window, not one batch's value: at
batch 16 a single step's L1 is noisy enough to hide the trend, and 100,000
individual points is not a curve anyone reads. The buffer is flushed at every
validation rather than only when full — the point of streaming a multi-hour run
is watching it while it runs.

`train/l1` is the quantity to watch against `val/l1`; `train/grad_norm` is the
one that tells you if the run is about to diverge, and it is worth having
precisely because gradient clipping is OFF (below).

## ⚠ `max_grad_norm = 0.0`

The clip walks every gradient slab through the host (`trainer.mojo:_SumSq`) —
a synchronisation point per parameter, the wrong shape for a hot loop. Zero is
also what the reference effectively does: it parses `--clip_max_norm 0.1` and
never applies it.

## What to expect

40 training episodes / 10 held out, ~12,400 training frames, a from-scratch
ResNet18, no augmentation. Validation L1 should fall considerably further than
the 5-episode store allowed before it turns; where it turns is the number worth
having, and the `best` checkpoint is what `act_so101_openloop_eval.mojo` should
be pointed at. The paper used 50 demonstrations per task, so this is the first
run at the data scale it assumes.
"""
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.act.config import (
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from mojo_rl.deep_agents.act.data import ACTDataset
from mojo_rl.deep_agents.act.trainer import ACTTrainer
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger

from std.python import Python, PythonObject


# ── shape (must match the store; see the header) ─────────────────────────
comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W

# ── model (see the table in the header) ──────────────────────────────────
comptime K = 60  # 2.0 s at 30 fps — the paper's horizon, not its frame count
comptime DIM = 256  # paper: 512
comptime HEADS = 8  # paper: 8
comptime FF = 1024  # paper: 3200
comptime LATENT = 32  # paper: 32 (unchanged — it is small already)
comptime N_ENC = 4  # paper: 4
comptime N_DEC = 1  # paper: 7, but output-equivalent to 1 (see config.mojo)
comptime BATCH = 16  # paper: 8

comptime DEFAULT_STEPS = 100000
"""~100 epochs over 40 episodes at batch 16 (966 steps/epoch). Override with
`ACT_STEPS`; the graph takes minutes to compile and extending a run must not
require rebuilding it."""
comptime VAL_EVERY = 1000
comptime VAL_BATCHES = 16
"""256 validation samples per pass, from a pinned RNG — see the header. Four
batches was fine for a 2000-step smoke and is far too noisy to select a
checkpoint from over a hundred passes."""
comptime VAL_SEED: UInt64 = 0x5DEECE66D
comptime LR = 1e-4
"""Paper: 1e-5, with a pretrained backbone it barely moves. This backbone is
random at step 0 and has to learn vision from scratch, so it needs the higher
rate — the same reasoning as the single-lr deviation in `ACT_PORT.md`."""
comptime KL_WEIGHT = 10.0  # paper: 10

comptime LOG_EVERY = 50
"""Training metrics stream as the MEAN over this window, not as one batch's
value. At batch 16 a single step's L1 is noisy enough to hide the trend, and
100,000 individual points is not a curve anyone reads — 2,000 windowed ones
is. Validation goes at `VAL_EVERY`, unaveraged: it is already a mean over
`VAL_BATCHES` fixed batches."""

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH, 0.1, "gpu",
]
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W


def store_path() raises -> String:
    """`$ACT_STORE` if set, else the recording the header names.

    The default is a specific recording, not a pattern: an example that
    silently picked up whichever store happened to be newest in the cache
    would report numbers nobody could attribute to a dataset. Point
    `ACT_STORE` at another store to train on it.
    """
    var os = Python.import_module("os")
    var env = String(
        os.environ.get(PythonObject("ACT_STORE"), PythonObject(""))
    )
    if env.byte_length() > 0:
        return env
    var home = String(os.path.expanduser(PythonObject("~")))
    return (
        home
        + "/.cache/mojo_rl/act_so101/"
        + "DenisLabs__record-test_20260825_094319_"
        + String(IMG_H) + "x" + String(IMG_W) + ".h5"
    )


def main() raises:
    var path = store_path()
    var os = Python.import_module("os")
    if not Bool(os.path.exists(PythonObject(path))):
        print("MISSING STORE: " + path)
        print("build it with tools/act/lerobot_v3_to_store.py — see the header")
        raise Error("store not found")

    var steps = Int(DEFAULT_STEPS)
    var env_steps = String(
        os.environ.get(PythonObject("ACT_STEPS"), PythonObject(""))
    )
    if env_steps.byte_length() > 0:
        steps = Int(env_steps)
        if steps < 1:
            raise Error("ACT_STEPS must be >= 1, got " + env_steps)

    var ctx = DeviceContext()
    print("ACT / SO-ARM101 — GPU training")
    print("  device  " + String(ctx.name()))
    print("  store   " + path)
    print(
        "  model   K=" + String(K) + " dim=" + String(DIM)
        + " heads=" + String(HEADS) + " ff=" + String(FF)
        + " enc=" + String(N_ENC) + " dec=" + String(N_DEC)
    )
    print(
        "  data    " + String(N_CAM) + " cameras at " + String(IMG_H) + "x"
        + String(IMG_W) + ", batch " + String(BATCH)
    )

    var ds = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](String(path), seed=7)
    print(
        "  split   " + String(len(ds.train_eps)) + " train / "
        + String(len(ds.val_eps)) + " val episodes of "
        + String(ds.n_episodes())
    )
    print("")

    # ── remote metrics ───────────────────────────────────────────────────
    # Configured from `.env` (`RL_MONITOR_URL`, `RL_MONITOR_API_KEY`). With
    # neither set the logger is inert and the run is unaffected — a training
    # run must not depend on a monitoring server being reachable.
    # `ACT_NO_MONITOR=1` forces it inert. A three-step smoke should not show
    # up in the dashboard next to a six-hour run, and blanking the shared
    # `.env` to achieve that is how a real run silently loses its metrics.
    var env_vars = load_dotenv()
    var no_monitor = String(
        os.environ.get(PythonObject("ACT_NO_MONITOR"), PythonObject(""))
    )
    var monitor_url = (
        String("") if no_monitor.byte_length() > 0
        else env_vars.get("RL_MONITOR_URL", "")
    )
    var logger = RemoteLogger(
        server_url=monitor_url,
        run_name="ACT SO-ARM101 (GPU)",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("algorithm", "ACT")
    logger.set_config("robot", "SO-ARM101")
    logger.set_config("target", "gpu")
    logger.set_config("device", String(ctx.name()))
    logger.set_config("store", path)
    logger.set_config("chunk_k", String(K))
    logger.set_config("hidden_dim", String(DIM))
    logger.set_config("dim_feedforward", String(FF))
    logger.set_config("heads", String(HEADS))
    logger.set_config("latent", String(LATENT))
    logger.set_config("enc_layers", String(N_ENC))
    logger.set_config("dec_layers", String(N_DEC))
    logger.set_config("batch", String(BATCH))
    logger.set_config("lr", String(LR))
    logger.set_config("kl_weight", String(KL_WEIGHT))
    logger.set_config("cameras", String(N_CAM))
    logger.set_config("image", String(IMG_H) + "x" + String(IMG_W))
    logger.set_config("steps", String(steps))
    logger.set_config("train_episodes", String(len(ds.train_eps)))
    logger.set_config("val_episodes", String(len(ds.val_eps)))
    print(
        "  metrics " + (
            "streaming to " + monitor_url if logger.is_active()
            else (
                "OFF (ACT_NO_MONITOR)" if no_monitor.byte_length() > 0
                else "local only (set RL_MONITOR_URL in .env)"
            )
        )
    )

    var tr = T.make(
        lr=Scalar[DT](LR),
        kl_weight=Scalar[DT](KL_WEIGHT),
        max_grad_norm=Scalar[DT](0.0),  # see the header
        ctx=ctx,
    )

    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)

    var best_val = Float64(1e30)
    var best_step = -1
    var best_ckpt = String("/tmp/act_so101_best_gpu.ckpt")
    var last_ckpt = String("/tmp/act_so101_last_gpu.ckpt")

    var train_frames = 0
    for i in range(len(ds.train_eps)):
        train_frames += ds.store.episodes.length_of(ds.train_eps[i])
    var steps_per_epoch = train_frames // BATCH
    if steps_per_epoch < 1:
        steps_per_epoch = 1
    print(
        "  run     " + String(steps) + " steps, " + String(steps_per_epoch)
        + " per epoch (" + String(train_frames) + " train frames)"
    )
    print("")

    var t0 = perf_counter_ns()
    var t_mark = t0
    var s_mark = 0

    # Windowed training means — see LOG_EVERY.
    var acc_l1 = Float64(0.0)
    var acc_kl = Float64(0.0)
    var acc_loss = Float64(0.0)
    var acc_gn = Float64(0.0)
    var acc_n = 0

    var names = List[String]()
    names.append(String("train/l1"))
    names.append(String("train/kl"))
    names.append(String("train/loss"))
    names.append(String("train/grad_norm"))
    names.append(String("train/epoch"))

    var val_names = List[String]()
    val_names.append(String("val/l1"))
    val_names.append(String("val/kl"))
    val_names.append(String("perf/s_per_step"))
    val_names.append(String("best/val_l1"))

    for s in range(steps):
        ds.sample_batch[K, BATCH](False, qpos, images, actions, valid)
        var r = tr.train_step(qpos, images, actions, valid)

        acc_l1 += r.l1
        acc_kl += r.kl
        acc_loss += r.loss
        acc_gn += r.grad_norm
        acc_n += 1
        if acc_n == LOG_EVERY:
            var vals = List[Float64]()
            vals.append(acc_l1 / Float64(acc_n))
            vals.append(acc_kl / Float64(acc_n))
            vals.append(acc_loss / Float64(acc_n))
            vals.append(acc_gn / Float64(acc_n))
            vals.append(Float64(s) / Float64(steps_per_epoch))
            logger.log_scalars(names, vals, s)
            acc_l1 = 0.0
            acc_kl = 0.0
            acc_loss = 0.0
            acc_gn = 0.0
            acc_n = 0

        if s % VAL_EVERY == 0 or s == steps - 1:
            # Validation L1 is the reference's model-selection metric
            # (`imitate_episodes.py` keeps the checkpoint with the lowest
            # validation loss), so that is what the best checkpoint tracks.
            #
            # ⚠ The sampler's stream is PINNED and restored around the pass, so
            # every validation scores the same batches. Left random, `best_val`
            # would be the minimum of a noisy estimate over ~100 passes — which
            # selects the luckiest draw, not the best model — and successive
            # points on the curve would not be comparable to each other.
            var saved_rng = ds.rng
            ds.rng = VAL_SEED
            var vl1 = Float64(0.0)
            var vkl = Float64(0.0)
            for _ in range(VAL_BATCHES):
                ds.sample_batch[K, BATCH](True, qpos, images, actions, valid)
                var v = tr.eval_step(qpos, images, actions, valid)
                vl1 += v.l1
                vkl += v.kl
            vl1 /= Float64(VAL_BATCHES)
            vkl /= Float64(VAL_BATCHES)
            ds.rng = saved_rng

            # Rate over the interval just finished, not since step 0 — the
            # cumulative average never sheds the first step's warm-up and
            # reads slow for the whole run.
            var now = perf_counter_ns()
            var recent = Float64(now - t_mark) / 1e9
            var d_steps = s - s_mark
            var sps = recent / Float64(d_steps) if d_steps > 0 else recent
            t_mark = now
            s_mark = s
            var eta = sps * Float64(steps - s) / 60.0

            print(
                "  step " + String(s)
                + " (epoch " + String(s // steps_per_epoch) + ")"
                + "  train l1 " + String(r.l1)
                + "  kl " + String(r.kl)
                + "  |  val l1 " + String(vl1)
                + "  |  " + String(sps) + " s/step, ~"
                + String(Int(eta)) + " min left"
            )
            # Written EVERY pass: a run killed at hour three otherwise leaves
            # nothing to evaluate or resume from.
            tr.save(last_ckpt)
            if vl1 < best_val:
                best_val = vl1
                best_step = s
                tr.save(best_ckpt)

            var vvals = List[Float64]()
            vvals.append(vl1)
            vvals.append(vkl)
            vvals.append(sps)
            vvals.append(best_val)
            logger.log_scalars(val_names, vvals, s)
            # Flushed at every validation rather than only when the buffer
            # fills: the point of streaming a multi-hour run is watching it
            # while it runs, and a partly-full buffer is invisible.
            logger.flush()

    logger.close()

    print("")
    print(
        "  best validation l1 " + String(best_val) + " at step "
        + String(best_step) + " (epoch "
        + String(best_step // steps_per_epoch) + ")"
    )
    print("  best -> " + best_ckpt)
    print("  last -> " + last_ckpt)
    print("")
    print(
        "  ⚠ validation L1 rising while training L1 falls is what a"
        " from-scratch"
    )
    print(
        "    backbone does on " + String(len(ds.train_eps))
        + " training episodes with no augmentation. WHERE it turns is the"
    )
    print(
        "    number worth having. Evaluate the BEST checkpoint; see"
        " act_so101_openloop_eval.mojo."
    )
