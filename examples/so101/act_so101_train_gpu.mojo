# +--------------------------------------------------------------------------+ #
# | ACT on the SO-ARM101 — GPU training
# +--------------------------------------------------------------------------+ #
"""Train ACT on the LeRobot v3 dataset, on GPU.

    pixi run python tools/act/lerobot_v3_to_store.py \\
        --repo DenisLabs/record-test_20260825_094319 --height 240 --width 320
    pixi run -e apple mojo build -I . -Xlinker -ld_classic -o /tmp/act_train_gpu \\
        examples/so101/act_so101_train_gpu.mojo && /tmp/act_train_gpu

(`-e nvidia` for CUDA. On NVIDIA the `-Xlinker` flag is not needed.)

⚠ `-Xlinker -ld_classic` is required: the fully-expanded graph type mangles to a
symbol longer than Apple's new linker accepts. The source is healthy; `mojo run`
JITs past it and never invokes ld.

## Configuration

Closer to the paper than the CPU example, but still not identical — the settings
below are what a single-GPU session absorbs comfortably. `config.mojo` carries
the paper's values (`hidden_dim 512`, `dim_feedforward 3200`, `chunk_size 100`,
`enc_layers 4`); raise these toward them as your hardware allows, and watch the
COMPILE time as much as the step time — the graph type expands with `K` and the
layer counts, and this file already takes minutes to build.

⚠ **`max_grad_norm = 0.0`** below. The clip walks every gradient slab through
the host (`trainer.mojo:_SumSq`), which on GPU is a synchronisation point per
parameter — the wrong shape for a hot loop. Zero is also what the reference
effectively does: it parses `--clip_max_norm 0.1` and never applies it.

## ⚠ And 4 training episodes will overfit

1997 frames, a from-scratch ResNet18, and no augmentation. Validation L1 is
expected to bottom out early and then rise while training L1 keeps falling. That
is the honest outcome at this data scale and it is what the curve below is for —
the deliverable here is a correct, trainable ACT, not a deployable policy. The
reference used 50 demonstrations per task.
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

from std.python import Python, PythonObject


# ── shape (must match the store; see the header) ─────────────────────────
comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W

# ── reduced model ────────────────────────────────────────────────────────
comptime K = 40  # paper: 100
comptime DIM = 128  # paper: 512
comptime HEADS = 8  # paper: 8
comptime FF = 512  # paper: 3200
comptime LATENT = 32  # paper: 32 (unchanged — it is small already)
comptime N_ENC = 2  # paper: 4
comptime N_DEC = 1  # paper: 7, but output-equivalent to 1 (see config.mojo)
comptime BATCH = 8  # paper: 8

comptime STEPS = 2000
comptime VAL_EVERY = 100
comptime VAL_BATCHES = 4
comptime LR = 1e-4
comptime KL_WEIGHT = 10.0

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
    var ckpt = String("/tmp/act_so101_best_gpu.ckpt")
    var t0 = perf_counter_ns()

    for s in range(STEPS):
        ds.sample_batch[K, BATCH](False, qpos, images, actions, valid)
        var r = tr.train_step(qpos, images, actions, valid)

        if s % VAL_EVERY == 0 or s == STEPS - 1:
            # Validation L1 is the reference's model-selection metric
            # (`imitate_episodes.py` keeps the checkpoint with the lowest
            # validation loss), so that is what the best checkpoint tracks.
            var vl1 = Float64(0.0)
            var vkl = Float64(0.0)
            for _ in range(VAL_BATCHES):
                ds.sample_batch[K, BATCH](True, qpos, images, actions, valid)
                var v = tr.eval_step(qpos, images, actions, valid)
                vl1 += v.l1
                vkl += v.kl
            vl1 /= Float64(VAL_BATCHES)
            vkl /= Float64(VAL_BATCHES)

            var el = Float64(perf_counter_ns() - t0) / 1e9
            print(
                "  step " + String(s)
                + "  train l1 " + String(r.l1)
                + "  kl " + String(r.kl)
                + "  |  val l1 " + String(vl1)
                + "  |  " + String(el / Float64(s + 1)) + " s/step"
            )
            if vl1 < best_val:
                best_val = vl1
                best_step = s
                tr.save(ckpt)

    print("")
    print(
        "  best validation l1 " + String(best_val) + " at step "
        + String(best_step)
    )
    print("  checkpoint -> " + ckpt)
    print("")
    print(
        "  ⚠ validation L1 rising while training L1 falls is what a"
        " from-scratch"
    )
    print(
        "    backbone does on " + String(len(ds.train_eps))
        + " training episodes with no augmentation."
    )
    print(
        "    The best checkpoint above is the one to evaluate; see"
        " act_so101_openloop_eval.mojo."
    )
