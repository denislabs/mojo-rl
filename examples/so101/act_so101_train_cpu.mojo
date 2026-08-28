# +--------------------------------------------------------------------------+ #
# | ACT on the SO-ARM101 — CPU training
# +--------------------------------------------------------------------------+ #
"""Train ACT on the LeRobot v3 dataset, on CPU.

    pixi run python tools/act/lerobot_v3_to_store.py \\
        --repo DenisLabs/record-test_20260825_094319 --height 240 --width 320
    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/act_train \\
        examples/so101/act_so101_train_cpu.mojo && /tmp/act_train

Set `ACT_STORE=<path/to/store.h5>` to train on a different recording — the
50-episode store, say. Nothing below is pinned to the row or episode count.

⚠ `-Xlinker -ld_classic` is required: the fully-expanded graph type mangles to a
symbol longer than Apple's new linker accepts. The source is healthy; `mojo run`
JITs past it and never invokes ld.

## ⚠ This is a REDUCED configuration, and deliberately so

The paper's settings (`hidden_dim 512`, `dim_feedforward 3200`, `chunk_size 100`,
480x640 x4 cameras, 2000 epochs) are a ~5-hour run on an RTX 2080 Ti. On CPU the
ResNet18 forward alone dominates, and the full configuration is not a run you
would wait for. The dimensions below are what fits a CPU session while keeping
every structural property intact — the CVAE, both transformer stacks, the
chunking, the masking. `config.mojo` carries the paper's values; M8 of the plan
is the GPU path where they become usable.

## ⚠ And 4 training episodes will overfit

1997 frames, a from-scratch ResNet18, and no augmentation. Validation L1 is
expected to bottom out early and then rise while training L1 keeps falling. That
is the honest outcome at this data scale and it is what the curve below is for —
the deliverable here is a correct, trainable ACT, not a deployable policy. The
reference used 50 demonstrations per task.
"""

from std.time import perf_counter_ns

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
comptime K = 20  # paper: 100
comptime DIM = 64  # paper: 512
comptime HEADS = 4  # paper: 8
comptime FF = 256  # paper: 3200
comptime LATENT = 32  # paper: 32 (unchanged — it is small already)
comptime N_ENC = 1  # paper: 4
comptime N_DEC = 1  # paper: 7, but output-equivalent to 1 (see config.mojo)
comptime BATCH = 4  # paper: 8

comptime STEPS = 400
comptime VAL_EVERY = 50
comptime VAL_BATCHES = 4
comptime LR = 1e-4
comptime KL_WEIGHT = 10.0

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH,
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

    print("ACT / SO-ARM101 — CPU training")
    print("  store   " + path)
    print(
        "  model   K=" + String(K) + " dim=" + String(DIM)
        + " heads=" + String(HEADS) + " ff=" + String(FF)
        + " enc=" + String(N_ENC) + " dec=" + String(N_DEC)
        + "  (REDUCED — see the header)"
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

    var tr = T.make(lr=Scalar[DT](LR), kl_weight=Scalar[DT](KL_WEIGHT))

    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)

    var best_val = Float64(1e30)
    var best_step = -1
    var ckpt = String("/tmp/act_so101_best.ckpt")
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
