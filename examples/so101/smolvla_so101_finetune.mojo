# +--------------------------------------------------------------------------+ #
# | SmolVLA on the SO-ARM101 — fine-tuning the published checkpoint
# +--------------------------------------------------------------------------+ #
"""Fine-tune `lerobot/smolvla_base` on a LeRobot v3 recording, on GPU.

**No robot is required.** Training is entirely offline — a store of recorded
frames in, gradients out. The arm is needed to *deploy* a policy, not to fit
one, and the offline claim this run can actually make is a HELD-OUT loss that
falls: train on some episodes, measure on episodes never trained on.

## Launching a run on a fresh NVIDIA box, start to finish

```bash
git clone <this repo> mojo-rl && cd mojo-rl
pixi install -e nvidia

# ── 1. disk ───────────────────────────────────────────────────────────────
#   pixi env    ~20 GB
#   checkpoint  0.9 GB   lerobot/smolvla_base model.safetensors
#   HF dataset  ~2 GB    the recording's snapshot
#   store        27 GB   the converted .h5 at 480x640 (see step 3)
# Budget 55 GB.
df -h .

# ── 2. auth, ONLY if the dataset repo is private ──────────────────────────
pixi run -e nvidia hf auth login          # or: export HF_TOKEN=hf_...

# ── 3. the dataset -> a TrajectoryStore ───────────────────────────────────
# ⚠ 480x640, NOT the 240x320 the ACT example uses. SmolVLA resizes to 512x512
# itself with `resize_with_pad`, and importing at a reduced size would resample
# twice — `test_store_vs_camera_frame.mojo` gates the training path against the
# deployment path at 640x480 and both must see the same pixels.
pixi run -e nvidia mojo run -I . \\
    examples/so101/act_so101_import_dataset.mojo \\
    --repo DenisLabs/record-test_20260828_092736 --height 480 --width 640
export SMOLVLA_STORE=~/.cache/mojo_rl/act_so101/DenisLabs__record-test_20260828_092736_480x640.h5

# ── 4. normalisation statistics — fetched automatically ──────────────────
# `meta/stats.json` is pulled from the SAME dataset repo as the store, so
# there is nothing to export. SmolVLA reads it rather than recomputing: ours
# and lerobot's differ by exactly sqrt(N/(N-1)) — sample vs population std —
# and training on one while the checkpoint was fit with the other is a silent
# scale error. `SMOLVLA_STATS` overrides it with a local file.

# ── 5. check the pieces before spending GPU hours ────────────────────────
pixi run -e nvidia test-vla                    # structural, no download
pixi run -e nvidia test-vla-gpu                # kernels at real shapes
pixi run -e nvidia test-vla-weights            # the real checkpoint loads

# ── 6. build, SMOKE, then run ────────────────────────────────────────────
pixi run -e nvidia mojo build -I . -o /tmp/smolvla_finetune \\
    examples/so101/smolvla_so101_finetune.mojo

# ⚠ THREE STEPS FIRST. Nothing in this port has ever run at the published
# depth with RECORD on — the tape is ~320 MB of per-layer activations that no
# gate has allocated — and the first thing to learn is whether it starts and
# what a step costs, not whether the loss falls.
SMOLVLA_STEPS=3 SMOLVLA_NO_MONITOR=1 /tmp/smolvla_finetune

# then the real thing. 2000 x 8 = 16k observations, about one pass over the
# recording. Read the s/step the smoke run printed before choosing a number.
/tmp/smolvla_finetune
```

⚠ **Run it from the project root** — `mojo_rl/io/hdf5` resolves libhdf5 through
a path relative to the working directory, and the tokenised instruction table
is read from `tools/vla/`.

### Environment variables

| | |
|---|---|
| `SMOLVLA_STORE` | the `.h5` to train on. **Required** — there is no default, because a fine-tune attributed to the wrong recording is worse than one that refuses to start |
| `SMOLVLA_STATS` | a local `meta/stats.json`. Optional — by default it is fetched from `SMOLVLA_REPO` |
| `SMOLVLA_REPO` | the dataset repo the statistics come from; defaults to the recording named above |
| `SMOLVLA_TASKS` | the tokenised instruction table; defaults to the checked-in one for this recording |
| `SMOLVLA_STEPS` | optimizer steps, without a rebuild |
| `SMOLVLA_ACCUM` | observations per optimizer step (default 8) |
| `SMOLVLA_LR` | default 1e-4 |
| `SMOLVLA_NO_MONITOR` | force the metrics logger inert |

## What this run is, and what it is not

**The trainable set is the shipped default minus one flag.**
`train_expert_only = True` and — here — `train_state_proj = False`: the action
expert, the action projections, and nothing else. `state_proj` IS supported
(`TRAIN_STATE_PROJ`, gated by `test_state_proj_grad.mojo`) and costs a full
backward through sixteen frozen VLM layers to train one 32x960 matrix. Turn it
on when the cheaper regime has been shown to work, not before.

⚠ **The batch is gradient ACCUMULATION at B = 1, not a batched forward.** B > 1
is a comptime parameter of every container here and it is not established: the
only B > 1 leg anywhere in the port is `test_kv_cache.mojo`'s, added when the
KV scratch turned out not to be batch-major. Accumulating micro-batches uses
only the B = 1 paths every gate covers, and `Linear.vjp` accumulates natively.

⚠ **The loss denominator is the GROUP's, not the micro-batch's.** Each
accumulation group is sampled FIRST, its total valid-timestep count summed,
and every micro-batch's `flow_mse` is given that total — so the accumulated
gradient is the mean over the whole group rather than a sum of per-micro-batch
means. Passing each micro-batch its own count would weight a chunk near an
episode boundary more heavily than a full one, by exactly the ratio of their
valid counts.

⚠ **Validation runs the backward it does not need.** `SmolVLATrainStep.run`
does forward and backward together, so a validation pass costs about twice what
it should and leaves gradients that the next training step's
`zero_trainable_grads` discards. Correct, wasteful, and named here rather than
left to be discovered in a profile.

⚠ **No checkpoint is written.** Deliberately: the trainable subset is ~100 M
parameters and checkpoint v2 truncates at 2 GiB, so a save path needs thought
rather than a hurried call. THIS RUN IS FOR ANSWERING "DOES THE LOSS FALL",
and the answer does not need to be resumable. Do not start a multi-hour run
expecting to keep the weights.

### The obvious optimisation, and why it is not here

Under `train_state_proj = False` the whole prefix — SigLIP over two 512x512
images, twelve vision layers, then sixteen VLM layers — is a CONSTANT for a
given (frame, instruction), and it is recomputed every time that frame is
drawn. Caching it would remove the dominant cost. It is 2.76 MB per row and
the recording is ~15 k rows, so caching all of it is 42 GB: the useful version
caches a subset, which is a design with a memory budget in it rather than a
one-line change. Measure first — this file exists to produce that measurement.
"""

from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.io.hf import hf_download_file, HF_MODEL, HF_DATASET
from mojo_rl.io.hdf5 import H5Dataset

from mojo_rl.deep_agents.smolvla.policy import SmolVLAPolicy
from mojo_rl.deep_agents.smolvla.normalize import SmolVLAStats
from mojo_rl.deep_agents.smolvla.tasks import TaskTokens
from mojo_rl.deep_agents.smolvla.dataset import SmolVLABatchSampler
from mojo_rl.deep_agents.smolvla.observation import fill_store_images
from mojo_rl.deep_agents.smolvla.train_step import SmolVLATrainStep
from mojo_rl.deep_agents.smolvla.finetune import (
    zero_trainable_grads, adam_step_trainables,
)
from mojo_rl.deep_agents.smolvla.flow_loss import (
    build_xt_ut, sample_noise, sample_times,
)
from mojo_rl.deep_agents.smolvla.heads import (
    SMOLVLA_ACTION_DIM, SMOLVLA_EXPERT_W, SMOLVLA_STATE_DIM,
)
from mojo_rl.deep_agents.smolvla.text import (
    SMOLLM_DIM, SMOLLM_LAYERS, SMOLLM_KV_W,
)
from mojo_rl.deep_agents.smolvla.expert import EXPERT_FF
from mojo_rl.deep_agents.smolvla.vision import SIGLIP_LAYERS

# ── the recording ────────────────────────────────────────────────────────
comptime SDIM = 6                # the SO-101's joints
comptime ADIM_REAL = 6
comptime N_CAM = 2
comptime SRC_H = 480
comptime SRC_W = 640
comptime N_LANG = 6
"""⚠ Pinned by the instruction table, not chosen. `test_tasks.mojo` refuses a
table whose tasks tokenise to different lengths, because a comptime N_LANG
cannot hold two — a multi-task fine-tune needs a padding decision first."""

comptime CHUNK = 50
comptime STEPS_EULER = 10        # inference only; training denoises ONCE
comptime B = 1
comptime PAD = SMOLVLA_ACTION_DIM

comptime REPO = String("lerobot/smolvla_base")
comptime DEFAULT_TASKS = String(
    "tools/vla/smolvla_tasks_record-test_20260828_092736.tsv"
)
comptime DEFAULT_DATA_REPO = String("DenisLabs/record-test_20260828_092736")

comptime DEFAULT_STEPS = 2000
"""⚠ 2000 x accum 8 is ~16 k observations, roughly one pass over the
recording. NOT a converged fine-tune — a first answer to whether the held-out
loss moves at all. Raise it with `SMOLVLA_STEPS` once a step time is known."""
comptime DEFAULT_ACCUM = 8
comptime LR = Scalar[DT](1.0e-4)
comptime VAL_EVERY = 200
comptime VAL_GROUPS = 8
comptime LOG_EVERY = 10
comptime VAL_SEED: UInt64 = 0x5DEECE66D
"""⚠ Validation redraws from the SAME seed every time, so every pass scores
the identical held-out observations. Otherwise the curve is a random walk over
which frames came up and "it improved" is unfalsifiable."""

comptime Pol = SmolVLAPolicy[
    N_CAM, N_LANG, CHUNK, STEPS_EULER, B, SMOLLM_LAYERS, SIGLIP_LAYERS, True
]
comptime Step = SmolVLATrainStep[
    CHUNK, ADIM_REAL, PAD, SMOLVLA_EXPERT_W, B, SMOLLM_LAYERS, EXPERT_FF,
    SMOLLM_DIM,
]
comptime Sampler = SmolVLABatchSampler[SDIM, ADIM_REAL, PAD, CHUNK, B]
comptime IMG_ELEMS = N_CAM * 3 * SRC_H * SRC_W
comptime AN = B * CHUNK * PAD


def _need(name: String) raises -> String:
    var v = getenv(name)
    if v.byte_length() == 0:
        raise Error(
            "$" + name + " is not set. This example refuses a default: a"
            " fine-tune attributed to the wrong recording or the wrong"
            " statistics is worse than one that will not start. See the"
            " header."
        )
    return v^


struct Group(Movable):
    """One accumulation group, drawn BEFORE any forward runs.

    ⚠ The whole group is sampled first so `total_valid` can be its sum. Each
    micro-batch's `flow_mse` is then given that total, which makes the
    accumulated gradient the mean over the group. Giving each micro-batch its
    own count instead would weight a chunk near an episode boundary more
    heavily than a full one, by the ratio of their valid counts — a silent,
    data-dependent reweighting of the loss.
    """

    var rows: List[Int]
    var tasks: List[Int]
    var raw_state: List[Float32]
    var actions: List[Scalar[DT]]
    var valid: List[Scalar[DT]]
    var total_valid: Int

    def __init__(out self):
        self.rows = List[Int]()
        self.tasks = List[Int]()
        self.raw_state = List[Float32]()
        self.actions = List[Scalar[DT]]()
        self.valid = List[Scalar[DT]]()
        self.total_valid = 0

    def __init__(out self, *, deinit move: Self):
        self.rows = move.rows^
        self.tasks = move.tasks^
        self.raw_state = move.raw_state^
        self.actions = move.actions^
        self.valid = move.valid^
        self.total_valid = move.total_valid


def draw_group(
    mut sam: Sampler, accum: Int, lo: Int, hi: Int,
    mut state_t: Tensor, mut acts_t: Tensor, mut valid_t: Tensor,
) raises -> Group:
    var gr = Group()
    for _ in range(accum):
        var tk = List[Int]()
        var rw = List[Int]()
        var rs = List[Float32]()
        var nv = sam.sample(state_t, acts_t, valid_t, tk, rw, rs, lo, hi)
        gr.total_valid += nv
        for b in range(B):
            gr.rows.append(rw[b])
            gr.tasks.append(tk[b])
            for j in range(SDIM):
                gr.raw_state.append(rs[b * SDIM + j])
        for i in range(AN):
            gr.actions.append(acts_t.data[i])
        for i in range(B * CHUNK):
            gr.valid.append(valid_t.data[i])
    return gr^


def main() raises:
    print("=" * 74)
    print("SmolVLA fine-tune — SO-ARM101")
    print("=" * 74)

    var store_path = _need(String("SMOLVLA_STORE"))
    # ⚠ Fetched from the dataset repo unless overridden, so the statistics
    # and the store cannot come from different snapshots by accident — which
    # would be a scale error with no symptom but a worse policy.
    var stats_path = getenv("SMOLVLA_STATS")
    if stats_path.byte_length() == 0:
        var data_repo = getenv("SMOLVLA_REPO")
        if data_repo.byte_length() == 0:
            data_repo = DEFAULT_DATA_REPO
        stats_path = hf_download_file(
            data_repo, String("meta/stats.json"), HF_DATASET
        )
    var tasks_path = getenv("SMOLVLA_TASKS")
    if tasks_path.byte_length() == 0:
        tasks_path = DEFAULT_TASKS

    var steps = DEFAULT_STEPS
    var e_steps = getenv("SMOLVLA_STEPS")
    if e_steps.byte_length() > 0:
        steps = Int(e_steps)
    var accum = DEFAULT_ACCUM
    var e_accum = getenv("SMOLVLA_ACCUM")
    if e_accum.byte_length() > 0:
        accum = Int(e_accum)

    var ctx = DeviceContext()
    print("  device  " + String(ctx.name()))
    print("  store   " + store_path)
    print("  stats   " + stats_path)
    print("  steps   " + String(steps) + " x accum " + String(accum)
          + "   lr " + String(LR))

    var tasks = TaskTokens(tasks_path)
    var n_lang = tasks.n_lang()
    if n_lang != N_LANG:
        raise Error(
            "the instruction table tokenises to " + String(n_lang)
            + " tokens and this build is pinned to " + String(N_LANG)
            + " — rebuild with N_LANG = " + String(n_lang)
        )
    print("  tasks   " + String(tasks.size()) + " instruction(s), "
          + String(n_lang) + " tokens   P = " + String(Pol.P))

    print("  loading lerobot/smolvla_base ...")
    var weights = hf_download_file(REPO, String("model.safetensors"), HF_MODEL)
    var pol = Pol.make["gpu", Deterministic](Optional(ctx))
    pol.load["gpu"](weights, Optional(ctx))
    pol.load_stats(stats_path)
    print("  policy  loaded")

    var sam = Sampler(store_path, SmolVLAStats.from_stats_json(stats_path))
    var n_ep = sam.store.n_episodes()
    var n_val_ep = n_ep // 5
    if n_val_ep < 1:
        n_val_ep = 1
    var split = sam.store.episodes.start_of(n_ep - n_val_ep)
    var n_rows = sam.n_rows()
    print(
        "  data    " + String(n_rows) + " rows, " + String(n_ep)
        + " episodes — train [0, " + String(split) + "), held out ["
        + String(split) + ", " + String(n_rows) + ")"
    )
    # ⚠ A CONTIGUOUS tail, not every fifth episode. Frames inside one episode
    # are near-duplicates of their neighbours, so an interleaved split puts a
    # training frame 33 ms from each held-out one and the held-out loss
    # measures memorisation instead of generalisation.

    var env_vars = load_dotenv()
    var no_mon = getenv("SMOLVLA_NO_MONITOR")
    var monitor_url = (
        String("") if no_mon.byte_length() > 0
        else env_vars.get("RL_MONITOR_URL", "")
    )
    var logger = RemoteLogger(
        server_url=monitor_url,
        run_name="SmolVLA SO-ARM101 fine-tune",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("algorithm", "SmolVLA")
    logger.set_config("robot", "SO-ARM101")
    logger.set_config("regime", "train_expert_only; state_proj FROZEN")
    logger.set_config("device", String(ctx.name()))
    logger.set_config("store", store_path)
    logger.set_config("chunk", String(CHUNK))
    logger.set_config("accum", String(accum))
    logger.set_config("lr", String(LR))

    var st = Step.make["gpu"](Optional(ctx))
    var opt = Adam(lr=LR)
    var sp_frozen = Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM].make[
        "gpu", Deterministic
    ](Optional(ctx))
    """⚠ A stand-in for the walks' `state_proj` argument. TRAIN_STATE_PROJ is
    False so neither walk touches it, and the policy's real one is never handed
    to the optimizer. That is what "frozen" means here — not a flag inside the
    optimizer but an object it never sees."""

    var images = Tensor.alloc(N_CAM * 3 * 512 * 512)
    var scratch = List[Float32]()
    var state_t = Tensor.alloc(B * PAD)
    var acts_t = Tensor.alloc(AN)
    var valid_t = Tensor.alloc(B * CHUNK)
    var noise_t = Tensor.alloc(AN)
    var times_t = Tensor.alloc(B)
    var x_t = Tensor.alloc(AN)
    var u_t = Tensor.alloc(AN)
    var row = List[Scalar[DType.uint8]](unsafe_uninit_length=IMG_ELEMS)
    var img_col = sam.store.open_column[DType.uint8](String("images"))

    var t0 = perf_counter_ns()

    for s in range(steps):
        zero_trainable_grads[
            "gpu", SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
            SMOLLM_KV_W, PAD,
        ](
            pol.expert, pol.action_in, pol.time_mlp_in, pol.time_mlp_out,
            pol.action_out, sp_frozen, Optional(ctx),
        )
        var gr = draw_group(sam, accum, 0, split, state_t, acts_t, valid_t)
        var loss = 0.0
        for m in range(accum):
            loss += run_one(
                m, gr, sam, tasks, pol, st, img_col, row, images, scratch,
                acts_t, valid_t, noise_t, times_t, x_t, u_t, ctx,
            )
        adam_step_trainables[
            "gpu", SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
            SMOLLM_KV_W, PAD,
        ](
            opt, pol.expert, pol.action_in, pol.time_mlp_in,
            pol.time_mlp_out, pol.action_out, sp_frozen, Optional(ctx),
        )

        if s % LOG_EVERY == 0:
            var el = Float64(perf_counter_ns() - t0) / 1.0e9
            print(
                "  step " + String(s) + "   train " + String(loss)
                + "   " + String(el / Float64(s + 1)) + " s/step"
            )
            var names = List[String]()
            var vals = List[Float64]()
            names.append(String("train/loss"))
            vals.append(loss)
            logger.log_scalars(names, vals, s)

        if s % VAL_EVERY == 0 or s == steps - 1:
            # ⚠ The seed is PINNED, so every validation scores the identical
            # held-out observations. It is restored afterwards so training
            # does not replay the same batches for ever.
            var keep_rng = sam.rng
            sam.rng = VAL_SEED
            var vsum = 0.0
            for _ in range(VAL_GROUPS):
                var vg = draw_group(
                    sam, accum, split, n_rows, state_t, acts_t, valid_t
                )
                for m in range(accum):
                    vsum += run_one(
                        m, vg, sam, tasks, pol, st, img_col, row, images,
                        scratch, acts_t, valid_t, noise_t, times_t, x_t, u_t,
                        ctx,
                    )
            sam.rng = keep_rng
            var vloss = vsum / Float64(VAL_GROUPS)
            print("  step " + String(s) + "   HELD-OUT " + String(vloss))
            var vn = List[String]()
            var vv = List[Float64]()
            vn.append(String("val/loss"))
            vv.append(vloss)
            logger.log_scalars(vn, vv, s)
            # ⚠ Validation ran `run_one`, which does a BACKWARD it does not
            # need — `SmolVLATrainStep.run` does both. The gradients it leaves
            # are discarded by the next step's `zero_trainable_grads`, above.
            # Correct, and about twice the cost it should be.

    logger.flush()
    print("done")


def run_one(
    m: Int,
    ref gr: Group,
    mut sam: Sampler,
    ref tasks: TaskTokens,
    mut pol: Pol,
    mut st: Step,
    mut img_col: H5Dataset,
    mut row: List[Scalar[DType.uint8]],
    mut images: Tensor,
    mut scratch: List[Float32],
    mut acts_t: Tensor,
    mut valid_t: Tensor,
    mut noise_t: Tensor,
    mut times_t: Tensor,
    mut x_t: Tensor,
    mut u_t: Tensor,
    ctx: DeviceContext,
) raises -> Float64:
    """One observation: its prefix, its interpolant, one denoising step."""
    var g = gr.rows[m * B]
    img_col.read_range[DType.uint8](g, g + 1, mptr(row))
    fill_store_images["gpu", N_CAM](
        row, SRC_W, SRC_H, images, scratch, Optional(ctx)
    )
    var lang = tasks.for_index(gr.tasks[m * B])
    var rs = List[Float32]()
    for j in range(SDIM):
        rs.append(gr.raw_state[m * B * SDIM + j])
    pol.build_prefix["gpu"](images, lang, rs, Optional(ctx))

    for i in range(AN):
        acts_t.data[i] = gr.actions[m * AN + i]
    for i in range(B * CHUNK):
        valid_t.data[i] = gr.valid[m * B * CHUNK + i]
    acts_t.upload_resident(ctx)
    valid_t.upload_resident(ctx)

    sample_noise(noise_t, AN)
    noise_t.upload_resident(ctx)
    var tl = sample_times(B)
    times_t.ensure(B)
    for b in range(B):
        times_t.data[b] = Scalar[DT](tl[b])
    times_t.upload_resident(ctx)
    build_xt_ut["gpu", B, CHUNK * PAD](
        noise_t, acts_t, times_t, x_t, u_t, Optional(ctx)
    )
    st.set_times["gpu"](tl, Optional(ctx))
    return st.run["gpu", Pol.P](
        pol.expert, pol.cache, pol.denoiser, pol.action_in, pol.time_mlp_in,
        pol.time_mlp_out, pol.action_out, x_t, u_t, valid_t, gr.total_valid,
        Optional(ctx),
    )
