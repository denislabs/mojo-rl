"""M2 — FB training on GPU, d = 128, from a collected dm_control dataset.

This is the run `docs/BFM_ZERO_SHOT_RL.md` §13 budgets at 2 M gradient steps and
"a few hours on a rented 4090". Everything it needs is device-resident: the
dataset is uploaded once, the sampler writes indices on device, and the batch is
assembled by the gather/pack kernels. Nothing crosses PCIe in the training loop
except the occasional logged loss.

Prerequisite: run `examples/fb/collect_dm_control.mojo` first. This reads the
store it writes.

    pixi run -e nvidia mojo run -I . examples/fb/fb_train_gpu.mojo

## Sweep interface

Every comptime constant below that a sweep needs to vary is ALSO a runtime flag,
so an arm costs a process launch rather than a rebuild (~90 s each, which is
most of a 17-minute arm):

    --steps N      --ortho X     --lr-b X     --bc X
    --obs-norm 0|1 --tag NAME

`--tag` is the one that matters for bookkeeping: it renames the checkpoint, the
CSV and the remote run together, so two arms cannot overwrite each other's
output. Absent flags keep the comptime defaults, so the no-argument invocation
above is byte-identical to what it was before the flags existed.
`examples/fb/fb_sweep.sh` drives the arms; §16.3 has the target values.

⚠⚠ **`want_loss` is on a stride, and that is not cosmetic.** Reading a loss back
from the GPU is a full pipeline stall. At 2 M steps, logging every step is the
difference between a few hours and most of a day, and the GRADIENTS are
identical either way because the loss value never enters the update.

⚠ `s'` is taken through a precomputed `next_row` table rather than `idx + 1`.
Without it, a row sampled at an episode boundary pairs a terminal state with the
NEXT episode's reset — a transition that never happened, injected into the
measure loss once per episode. The table is built from the store's OWN episode
index (`ep_offset` / `ep_len`), never from an assumed episode length, and the
count of self-transitions it creates is asserted against `n_episodes` below.

⚠ `s+` comes from a SECOND, INDEPENDENT draw. It is not `s'`. See
`fb/loss.mojo`: the successor measure asks "starting from s, how often is s+
visited", and if `s+` were the batch's own next-states the matrix would only
ever be evaluated on pairs one step apart.
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.random import random_float64, seed
from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.nn.random.box_muller import box_muller_normal_gpu

from mojo_rl.data.store import TrajectoryStore
from mojo_rl.data.resident import ResidentColumn, IDX_DT
from mojo_rl.data.sampler import UniformDeviceSampler

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import CsvLogger, RemoteLogger, CompositeLogger
from mojo_rl.cuda import CUDAGraph, maybe_capture_replay
from mojo_rl.deep_agents.fb.trainer import FBTrainer, FBLosses
from mojo_rl.deep_agents.fb.obs_norm import ObsNorm
from mojo_rl.deep_agents.fb.kernels import (
    gather_rows_kernel,
    gather_idx_kernel,
    z_mixture_kernel,
    project_sphere_kernel,
    ensure_t,
    _blocks,
)


# ── the dataset this reads ───────────────────────────────────────────────
comptime STORE_PATH: StaticString = "fb_walker_all_sac.h5"
comptime NQ: Int = 9
comptime NV: Int = 9
comptime NACT: Int = 6
comptime OBS: Int = NQ + NV
# ⚠ There is deliberately NO `EP_LEN` here. It used to be one, and it was a
# silent correctness bug: `next_row` marked a boundary at every multiple of a
# COMPTIME 250 while the collected store runs 1000-step episodes, so 3 of every
# 4 "boundaries" were fabricated and 0.3% of rows got a spurious
# self-transition. Boundaries now come from the store's own episode index, so a
# dataset with different episode lengths — or ragged ones, which early
# termination produces — cannot desynchronise from this file.

# ── the run ──────────────────────────────────────────────────────────────
comptime D: Int = 128  # §13: 128 at M2 (50 was the M1 setting)
comptime BATCH: Int = 1024
comptime HID: Int = 1024
comptime TRAIN_STEPS: Int = 2_000_000
comptime LOG_EVERY: Int = 2000  # see the want_loss note in the header
comptime CKPT_EVERY: Int = 50_000
comptime CKPT_PATH: StaticString = "fb_walker_all_d128.ckpt"

# ⚠⚠ Global grad-norm clip. FB's measure loss scales as (||F||·sqrt(d))^2 and
# was measured spiking to +2559 on walker at 1 M rows; the gradients spike with
# it. `L_ortho` and the norm layer bound `B`, and NOTHING bounds `F` — clipping
# is the cheap standard remedy and every reference FB implementation uses one.
# 0 disables it.
comptime MAX_GRAD_NORM: Float64 = 1.0

# ⚠⚠ Behaviour-cloning weight on the actor. Measured NECESSARY, not optional:
# at `bc_weight = 0` and 200 k steps, 95-98% of pi_z's actions had |a| > 0.99
# — bang-bang torque — and the policy scored WORSE than random on all three
# walker tasks (t = -6.0 on `walk`). The actor maximises `F·z` over [-1,1],
# `F` is near-linear in `a`, and a linear function's maximum on a box is a
# corner.
#
# ⚠ Too large collapses every `z` onto the data's mean action and destroys the
# z-conditioned policy family — the thing FB exists for. 1.0 is a starting
# point; the eval prints `mean|a|` and `saturated`, so tune against those
# rather than against the loss.
comptime BC_WEIGHT: Float64 = 1.0

# ⚠⚠ **BFM-Zero ships `ortho_coef = 100`; this has always run 1.0.**
# `docs/BFM_ZERO_SHOT_RL.md` §16.3 — arXiv 2511.04131 Table 1 AND the released
# `fb_cpr/configs.py` both carry 100, a factor of 100 above `FBTrainer.make`'s
# default, which is what every §13 measurement was taken at. It is left at 1.0
# here so the existing numbers stay comparable; `--ortho 100` is the arm.
comptime ORTHO_WEIGHT: Float64 = 1.0

# ⚠ **B's learning rate, SEPARATE from F's.** The reference trains B at 1e-5
# against F's 3e-4 — B is the shared representation and F chases it, so a B
# moving at F's rate is a target that will not sit still. -1 inherits `lr`,
# which is what this script did implicitly before the flag existed.
comptime LR_B: Float64 = -1.0

# ⚠⚠ **Observation standardisation.** BFM-Zero normalises every observation
# entering F, B and the actor (`BatchNorm1d(affine=False)`); we fed raw
# `qpos | qvel`, and on walker `qvel` spans about an order of magnitude more
# than `qpos`. See `fb/obs_norm.mojo` — in particular why the statistics are
# written NEXT TO THE CHECKPOINT rather than recomputed at eval time.
comptime OBS_NORM: Bool = False
# ⚠⚠ CUDA-graph capture of the train step. NVIDIA only — `CUDAGraph` is a
# compile-time no-op elsewhere, so this is bit-identical to eager on Apple and
# the Apple gates cannot prove the capture works. Measured motivation: an FB
# step issues ~150 kernel launches and NOTHING was captured, while the SAC
# driver has captured since day one — that asymmetry, not the arithmetic, is
# what makes FB feel slow next to SAC.
#
# ⚠ Capture requires the trainer's capture-safe path, which is why
# `FBTrainer.make` now `adopt`s its optimizers and the target-smoothing noise
# draws its Philox offset from a DEVICE buffer. A host offset would be baked in
# at capture and every replay would redraw identical noise — silently, since
# frozen noise still trains. `tests/deep_agents/test_fb_cuda_graph_safety.mojo`
# gates exactly that.
comptime USE_TRAIN_CUDA_GRAPH: Bool = True

# ── logging ──────────────────────────────────────────────────────────────
# ⚠ The run logs to a LOCAL CSV **and** the remote monitor, via
# `CompositeLogger`. Not redundancy for its own sake: a previous run's console
# output was lost mid-arc and the interesting window went with it, because the
# only record was a terminal scrollback. The CSV survives a dropped ssh session,
# a killed monitor, and a laptop reboot.
comptime CSV_PATH: StaticString = "fb_walker_all_d128_metrics.csv"
comptime RUN_NAME: StaticString = "FB walker all-tasks d128"
comptime SEED: Int = 20260805

comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D

comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
# ⚠⚠ **`B` MUST end in a normalisation.** Meta Motivo's config carries
# `"b": {..., "norm": true}` and §6's reuse table names `layer_norm.mojo` for
# exactly this; the first version of this script omitted it and the run
# DIVERGED. The signature is unmistakable once you know to look: `L_ortho`
# went POSITIVE and grew 8x (21 -> 172 over 24 k steps) while `|B|` climbed
# 7.2 -> 9.3 and the measure loss fell without bound (-59 -> -295).
#
# ⚠⚠ **AND THE NORMALISATION MUST NOT BE LEARNABLE.** `LayerNorm[D]` pins
# `||B||` at sqrt(d) only while its gamma is 1. Over the first 1 M-row run it
# did not stay there: gamma grew and `|B|` drifted 11.31 -> 17.54 across 100 k
# steps, monotonically, while `L_ortho` rose 8.8x tracking it (113.9 -> 998.6).
#
# `L_ortho` is `Q - 2S`, with `Q = (1/N^2) sum_ij (B(s_i)·B(s'_j))^2` and
# `S = (1/N) sum_i ||B(s_i)||^2`. Under `B -> cB` those scale as `c^4` and
# `c^2`, so the rise decomposes cleanly:
#
#            |B|      L_ortho      anchor -2|B|^2     implied Q
#     2 k   11.99       113.9           -287.5           401.4
#   100 k   17.54       998.6           -615.3          1613.9
#
# Q grew 4.02x against the 4.58x that pure scaling (c^4, c = 1.463) predicts —
# slightly LESS, i.e. B's rows became marginally MORE orthogonal. So the whole
# rise is the EXPANSION; there is no directional collapse. The optimizer grew B
# because a larger B fits the TD targets more easily, and the quartic penalty
# for doing so outruns the quadratic anchor that is supposed to hold it.
#
# ⚠ An earlier version of this comment called `L_ortho` a "pure quartic" and
# read the undecomposed 8.8x rise as directional collapse. Both were wrong —
# from stopping 40 lines into `fb_ortho_loss` and not reading the `-2S` term.
#
# `LayerNormNoAffine[D]` removes the degenerate direction outright: the output
# row norm is sqrt(d) exactly, with no learnable escape. Nothing the theory
# wants is lost — B's DIRECTION stays fully learnable in the Linear before it,
# and FB wants B on the sqrt(d) sphere anyway (Meta Motivo's `"norm": true`).
# It also conditions `L_ortho` better: with |B| fixed, ortho can only be
# reduced by DECORRELATING, which is its actual job.
#
# ⚠ The pin is conditional on B's pre-norm per-row std staying above ~1e-3
# (eps = 1e-6); below that the projection quietly stops projecting. Because it
# is otherwise exact, `|B| != sqrt(d)` in the log below IS that alarm.
# Gated by `tests/nn/test_layer_norm_no_affine.mojo`, which asserts sqrt(d) at
# d = 128 and the closed-form degradation below eps.
#
# ⚠ Historical note, kept because it was the FIRST failure and the sign rule
# it produced is WRONG now: with an UNNORMALISED B the diagnostic was "ortho
# went POSITIVE and grew 8x while |B| climbed 7.2 -> 9.3". Once B is pinned,
# ortho's sign carries no such meaning — read TRENDS, not signs.
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, NACT, D, BATCH, "gpu"]


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent.

    ⚠ A flag given WITHOUT a value raises rather than falling back to the
    default. `--ortho` with a missing argument would otherwise run a full arm
    at 1.0 and label it 100 in the CSV — a sweep whose rows lie about what
    produced them is worse than one that refuses to start.
    """
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    # ── sweep flags (see the header) ─────────────────────────────────────
    var train_steps = atol(_flag(String("--steps"), String(TRAIN_STEPS)))
    var ortho_w = atof(_flag(String("--ortho"), String(ORTHO_WEIGHT)))
    var lr_b = atof(_flag(String("--lr-b"), String(LR_B)))
    var bc_w = atof(_flag(String("--bc"), String(BC_WEIGHT)))
    var obs_norm_on = atol(_flag(String("--obs-norm"),
                                 String(Int(OBS_NORM)))) != 0
    var tag = _flag(String("--tag"), String(""))
    var ckpt_path = String(CKPT_PATH)
    var csv_path = String(CSV_PATH)
    var run_name = String(RUN_NAME)
    if tag.byte_length() > 0:
        ckpt_path = "fb_walker_" + tag + ".ckpt"
        csv_path = "fb_walker_" + tag + "_metrics.csv"
        run_name = String(RUN_NAME) + " [" + tag + "]"
    print(
        "[0] arm: steps", train_steps, " ortho", ortho_w, " lr_b", lr_b,
        " bc", bc_w, " obs_norm", obs_norm_on, " tag '", tag, "'",
    )

    var ctx = DeviceContext()
    print("[1] loading", STORE_PATH, "...")
    var store = TrajectoryStore(String(STORE_PATH))
    var n_rows = store.n_rows()
    var qpos = ResidentColumn[DType.float32].load(store, String("qpos"))
    var qvel = ResidentColumn[DType.float32].load(store, String("qvel"))
    var action = ResidentColumn[DType.float32].load(store, String("action"))
    print("      ", n_rows, "rows")
    if n_rows < BATCH * 4:
        raise Error(
            "store has only "
            + String(n_rows)
            + " rows; collect more before training at BATCH="
            + String(BATCH)
        )

    # ⚠⚠ How many times will each transition be seen? This is the check that
    # would have prevented the first M2 launch. A 10 000-row store with a 2 M
    # step run at batch 1024 is ~205 000 epochs: the measure loss cycled to
    # +2048 and back, `actor` swung -7 -> -317 -> -7, and `|B|` dipped to 8.4
    # before recovering — a TD bootstrap oscillating on memorised data, with
    # the excursion amplitude GROWING (+-100 early, +-2000 later).
    #
    # None of that looks like a code bug and none of it is one. Offline RL on
    # dm_control normally runs 1 M - 10 M transitions; a few hundred epochs is
    # ordinary, a few hundred THOUSAND is not.
    var epochs = Float64(train_steps) * Float64(BATCH) / Float64(n_rows)
    print("       each transition will be seen ~", epochs, "times")
    if epochs > 5000.0:
        raise Error(
            "dataset far too small: "
            + String(n_rows)
            + " rows against "
            + String(train_steps)
            + " steps at batch "
            + String(BATCH)
            + " is ~"
            + String(epochs)
            + " epochs. FB will overfit and the TD"
            " bootstrap will oscillate rather than converge — this exact"
            " configuration blew up on the first M2 run. Collect ~1 M rows"
            " (raise N_EPISODES in collect_dm_control.mojo; walker runs at"
            " ~13.3 k steps/s, so 10 M transitions is ~12 minutes), or lower"
            " TRAIN_STEPS."
        )

    # ── upload the dataset ONCE ──────────────────────────────────────────
    # obs = [qpos | qvel], built on the host because it is a one-off.
    var obs_host = Tensor()
    obs_host.ensure(n_rows * OBS)
    for r in range(n_rows):
        for k in range(NQ):
            obs_host.data[r * OBS + k] = Scalar[DT](
                Float64(qpos.host[r * NQ + k])
            )
        for k in range(NV):
            obs_host.data[r * OBS + NQ + k] = Scalar[DT](
                Float64(qvel.host[r * NV + k])
            )
    # ⚠⚠ Standardise BEFORE the upload, so every consumer on device — the
    # gather kernels, B, F, the actor — sees one representation. Normalising
    # after upload, or in only some of the three gathers, is the kind of split
    # that trains fine and evaluates to noise.
    var onorm = ObsNorm[OBS]()
    if obs_norm_on:
        onorm = ObsNorm[OBS].fit(obs_host, n_rows)
        onorm.apply_rows(obs_host, n_rows)
        print("       obs standardised; dim 0 mu/sd", onorm.mu[0], onorm.sd[0],
              " dim", NQ, "mu/sd", onorm.mu[NQ], onorm.sd[NQ])
    obs_host.upload(ctx)

    var act_host = Tensor()
    act_host.ensure(n_rows * NACT)
    for i in range(n_rows * NACT):
        act_host.data[i] = Scalar[DT](Float64(action.host[i]))
    act_host.upload(ctx)

    # `next_row`, episode-safe. The last row of an episode maps to ITSELF
    # rather than to the next episode's first row — a self-transition is a
    # harmless approximation; a cross-episode one is a fabricated transition.
    #
    # ⚠ Boundaries come from the store's OWN index, not an assumed episode
    # length. The previous version tested `(r + 1) % EP_LEN == 0` against a
    # comptime 250 while the store held 1000-step episodes: every real boundary
    # was still caught (1000 is a multiple of 250), so nothing was fabricated —
    # but three quarters of the marks landed MID-EPISODE and threw away a real
    # transition each, silently, at 0.3% of rows. Nothing in the loss curve
    # could show that.
    var nxt = ctx.enqueue_create_host_buffer[IDX_DT](n_rows)
    for r in range(n_rows):
        var n = r + 1
        if n >= n_rows:
            n = r
        nxt[r] = Scalar[IDX_DT](n)
    var n_eps = store.episodes.n_episodes()
    var marked = 0
    for e in range(n_eps):
        var off = Int(store.episodes.ep_offset[e])
        var ln = Int(store.episodes.ep_len[e])
        if ln <= 0:
            continue
        var last = off + ln - 1
        if last < n_rows:
            nxt[last] = Scalar[IDX_DT](last)
            marked += 1
    # A store whose episode index disagrees with its row count would silently
    # mis-mark boundaries, which is the failure this replaced. Assert instead.
    if marked != n_eps:
        raise Error(
            "episode index is inconsistent with the row count: marked "
            + String(marked) + " of " + String(n_eps) + " episode ends"
        )
    print(
        "      episode index:", n_eps, "episodes,", marked,
        "self-transitions (", Float64(marked) * 100.0 / Float64(n_rows), "% of rows )",
    )
    var nxt_dev = ctx.enqueue_create_buffer[IDX_DT](n_rows)
    ctx.enqueue_copy(nxt_dev, nxt)
    ctx.synchronize()
    print("      uploaded obs/action/next_row to device")

    # ── device scratch ───────────────────────────────────────────────────
    var idx_s = ctx.enqueue_create_buffer[IDX_DT](BATCH)
    var idx_sn = ctx.enqueue_create_buffer[IDX_DT](BATCH)
    var idx_sp = ctx.enqueue_create_buffer[IDX_DT](BATCH)
    var samp_a = UniformDeviceSampler(n_rows, seed=UInt64(SEED))
    var samp_b = UniformDeviceSampler(n_rows, seed=UInt64(SEED) + 977)

    var t = Trainer.make(
        lr=3e-4,
        gamma=0.98,
        tau=0.01,
        ortho_weight=ortho_w,
        ctx=ctx,
        seed=UInt64(SEED) + 13,
        max_grad_norm=MAX_GRAD_NORM,
        bc_weight=bc_w,
        lr_b=lr_b,
    )
    # Size the owned batch buffers before gathering straight into them.
    t.ensure_sized()

    var gauss = Tensor()
    var pick = Tensor()
    ensure_t["gpu"](gauss, BATCH * D, ctx)
    ensure_t["gpu"](pick, BATCH * 2, ctx)
    var rng_off = UInt64(1)

    # ─── logging ─────────────────────────────────────────────────────────
    var env_vars = load_dotenv()
    var logger = CompositeLogger(
        CsvLogger(csv_path, buffer_size=64),
        RemoteLogger(
            server_url=env_vars.get("RL_MONITOR_URL", ""),
            run_name=run_name,
            buffer_size=64,
            api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
        ),
    )
    logger.set_config("algorithm", "FB")
    logger.set_config("env", "dm_control/walker-all")
    logger.set_config("store", String(STORE_PATH))
    logger.set_config("rows", String(n_rows))
    logger.set_config("d", String(D))
    logger.set_config("batch", String(BATCH))
    logger.set_config("hidden", String(HID))
    logger.set_config("train_steps", String(train_steps))
    logger.set_config("max_grad_norm", String(MAX_GRAD_NORM))
    # ⚠ The swept values are logged as CONFIG, not left implicit in the file
    # name. A CSV that records only its own name cannot be re-read six weeks
    # later without the shell history that produced it.
    logger.set_config("bc_weight", String(bc_w))
    logger.set_config("ortho_weight", String(ortho_w))
    logger.set_config("lr_b", String(lr_b if lr_b >= 0.0 else 3e-4))
    logger.set_config("obs_norm", String(obs_norm_on))
    logger.set_config("tag", tag)
    logger.set_config("cuda_graph", String(USE_TRAIN_CUDA_GRAPH))
    logger.set_config("epochs_over_dataset", String(epochs))

    # Lazily captured on the first non-logging step; replayed thereafter.
    var train_graph = Optional[CUDAGraph](None)
    comptime SQRT_D = sqrt(Float64(D))
    var t_log = perf_counter_ns()
    var last_log_step = 0
    var gn_f1 = Float64(0)
    var gn_f2 = Float64(0)
    var gn_b = Float64(0)

    print(
        "[2] training", train_steps, "steps  (d =", D, ", batch =", BATCH, ")"
    )
    print("      USE_TRAIN_CUDA_GRAPH =", USE_TRAIN_CUDA_GRAPH)
    for step in range(train_steps):
        # Two INDEPENDENT draws.
        samp_a.draw_into_device(ctx, idx_s, BATCH)
        samp_b.draw_into_device(ctx, idx_sp, BATCH)
        ctx.enqueue_function[gather_idx_kernel[BATCH]](
            nxt_dev.unsafe_ptr(),
            idx_s.unsafe_ptr(),
            idx_sn.unsafe_ptr(),
            grid_dim=_blocks(BATCH),
            block_dim=TPB,
        )

        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(),
            idx_s.unsafe_ptr(),
            t.bs.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS),
            block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(),
            idx_sn.unsafe_ptr(),
            t.bsn.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS),
            block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(),
            idx_sp.unsafe_ptr(),
            t.bsp.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS),
            block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[NACT, BATCH]](
            act_host.dev.value().unsafe_ptr(),
            idx_s.unsafe_ptr(),
            t.ba.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * NACT),
            block_dim=TPB,
        )

        # z: half uniform on the sphere, half from B(s+). The projection runs
        # unconditionally on BOTH branches — a z off the sphere trains to a
        # policy that emits plausible garbage and reports nothing.
        t.embed_sp()
        box_muller_normal_gpu[BATCH * D](
            ctx, mptr(gauss.dev.value().unsafe_ptr()), UInt64(SEED), rng_off
        )
        rng_off += UInt64(BATCH * D)
        box_muller_normal_gpu[BATCH * 2](
            ctx, mptr(pick.dev.value().unsafe_ptr()), UInt64(SEED) + 31, rng_off
        )
        rng_off += UInt64(BATCH * 2)
        ctx.enqueue_function[z_mixture_kernel[D, BATCH]](
            t.bz.dev.value().unsafe_ptr(),
            gauss.dev.value().unsafe_ptr(),
            t.b_sp.dev.value().unsafe_ptr(),
            pick.dev.value().unsafe_ptr(),
            Scalar[DT](0.5),
            Int32(BATCH),  # fixed-width: `Int` is not DevicePassable
            grid_dim=_blocks(BATCH),
            block_dim=TPB,
        )
        ctx.enqueue_function[project_sphere_kernel[D, BATCH]](
            t.bz.dev.value().unsafe_ptr(),
            Scalar[DT](sqrt(Float64(D))),
            grid_dim=_blocks(BATCH),
            block_dim=TPB,
        )

        # ⚠ Logging steps take the EAGER path: `want_loss=True` reads losses
        # back, and a D2H inside a capture is illegal. Both paths advance the
        # same device counters (RNG offset, Adam beta^t), so interleaving them
        # is consistent — the graph is simply not replayed on those steps.
        var want = (step % LOG_EVERY) == 0 or step == train_steps - 1
        var l = FBLosses(0.0, 0.0, 0.0, 0.0, 0.0)
        comptime if USE_TRAIN_CUDA_GRAPH:
            if want:
                l = t.train_step(want_loss=True)
            else:
                def _captured_step() capturing raises -> None:
                    t.train_device_kernels()

                maybe_capture_replay[_captured_step](train_graph, ctx)
        else:
            l = t.train_step(want_loss=want)
        if want:
            # ⚠ `|F|` was computed by `FBLosses` from the first version of this
            # script and NEVER PRINTED — the struct's own docstring calls
            # `f_norm` and `b_norm` "the collapse detectors" and only one of
            # them reached the log. It cost a whole 484 k-step run's worth of
            # ambiguity: the measure loss drifted from -129 to -305 and the
            # single quantity that separates "F is learning" from "F is running
            # away" was being discarded every 2000 steps.
            t.read_grad_norms(gn_f1, gn_f2, gn_b)
            var now = perf_counter_ns()
            # ⚠ Steps SINCE the last emit, not LOG_EVERY. At step 0 exactly one
            # step has run, so dividing by LOG_EVERY reported 1913 st/s against
            # a true ~73 — a 26x-wrong first point in a CSV somebody will plot.
            var since = step - last_log_step
            var sps = 0.0
            if since > 0:
                sps = Float64(since) * 1e9 / Float64(now - t_log)
            t_log = now
            last_log_step = step

            var names = List[String]()
            var vals = List[Float64]()
            names.append(String("fb/measure")); vals.append(l.measure)
            names.append(String("fb/ortho")); vals.append(l.ortho)
            names.append(String("fb/actor")); vals.append(l.actor)
            names.append(String("fb/f_norm")); vals.append(l.f_norm)
            names.append(String("fb/b_norm")); vals.append(l.b_norm)
            # Derived, because reading the raw numbers needed hand arithmetic
            # three separate times during the last run:
            #   b_norm_deficit — the sqrt(d) pin is exact, so ANY deficit is the
            #     eps floor being approached. Alarm at > 0.11 (|B| < 11.2).
            #   ortho_Q — L_ortho = Q - 2*||B||^2. With |B| pinned the anchor is
            #     a constant, so Q is the part that carries information.
            names.append(String("fb/b_norm_deficit"))
            vals.append(SQRT_D - l.b_norm)
            names.append(String("fb/ortho_Q"))
            vals.append(l.ortho + 2.0 * l.b_norm * l.b_norm)
            names.append(String("fb/grad_norm_f1")); vals.append(gn_f1)
            names.append(String("fb/grad_norm_f2")); vals.append(gn_f2)
            names.append(String("fb/grad_norm_b")); vals.append(gn_b)
            names.append(String("perf/steps_per_s")); vals.append(sps)
            logger.log_scalars(names, vals, step)

            print(
                "   step", step,
                " measure", l.measure,
                " ortho", l.ortho,
                " actor", l.actor,
                " |F|", l.f_norm,
                " |B|", l.b_norm,
                " gF", gn_f1,
                " ", sps, "st/s",
            )
            # ⚠ Read TRENDS, not signs. `L_ortho = Q - 2S` (quartic minus
            # the `-2E||B||^2` anchor) can sit either side of zero depending on
            # the scale of B: M1's UNNORMALISED B (||B|| ~ 1.9) converged
            # NEGATIVE at -2.2..-3.5, while at ||B|| ~ sqrt(128) the quartic
            # dominates and a POSITIVE value is the constrained optimum. The
            # sign is a fact about the scale, not about health.
            #
            # ⚠ With `LayerNormNoAffine` the anchor `-2S` is now a CONSTANT
            # (-2d = -256), so L_ortho is `Q - 256` and minimising it is
            # exactly minimising the cross-correlations. That is the whole
            # point of pinning the norm: the scale route out is closed, and
            # the only way left to reduce ortho is to DECORRELATE.
            #
            # ⚠⚠ **`|B|` is now a HARD INVARIANT, not a trend.** With
            # `LayerNormNoAffine[D]` it must read sqrt(128) = 11.314 at EVERY
            # step. Any drift means B's pre-norm rows have collapsed below the
            # eps floor (per-row std < ~1e-3) and the projection has stopped
            # projecting — see the BNet comment. The previous run, on learnable
            # `LayerNorm`, drifted 11.31 -> 17.54 over 100 k steps with ortho
            # rising 8.8x behind it; that mode is now unreachable, so treat
            # |B| != 11.314 as a hard fault rather than something to watch.
            #
            # The failure modes that REMAIN observable here:
            #   ortho GROWING without bound   B's rows becoming co-linear
            #                                 (directional collapse — the
            #                                 magnitude route is now closed)
            #   measure falling without bound  F running away; MAX_GRAD_NORM
            #                                  is the guard
            # The measure loss descends in every case, so it diagnoses none.
        if step > 0 and (step % CKPT_EVERY) == 0:
            # ⚠⚠ STEP-STAMPED, not a single overwritten path. FB's loss can be
            # healthy at 50 k and cycling at 116 k; overwriting means the run
            # ends holding its WORST state and the good early one is gone. That
            # happened: a stable 50 k checkpoint was replaced by a 100 k one
            # from the oscillating phase before it could be evaluated.
            var p = ckpt_path + "." + String(step)
            t.save_state(p)
            # ⚠⚠ The normalisation statistics travel WITH the checkpoint, one
            # sidecar per rung. `fb_eval_walker` loads `<ckpt>.norm` and applies
            # it or not on that basis alone — there is no eval-side flag that
            # could disagree with this run. See `fb/obs_norm.mojo`.
            if obs_norm_on:
                onorm.save(p + ".norm")
            print("      checkpoint ->", p)
    var pf = ckpt_path + ".final"
    t.save_state(pf)
    if obs_norm_on:
        onorm.save(pf + ".norm")
    # ⚠ Without this the tail of the buffer is lost — CsvLogger flushes at
    # `buffer_size`, so up to 63 entries (the most recent ones) would never
    # reach disk on a clean exit.
    logger.close()
    print("[3] done. final checkpoint ->", pf)
    print("      metrics CSV ->", csv_path)
