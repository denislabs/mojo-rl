"""FB training — SHORT run for nsys profiling.

Profiling harness around the FB GPU train step, the counterpart of
`examples/walker2d/sac_walker2d_profile_graph_nn.mojo`. Same dataset, same
architecture and same dims as `fb_train_gpu.mojo` so the numbers transfer;
only the step count is cut and logging/checkpointing removed.

Run with:
    pixi run -e nvidia nsys profile --trace=cuda --cuda-graph-trace=node \
        --stats=true mojo run -I . \
        examples/fb/fb_train_profile_gpu.mojo

Prints host wall-clock so you can compare it against nsys's GPU-busy total:
if elapsed >> GPU busy, the loop is CPU/launch-bound — the regime the capture
toggle below exists for. An FB step issues ~150 kernel launches, and until
this landed NONE of them were captured while the SAC driver captured from day
one; that asymmetry is the first thing to measure, not the arithmetic.

Profiling knobs:
  * USE_TRAIN_CUDA_GRAPH — capture the per-update device kernel sequence into a
    CUDA graph and replay it, vs eager per-kernel launch. NVIDIA only (no-op on
    Apple/Metal, so an Apple run cannot tell the two apart).
  * LOG_EVERY — logging steps take the EAGER path because `want_loss=True`
    D2Hs, which is illegal mid-capture. Set it above TRAIN_STEPS to profile a
    pure-replay loop with no eager steps at all; keep it small to profile the
    interleaving the real run actually does.
  * TRAIN_STEPS — 3000 is enough for a stable profile once the first-call
    capture (a settle run plus the captured run) has amortised.

⚠ The batch assembly ahead of the step — samplers, gathers, z mixture — stays
EAGER and is NOT captured. It is a dozen small launches per step against the
step's ~150, so it was not worth the extra capture-safety surface (both
`box_muller_normal_gpu` calls there still take HOST offsets and would freeze
under capture). If the profile shows those launches mattering, that is the next
thing to move, and it needs the device-offset treatment first.
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.random import random_float64, seed

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

from mojo_rl.cuda import CUDAGraph, maybe_capture_replay
from mojo_rl.deep_agents.fb.trainer import FBTrainer, FBLosses
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
comptime TRAIN_STEPS: Int = 3_000
comptime LOG_EVERY: Int = 1_000  # see the want_loss note in the header
comptime CKPT_EVERY: Int = 10_000_000  # effectively off
comptime CKPT_PATH: StaticString = "/tmp/fb_profile_unused.ckpt"

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


def main() raises:
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
    var epochs = Float64(TRAIN_STEPS) * Float64(BATCH) / Float64(n_rows)
    print("       each transition will be seen ~", epochs, "times")
    if epochs > 5000.0:
        raise Error(
            "dataset far too small: "
            + String(n_rows)
            + " rows against "
            + String(TRAIN_STEPS)
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
        ctx=ctx,
        seed=UInt64(SEED) + 13,
        max_grad_norm=MAX_GRAD_NORM,
        bc_weight=BC_WEIGHT,
    )
    # Size the owned batch buffers before gathering straight into them.
    t.ensure_sized()

    var gauss = Tensor()
    var pick = Tensor()
    ensure_t["gpu"](gauss, BATCH * D, ctx)
    ensure_t["gpu"](pick, BATCH * 2, ctx)
    var rng_off = UInt64(1)

    # Lazily captured on the first non-logging step; replayed thereafter.
    var train_graph = Optional[CUDAGraph](None)

    print(
        "[2] training", TRAIN_STEPS, "steps  (d =", D, ", batch =", BATCH, ")"
    )
    print("      USE_TRAIN_CUDA_GRAPH =", USE_TRAIN_CUDA_GRAPH)
    for step in range(TRAIN_STEPS):
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
        var want = (step % LOG_EVERY) == 0 or step == TRAIN_STEPS - 1
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
            print(
                "   step",
                step,
                " measure",
                l.measure,
                " ortho",
                l.ortho,
                " actor",
                l.actor,
                " |B|",
                l.b_norm,
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
            var p = String(CKPT_PATH) + "." + String(step)
            t.save_state(p)
            print("      checkpoint ->", p)
    var pf = String(CKPT_PATH) + ".final"
    t.save_state(pf)
    print("[3] done. final checkpoint ->", pf)
