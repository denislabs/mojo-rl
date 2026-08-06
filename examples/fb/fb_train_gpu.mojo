"""M2 — FB training on GPU, d = 128, from a collected dm_control dataset.

This is the run `docs/BFM_ZERO_SHOT_RL.md` §13 budgets at 2 M gradient steps and
"a few hours on a rented 4090". Everything it needs is device-resident: the
dataset is uploaded once, the sampler writes indices on device, and the batch is
assembled by the gather/pack kernels. Nothing crosses PCIe in the training loop
except the occasional logged loss.

Prerequisite: run `examples/fb/collect_dm_control.mojo` first. This reads the
store it writes.

    pixi run -e nvidia mojo run -I . examples/fb/fb_train_gpu.mojo

⚠⚠ **`want_loss` is on a stride, and that is not cosmetic.** Reading a loss back
from the GPU is a full pipeline stall. At 2 M steps, logging every step is the
difference between a few hours and most of a day, and the GRADIENTS are
identical either way because the loss value never enters the update.

⚠ `s'` is taken through a precomputed `next_row` table rather than `idx + 1`.
Without it, a row sampled at an episode boundary pairs a terminal state with the
NEXT episode's reset — a transition that never happened, injected into the
measure loss at a rate of 1/EP_LEN. With EP_LEN = 250 that is 0.4% of every
batch, which is small enough to never look wrong and large enough to matter.

⚠ `s+` comes from a SECOND, INDEPENDENT draw. It is not `s'`. See
`fb/loss.mojo`: the successor measure asks "starting from s, how often is s+
visited", and if `s+` were the batch's own next-states the matrix would only
ever be evaluated on pairs one step apart.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.random import random_float64, seed

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.nn.random.box_muller import box_muller_normal_gpu

from mojo_rl.data.store import TrajectoryStore
from mojo_rl.data.resident import ResidentColumn, IDX_DT
from mojo_rl.data.sampler import UniformDeviceSampler

from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb.kernels import (
    gather_rows_kernel, gather_idx_kernel, z_mixture_kernel,
    project_sphere_kernel, ensure_t, _blocks,
)


# ── the dataset this reads ───────────────────────────────────────────────
comptime STORE_PATH: StaticString = "/tmp/fb_walker_wide.h5"
comptime NQ: Int = 9
comptime NV: Int = 9
comptime NACT: Int = 6
comptime OBS: Int = NQ + NV
comptime EP_LEN: Int = 250

# ── the run ──────────────────────────────────────────────────────────────
comptime D: Int = 128            # §13: 128 at M2 (50 was the M1 setting)
comptime BATCH: Int = 1024
comptime HID: Int = 1024
comptime TRAIN_STEPS: Int = 2_000_000
comptime LOG_EVERY: Int = 2000   # see the want_loss note in the header
comptime CKPT_EVERY: Int = 50_000
comptime CKPT_PATH: StaticString = "/tmp/fb_walker_d128.ckpt"

# ⚠⚠ Global grad-norm clip. FB's measure loss scales as (||F||·sqrt(d))^2 and
# was measured spiking to +2559 on walker at 1 M rows; the gradients spike with
# it. `L_ortho` and the LayerNorm bound `B`, and NOTHING bounds `F` — clipping
# is the cheap standard remedy and every reference FB implementation uses one.
# 0 disables it.
comptime MAX_GRAD_NORM: Float64 = 1.0
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
# At the orthonormality optimum `L_ortho = E[(B·B+)^2] - 2E[||B||^2]` is
# NEGATIVE (M1's converged point_mass run sat at -2.2 .. -3.5). A positive,
# growing ortho loss means B is running away from orthonormal and the
# regulariser is losing to the anchor term, which is unbounded below when F
# and B both grow. Watch the SIGN of ortho, not just `|B|`.
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNorm[D]
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
            "store has only " + String(n_rows) + " rows; collect more before"
            " training at BATCH=" + String(BATCH)
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
    var epochs = (
        Float64(TRAIN_STEPS) * Float64(BATCH) / Float64(n_rows)
    )
    print("       each transition will be seen ~", epochs, "times")
    if epochs > 5000.0:
        raise Error(
            "dataset far too small: " + String(n_rows) + " rows against "
            + String(TRAIN_STEPS) + " steps at batch " + String(BATCH)
            + " is ~" + String(epochs) + " epochs. FB will overfit and the TD"
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
    var nxt = ctx.enqueue_create_host_buffer[IDX_DT](n_rows)
    for r in range(n_rows):
        var n = r + 1
        if n >= n_rows or (n % EP_LEN) == 0:
            n = r
        nxt[r] = Scalar[IDX_DT](n)
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
        lr=3e-4, gamma=0.98, tau=0.01, ctx=ctx, seed=UInt64(SEED) + 13,
        max_grad_norm=MAX_GRAD_NORM,
    )
    # Size the owned batch buffers before gathering straight into them.
    t.ensure_sized()

    var gauss = Tensor()
    var pick = Tensor()
    ensure_t["gpu"](gauss, BATCH * D, ctx)
    ensure_t["gpu"](pick, BATCH * 2, ctx)
    var rng_off = UInt64(1)

    print("[2] training", TRAIN_STEPS, "steps  (d =", D, ", batch =", BATCH, ")")
    for step in range(TRAIN_STEPS):
        # Two INDEPENDENT draws.
        samp_a.draw_into_device(ctx, idx_s, BATCH)
        samp_b.draw_into_device(ctx, idx_sp, BATCH)
        ctx.enqueue_function[gather_idx_kernel[BATCH]](
            nxt_dev.unsafe_ptr(), idx_s.unsafe_ptr(), idx_sn.unsafe_ptr(),
            grid_dim=_blocks(BATCH), block_dim=TPB,
        )

        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(), idx_s.unsafe_ptr(),
            t.bs.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS), block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(), idx_sn.unsafe_ptr(),
            t.bsn.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS), block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(), idx_sp.unsafe_ptr(),
            t.bsp.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS), block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[NACT, BATCH]](
            act_host.dev.value().unsafe_ptr(), idx_s.unsafe_ptr(),
            t.ba.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * NACT), block_dim=TPB,
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
            Scalar[DT](0.5), BATCH,
            grid_dim=_blocks(BATCH), block_dim=TPB,
        )
        ctx.enqueue_function[project_sphere_kernel[D, BATCH]](
            t.bz.dev.value().unsafe_ptr(), Scalar[DT](sqrt(Float64(D))),
            grid_dim=_blocks(BATCH), block_dim=TPB,
        )

        var want = (step % LOG_EVERY) == 0 or step == TRAIN_STEPS - 1
        var l = t.train_step(want_loss=want)
        if want:
            print(
                "   step", step, " measure", l.measure, " ortho", l.ortho,
                " actor", l.actor, " |B|", l.b_norm,
            )
            # ⚠ Read TRENDS, not signs. With `LayerNorm[D]` on B, ||B|| is
            # pinned near sqrt(D) = 11.3 (observed ~13.6 once the learned gain
            # settles), and at that scale `E[(B·B+)^2]` (~||B||^4) outweighs
            # `-2E[||B||^2]`, so a POSITIVE ortho is the constrained optimum —
            # not a failure. An earlier version of this comment demanded a
            # NEGATIVE ortho, which was read off M1's UNNORMALISED B where
            # ||B|| ~ 1.9; that criterion does not transfer.
            #
            # The two real failure modes:
            #   |B| -> 0                  collapse; L_ortho exists to stop it
            #   ortho GROWING without bound   B running away from orthonormal
            # Measured healthy here: measure ~-70 flat, |B| 13.5-14.5 flat,
            # ortho oscillating 16-140 with no trend, over 54 k steps.
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
