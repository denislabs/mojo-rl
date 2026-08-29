# +--------------------------------------------------------------------------+ #
# | ACT device-resident logging — the window must equal what the D2H reported
# +--------------------------------------------------------------------------+ #
"""`train_metrics` / `val_metrics` against the values `_read_terms` downloads.

    pixi run -e apple  mojo run -I . tests/deep_agents/act/test_act_device_metrics.mojo
    pixi run -e nvidia mojo run -I . tests/deep_agents/act/test_act_device_metrics.mojo

The device accumulators exist to delete the per-step D2H from an ACT step
(`docs/ACT_GPU_DATA_PATH.md`, "no D2H in the step"). What makes that worth
gating is the failure mode: a wrong reduction, a dropped fold or a stale
accumulator all produce a *plausible* number. The training curve would still
descend; it would just be reporting something other than the loss, and nothing
downstream would notice.

So this compares the two paths on the SAME forward. `eval_step_resident`
downloads the three `[BATCH]` loss vectors and means them on the host;
`eval_step_resident_accum` reduces the identical buffers on device into the
window. In eval mode the forward is deterministic — the latent is pinned,
dropout is off and BatchNorm reads running statistics — so repeating it must
give the same numbers, and the window mean over N repeats must equal the single
downloaded value.

⚠ VACUITY IS THE FAILURE MODE HERE. A comparison of two zeros passes, and both
paths report zero if the accumulator is never fed. Every check below therefore
also asserts the value is nonzero, and the count is checked separately from the
means — `n` is the host counter, the means come off the device, and a fold that
silently no-ops would leave the count right and the mean wrong.

A STUB backbone for the same reason `test_act_gpu_vs_cpu` uses one: this gates
the metrics plumbing, which does not care what produced the loss.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.act.trainer import ACTTrainer


comptime QPOS = 6
comptime ADIM = 6
comptime N_CAM = 1
comptime IMG_H = 64
comptime IMG_W = 64
comptime K = 5
comptime DIM = 16
comptime HEADS = 2
comptime FF = 32
comptime LATENT = 8
comptime N_ENC = 1
comptime N_DEC = 1
comptime BATCH = 4
comptime P = 0.0

comptime FEAT_CH = 8
comptime STUB = Sequential[
    Conv2DBatchNormReLU[3, FEAT_CH, 3, 2, 1, IMG_H, IMG_W],
    Conv2DBatchNormReLU[FEAT_CH, FEAT_CH, 3, 2, 1, IMG_H // 2, IMG_W // 2],
]
comptime SOH = IMG_H // 4
comptime SOW = IMG_W // 4

comptime TG = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH, P, "gpu", FEAT_CH, SOH, SOW, STUB,
]
comptime TC = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH, P, "cpu", FEAT_CH, SOH, SOW, STUB,
]
"""The CPU leg exists to make the host branch of `_accum_terms` REACHABLE.

`comptime if` prunes, so a CPU-only type error in a method no CPU caller
invokes is invisible — the GPU gate above would pass over a branch that never
compiles. Instantiating a CPU trainer and running the same comparison forces
it, and checks the host arithmetic while it is there."""
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W
comptime ENC_SEQ = K + 2

# One forward, meaned two ways, on the same buffers. The gap is the reduction
# ORDER (a block tree-sum on device vs a left-to-right host sweep over BATCH
# values) plus, on NVIDIA, nothing at all — the values compared are already
# computed, not recomputed. fp32 over 4 values leaves this far tighter than the
# tolerance; it is loose enough only to not be brittle.
comptime RTOL: Float64 = 1e-5
comptime ATOL: Float64 = 1e-7

comptime REPEATS = 3
"""Folds per window. >1 on purpose: with a single fold, `sum/count` equals the
value whether or not `count` is being advanced, so the divisor is untested."""


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def close(a: Float64, b: Float64) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    var m = a if a > 0.0 else -a
    var mb = b if b > 0.0 else -b
    if mb > m:
        m = mb
    return d <= ATOL + RTOL * m


def nonzero(v: Float64) -> Bool:
    """A metric that is exactly 0 is the vacuous pass this gate exists to
    avoid — an unfed accumulator and an unfed host read agree perfectly."""
    return v > 1e-12 or v < -1e-12


def main() raises:
    var fails = 0
    print("ACT device-metrics gate (window == downloaded value)")
    print("")

    var ctx = DeviceContext()
    print("  device: " + String(ctx.name()))
    var tr = TG.make(lr=Scalar[DT](1e-4), ctx=ctx)

    # ── one fixed batch, structured rather than noise ────────────────────
    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * ENC_SEQ)
    for i in range(len(qpos)):
        qpos[i] = Scalar[DT](0.01 * Float64(i % 17) - 0.08)
    for i in range(len(images)):
        images[i] = Scalar[DT](0.005 * Float64(i % 23) - 0.05)
    for i in range(len(actions)):
        actions[i] = Scalar[DT](0.02 * Float64(i % 13) - 0.12)
    for i in range(len(valid)):
        valid[i] = Scalar[DT](1.0)

    # ── 1. the window mean equals the downloaded mean ────────────────────
    # `eval_step` seeds the inputs and reads the terms the old way; the
    # `_resident` calls that follow re-run the SAME forward on the SAME slots.
    var r = tr.eval_step(qpos, images, actions, valid)
    for _ in range(REPEATS):
        tr.eval_step_resident_accum()
    var w = tr.val_metrics()

    check(fails, "val window is not empty", w.n == REPEATS,
          "n=" + String(w.n) + " expected " + String(REPEATS))
    check(fails, "downloaded loss is nonzero", nonzero(r.loss),
          "loss=" + String(r.loss))
    check(fails, "downloaded l1 is nonzero", nonzero(r.l1),
          "l1=" + String(r.l1))
    check(fails, "window loss == downloaded loss", close(w.loss, r.loss),
          String(w.loss) + " vs " + String(r.loss))
    check(fails, "window l1   == downloaded l1", close(w.l1, r.l1),
          String(w.l1) + " vs " + String(r.l1))
    check(fails, "window kl   == downloaded kl", close(w.kl, r.kl),
          String(w.kl) + " vs " + String(r.kl))

    # ── 2. the flush actually resets ─────────────────────────────────────
    # A window that is never cleared reports a lifetime mean, which drifts away
    # from the truth slowly enough to look like convergence.
    var w2 = tr.val_metrics()
    check(fails, "flush cleared the window", w2.n == 0 and w2.loss == 0.0,
          "n=" + String(w2.n) + " loss=" + String(w2.loss))

    # ── 3. peek does NOT reset ───────────────────────────────────────────
    tr.eval_step_resident_accum()
    var p1 = tr.val_metrics(False)
    var p2 = tr.val_metrics(False)
    check(fails, "peek leaves the window intact",
          p1.n == 1 and p2.n == 1 and close(p1.loss, p2.loss),
          "n=" + String(p1.n) + "," + String(p2.n))
    _ = tr.val_metrics()

    # ── 4. the grad norm, folded off its device buffer ───────────────────
    # ⚠ Reaching into `_accum_grad_norm` / `_train_acc` on purpose. The public
    # entry point is `train_step_device_accum`, which needs an
    # `ACTDeviceDataset` — a real HDF5 store this gate has no business
    # requiring to check a one-element reduction. What is being checked is the
    # wiring: that `Adam.clip_norm_dev` names the SAME buffer `read_clip_norm`
    # downloads, and that a `[1]` block reduction returns the value rather than
    # a zero from an empty tree-sum.
    var rt = tr.train_step(qpos, images, actions, valid)
    check(fails, "train_step reports a nonzero grad norm",
          nonzero(rt.grad_norm), "gn=" + String(rt.grad_norm))
    check(fails, "the optimizer has a device clip-norm buffer",
          tr.opt.has_clip_norm_dev())

    var tw0 = tr.train_metrics()
    check(fails, "train window starts empty",
          tw0.n == 0 and tw0.grad_norm == 0.0, "n=" + String(tw0.n))

    # `train_step` left the pre-clip norm in the device buffer and the loss
    # nodes in the graph, and nothing has written either since — so folding a
    # full step's worth now must reproduce what it downloaded. Both terms
    # together, because that is the shape the real step folds; folding the
    # grad norm alone would exercise a window no caller ever builds.
    for _ in range(REPEATS):
        tr._accum_terms[False]()
        tr._accum_grad_norm()
    var tw = tr.train_metrics()
    check(fails, "train window counts the folds", tw.n == REPEATS,
          "n=" + String(tw.n) + " expected " + String(REPEATS))
    check(fails, "window grad_norm == read_clip_norm",
          close(tw.grad_norm, rt.grad_norm),
          String(tw.grad_norm) + " vs " + String(rt.grad_norm))
    check(fails, "window grad_norm is nonzero", nonzero(tw.grad_norm),
          "gn=" + String(tw.grad_norm))
    check(fails, "window loss == train_step loss", close(tw.loss, rt.loss),
          String(tw.loss) + " vs " + String(rt.loss))

    # ── 5. the CPU leg ───────────────────────────────────────────────────
    # Same comparison, host accumulators. Not a parity check against the GPU
    # numbers (different weights — the two trainers init independently); a
    # check that the CPU branch exists, compiles, and means what it says.
    print("")
    print("  CPU leg:")
    var trc = TC.make(lr=Scalar[DT](1e-4))
    var rc = trc.eval_step(qpos, images, actions, valid)
    for _ in range(REPEATS):
        trc.eval_step_resident_accum()
    var wc = trc.val_metrics()
    check(fails, "cpu window is not empty", wc.n == REPEATS,
          "n=" + String(wc.n))
    check(fails, "cpu downloaded loss is nonzero", nonzero(rc.loss),
          "loss=" + String(rc.loss))
    check(fails, "cpu window loss == step loss", close(wc.loss, rc.loss),
          String(wc.loss) + " vs " + String(rc.loss))
    check(fails, "cpu window l1   == step l1", close(wc.l1, rc.l1),
          String(wc.l1) + " vs " + String(rc.l1))
    check(fails, "cpu flush cleared the window",
          trc.val_metrics().n == 0)

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        raise Error("device-metrics gate failed")
