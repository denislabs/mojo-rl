"""`FBOnlineAgent` smoke gate — the online FB loop on a batched GPU env.

`docs/BFM_ZERO_SHOT_RL.md` §18.3 A3. Pendulum (`BatchedGpuEnv[PendulumV2]`),
four lanes, tiny nets: this gates WIRING, not learning. Every check below is
a property the loss curve could not show:

  [1] every lane's `z` sits on the radius-sqrt(D) sphere at init and after
      `z_hold` resamples, and every lane has CHANGED at least once by then —
      a lane whose z never moves trains a single-task policy.
  [2] the replay ring WRAPS: after 3·CAP env steps the fill is CAP, the write
      head is back at 0, and every stored `z` is on the sphere.
  [3] `relabel_ratio` does what it says: `keep_frac = 1` keeps every stored
      row, `0` keeps none, `0.2` keeps a binomial share — checked on a marker
      z, and every relabelled row is back on the sphere.
  [4] the driver runs end to end: train steps happen, the flush prints finite
      diagnostics, and `B`'s rows have not collapsed onto one direction.
  [5] capture safety: two `train_device_kernels` calls draw DIFFERENT batch z
      and move `B` differently — the device counters advance in-sequence.
  [6] a checkpoint written by the online agent loads into the plain CPU
      `FBTrainer` the eval scripts build, and `B` agrees with the GPU copy.

Run (Apple Metal or NVIDIA):
    pixi run -e apple mojo run -I . tests/deep_agents/test_fb_online_smoke.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.deep_agents.fb.online import FBOnlineAgent
from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.training.batched_env import BatchedGpuEnv
from mojo_rl.deep_agents.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)
from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS = 3
comptime ACT = 1
comptime D = 16
comptime BATCH = 64
comptime CAP = 1024
comptime LANES = 4
comptime ZBUF = 256
comptime HID = 32
comptime SEED = 20260907
comptime Z_HOLD = 20
comptime LEARNING_STARTS = 256
comptime SQRT_D = sqrt(Float64(D))

comptime FNet = Sequential[Linear[OBS + ACT + D, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[OBS + D, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]
comptime Agent = FBOnlineAgent[
    FNet, BNet, ANet, OBS, ACT, D, BATCH, CAP, LANES, ZBUF
]
comptime CpuTrainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]
comptime EnvT = BatchedGpuEnv[PendulumV2[DT], LANES, OBS, ACT]


def _download_rows(
    ctx: DeviceContext, ref src: Tensor, n: Int
) raises -> Tensor:
    """Host copy of a device-only tensor (the ring has no host mirror)."""
    var h = Tensor.alloc(n)
    h.ensure_gpu(ctx, n)
    ctx.enqueue_copy(h.dev.value(), src.dev.value())
    h.download(ctx)
    return h^


def _rows_on_sphere(ref t: Tensor, rows: Int, tol: Float64) -> Int:
    """Rows whose norm is within `tol` of sqrt(D)."""
    var ok = 0
    for i in range(rows):
        var s = Float64(0)
        for k in range(D):
            var v = Float64(t.data[i * D + k])
            s += v * v
        if abs(sqrt(s) - SQRT_D) < tol:
            ok += 1
    return ok


def _rt(n: Int, ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    t.upload(ctx)
    return t^


def _make(ctx: DeviceContext) raises -> Agent:
    return Agent.make(
        ctx,
        lr=1e-3,
        learning_starts=LEARNING_STARTS,
        action_scale=2.0,
        z_hold=Z_HOLD,
        window_size=8,
        seed=UInt64(SEED),
    )


def test_lane_z(ctx: DeviceContext) raises:
    print("[1] per-lane z: on the sphere, and resampled within z_hold ...")
    var a = _make(ctx)
    var z0 = _download_rows(ctx, a.z_lane, LANES * D)
    var on0 = _rows_on_sphere(z0, LANES, 1e-4)
    print("      init: on sphere", on0, "/", LANES)
    assert_true(on0 == LANES, "a lane's initial z is off the sphere")

    var changed = List[Bool](length=LANES, fill=False)
    for _ in range(Z_HOLD + 1):
        a._resample_lanes(force=False)
        a._act_iter += 1
        var z1 = _download_rows(ctx, a.z_lane, LANES * D)
        assert_true(
            _rows_on_sphere(z1, LANES, 1e-4) == LANES,
            "a lane left the sphere after a resample",
        )
        for e in range(LANES):
            var d = Float64(0)
            for k in range(D):
                d += abs(Float64(z1.data[e * D + k]) - Float64(z0.data[e * D + k]))
            if d > 1e-6:
                changed[e] = True
    var n_changed = 0
    for e in range(LANES):
        if changed[e]:
            n_changed += 1
    print("      lanes resampled within", Z_HOLD + 1, "iters:", n_changed, "/", LANES)
    assert_true(
        n_changed == LANES,
        "a lane's z never changed across z_hold+1 iterations — the stagger"
        " or the hold test is wrong and that lane trains one task forever",
    )
    print("      OK")


def _kept(
    ctx: DeviceContext, mut ag: Agent, keep: Float64,
    mut s: Tensor, mut ac: Tensor, mut sn: Tensor, mut sp: Tensor,
    mut zm: Tensor,
) raises -> Int:
    """Rows still equal to the marker after one relabel at `keep`."""
    ag.keep_frac = keep
    ag.t.load_batch(s, ac, sn, sp, zm)
    ag._relabel_z()
    var out = _download_rows(ctx, ag.t.bz, BATCH * D)
    assert_true(
        _rows_on_sphere(out, BATCH, 1e-4) == BATCH,
        "a relabelled z row is off the sphere",
    )
    var kept = 0
    for i in range(BATCH):
        var d = Float64(0)
        for k in range(D):
            d += abs(Float64(out.data[i * D + k]) - Float64(zm.data[i * D + k]))
        if d < 1e-6:
            kept += 1
    return kept


def test_relabel(ctx: DeviceContext) raises:
    print("[3] relabel: keep_frac 1 keeps all, 0 keeps none, 0.2 a share ...")
    var a = _make(ctx)
    # Marker z: sqrt(D) on axis 0 — already on the sphere, so a KEPT row is
    # exactly the marker and a relabelled one is not.
    var zm = Tensor.alloc(BATCH * D)
    for i in range(BATCH):
        for k in range(D):
            zm.data[i * D + k] = Scalar[DT](SQRT_D if k == 0 else 0.0)
    zm.upload(ctx)
    var s = _rt(BATCH * OBS, ctx)
    var ac = _rt(BATCH * ACT, ctx)
    var sn = _rt(BATCH * OBS, ctx)
    var sp = _rt(BATCH * OBS, ctx)

    var k1 = _kept(ctx, a, 1.0, s, ac, sn, sp, zm)
    var k0 = _kept(ctx, a, 0.0, s, ac, sn, sp, zm)
    var k2 = _kept(ctx, a, 0.2, s, ac, sn, sp, zm)
    print("      kept: keep_frac=1.0 ->", k1, " 0.0 ->", k0, " 0.2 ->", k2, "of", BATCH)
    assert_true(k1 == BATCH, "keep_frac=1.0 relabelled a row")
    assert_true(k0 == 0, "keep_frac=0.0 kept a row")
    # Binomial(64, 0.2): mean 12.8, sd 3.2. ±4 sd.
    assert_true(k2 >= 1 and k2 <= 26, "keep_frac=0.2 kept an implausible share")
    print("      OK")


def test_driver_ring_and_capture(ctx: DeviceContext) raises:
    print("[2]+[4]+[5]+[6] driver end to end, ring wrap, capture, checkpoint ...")
    seed(SEED)
    var a = _make(ctx)
    var env = EnvT(ctx)
    comptime TOTAL = 3 * CAP
    _ = run_offpolicy_train_batched[
        Agent, EnvT, N_ENVS=LANES,
        USE_TRAIN_CUDA_GRAPH=False, USE_ENV_CUDA_GRAPH=False,
    ](
        Optional(ctx), a, env, TOTAL,
        rng_seed=UInt64(SEED), updates_per_step=1,
        print_every=CAP, verbose=True,
    )
    ctx.synchronize()

    print("      [2] ring: size", a.size, " pos", a.pos, " (CAP", CAP, ")")
    assert_true(a.size == CAP, "ring did not fill to CAP after 3*CAP steps")
    assert_true(a.pos == 0, "write head did not wrap back to 0")
    var rz = _download_rows(ctx, a.r_z, CAP * D)
    var on = _rows_on_sphere(rz, CAP, 1e-3)
    print("      [2] stored z on sphere:", on, "/", CAP)
    assert_true(on == CAP, "a stored z row is off the sphere")

    print("      [4] train steps:", a.total_train_steps())
    var want_min = (TOTAL - LEARNING_STARTS) // LANES - 2
    assert_true(
        a.total_train_steps() >= want_min,
        "too few train steps (" + String(a.total_train_steps()) + " < "
        + String(want_min) + ") — the warmup gate or the size gate is wrong",
    )
    var measure = Float64(0)
    var ortho = Float64(0)
    var actor = Float64(0)
    var fnv = Float64(0)
    var bn = Float64(0)
    a.peek_losses(measure, ortho, actor, fnv, bn)
    print("      [4] losses: measure", measure, " ortho", ortho, " actor", actor,
          " |F|", fnv, " |B|", bn)
    assert_true(measure == measure and ortho == ortho, "loss is NaN")
    assert_true(abs(bn - SQRT_D) < 1e-2, "|B| off the sqrt(D) pin: " + String(bn))

    # B row spread on a probe: collapsed B => every row the same direction.
    var probe = _rt(BATCH * OBS, ctx)
    var b0 = Tensor()
    a.t.backward_embed[BATCH](probe, b0)
    b0.download(ctx)
    var spread = Float64(0)
    var pairs = 0
    for i in range(0, BATCH, 4):
        for j in range(i + 1, BATCH, 4):
            var d = Float64(0)
            for k in range(D):
                d += abs(Float64(b0.data[i * D + k]) - Float64(b0.data[j * D + k]))
            spread += d
            pairs += 1
    spread /= Float64(pairs)
    print("      [4] B row spread (mean L1 between probe rows):", spread)
    assert_true(spread > 0.1, "B rows collapsed onto one direction")

    print("      [5] two captured-path steps draw different z and move B ...")
    a.train_device_kernels()
    var z1 = _download_rows(ctx, a.t.bz, BATCH * D)
    var b1 = Tensor()
    a.t.backward_embed[BATCH](probe, b1)
    b1.download(ctx)
    a.train_device_kernels()
    var z2 = _download_rows(ctx, a.t.bz, BATCH * D)
    var b2 = Tensor()
    a.t.backward_embed[BATCH](probe, b2)
    b2.download(ctx)
    var dz = Float64(0)
    var db = Float64(0)
    for i in range(BATCH * D):
        var x = abs(Float64(z1.data[i]) - Float64(z2.data[i]))
        var y = abs(Float64(b1.data[i]) - Float64(b2.data[i]))
        if x > dz:
            dz = x
        if y > db:
            db = y
    print("      [5] max|Δz|", dz, " max|ΔB|", db)
    assert_true(dz > 1e-4, "the batch z did not change between two device steps")
    assert_true(db > 1e-9, "B did not move on the second device step")

    print("      [6] checkpoint loads into the CPU FBTrainer ...")
    var path = String("/tmp/fb_online_smoke.ckpt")
    a.save_state(path)
    var tc = CpuTrainer.make(lr=1e-3, ctx=None)
    tc.load_state(path)
    var probe_h = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        probe_h.data[i] = probe.data[i]
    var bg = Tensor()
    a.t.backward_embed[BATCH](probe, bg)
    bg.download(ctx)
    var bc = Tensor()
    tc.backward_embed[BATCH](probe_h, bc)
    var worst = Float64(0)
    for i in range(BATCH * D):
        var d = abs(Float64(bg.data[i]) - Float64(bc.data[i]))
        if d > worst:
            worst = d
    print("      [6] max |B_gpu - B_cpu| on the probe:", worst)
    assert_true(worst < 1e-3, "checkpoint round trip disagrees: " + String(worst))
    print("      OK")


def main() raises:
    print("=" * 70)
    print("FB online agent smoke — Pendulum, 4 lanes")
    print("=" * 70)
    var ctx = DeviceContext()
    test_lane_z(ctx)
    test_relabel(ctx)
    test_driver_ring_and_capture(ctx)
    print("\n[PASS] FB online smoke")
