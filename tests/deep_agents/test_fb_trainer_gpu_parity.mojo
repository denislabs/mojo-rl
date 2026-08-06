"""`FBTrainer` CPU vs GPU parity — the control on the M2 GPU path.

`FBTrainer` is written ONCE and dispatches on its `TARGET` parameter. That is
the design that avoids a duplicated 400-line step body drifting from its twin,
and this file is what makes the claim checkable: identical initial weights and
identical batches must produce identical losses and identical `B` outputs on
both devices.

⚠⚠ **Exploration noise is turned OFF (`policy_noise = 0`).** The CPU path draws
its target-policy smoothing noise from the host RNG and the GPU path from
Philox — deliberately, because the alternative is a host draw uploaded every
step, i.e. a PCIe round trip in the hot loop to make a test easier. With noise
on, the two devices see different target actions and there is nothing to
compare. Turning it off is what makes the REST of the step comparable; it also
means this gate does NOT cover `smooth_action_t`, which is checked separately
below on a fixed noise buffer.

The tolerance is 1e-5 relative, chosen from the MEASURED agreement rather than
guessed: at BATCH=16 the two devices agree to ~7e-8 on every loss, ~6e-8 on
`B(s)`, and `smooth_action_t` is bit-exact. The bound keeps ~100x headroom for
the reduction-order divergence to grow with BATCH — `block.sum` over a strided
grid versus a sequential host loop, and fp32 addition is not associative — while
staying tight enough to fail on a dropped or mis-scaled term. The first version
of this file used a defensive 2e-3, which would have passed a regression three
orders of magnitude larger than anything real.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_trainer_gpu_parity.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh, ReLU
from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb.kernels import smooth_action_t, ensure_t


comptime OBS: Int = 5
comptime ACT: Int = 3
comptime D: Int = 8
comptime BATCH: Int = 16
comptime HID: Int = 24
comptime STEPS: Int = 6
comptime SEED: Int = 20260805
comptime REL_TOL: Float64 = 1e-5

comptime F_IN = OBS + ACT + D
comptime A_IN = OBS + D

comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[Linear[OBS, HID], ReLU[HID], Linear[HID, D]]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]

comptime CpuTrainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]
comptime GpuTrainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "gpu"]


def _fill(mut t: Tensor, n: Int, a: Float64, b: Float64) raises:
    """Deterministic, and NOT the same pattern for every buffer — a shared
    pattern would let an index mix-up between s / s' / s+ go unnoticed."""
    ensure_t["cpu"](t, n, None)
    for i in range(n):
        t.data[i] = Scalar[DT](
            a * Float64((i * 7) % 13) - b * Float64((i * 3) % 5) + 0.1
        )


def _rel(x: Float64, y: Float64) -> Float64:
    var d = abs(x - y)
    var m = abs(x) if abs(x) > abs(y) else abs(y)
    return d / m if m > 1.0 else d


def test_step_parity() raises:
    print("[1] CPU vs GPU:", STEPS, "steps, noise off ...")
    var ctx = DeviceContext()

    # `Deterministic` init: both trainers must start from the SAME weights, or
    # nothing downstream is comparable.
    var tc = CpuTrainer.make[Deterministic](
        lr=1e-3, ortho_weight=1.0, ctx=None
    )
    var tg = GpuTrainer.make[Deterministic](
        lr=1e-3, ortho_weight=1.0, ctx=ctx
    )
    tc.policy_noise = 0.0
    tg.policy_noise = 0.0

    var worst_m = Float64(0)
    var worst_o = Float64(0)
    var worst_a = Float64(0)
    var moved = Float64(0)

    for step in range(STEPS):
        var s = Tensor()
        var a = Tensor()
        var sn = Tensor()
        var sp = Tensor()
        var z = Tensor()
        _fill(s, BATCH * OBS, 0.11 + 0.01 * Float64(step), 0.07)
        _fill(a, BATCH * ACT, 0.13, 0.05 + 0.01 * Float64(step))
        _fill(sn, BATCH * OBS, 0.09, 0.12)
        _fill(sp, BATCH * OBS, 0.17, 0.03)
        _fill(z, BATCH * D, 0.21, 0.09)

        tc.load_batch(s, a, sn, sp, z)
        var lc = tc.train_step()

        # The GPU trainer needs the same numbers ON DEVICE. `load_batch` sizes
        # the owned buffers and uploads through the same path production uses.
        var sg = Tensor()
        var ag = Tensor()
        var sng = Tensor()
        var spg = Tensor()
        var zg = Tensor()
        _fill(sg, BATCH * OBS, 0.11 + 0.01 * Float64(step), 0.07)
        _fill(ag, BATCH * ACT, 0.13, 0.05 + 0.01 * Float64(step))
        _fill(sng, BATCH * OBS, 0.09, 0.12)
        _fill(spg, BATCH * OBS, 0.17, 0.03)
        _fill(zg, BATCH * D, 0.21, 0.09)
        sg.upload(ctx)
        ag.upload(ctx)
        sng.upload(ctx)
        spg.upload(ctx)
        zg.upload(ctx)
        tg.load_batch(sg, ag, sng, spg, zg)
        var lg = tg.train_step()

        var rm = _rel(lc.measure, lg.measure)
        var ro = _rel(lc.ortho, lg.ortho)
        var ra = _rel(lc.actor, lg.actor)
        if rm > worst_m:
            worst_m = rm
        if ro > worst_o:
            worst_o = ro
        if ra > worst_a:
            worst_a = ra
        if abs(lc.measure) > moved:
            moved = abs(lc.measure)
        print(
            "      step", step, " measure cpu", lc.measure, " gpu", lg.measure,
        )

    print(
        "      worst relative: measure", worst_m, " ortho", worst_o,
        " actor", worst_a,
    )
    # A loss pinned at zero on both devices would pass any tolerance.
    assert_true(
        moved > 1e-4,
        "the measure loss never left zero on either device — this gate would"
        " pass on a trainer that computes nothing",
    )
    assert_true(worst_m < REL_TOL, "measure loss: " + String(worst_m))
    assert_true(worst_o < REL_TOL, "ortho loss: " + String(worst_o))
    assert_true(worst_a < REL_TOL, "actor loss: " + String(worst_a))


def test_backward_embed_parity() raises:
    """`B(s)` after training must agree — the losses could match while the
    PARAMETERS diverged, if a gradient reached the optimizer on only one path.
    """
    print("[2] B(s) after", STEPS, "steps agrees across devices ...")
    var ctx = DeviceContext()
    var tc = CpuTrainer.make[Deterministic](lr=1e-3, ctx=None)
    var tg = GpuTrainer.make[Deterministic](lr=1e-3, ctx=ctx)
    tc.policy_noise = 0.0
    tg.policy_noise = 0.0

    for step in range(STEPS):
        var s = Tensor()
        var a = Tensor()
        var sn = Tensor()
        var sp = Tensor()
        var z = Tensor()
        _fill(s, BATCH * OBS, 0.11 + 0.01 * Float64(step), 0.07)
        _fill(a, BATCH * ACT, 0.13, 0.05)
        _fill(sn, BATCH * OBS, 0.09, 0.12)
        _fill(sp, BATCH * OBS, 0.17, 0.03)
        _fill(z, BATCH * D, 0.21, 0.09)
        tc.load_batch(s, a, sn, sp, z)
        _ = tc.train_step(want_loss=False)

        var sg = Tensor()
        var ag = Tensor()
        var sng = Tensor()
        var spg = Tensor()
        var zg = Tensor()
        _fill(sg, BATCH * OBS, 0.11 + 0.01 * Float64(step), 0.07)
        _fill(ag, BATCH * ACT, 0.13, 0.05)
        _fill(sng, BATCH * OBS, 0.09, 0.12)
        _fill(spg, BATCH * OBS, 0.17, 0.03)
        _fill(zg, BATCH * D, 0.21, 0.09)
        sg.upload(ctx)
        ag.upload(ctx)
        sng.upload(ctx)
        spg.upload(ctx)
        zg.upload(ctx)
        tg.load_batch(sg, ag, sng, spg, zg)
        _ = tg.train_step(want_loss=False)

    var probe = Tensor()
    _fill(probe, BATCH * OBS, 0.23, 0.11)
    var bc = Tensor()
    tc.backward_embed[BATCH](probe, bc)

    var probe_g = Tensor()
    _fill(probe_g, BATCH * OBS, 0.23, 0.11)
    probe_g.upload(ctx)
    var bg = Tensor()
    tg.backward_embed[BATCH](probe_g, bg)
    bg.download(ctx)

    var worst = Float64(0)
    var mag = Float64(0)
    for i in range(BATCH * D):
        var r = _rel(Float64(bc.data[i]), Float64(bg.data[i]))
        if r > worst:
            worst = r
        if abs(Float64(bc.data[i])) > mag:
            mag = abs(Float64(bc.data[i]))
    print("      worst relative |B_cpu - B_gpu| =", worst, " (max |B| =", mag, ")")
    assert_true(mag > 1e-4, "B(probe) is ~0 on CPU — nothing was compared")
    assert_true(worst < REL_TOL, "B(s) diverged: " + String(worst))


def test_smooth_action_parity() raises:
    """`smooth_action_t` on a FIXED noise buffer.

    The parity gates above run with noise off, so this op is otherwise
    untested on GPU. Feeding both devices the same noise is what isolates the
    clamp/scale arithmetic from the RNG difference.
    """
    print("[3] smooth_action_t agrees on a fixed noise buffer ...")
    var ctx = DeviceContext()
    comptime N = BATCH * ACT

    var pi = Tensor()
    var noise = Tensor()
    _fill(pi, N, 0.4, 0.3)
    _fill(noise, N, 0.9, 0.6)

    var out_c = Tensor()
    smooth_action_t["cpu", N](
        out_c, pi, noise, Scalar[DT](0.2), Scalar[DT](0.3), None
    )

    var pi_g = Tensor()
    var noise_g = Tensor()
    _fill(pi_g, N, 0.4, 0.3)
    _fill(noise_g, N, 0.9, 0.6)
    pi_g.upload(ctx)
    noise_g.upload(ctx)
    var out_g = Tensor()
    ensure_t["gpu"](out_g, N, ctx)
    smooth_action_t["gpu", N](
        out_g, pi_g, noise_g, Scalar[DT](0.2), Scalar[DT](0.3), ctx
    )
    out_g.download(ctx)

    var worst = Float64(0)
    var hit_clamp = False
    for i in range(N):
        var e = abs(Float64(out_c.data[i]) - Float64(out_g.data[i]))
        if e > worst:
            worst = e
        if abs(Float64(out_c.data[i])) > 0.999:
            hit_clamp = True
    print("      worst |cpu - gpu| =", worst, " (clamp exercised:", hit_clamp, ")")
    assert_true(
        hit_clamp,
        "no output reached the +-1 clamp, so the clamp branch was never"
        " exercised — pick a larger pi",
    )
    assert_true(worst < 1e-6, "smooth_action mismatch: " + String(worst))


def main() raises:
    test_step_parity()
    test_backward_embed_parity()
    test_smooth_action_parity()
    print("\n[PASS] FBTrainer CPU/GPU parity")
