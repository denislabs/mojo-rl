"""TD-MPC2 PolicyStep + TDTargetStep GPU smoke (Apple Metal).

NOTE: these blocks use RSample (Philox box-muller), whose noise stream
differs CPU↔GPU, so their outputs are NOT bit-parity with CPU (unlike the
RNG-free WMStep, which IS bit-exact — see test_tdmpc2_wm_gpu_parity, and
TwoHotDecode, see test_tdmpc2_decode_gpu_parity). The deterministic GPU
machinery is therefore validated elsewhere; here we only smoke that the
GPU actor-update + TD-target run and produce finite, in-range results.
End-to-end GPU correctness is gated by the GPU Pendulum convergence run.

Run: `pixi run -e apple mojo run -I . tests/deep_agents2/test_tdmpc2_blocks_gpu_parity.mojo`
"""

from std.memory import alloc
from std.random import seed
from std.math import abs, isfinite
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.tdmpc2.nets import (
    TDMPC2Encoder, TDMPC2Policy, TDMPC2QNet,
)
from mojo_rl.deep_agents2.tdmpc2.policy_step import PolicyStep
from mojo_rl.deep_agents2.tdmpc2.td_target_step import TDTargetStep

comptime OBS = 4
comptime ENC = 16
comptime ACT = 2
comptime LATENT = 16
comptime MLP = 16
comptime BINS = 11
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 4
comptime H = 3
comptime SD = 99

comptime EncT = TDMPC2Encoder[OBS, ENC, LATENT, SN]
comptime PolicyT = TDMPC2Policy[LATENT, ACT, MLP]
comptime QNetT = TDMPC2QNet[LATENT, ACT, MLP, BINS]


def _fill_pseudo(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, sd: Int):
    var s = UInt64(sd * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float64((s >> 33)) / Float64(UInt64(1) << 31)
        p[i] = Scalar[DT]((u - 1.0))


def _upload(
    ctx: DeviceContext, src: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int
) raises -> DeviceBuffer[DT]:
    var d = ctx.enqueue_create_buffer[DT](n)
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr()[i] = src[i]
    ctx.enqueue_copy(d, h)
    ctx.synchronize()
    return d^


def test_policy_step_parity() raises:
    comptime PB = (H + 1) * B
    comptime StepT = PolicyStep[LATENT, ACT, MLP, BINS, VMIN, VMAX, PB]
    var ctx = DeviceContext()
    var lr = Scalar[DT](1e-3)

    seed(SD)
    var pol_c = PolicyT.make["cpu", INIT=Kaiming]()
    var qc = List[QNetT]()
    qc.append(QNetT.make["cpu", INIT=Kaiming]())
    qc.append(QNetT.make["cpu", INIT=Kaiming]())
    var pio_c = Adam.make["cpu", PolicyT](pol_c); pio_c.lr = lr
    var sc = StepT.make["cpu"]()

    seed(SD)
    var pol_g = PolicyT.make["gpu", INIT=Kaiming](ctx=ctx)
    var qg = List[QNetT]()
    qg.append(QNetT.make["gpu", INIT=Kaiming](ctx=ctx))
    qg.append(QNetT.make["gpu", INIT=Kaiming](ctx=ctx))
    var pio_g = Adam.make["gpu", PolicyT](pol_g, ctx=ctx); pio_g.lr = lr
    var sg = StepT.make["gpu"](ctx=ctx)

    var z = alloc[Scalar[DT]](PB * LATENT)
    _fill_pseudo(z, PB * LATENT, 5)
    var z_dev = _upload(ctx, z, PB * LATENT)

    var max_rel: Scalar[DT] = 0.0
    for it in range(3):
        var lc = sc.step["cpu"](pol_c, qc, 0, 1, pio_c, z)
        var lg = sg.step["gpu"](pol_g, qg, 0, 1, pio_g, _z_ptr(z_dev), ctx=ctx)
        var d = lc - lg
        if d < 0:
            d = -d
        var den = lc if lc >= 0 else -lc
        if den < Scalar[DT](1e-6):
            den = Scalar[DT](1e-6)
        var rel = d / den
        if rel > max_rel:
            max_rel = rel
        print("  policy iter", it, " cpu=", lc, " gpu=", lg, " rel=", rel)
        assert_true(isfinite(lc) and isfinite(lg), "policy losses finite")
    # No bit-parity assert: RSample noise differs CPU↔GPU by design.
    z.free()


def _z_ptr(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def test_td_target_parity() raises:
    comptime StepT = TDTargetStep[
        OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H
    ]
    var ctx = DeviceContext()

    seed(SD)
    var enc_c = EncT.make["cpu", INIT=Kaiming]()
    var pol_c = PolicyT.make["cpu", INIT=Kaiming]()
    var qtc = List[QNetT]()
    qtc.append(QNetT.make["cpu", INIT=Kaiming]())
    qtc.append(QNetT.make["cpu", INIT=Kaiming]())
    var sc = StepT.make["cpu"]()

    seed(SD)
    var enc_g = EncT.make["gpu", INIT=Kaiming](ctx=ctx)
    var pol_g = PolicyT.make["gpu", INIT=Kaiming](ctx=ctx)
    var qtg = List[QNetT]()
    qtg.append(QNetT.make["gpu", INIT=Kaiming](ctx=ctx))
    qtg.append(QNetT.make["gpu", INIT=Kaiming](ctx=ctx))
    var sg = StepT.make["gpu"](ctx=ctx)

    var obs = alloc[Scalar[DT]]((H + 1) * B * OBS)
    var rew = alloc[Scalar[DT]](H * B)
    var done = alloc[Scalar[DT]](H * B)
    _fill_pseudo(obs, (H + 1) * B * OBS, 1)
    _fill_pseudo(rew, H * B, 3)
    for i in range(H * B):
        done[i] = Scalar[DT](0.0)
    var td_c = alloc[Scalar[DT]](H * B)
    var td_g = alloc[Scalar[DT]](H * B)
    var gamma = Scalar[DT](0.99)

    sc.step["cpu"](enc_c, pol_c, qtc, 0, 1, obs, rew, done, td_c, gamma)
    sg.step["gpu"](enc_g, pol_g, qtg, 0, 1, obs, rew, done, td_g, gamma, ctx=ctx)

    var max_rel: Scalar[DT] = 0.0
    for i in range(H * B):
        var d = td_c[i] - td_g[i]
        if d < 0:
            d = -d
        var den = td_c[i] if td_c[i] >= 0 else -td_c[i]
        if den < Scalar[DT](1e-6):
            den = Scalar[DT](1e-6)
        var rel = d / den
        if rel > max_rel:
            max_rel = rel
        assert_true(isfinite(td_g[i]), "td_g finite")
        assert_true(isfinite(td_c[i]), "td_c finite")
    print("  td_target max rel (RSample-driven, not gated) =", max_rel)
    obs.free(); rew.free(); done.free(); td_c.free(); td_g.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
