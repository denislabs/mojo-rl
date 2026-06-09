"""TD-MPC2 WMStep CPU vs GPU parity (Apple Metal).

Builds identical CPU + GPU world-model module sets (same seed → same init
since Kaiming inits on host for both) and runs the WM BPTT step on the same
fixed batch for a few iterations, comparing the returned total loss. On
Apple (fp32) the paths should match to a tight tolerance; this gates the
GPU BPTT marshalling (carry extraction / seed assembly / grad threading /
metric reduction) against the validated CPU path.

Run: `pixi run -e apple mojo run -I . tests/deep_agents2/test_tdmpc2_wm_gpu_parity.mojo`
"""

from std.memory import alloc
from std.random import seed
from std.math import abs, isfinite
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.tdmpc2.nets import (
    TDMPC2Encoder, TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Termination,
)
from mojo_rl.deep_agents2.tdmpc2.wm_graph import TDMPC2WMGraph
from mojo_rl.deep_agents2.tdmpc2.wm_step import WMStep

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
comptime SD = 123

comptime EncT = TDMPC2Encoder[OBS, ENC, LATENT, SN]
comptime DynT = TDMPC2Dynamics[LATENT, ACT, MLP, SN]
comptime RewT = TDMPC2Reward[LATENT, ACT, MLP, BINS]
comptime QNetT = TDMPC2QNet[LATENT, ACT, MLP, BINS]
comptime TermT = TDMPC2Termination[LATENT, ACT, MLP]
comptime GraphT = TDMPC2WMGraph[LATENT, ACT, MLP, BINS, SN, VMIN, VMAX]
comptime StepT = WMStep[OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H]


def _fill_pseudo(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, sd: Int):
    var s = UInt64(sd * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float64((s >> 33)) / Float64(UInt64(1) << 31)
        p[i] = Scalar[DT]((u - 1.0))


def main() raises:
    print("=" * 70)
    print("TD-MPC2 WMStep CPU vs GPU parity (Apple)")
    print("=" * 70)
    var ctx = DeviceContext()
    var lr = Scalar[DT](3e-3)

    # ── CPU set ────────────────────────────────────────────────────────
    seed(SD)
    var enc_c = EncT.make["cpu", INIT=Kaiming]()
    var dyn_c = DynT.make["cpu", INIT=Kaiming]()
    var rew_c = RewT.make["cpu", INIT=Kaiming]()
    var graph_c = GraphT.make["cpu", INIT=Kaiming]()
    var q_c = List[QNetT]()
    var qo_c = List[Adam]()
    for _ in range(5):
        var qn = QNetT.make["cpu", INIT=Kaiming]()
        var qo = Adam.make["cpu", QNetT](qn)
        qo.lr = lr
        q_c.append(qn^)
        qo_c.append(qo^)
    var eo_c = Adam.make["cpu", EncT](enc_c); eo_c.lr = lr
    var do_c = Adam.make["cpu", DynT](dyn_c); do_c.lr = lr
    var ro_c = Adam.make["cpu", RewT](rew_c); ro_c.lr = lr
    var term_c = TermT.make["cpu", INIT=Kaiming]()
    var to_c = Adam.make["cpu", TermT](term_c); to_c.lr = lr
    var step_c = StepT.make["cpu"]()

    # ── GPU set (same seed → same Kaiming draws → same init) ───────────
    seed(SD)
    var enc_g = EncT.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn_g = DynT.make["gpu", INIT=Kaiming](ctx=ctx)
    var rew_g = RewT.make["gpu", INIT=Kaiming](ctx=ctx)
    var graph_g = GraphT.make["gpu", INIT=Kaiming](ctx=ctx)
    var q_g = List[QNetT]()
    var qo_g = List[Adam]()
    for _ in range(5):
        var qn = QNetT.make["gpu", INIT=Kaiming](ctx=ctx)
        var qo = Adam.make["gpu", QNetT](qn, ctx=ctx)
        qo.lr = lr
        q_g.append(qn^)
        qo_g.append(qo^)
    var eo_g = Adam.make["gpu", EncT](enc_g, ctx=ctx); eo_g.lr = lr
    var do_g = Adam.make["gpu", DynT](dyn_g, ctx=ctx); do_g.lr = lr
    var ro_g = Adam.make["gpu", RewT](rew_g, ctx=ctx); ro_g.lr = lr
    var term_g = TermT.make["gpu", INIT=Kaiming](ctx=ctx)
    var to_g = Adam.make["gpu", TermT](term_g, ctx=ctx); to_g.lr = lr
    var step_g = StepT.make["gpu"](ctx=ctx)

    # ── fixed shared batch (t-major host) ──────────────────────────────
    var obs = alloc[Scalar[DT]]((H + 1) * B * OBS)
    var act = alloc[Scalar[DT]](H * B * ACT)
    var rew = alloc[Scalar[DT]](H * B)
    var td = alloc[Scalar[DT]](H * B)
    var done = alloc[Scalar[DT]](H * B)
    _fill_pseudo(obs, (H + 1) * B * OBS, 1)
    _fill_pseudo(act, H * B * ACT, 2)
    _fill_pseudo(rew, H * B, 3)
    _fill_pseudo(td, H * B, 4)
    for i in range(H * B):
        done[i] = Scalar[DT](0.0)

    var max_rel: Scalar[DT] = 0.0
    for it in range(4):
        var lc = step_c.step["cpu"](
            graph_c, enc_c, dyn_c, rew_c, q_c, term_c,
            eo_c, do_c, ro_c, qo_c, to_c,
            obs, act, rew, td, done,
        ).total()
        var lg = step_g.step["gpu"](
            graph_g, enc_g, dyn_g, rew_g, q_g, term_g,
            eo_g, do_g, ro_g, qo_g, to_g,
            obs, act, rew, td, done, ctx=ctx,
        ).total()
        var d = lc - lg
        if d < 0:
            d = -d
        var denom = lc if lc >= 0 else -lc
        if denom < Scalar[DT](1e-6):
            denom = Scalar[DT](1e-6)
        var rel = d / denom
        if rel > max_rel:
            max_rel = rel
        print("  iter", it, " cpu=", lc, " gpu=", lg, " rel=", rel)
        assert_true(isfinite(lc) and isfinite(lg), "losses finite")

    print("  max rel diff =", max_rel)
    assert_true(max_rel < Scalar[DT](1e-2), "CPU/GPU WM loss must match")
    print("=" * 70)
    print("PARITY PASSED — TD-MPC2 WMStep GPU matches CPU")
    print("=" * 70)
    obs.free(); act.free(); rew.free(); td.free(); done.free()
