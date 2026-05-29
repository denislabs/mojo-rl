"""SPIKE (PR5c Step 3 AC): imagination rollout + actor-critic loss + step.

Frozen WM (imagine graph + reward/cont heads, fixed random params),
trainable value + policy (standalone `DreamerValue`/`DreamerPolicy`
Modules). Roll out `T` imagination steps, sample bounded_normal actions
(deterministic pseudo-noise placeholder for Philox), build the imag_loss
inputs, run the validated `imag_loss_cpu` + `imag_loss_backward`, vjp into
value/policy, `DreamerOpt`-step. Smoke = AC loss finite + decreases.

Run: `pixi run mojo run -I . tests/nn2/spike_wm_imag_ac.mojo`
"""

from std.memory import alloc
from std.math import tanh, exp
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.dreamer_opt import DreamerOpt
from mojo_rl.deep_agents2.dreamerv3.wm import WMImagineGraph
from mojo_rl.deep_agents2.dreamerv3.nets import (
    DreamerValue, DreamerPolicy, DreamerRewardMLP, DreamerContMLP,
)
from mojo_rl.deep_agents2.dreamerv3.twohot import twohot_pred, symexp_twohot_bins
from mojo_rl.deep_agents2.dreamerv3.dists import bounded_std
from mojo_rl.deep_agents2.dreamerv3.normalize import PercentileNormalize
from mojo_rl.deep_agents2.dreamerv3.imag_loss import (
    imag_loss_cpu, imag_loss_backward,
)

comptime BK = 2
comptime T = 5
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime ACT = 1
comptime SC = STOCH * CLASSES
comptime FEAT = DETER + SC
comptime VU = 8
comptime PU = 8
comptime RU = 8
comptime BINS = 7

comptime Imag = WMImagineGraph[DETER, H, STOCH, CLASSES, BLOCKS, ACT]
comptime Val = DreamerValue[FEAT, VU, BINS]
comptime Pol = DreamerPolicy[FEAT, PU, ACT]
comptime Rew = DreamerRewardMLP[FEAT, RU, BINS]
comptime Con = DreamerContMLP[FEAT, RU]


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _pseudo(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, seed: Int):
    var s = UInt64(seed * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float64((s >> 33)) / Float64(UInt64(1) << 31)
        p[i] = Scalar[DT]((u - 1.0))


def ac_step(
    mut imag: Imag, mut val: Val, mut sval: Val, mut pol: Pol,
    mut rew: Rew, mut con: Con,
    mut ov: DreamerOpt, mut op: DreamerOpt,
    mut retnorm: PercentileNormalize,
    bins: UnsafePointer[Scalar[DT], MutAnyOrigin],
    deter0: UnsafePointer[Scalar[DT], MutAnyOrigin],
    stoch0: UnsafePointer[Scalar[DT], MutAnyOrigin],
    noise: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [T,BK,ACT]
) raises -> Scalar[DT]:
    comptime MINSTD = Scalar[DT](0.1)
    comptime MAXSTD = Scalar[DT](1.0)
    # rollout buffers
    var feats = _alloc(BK * T * FEAT)
    var acts = _alloc(BK * T * ACT)
    var pmean = _alloc(BK * T * ACT)
    var pstd = _alloc(BK * T * ACT)
    var vlog = _alloc(BK * T * BINS)
    var svlog = _alloc(BK * T * BINS)
    var rewv = _alloc(BK * T)
    var conv = _alloc(BK * T)

    # carry_0
    var cd = _alloc(BK * DETER)
    var cs = _alloc(BK * SC)
    for i in range(BK * DETER):
        cd[i] = deter0[i]
    for i in range(BK * SC):
        cs[i] = stoch0[i]

    var featbuf = _alloc(BK * FEAT)
    var polbuf = _alloc(BK * 2 * ACT)
    var vbuf = _alloc(BK * BINS)
    var svbuf = _alloc(BK * BINS)
    var rbuf = _alloc(BK * BINS)
    var cbuf = _alloc(BK * 1)

    for t in range(T):
        # feat_t = concat([deter, stoch])
        for b in range(BK):
            for k in range(DETER):
                featbuf[b * FEAT + k] = cd[b * DETER + k]
            for k in range(SC):
                featbuf[b * FEAT + DETER + k] = cs[b * SC + k]
            for k in range(FEAT):
                feats[(b * T + t) * FEAT + k] = featbuf[b * FEAT + k]
        var ft = TileTensor(featbuf, row_major[BK, FEAT]())
        # policy → mean_raw, std_raw
        var pt = TileTensor(polbuf, row_major[BK, 2 * ACT]())
        pol.forward["cpu", BK](ft, output=pt)
        for b in range(BK):
            for a in range(ACT):
                var mr = polbuf[b * 2 * ACT + a]
                var sr = polbuf[b * 2 * ACT + ACT + a]
                pmean[(b * T + t) * ACT + a] = mr
                pstd[(b * T + t) * ACT + a] = sr
                var mean = tanh(mr)
                var std = bounded_std(sr, MINSTD, MAXSTD)
                var z = noise[(t * BK + b) * ACT + a]
                acts[(b * T + t) * ACT + a] = mean + std * z
        # value / slowvalue
        var ft2 = TileTensor(featbuf, row_major[BK, FEAT]())
        var vt = TileTensor(vbuf, row_major[BK, BINS]())
        val.forward["cpu", BK](ft2, output=vt)
        var ft3 = TileTensor(featbuf, row_major[BK, FEAT]())
        var svt = TileTensor(svbuf, row_major[BK, BINS]())
        sval.forward["cpu", BK](ft3, output=svt)
        for b in range(BK):
            for c in range(BINS):
                vlog[(b * T + t) * BINS + c] = vbuf[b * BINS + c]
                svlog[(b * T + t) * BINS + c] = svbuf[b * BINS + c]
        # reward / cont preds
        var ft4 = TileTensor(featbuf, row_major[BK, FEAT]())
        var rt = TileTensor(rbuf, row_major[BK, BINS]())
        rew.forward["cpu", BK](ft4, output=rt)
        var ft5 = TileTensor(featbuf, row_major[BK, FEAT]())
        var ct = TileTensor(cbuf, row_major[BK, 1]())
        con.forward["cpu", BK](ft5, output=ct)
        for b in range(BK):
            rewv[b * T + t] = twohot_pred[BINS](rbuf, b * BINS, bins)
            conv[b * T + t] = Scalar[DT](1.0) / (
                Scalar[DT](1.0) + exp(-cbuf[b])
            )
        # imagine_step → carry_{t+1}
        var at = _alloc(BK * ACT)
        for b in range(BK):
            for a in range(ACT):
                at[b * ACT + a] = acts[(b * T + t) * ACT + a]
        imag.set_input["deter", BK](TileTensor(cd, row_major[BK, DETER]()))
        imag.set_input["stoch", BK](TileTensor(cs, row_major[BK, SC]()))
        imag.set_input["action", BK](TileTensor(at, row_major[BK, ACT]()))
        var fo = TileTensor(featbuf, row_major[BK, FEAT]())
        imag.forward["cpu", BK](fo)   # output feat = concat(nd, stoch_new)
        var nd = imag.node_out_ptr["nd"]()
        var sn = imag.node_out_ptr["stoch_new"]()
        for i in range(BK * DETER):
            cd[i] = nd[i]
        for i in range(BK * SC):
            cs[i] = sn[i]
        at.free()

    # ── imag_loss ────────────────────────────────────────────────────
    comptime TM1 = T - 1
    var pol_loss = _alloc(BK * TM1)
    var val_loss = _alloc(BK * TM1)
    var ret = _alloc(BK * TM1)
    imag_loss_cpu[BK, T, ACT, BINS](
        acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
        MINSTD, MAXSTD, Scalar[DT](0.95), Scalar[DT](3e-4), Scalar[DT](1.0),
        retnorm, pol_loss, val_loss, ret,
    )
    var total: Scalar[DT] = 0.0
    for i in range(BK * TM1):
        total += pol_loss[i] + val_loss[i]

    # ── backward ─────────────────────────────────────────────────────
    var rstats = retnorm.stats()
    var rscale = rstats[1]
    var d_pol = _alloc(BK * TM1)
    var d_val = _alloc(BK * TM1)
    for i in range(BK * TM1):
        d_pol[i] = 1.0
        d_val[i] = 1.0
    var g_vlog = _alloc(BK * T * BINS)
    var g_pmean = _alloc(BK * T * ACT)
    var g_pstd = _alloc(BK * T * ACT)
    imag_loss_backward[BK, T, ACT, BINS](
        acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
        MINSTD, MAXSTD, Scalar[DT](0.95), Scalar[DT](3e-4), Scalar[DT](1.0),
        rscale, d_pol, d_val, g_vlog, g_pmean, g_pstd,
    )

    ov.zero_grad["cpu", Val](val)
    op.zero_grad["cpu", Pol](pol)
    # per-step vjp (recompute forward to refresh caches)
    var gfeat = _alloc(BK * FEAT)
    var polg = _alloc(BK * 2 * ACT)
    var vscr = _alloc(BK * BINS)
    var pscr = _alloc(BK * 2 * ACT)
    for t in range(T):
        var ftt = _alloc(BK * FEAT)
        for b in range(BK):
            for k in range(FEAT):
                ftt[b * FEAT + k] = feats[(b * T + t) * FEAT + k]
        # value vjp
        var fvt = TileTensor(ftt, row_major[BK, FEAT]())
        var vot = TileTensor(vscr, row_major[BK, BINS]())
        val.forward["cpu", BK](fvt, output=vot)
        var gv = _alloc(BK * BINS)
        for b in range(BK):
            for c in range(BINS):
                gv[b * BINS + c] = g_vlog[(b * T + t) * BINS + c]
        var gvt = TileTensor(gv, row_major[BK, BINS]())
        var gft = TileTensor(gfeat, row_major[BK, FEAT]())
        val.vjp["cpu", BK](gvt, gft)
        # policy vjp
        var fpt = TileTensor(ftt, row_major[BK, FEAT]())
        var pot = TileTensor(pscr, row_major[BK, 2 * ACT]())
        pol.forward["cpu", BK](fpt, output=pot)
        for b in range(BK):
            for a in range(ACT):
                polg[b * 2 * ACT + a] = g_pmean[(b * T + t) * ACT + a]
                polg[b * 2 * ACT + ACT + a] = g_pstd[(b * T + t) * ACT + a]
        var pgt = TileTensor(polg, row_major[BK, 2 * ACT]())
        var gft2 = TileTensor(gfeat, row_major[BK, FEAT]())
        pol.vjp["cpu", BK](pgt, gft2)
        gv.free(); ftt.free()

    ov.step["cpu", Val](val)
    op.step["cpu", Pol](pol)

    feats.free(); acts.free(); pmean.free(); pstd.free(); vlog.free()
    svlog.free(); rewv.free(); conv.free(); cd.free(); cs.free()
    featbuf.free(); polbuf.free(); vbuf.free(); svbuf.free(); rbuf.free()
    cbuf.free(); pol_loss.free(); val_loss.free(); ret.free()
    d_pol.free(); d_val.free(); g_vlog.free(); g_pmean.free(); g_pstd.free()
    gfeat.free(); polg.free(); vscr.free(); pscr.free()
    return total


def main() raises:
    print("=" * 70)
    print("SPIKE (PR5c Step 3 AC): imagination rollout + AC loss + step")
    print("=" * 70)
    var imag = Imag.make["cpu", INIT=Kaiming]()
    var val = Val.make["cpu", INIT=Kaiming]()
    var sval = Val.make["cpu", INIT=Kaiming]()
    var pol = Pol.make["cpu", INIT=Kaiming]()
    var rew = Rew.make["cpu", INIT=Kaiming]()
    var con = Con.make["cpu", INIT=Kaiming]()
    var ov = DreamerOpt.make["cpu", Val](val)
    var op = DreamerOpt.make["cpu", Pol](pol)
    ov.lr = Scalar[DT](3e-3)
    op.lr = Scalar[DT](3e-3)
    var retnorm = PercentileNormalize.make(
        String("perc"), Scalar[DT](0.01), Scalar[DT](5.0), Scalar[DT](95.0),
        Scalar[DT](1.0), False,
    )
    var bins = _alloc(BINS)
    symexp_twohot_bins[BINS](bins)

    var deter0 = _alloc(BK * DETER)
    var stoch0 = _alloc(BK * SC)
    var noise = _alloc(T * BK * ACT)
    _pseudo(deter0, BK * DETER, 7)
    _pseudo(stoch0, BK * SC, 8)
    _pseudo(noise, T * BK * ACT, 9)

    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    comptime ITERS = 30
    for it in range(ITERS):
        var l = ac_step(
            imag, val, sval, pol, rew, con, ov, op, retnorm,
            bins, deter0, stoch0, noise,
        )
        assert_true(l == l, "AC loss finite")
        if it == 0:
            first = l
            print("  iter 0   AC loss =", l)
        if it == ITERS - 1:
            last = l
            print("  iter", ITERS - 1, "  AC loss =", l)
    print("  decrease:", first, "->", last)
    assert_true(last < first, "AC loss must decrease")
    print("=" * 70)
    print("SPIKE PASSED — imagination AC trains (loss decreases), no NaN")
    print("=" * 70)
    bins.free(); deter0.free(); stoch0.free(); noise.free()
