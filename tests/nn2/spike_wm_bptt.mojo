"""SPIKE (PR5c Step 4 core): WM-BPTT scan + multi-DreamerOpt step.

De-risks the trainer's hardest mechanism: a T-step BPTT over the
`WMCoreGraph` with carry threading via the passthrough output columns,
standalone head-loss mini-graphs (decoder/reward/cont) whose feat grads
fold back into the carry, recompute-in-backward, and one `DreamerOpt` per
module. Validation = the total WM loss DECREASES over N steps on a fixed
synthetic batch (smoke; no fixture — this is orchestration, not new math).

Run: `pixi run mojo run -I . tests/nn2/spike_wm_bptt.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.dreamer_opt import DreamerOpt
from mojo_rl.deep_agents2.dreamerv3.wm import (
    WMCoreGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from mojo_rl.deep_agents2.dreamerv3.nets import DreamerEncoder

comptime B = 2
comptime T = 3
comptime OBS = 3
comptime ACT = 1
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime SC = STOCH * CLASSES
comptime TOKEN = 8
comptime DEC_U = 8
comptime HU = 8
comptime BINS = 7
comptime CARRY = 2 + DETER + SC      # WMCoreGraph output width

comptime Enc = DreamerEncoder[OBS, TOKEN]
comptime WMCore = WMCoreGraph[DETER, H, STOCH, CLASSES, BLOCKS, ACT, TOKEN]
comptime Dec = DecLossGraph[SC, DETER, OBS, DEC_U]
comptime Rew = RewLossGraph[DETER, SC, HU, BINS]
comptime Con = ConLossGraph[DETER, SC, HU]

comptime DYN_SCALE = Scalar[DT](1.0)
comptime REP_SCALE = Scalar[DT](0.1)


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _fill_pseudo(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, seed: Int):
    # deterministic pseudo-random in [-1,1] (no RNG dependency).
    var s = UInt64(seed * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float64((s >> 33)) / Float64(UInt64(1) << 31)
        p[i] = Scalar[DT]((u - 1.0))


def wm_step(
    mut enc: Enc, mut wm: WMCore, mut dec: Dec, mut rew: Rew, mut con: Con,
    mut oe: DreamerOpt, mut ow: DreamerOpt, mut od: DreamerOpt,
    mut orw: DreamerOpt, mut oc: DreamerOpt,
    obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B,T,OBS]
    act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B,T,ACT]
    rtgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B,T] reward target
    ctgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B,T] cont target
    cdeter: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [(T+1),B,DETER]
    cstoch: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [(T+1),B,SC]
    toks: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [T,B,TOKEN]
) raises -> Scalar[DT]:
    # ── encode tokens for every step ────────────────────────────────
    for t in range(T):
        var ob = _alloc(B * OBS)
        for b in range(B):
            for k in range(OBS):
                ob[b * OBS + k] = obs[(b * T + t) * OBS + k]
        var tk = toks + t * B * TOKEN
        var tkt = TileTensor(tk, row_major[B, TOKEN]())
        enc.forward["cpu", B](TileTensor(ob, row_major[B, OBS]()), output=tkt)
        ob.free()

    # ── forward scan (carry_0 = zeros) ──────────────────────────────
    for i in range(B * DETER):
        cdeter[i] = 0.0
    for i in range(B * SC):
        cstoch[i] = 0.0

    var total: Scalar[DT] = 0.0
    var outbuf = _alloc(B * CARRY)
    var dloss = _alloc(B)
    var rloss = _alloc(B)
    var closs = _alloc(B)
    for t in range(T):
        var dt = cdeter + t * B * DETER
        var st = cstoch + t * B * SC
        var at = _alloc(B * ACT)
        for b in range(B):
            for k in range(ACT):
                at[b * ACT + k] = act[(b * T + t) * ACT + k]
        wm.set_input["deter", B](TileTensor(dt, row_major[B, DETER]()))
        wm.set_input["stoch", B](TileTensor(st, row_major[B, SC]()))
        wm.set_input["action", B](TileTensor(at, row_major[B, ACT]()))
        wm.set_input["tokens", B](TileTensor(toks + t * B * TOKEN, row_major[B, TOKEN]()))
        var ot = TileTensor(outbuf, row_major[B, CARRY]())
        wm.forward["cpu", B](ot)
        # extract carry t+1
        var ndn = cdeter + (t + 1) * B * DETER
        var snn = cstoch + (t + 1) * B * SC
        for b in range(B):
            for k in range(DETER):
                ndn[b * DETER + k] = outbuf[b * CARRY + 2 + k]
            for k in range(SC):
                snn[b * SC + k] = outbuf[b * CARRY + 2 + DETER + k]
            total += DYN_SCALE * outbuf[b * CARRY + 0]
            total += REP_SCALE * outbuf[b * CARRY + 1]
        # head losses on carry t+1
        var rtg = _alloc(B * OBS)
        for b in range(B):
            for k in range(OBS):
                rtg[b * OBS + k] = obs[(b * T + t) * OBS + k]
        dec.set_input["stoch_new", B](TileTensor(snn, row_major[B, SC]()))
        dec.set_input["nd", B](TileTensor(ndn, row_major[B, DETER]()))
        dec.set_input["rtgt", B](TileTensor(rtg, row_major[B, OBS]()))
        var dlt = TileTensor(dloss, row_major[B, 1]())
        dec.forward["cpu", B](dlt)
        var rwt = _alloc(B)
        var cnt = _alloc(B)
        for b in range(B):
            rwt[b] = rtgt[b * T + t]
            cnt[b] = ctgt[b * T + t]
        rew.set_input["nd", B](TileTensor(ndn, row_major[B, DETER]()))
        rew.set_input["stoch_new", B](TileTensor(snn, row_major[B, SC]()))
        rew.set_input["rtgt", B](TileTensor(rwt, row_major[B, 1]()))
        var rlt = TileTensor(rloss, row_major[B, 1]())
        rew.forward["cpu", B](rlt)
        con.set_input["nd", B](TileTensor(ndn, row_major[B, DETER]()))
        con.set_input["stoch_new", B](TileTensor(snn, row_major[B, SC]()))
        con.set_input["ctgt", B](TileTensor(cnt, row_major[B, 1]()))
        var clt = TileTensor(closs, row_major[B, 1]())
        con.forward["cpu", B](clt)
        for b in range(B):
            total += dloss[b] + rloss[b] + closs[b]
        at.free(); rtg.free(); rwt.free(); cnt.free()

    # ── backward scan ───────────────────────────────────────────────
    oe.zero_grad["cpu", Enc](enc)
    ow.zero_grad_graph["cpu"](wm)
    od.zero_grad_graph["cpu"](dec)
    orw.zero_grad_graph["cpu"](rew)
    oc.zero_grad_graph["cpu"](con)

    var gcd = _alloc(B * DETER)   # grad on carry deter from t+1
    var gcs = _alloc(B * SC)
    for i in range(B * DETER):
        gcd[i] = 0.0
    for i in range(B * SC):
        gcs[i] = 0.0
    var ones1 = _alloc(B)
    for b in range(B):
        ones1[b] = 1.0
    var seed = _alloc(B * CARRY)
    var scratch = _alloc(B * CARRY)
    var dl1 = _alloc(B)

    for rev in range(T):
        var t = T - 1 - rev
        var dt = cdeter + t * B * DETER
        var st = cstoch + t * B * SC
        var ndn = cdeter + (t + 1) * B * DETER
        var snn = cstoch + (t + 1) * B * SC
        var at = _alloc(B * ACT)
        for b in range(B):
            for k in range(ACT):
                at[b * ACT + k] = act[(b * T + t) * ACT + k]
        var rtg = _alloc(B * OBS)
        for b in range(B):
            for k in range(OBS):
                rtg[b * OBS + k] = obs[(b * T + t) * OBS + k]
        var rwt = _alloc(B)
        var cnt = _alloc(B)
        for b in range(B):
            rwt[b] = rtgt[b * T + t]
            cnt[b] = ctgt[b * T + t]

        # recompute head forwards (refresh caches), then vjp(ones)
        dec.set_input["stoch_new", B](TileTensor(snn, row_major[B, SC]()))
        dec.set_input["nd", B](TileTensor(ndn, row_major[B, DETER]()))
        dec.set_input["rtgt", B](TileTensor(rtg, row_major[B, OBS]()))
        var dlt = TileTensor(dl1, row_major[B, 1]())
        dec.forward["cpu", B](dlt)
        var dseed = TileTensor(ones1, row_major[B, 1]())
        dec.vjp["cpu", B](dseed)

        rew.set_input["nd", B](TileTensor(ndn, row_major[B, DETER]()))
        rew.set_input["stoch_new", B](TileTensor(snn, row_major[B, SC]()))
        rew.set_input["rtgt", B](TileTensor(rwt, row_major[B, 1]()))
        var rlt = TileTensor(dl1, row_major[B, 1]())
        rew.forward["cpu", B](rlt)
        var rseed = TileTensor(ones1, row_major[B, 1]())
        rew.vjp["cpu", B](rseed)

        con.set_input["nd", B](TileTensor(ndn, row_major[B, DETER]()))
        con.set_input["stoch_new", B](TileTensor(snn, row_major[B, SC]()))
        con.set_input["ctgt", B](TileTensor(cnt, row_major[B, 1]()))
        var clt = TileTensor(dl1, row_major[B, 1]())
        con.forward["cpu", B](clt)
        var cseed = TileTensor(ones1, row_major[B, 1]())
        con.vjp["cpu", B](cseed)

        # carry grad at t+1 = incoming (gcd/gcs) + head feat grads
        var dnd = dec.grad_input_ptr["nd"]()
        var dsn = dec.grad_input_ptr["stoch_new"]()
        var rnd = rew.grad_input_ptr["nd"]()
        var rsn = rew.grad_input_ptr["stoch_new"]()
        var cnd = con.grad_input_ptr["nd"]()
        var csn = con.grad_input_ptr["stoch_new"]()
        # seed wmcore vjp
        for b in range(B):
            seed[b * CARRY + 0] = DYN_SCALE
            seed[b * CARRY + 1] = REP_SCALE
            for k in range(DETER):
                seed[b * CARRY + 2 + k] = (
                    gcd[b * DETER + k] + dnd[b * DETER + k]
                    + rnd[b * DETER + k] + cnd[b * DETER + k]
                )
            for k in range(SC):
                seed[b * CARRY + 2 + DETER + k] = (
                    gcs[b * SC + k] + dsn[b * SC + k]
                    + rsn[b * SC + k] + csn[b * SC + k]
                )
        # recompute wmcore forward (refresh caches) then vjp
        wm.set_input["deter", B](TileTensor(dt, row_major[B, DETER]()))
        wm.set_input["stoch", B](TileTensor(st, row_major[B, SC]()))
        wm.set_input["action", B](TileTensor(at, row_major[B, ACT]()))
        wm.set_input["tokens", B](TileTensor(toks + t * B * TOKEN, row_major[B, TOKEN]()))
        var sct = TileTensor(scratch, row_major[B, CARRY]())
        wm.forward["cpu", B](sct)
        var seedt = TileTensor(seed, row_major[B, CARRY]())
        wm.vjp["cpu", B](seedt)

        # carry grad to step t
        var gdt = wm.grad_input_ptr["deter"]()
        var gst = wm.grad_input_ptr["stoch"]()
        for i in range(B * DETER):
            gcd[i] = gdt[i]
        for i in range(B * SC):
            gcs[i] = gst[i]
        # encoder backward: grad_tokens → encoder.vjp (recompute fwd first)
        var gtok = wm.grad_input_ptr["tokens"]()
        var ob = _alloc(B * OBS)
        for b in range(B):
            for k in range(OBS):
                ob[b * OBS + k] = obs[(b * T + t) * OBS + k]
        var tkscr = _alloc(B * TOKEN)
        var tkt = TileTensor(tkscr, row_major[B, TOKEN]())
        enc.forward["cpu", B](TileTensor(ob, row_major[B, OBS]()), output=tkt)
        var gobs = _alloc(B * OBS)
        var gobst = TileTensor(gobs, row_major[B, OBS]())
        enc.vjp["cpu", B](TileTensor(gtok, row_major[B, TOKEN]()), gobst)
        at.free(); rtg.free(); rwt.free(); cnt.free(); ob.free()
        tkscr.free(); gobs.free()

    # ── opt step ────────────────────────────────────────────────────
    oe.step["cpu", Enc](enc)
    ow.step_graph["cpu"](wm)
    od.step_graph["cpu"](dec)
    orw.step_graph["cpu"](rew)
    oc.step_graph["cpu"](con)

    outbuf.free(); dloss.free(); rloss.free(); closs.free()
    gcd.free(); gcs.free(); ones1.free(); seed.free(); scratch.free(); dl1.free()
    return total


def main() raises:
    print("=" * 70)
    print("SPIKE (PR5c Step 4 core): WM-BPTT scan + multi-DreamerOpt step")
    print("=" * 70)
    var enc = Enc.make["cpu", INIT=Kaiming]()
    var wm = WMCore.make["cpu", INIT=Kaiming]()
    var dec = Dec.make["cpu", INIT=Kaiming]()
    var rew = Rew.make["cpu", INIT=Kaiming]()
    var con = Con.make["cpu", INIT=Kaiming]()
    var oe = DreamerOpt.make["cpu", Enc](enc)
    var ow = DreamerOpt.make_graph["cpu"](wm)
    var od = DreamerOpt.make_graph["cpu"](dec)
    var orw = DreamerOpt.make_graph["cpu"](rew)
    var oc = DreamerOpt.make_graph["cpu"](con)
    var lr = Scalar[DT](3e-3)
    oe.lr = lr; ow.lr = lr; od.lr = lr; orw.lr = lr; oc.lr = lr

    # fixed synthetic batch
    var obs = _alloc(B * T * OBS)
    var act = _alloc(B * T * ACT)
    var rtgt = _alloc(B * T)
    var ctgt = _alloc(B * T)
    _fill_pseudo(obs, B * T * OBS, 1)
    _fill_pseudo(act, B * T * ACT, 2)
    _fill_pseudo(rtgt, B * T, 3)
    for i in range(B * T):
        ctgt[i] = 1.0   # cont target = 1 (not done)

    var cdeter = _alloc((T + 1) * B * DETER)
    var cstoch = _alloc((T + 1) * B * SC)
    var toks = _alloc(T * B * TOKEN)

    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    comptime ITERS = 40
    for it in range(ITERS):
        var l = wm_step(
            enc, wm, dec, rew, con, oe, ow, od, orw, oc,
            obs, act, rtgt, ctgt, cdeter, cstoch, toks,
        )
        if it == 0:
            first = l
            print("  iter 0   total WM loss =", l)
        if it == ITERS - 1:
            last = l
            print("  iter", ITERS - 1, "  total WM loss =", l)
        # finite check
        assert_true(l == l, "loss finite")

    print("  decrease:", first, "->", last)
    assert_true(last < first, "WM BPTT loss must decrease")
    print("=" * 70)
    print("SPIKE PASSED — WM-BPTT scan trains (loss decreases), no NaN")
    print("=" * 70)
    obs.free(); act.free(); rtgt.free(); ctgt.free()
    cdeter.free(); cstoch.free(); toks.free()
