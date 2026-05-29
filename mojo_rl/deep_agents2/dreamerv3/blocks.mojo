"""DreamerV3 trainer blocks — SAC-style composable make/step units.

Mirrors `deep_agents2/sac/` + `training/blocks/`: each block is a concrete
struct with `make[target](ctx)` + `step[target](mut state, mut <modules/opts>)`.
`target: StaticString` is comptime ("cpu"/"gpu"); `ctx: Optional[DeviceContext]`
is threaded at runtime via `DreamerState`. CPU/GPU split inside each step is a
`comptime if target == "cpu": ... else: ...`.

v1 lands the CPU branch (the validated `spike_wm_bptt` / `spike_wm_imag_ac`
logic). The GPU branch is gated until PR5c Step 5 adds GPU kernels to the 5
custom RSSM ops (ActionSquash / BlockGroupAssemble / GRUGate / OneHotKLLoss /
StraightThroughSample) — everything else (Linear/BlockLinear/RMSNorm/
Sequential/ComputeGraph/DreamerOpt) is already GPU. The block structure means
that swap is localized to each `else:` branch; the trainer/composition + the
target/ctx plumbing are GPU-ready now.

Shared state (`DreamerState`) holds the cross-block buffers (sampled batch,
RSSM carries, imagination noise) + inter-block scalars + ctx. v1 buffers are
plain CPU allocs sized once in `make`; the GPU upgrade swaps them for
`Scratch[..]` device buffers (same `target_ptr[target]` access shape SAC uses).
"""

from std.memory import alloc
from std.math import tanh, exp, sqrt
from std.random import random_float64
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents2.dreamerv3.twohot import twohot_pred
from mojo_rl.deep_agents2.dreamerv3.dists import bounded_std
from mojo_rl.deep_agents2.dreamerv3.normalize import PercentileNormalize
from mojo_rl.deep_agents2.dreamerv3.repl_loss import repl_loss_backward
from mojo_rl.deep_agents2.dreamerv3.imag_loss import (
    imag_loss_cpu, imag_loss_backward,
)
from mojo_rl.deep_agents2.dreamerv3.param_sync import (
    collect_params, apply_params,
)
from mojo_rl.deep_agents2.dreamerv3.polyak import polyak_module
from mojo_rl.deep_agents2.dreamerv3.wm import (
    WMCoreGraph, WMImagineGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from mojo_rl.deep_agents2.dreamerv3.nets import (
    DreamerEncoder, DreamerValue, DreamerPolicy,
)
from mojo_rl.nn2.optimizer.dreamer_opt import DreamerOpt


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


@always_inline
def _dp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


@always_inline
def _lt[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


# ── GPU marshalling kernels (validated in spike_wm_bptt_gpu) ──────────


def _bcopy[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    """Contiguous device→device copy (carry / grad threading)."""
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i])


def _seed_asm_k[B_: Int, CARRY_: Int, D_: Int, SC_: Int](
    seed: LayoutTensor[DT, Layout.row_major(B_ * CARRY_), MutAnyOrigin],
    gcd: LayoutTensor[DT, Layout.row_major(B_ * D_), MutAnyOrigin],
    gcs: LayoutTensor[DT, Layout.row_major(B_ * SC_), MutAnyOrigin],
    dnd: LayoutTensor[DT, Layout.row_major(B_ * D_), MutAnyOrigin],
    rnd: LayoutTensor[DT, Layout.row_major(B_ * D_), MutAnyOrigin],
    cnd: LayoutTensor[DT, Layout.row_major(B_ * D_), MutAnyOrigin],
    dsn: LayoutTensor[DT, Layout.row_major(B_ * SC_), MutAnyOrigin],
    rsn: LayoutTensor[DT, Layout.row_major(B_ * SC_), MutAnyOrigin],
    csn: LayoutTensor[DT, Layout.row_major(B_ * SC_), MutAnyOrigin],
    dyn: Scalar[DT],
    rep: Scalar[DT],
):
    """Assemble the BPTT carry seed = [dyn, rep, carry-grad + Σ head nd-grads,
    carry-grad + Σ head stoch-grads] for one batch row."""
    var b = Int(global_idx.x)
    if b < B_:
        seed[b * CARRY_] = dyn
        seed[b * CARRY_ + 1] = rep
        for k in range(D_):
            seed[b * CARRY_ + 2 + k] = (
                rebind[Scalar[DT]](gcd[b * D_ + k])
                + rebind[Scalar[DT]](dnd[b * D_ + k])
                + rebind[Scalar[DT]](rnd[b * D_ + k])
                + rebind[Scalar[DT]](cnd[b * D_ + k])
            )
        for k in range(SC_):
            seed[b * CARRY_ + 2 + D_ + k] = (
                rebind[Scalar[DT]](gcs[b * SC_ + k])
                + rebind[Scalar[DT]](dsn[b * SC_ + k])
                + rebind[Scalar[DT]](rsn[b * SC_ + k])
                + rebind[Scalar[DT]](csn[b * SC_ + k])
            )


def _feat_concat_k[B_: Int, D_: Int, SC_: Int](
    deter: LayoutTensor[DT, Layout.row_major(B_ * D_), MutAnyOrigin],
    stoch: LayoutTensor[DT, Layout.row_major(B_ * SC_), MutAnyOrigin],
    feat: LayoutTensor[DT, Layout.row_major(B_ * (D_ + SC_)), MutAnyOrigin],
):
    """feat[b] = concat(deter[b], stoch[b])  (FEAT = D + SC)."""
    var b = Int(global_idx.x)
    if b < B_:
        var F = D_ + SC_
        for k in range(D_):
            feat[b * F + k] = rebind[Scalar[DT]](deter[b * D_ + k])
        for k in range(SC_):
            feat[b * F + D_ + k] = rebind[Scalar[DT]](stoch[b * SC_ + k])


# ──────────────────────────────────────────────────────────────────────
# DreamerState — cross-block shared buffers + ctx + inter-block scalars.
# v1 CPU allocs (sized once); GPU upgrade → Scratch device buffers.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct DreamerState[
    OBS: Int, ACT: Int, DETER: Int, SC: Int, TOKEN: Int,
    B: Int, T: Int, T_IMAG: Int,
](Movable & ImplicitlyDestructible):
    var ctx: Optional[DeviceContext]
    # sampled batch (filled by the trainer from replay) — HOST, batch-major,
    # both targets (GPU uploads from here).
    var mb_obs: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [B,T+1,OBS]
    var mb_act: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [B,T,ACT]
    var mb_rew: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [B,T]
    var mb_dne: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [B,T]
    # RSSM carries (CPU branch). GPU uses cdeter_d / cstoch_d / toks_d.
    var cdeter: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [(T+1)*B*DETER]
    var cstoch: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [(T+1)*B*SC]
    var toks: UnsafePointer[Scalar[DT], MutAnyOrigin]     # [T*B*TOKEN]
    var noise: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [T_IMAG*T*B*ACT] HOST (NS=T*B imag starts)
    var last_wm_loss: Scalar[DT]
    var last_ac_loss: Scalar[DT]
    # ── diagnostics (filled per train_step; see docs/runbook) ──
    var dbg_real_rew: Scalar[DT]    # mean replay reward in the batch
    var dbg_rew_pred: Scalar[DT]    # mean imagined reward (rew head pred)
    var dbg_ret_mean: Scalar[DT]    # mean λ-return in imagination
    var dbg_ret_std: Scalar[DT]     # std of λ-return (degenerate ⇒ no signal)
    var dbg_pmean_abs: Scalar[DT]   # mean |policy mean output| (policy moving?)
    # ── GPU set (None on CPU). Time-major device minibatch + carries. ──
    var tm_obs: UnsafePointer[Scalar[DT], MutAnyOrigin]   # host staging [T,B,OBS]
    var tm_act: UnsafePointer[Scalar[DT], MutAnyOrigin]   # host staging [T,B,ACT]
    var tm_rew: UnsafePointer[Scalar[DT], MutAnyOrigin]   # host staging [T,B]
    var tm_cont: UnsafePointer[Scalar[DT], MutAnyOrigin]  # host staging [T,B]
    var d_obs: Optional[DeviceBuffer[DT]]                 # [T*B*OBS]
    var d_act: Optional[DeviceBuffer[DT]]                 # [T*B*ACT]
    var d_rew: Optional[DeviceBuffer[DT]]                 # [T*B]
    var d_cont: Optional[DeviceBuffer[DT]]                # [T*B]
    var d_cdeter: Optional[DeviceBuffer[DT]]              # [(T+1)*B*DETER]
    var d_cstoch: Optional[DeviceBuffer[DT]]              # [(T+1)*B*SC]
    var d_toks: Optional[DeviceBuffer[DT]]                # [T*B*TOKEN]

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self(
            ctx=ctx,
            mb_obs=_alloc(Self.B * (Self.T + 1) * Self.OBS),
            mb_act=_alloc(Self.B * Self.T * Self.ACT),
            mb_rew=_alloc(Self.B * Self.T),
            mb_dne=_alloc(Self.B * Self.T),
            cdeter=_alloc((Self.T + 1) * Self.B * Self.DETER),
            cstoch=_alloc((Self.T + 1) * Self.B * Self.SC),
            toks=_alloc(Self.T * Self.B * Self.TOKEN),
            noise=_alloc(Self.T_IMAG * Self.T * Self.B * Self.ACT),
            last_wm_loss=Scalar[DT](0.0),
            last_ac_loss=Scalar[DT](0.0),
            dbg_real_rew=Scalar[DT](0.0),
            dbg_rew_pred=Scalar[DT](0.0),
            dbg_ret_mean=Scalar[DT](0.0),
            dbg_ret_std=Scalar[DT](0.0),
            dbg_pmean_abs=Scalar[DT](0.0),
            tm_obs=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            tm_act=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            tm_rew=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            tm_cont=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            d_obs=None, d_act=None, d_rew=None, d_cont=None,
            d_cdeter=None, d_cstoch=None, d_toks=None,
        )
        comptime if target == "gpu":
            var c = ctx.value()
            s.tm_obs = _alloc(Self.T * Self.B * Self.OBS)
            s.tm_act = _alloc(Self.T * Self.B * Self.ACT)
            s.tm_rew = _alloc(Self.T * Self.B)
            s.tm_cont = _alloc(Self.T * Self.B)
            s.d_obs = c.enqueue_create_buffer[DT](Self.T * Self.B * Self.OBS)
            s.d_act = c.enqueue_create_buffer[DT](Self.T * Self.B * Self.ACT)
            s.d_rew = c.enqueue_create_buffer[DT](Self.T * Self.B)
            s.d_cont = c.enqueue_create_buffer[DT](Self.T * Self.B)
            s.d_cdeter = c.enqueue_create_buffer[DT]((Self.T + 1) * Self.B * Self.DETER)
            s.d_cstoch = c.enqueue_create_buffer[DT]((Self.T + 1) * Self.B * Self.SC)
            s.d_toks = c.enqueue_create_buffer[DT](Self.T * Self.B * Self.TOKEN)
        return s^


# ──────────────────────────────────────────────────────────────────────
# WMStep — WM-BPTT over one sampled length-T window. Trains enc/core/dec/
# rew/con; fills state.cdeter / cstoch with the posterior carry sequence.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct WMStep[
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, BINS: Int, B: Int, T: Int,
](Movable & ImplicitlyDestructible):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime CARRY = 2 + Self.DETER + Self.SC
    comptime EncT = DreamerEncoder[Self.OBS, Self.TOKEN, SwishOp]
    comptime CoreT = WMCoreGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN, SwishOp,
    ]
    comptime DecT = DecLossGraph[Self.SC, Self.DETER, Self.OBS, Self.DEC_U, SwishOp]
    comptime RewT = RewLossGraph[Self.DETER, Self.SC, Self.HU, Self.BINS, SwishOp]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU, SwishOp]
    comptime StateT = DreamerState[
        Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, 1,
    ]
    var dyn_scale: Scalar[DT]
    var rep_scale: Scalar[DT]
    var horizon: Scalar[DT]   # contdisc continue target = (1-term)·(1-1/horizon)

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self(
            dyn_scale=Scalar[DT](1.0), rep_scale=Scalar[DT](0.1),
            horizon=Scalar[DT](333.0),
        )

    def step[
        target: StaticString, T_IMAG: Int,
    ](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, T_IMAG],
        mut enc: Self.EncT,
        mut core: Self.CoreT,
        mut dec: Self.DecT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oe: DreamerOpt,
        mut ocore: DreamerOpt,
        mut odec: DreamerOpt,
        mut orew: DreamerOpt,
        mut ocon: DreamerOpt,
    ) raises:
        comptime if target == "cpu":
            self._wm_cpu[target, T_IMAG](
                st, enc, core, dec, rew, con, oe, ocore, odec, orew, ocon
            )
        else:
            self._wm_gpu[target, T_IMAG](
                st, enc, core, dec, rew, con, oe, ocore, odec, orew, ocon
            )

    def _wm_cpu[
        target: StaticString, T_IMAG: Int,
    ](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, T_IMAG],
        mut enc: Self.EncT,
        mut core: Self.CoreT,
        mut dec: Self.DecT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oe: DreamerOpt,
        mut ocore: DreamerOpt,
        mut odec: DreamerOpt,
        mut orew: DreamerOpt,
        mut ocon: DreamerOpt,
    ) raises:
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime CARRYl = Self.CARRY
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        var DYN = self.dyn_scale
        var REP = self.rep_scale
        var obs = st.mb_obs
        var act = st.mb_act
        var rew_t = st.mb_rew
        var dne = st.mb_dne
        var cdeter = st.cdeter
        var cstoch = st.cstoch
        var toks = st.toks
        # encode tokens
        for t in range(Self.T):
            var ob = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    ob[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var tk = toks + t * Self.B * TOK
            var tkt = TileTensor(tk, row_major[Self.B, TOK]())
            enc.forward[target, Self.B](
                TileTensor(ob, row_major[Self.B, OBSD]()), output=tkt
            )
            ob.free()
        for i in range(Self.B * D):
            cdeter[i] = 0.0
        for i in range(Self.B * SCl):
            cstoch[i] = 0.0
        var total: Scalar[DT] = 0.0
        var outbuf = _alloc(Self.B * CARRYl)
        var dl = _alloc(Self.B)
        for t in range(Self.T):
            var dtp = cdeter + t * Self.B * D
            var stp = cstoch + t * Self.B * SCl
            var at = _alloc(Self.B * ACTD)
            for b in range(Self.B):
                for k in range(ACTD):
                    at[b * ACTD + k] = act[(b * Self.T + t) * ACTD + k]
            core.set_input["deter", Self.B](TileTensor(dtp, row_major[Self.B, D]()))
            core.set_input["stoch", Self.B](TileTensor(stp, row_major[Self.B, SCl]()))
            core.set_input["action", Self.B](TileTensor(at, row_major[Self.B, ACTD]()))
            core.set_input["tokens", Self.B](TileTensor(toks + t * Self.B * TOK, row_major[Self.B, TOK]()))
            var ot = TileTensor(outbuf, row_major[Self.B, CARRYl]())
            core.forward[target, Self.B](ot)
            var ndn = cdeter + (t + 1) * Self.B * D
            var snn = cstoch + (t + 1) * Self.B * SCl
            for b in range(Self.B):
                for k in range(D):
                    ndn[b * D + k] = outbuf[b * CARRYl + 2 + k]
                for k in range(SCl):
                    snn[b * SCl + k] = outbuf[b * CARRYl + 2 + D + k]
                total += DYN * outbuf[b * CARRYl + 0] + REP * outbuf[b * CARRYl + 1]
            var rtg = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    rtg[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var rwt = _alloc(Self.B)
            var cnt = _alloc(Self.B)
            for b in range(Self.B):
                rwt[b] = rew_t[b * Self.T + t]
                # contdisc: continue target = (1-term)·(1-1/horizon) so the
                # cont head learns ~0.997 → geometric discounting (disc=1).
                # (term≈done here; Pendulum truncates, so this is ~0.997.)
                cnt[b] = (Scalar[DT](1.0) - dne[b * Self.T + t]) * (
                    Scalar[DT](1.0) - Scalar[DT](1.0) / self.horizon
                )
            dec.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            dec.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            dec.set_input["rtgt", Self.B](TileTensor(rtg, row_major[Self.B, OBSD]()))
            var dlt = TileTensor(dl, row_major[Self.B, 1]())
            dec.forward[target, Self.B](dlt)
            for b in range(Self.B):
                total += dl[b]
            rew.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            rew.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            rew.set_input["rtgt", Self.B](TileTensor(rwt, row_major[Self.B, 1]()))
            var rlt = TileTensor(dl, row_major[Self.B, 1]())
            rew.forward[target, Self.B](rlt)
            for b in range(Self.B):
                total += dl[b]
            con.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            con.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            con.set_input["ctgt", Self.B](TileTensor(cnt, row_major[Self.B, 1]()))
            var clt = TileTensor(dl, row_major[Self.B, 1]())
            con.forward[target, Self.B](clt)
            for b in range(Self.B):
                total += dl[b]
            at.free(); rtg.free(); rwt.free(); cnt.free()

        oe.zero_grad[target, Self.EncT](enc)
        ocore.zero_grad_graph[target](core)
        odec.zero_grad_graph[target](dec)
        orew.zero_grad_graph[target](rew)
        ocon.zero_grad_graph[target](con)
        var gcd = _alloc(Self.B * D)
        var gcs = _alloc(Self.B * SCl)
        for i in range(Self.B * D):
            gcd[i] = 0.0
        for i in range(Self.B * SCl):
            gcs[i] = 0.0
        var ones1 = _alloc(Self.B)
        for b in range(Self.B):
            ones1[b] = 1.0
        var seed = _alloc(Self.B * CARRYl)
        var scratch = _alloc(Self.B * CARRYl)
        var dl1 = _alloc(Self.B)
        for rev in range(Self.T):
            var t = Self.T - 1 - rev
            var dtp = cdeter + t * Self.B * D
            var stp = cstoch + t * Self.B * SCl
            var ndn = cdeter + (t + 1) * Self.B * D
            var snn = cstoch + (t + 1) * Self.B * SCl
            var at = _alloc(Self.B * ACTD)
            for b in range(Self.B):
                for k in range(ACTD):
                    at[b * ACTD + k] = act[(b * Self.T + t) * ACTD + k]
            var rtg = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    rtg[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var rwt = _alloc(Self.B)
            var cnt = _alloc(Self.B)
            for b in range(Self.B):
                rwt[b] = rew_t[b * Self.T + t]
                # contdisc: continue target = (1-term)·(1-1/horizon) so the
                # cont head learns ~0.997 → geometric discounting (disc=1).
                # (term≈done here; Pendulum truncates, so this is ~0.997.)
                cnt[b] = (Scalar[DT](1.0) - dne[b * Self.T + t]) * (
                    Scalar[DT](1.0) - Scalar[DT](1.0) / self.horizon
                )
            dec.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            dec.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            dec.set_input["rtgt", Self.B](TileTensor(rtg, row_major[Self.B, OBSD]()))
            var dlt = TileTensor(dl1, row_major[Self.B, 1]())
            dec.forward[target, Self.B](dlt)
            var ds = TileTensor(ones1, row_major[Self.B, 1]())
            dec.vjp[target, Self.B](ds)
            rew.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            rew.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            rew.set_input["rtgt", Self.B](TileTensor(rwt, row_major[Self.B, 1]()))
            var rlt = TileTensor(dl1, row_major[Self.B, 1]())
            rew.forward[target, Self.B](rlt)
            var rs = TileTensor(ones1, row_major[Self.B, 1]())
            rew.vjp[target, Self.B](rs)
            con.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            con.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            con.set_input["ctgt", Self.B](TileTensor(cnt, row_major[Self.B, 1]()))
            var clt = TileTensor(dl1, row_major[Self.B, 1]())
            con.forward[target, Self.B](clt)
            var cs = TileTensor(ones1, row_major[Self.B, 1]())
            con.vjp[target, Self.B](cs)
            var dnd = dec.grad_input_ptr["nd"]()
            var dsn = dec.grad_input_ptr["stoch_new"]()
            var rnd = rew.grad_input_ptr["nd"]()
            var rsn = rew.grad_input_ptr["stoch_new"]()
            var cnd = con.grad_input_ptr["nd"]()
            var csn = con.grad_input_ptr["stoch_new"]()
            for b in range(Self.B):
                seed[b * CARRYl + 0] = DYN
                seed[b * CARRYl + 1] = REP
                for k in range(D):
                    seed[b * CARRYl + 2 + k] = (
                        gcd[b * D + k] + dnd[b * D + k] + rnd[b * D + k]
                        + cnd[b * D + k]
                    )
                for k in range(SCl):
                    seed[b * CARRYl + 2 + D + k] = (
                        gcs[b * SCl + k] + dsn[b * SCl + k] + rsn[b * SCl + k]
                        + csn[b * SCl + k]
                    )
            core.set_input["deter", Self.B](TileTensor(dtp, row_major[Self.B, D]()))
            core.set_input["stoch", Self.B](TileTensor(stp, row_major[Self.B, SCl]()))
            core.set_input["action", Self.B](TileTensor(at, row_major[Self.B, ACTD]()))
            core.set_input["tokens", Self.B](TileTensor(toks + t * Self.B * TOK, row_major[Self.B, TOK]()))
            var sct = TileTensor(scratch, row_major[Self.B, CARRYl]())
            core.forward[target, Self.B](sct)
            var seedt = TileTensor(seed, row_major[Self.B, CARRYl]())
            core.vjp[target, Self.B](seedt)
            var gdt = core.grad_input_ptr["deter"]()
            var gst = core.grad_input_ptr["stoch"]()
            for i in range(Self.B * D):
                gcd[i] = gdt[i]
            for i in range(Self.B * SCl):
                gcs[i] = gst[i]
            var gtok = core.grad_input_ptr["tokens"]()
            var ob = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    ob[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var tkscr = _alloc(Self.B * TOK)
            var tkt = TileTensor(tkscr, row_major[Self.B, TOK]())
            enc.forward[target, Self.B](TileTensor(ob, row_major[Self.B, OBSD]()), output=tkt)
            var gobs = _alloc(Self.B * OBSD)
            var gobst = TileTensor(gobs, row_major[Self.B, OBSD]())
            enc.vjp[target, Self.B](TileTensor(gtok, row_major[Self.B, TOK]()), gobst)
            at.free(); rtg.free(); rwt.free(); cnt.free(); ob.free()
            tkscr.free(); gobs.free()
        oe.step[target, Self.EncT](enc)
        ocore.step_graph[target](core)
        odec.step_graph[target](dec)
        orew.step_graph[target](rew)
        ocon.step_graph[target](con)
        outbuf.free(); dl.free(); gcd.free(); gcs.free(); ones1.free()
        seed.free(); scratch.free(); dl1.free()
        st.last_wm_loss = total

    def _wm_gpu[
        target: StaticString, T_IMAG: Int,
    ](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, T_IMAG],
        mut enc: Self.EncT,
        mut core: Self.CoreT,
        mut dec: Self.DecT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oe: DreamerOpt,
        mut ocore: DreamerOpt,
        mut odec: DreamerOpt,
        mut orew: DreamerOpt,
        mut ocon: DreamerOpt,
    ) raises:
        # GPU WM-BPTT scan — ports the validated `spike_wm_bptt_gpu` onto the
        # block's device state. Replay sampled batch-major into host `mb_*`; we
        # transpose → time-major host staging → H2D, then run the proven scan.
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime CARRYl = Self.CARRY
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime BV = Self.B
        comptime TV = Self.T
        var DYN = self.dyn_scale
        var REP = self.rep_scale
        var ctx = st.ctx.value()

        # transpose host minibatch (batch-major) → time-major host staging
        var mb_obs = st.mb_obs; var mb_act = st.mb_act
        var mb_rew = st.mb_rew; var mb_dne = st.mb_dne
        for t in range(TV):
            for b in range(BV):
                for k in range(OBSD):
                    st.tm_obs[(t * BV + b) * OBSD + k] = mb_obs[(b * (TV + 1) + t) * OBSD + k]
                for k in range(ACTD):
                    st.tm_act[(t * BV + b) * ACTD + k] = mb_act[(b * TV + t) * ACTD + k]
                st.tm_rew[t * BV + b] = mb_rew[b * TV + t]
                st.tm_cont[t * BV + b] = (Scalar[DT](1.0) - mb_dne[b * TV + t]) * (
                Scalar[DT](1.0) - Scalar[DT](1.0) / self.horizon
            )
        ctx.enqueue_copy(st.d_obs.value(), st.tm_obs)
        ctx.enqueue_copy(st.d_act.value(), st.tm_act)
        ctx.enqueue_copy(st.d_rew.value(), st.tm_rew)
        ctx.enqueue_copy(st.d_cont.value(), st.tm_cont)

        var obs = _dp(st.d_obs.value())
        var act = _dp(st.d_act.value())
        var rewb = _dp(st.d_rew.value())
        var contb = _dp(st.d_cont.value())
        var cdeter = _dp(st.d_cdeter.value())
        var cstoch = _dp(st.d_cstoch.value())
        var toks = _dp(st.d_toks.value())

        # zero the carry buffers (whole thing; slot 0 = the zero carry_0)
        var zc = _alloc((TV + 1) * BV * D)
        for i in range((TV + 1) * BV * D):
            zc[i] = 0.0
        ctx.enqueue_copy(st.d_cdeter.value(), zc)
        var zs = _alloc((TV + 1) * BV * SCl)
        for i in range((TV + 1) * BV * SCl):
            zs[i] = 0.0
        ctx.enqueue_copy(st.d_cstoch.value(), zs)

        # working device buffers (allocated once per step)
        var outbuf = ctx.enqueue_create_buffer[DT](BV * CARRYl)
        var dl = ctx.enqueue_create_buffer[DT](BV)
        var seed = ctx.enqueue_create_buffer[DT](BV * CARRYl)
        var gcd = ctx.enqueue_create_buffer[DT](BV * D)
        var gcs = ctx.enqueue_create_buffer[DT](BV * SCl)
        var ones1 = ctx.enqueue_create_buffer[DT](BV)
        var gobs = ctx.enqueue_create_buffer[DT](BV * OBSD)
        var tokscr = ctx.enqueue_create_buffer[DT](BV * TOK)
        var ho = _alloc(BV)                 # head-loss readback
        var hcarry = _alloc(BV * CARRYl)    # dyn/rep readback
        var hones = _alloc(BV)
        for b in range(BV):
            hones[b] = 1.0
        ctx.enqueue_copy(ones1, hones)
        var zgd = _alloc(BV * D)
        var zgs = _alloc(BV * SCl)
        for i in range(BV * D):
            zgd[i] = 0.0
        for i in range(BV * SCl):
            zgs[i] = 0.0

        comptime CD = BV * D
        comptime CS = BV * SCl
        comptime nbD = (CD + TPB - 1) // TPB
        comptime nbS = (CS + TPB - 1) // TPB
        comptime nbB = (BV + TPB - 1) // TPB
        comptime ckND = _bcopy[CD]
        comptime ckSC = _bcopy[CS]
        comptime ksa = _seed_asm_k[BV, CARRYl, D, SCl]

        var out_t = TileTensor(_dp(outbuf), row_major[BV, CARRYl]())
        var dl_t = TileTensor(_dp(dl), row_major[BV, 1]())
        var seed_t = TileTensor(_dp(seed), row_major[BV, CARRYl]())
        var ones_t = TileTensor(_dp(ones1), row_major[BV, 1]())
        var tokscr_t = TileTensor(_dp(tokscr), row_major[BV, TOK]())
        var gobs_t = TileTensor(_dp(gobs), row_major[BV, OBSD]())

        var total: Scalar[DT] = 0.0
        # ── forward scan ──
        for t in range(TV):
            var obt = obs + t * BV * OBSD
            var tkt = toks + t * BV * TOK
            var tkt_t = TileTensor(tkt, row_major[BV, TOK]())
            enc.forward[target, BV](TileTensor(obt, row_major[BV, OBSD]()), output=tkt_t)
            var dtp = cdeter + t * BV * D
            var stp = cstoch + t * BV * SCl
            core.set_input["deter", BV](TileTensor(dtp, row_major[BV, D]()))
            core.set_input["stoch", BV](TileTensor(stp, row_major[BV, SCl]()))
            core.set_input["action", BV](TileTensor(act + t * BV * ACTD, row_major[BV, ACTD]()))
            core.set_input["tokens", BV](TileTensor(tkt, row_major[BV, TOK]()))
            core.forward[target, BV](out_t)
            var ndn = cdeter + (t + 1) * BV * D
            var snn = cstoch + (t + 1) * BV * SCl
            ctx.enqueue_function[ckND](_lt[CD](core.node_out_ptr["nd"]()), _lt[CD](ndn), grid_dim=nbD, block_dim=TPB)
            ctx.enqueue_function[ckSC](_lt[CS](core.node_out_ptr["stoch_new"]()), _lt[CS](snn), grid_dim=nbS, block_dim=TPB)
            ctx.synchronize()
            ctx.enqueue_copy(hcarry, outbuf)
            ctx.synchronize()
            for b in range(BV):
                total += DYN * hcarry[b * CARRYl + 0] + REP * hcarry[b * CARRYl + 1]
            dec.set_input["stoch_new", BV](TileTensor(snn, row_major[BV, SCl]()))
            dec.set_input["nd", BV](TileTensor(ndn, row_major[BV, D]()))
            dec.set_input["rtgt", BV](TileTensor(obt, row_major[BV, OBSD]()))
            dec.forward[target, BV](dl_t)
            ctx.synchronize(); ctx.enqueue_copy(ho, dl); ctx.synchronize()
            for b in range(BV):
                total += ho[b]
            rew.set_input["nd", BV](TileTensor(ndn, row_major[BV, D]()))
            rew.set_input["stoch_new", BV](TileTensor(snn, row_major[BV, SCl]()))
            rew.set_input["rtgt", BV](TileTensor(rewb + t * BV, row_major[BV, 1]()))
            rew.forward[target, BV](dl_t)
            ctx.synchronize(); ctx.enqueue_copy(ho, dl); ctx.synchronize()
            for b in range(BV):
                total += ho[b]
            con.set_input["nd", BV](TileTensor(ndn, row_major[BV, D]()))
            con.set_input["stoch_new", BV](TileTensor(snn, row_major[BV, SCl]()))
            con.set_input["ctgt", BV](TileTensor(contb + t * BV, row_major[BV, 1]()))
            con.forward[target, BV](dl_t)
            ctx.synchronize(); ctx.enqueue_copy(ho, dl); ctx.synchronize()
            for b in range(BV):
                total += ho[b]

        # ── backward scan ──
        oe.zero_grad[target, Self.EncT](enc)
        ocore.zero_grad_graph[target](core)
        odec.zero_grad_graph[target](dec)
        orew.zero_grad_graph[target](rew)
        ocon.zero_grad_graph[target](con)
        ctx.enqueue_copy(gcd, zgd)
        ctx.enqueue_copy(gcs, zgs)
        for rev in range(TV):
            var t = TV - 1 - rev
            var dtp = cdeter + t * BV * D
            var stp = cstoch + t * BV * SCl
            var ndn = cdeter + (t + 1) * BV * D
            var snn = cstoch + (t + 1) * BV * SCl
            var obt = obs + t * BV * OBSD
            dec.set_input["stoch_new", BV](TileTensor(snn, row_major[BV, SCl]()))
            dec.set_input["nd", BV](TileTensor(ndn, row_major[BV, D]()))
            dec.set_input["rtgt", BV](TileTensor(obt, row_major[BV, OBSD]()))
            dec.forward[target, BV](dl_t)
            dec.vjp[target, BV](ones_t)
            rew.set_input["nd", BV](TileTensor(ndn, row_major[BV, D]()))
            rew.set_input["stoch_new", BV](TileTensor(snn, row_major[BV, SCl]()))
            rew.set_input["rtgt", BV](TileTensor(rewb + t * BV, row_major[BV, 1]()))
            rew.forward[target, BV](dl_t)
            rew.vjp[target, BV](ones_t)
            con.set_input["nd", BV](TileTensor(ndn, row_major[BV, D]()))
            con.set_input["stoch_new", BV](TileTensor(snn, row_major[BV, SCl]()))
            con.set_input["ctgt", BV](TileTensor(contb + t * BV, row_major[BV, 1]()))
            con.forward[target, BV](dl_t)
            con.vjp[target, BV](ones_t)
            ctx.enqueue_function[ksa](
                _lt[BV * CARRYl](_dp(seed)),
                _lt[CD](_dp(gcd)), _lt[CS](_dp(gcs)),
                _lt[CD](dec.grad_input_ptr["nd"]()), _lt[CD](rew.grad_input_ptr["nd"]()), _lt[CD](con.grad_input_ptr["nd"]()),
                _lt[CS](dec.grad_input_ptr["stoch_new"]()), _lt[CS](rew.grad_input_ptr["stoch_new"]()), _lt[CS](con.grad_input_ptr["stoch_new"]()),
                DYN, REP, grid_dim=nbB, block_dim=TPB,
            )
            core.set_input["deter", BV](TileTensor(dtp, row_major[BV, D]()))
            core.set_input["stoch", BV](TileTensor(stp, row_major[BV, SCl]()))
            core.set_input["action", BV](TileTensor(act + t * BV * ACTD, row_major[BV, ACTD]()))
            core.set_input["tokens", BV](TileTensor(toks + t * BV * TOK, row_major[BV, TOK]()))
            core.forward[target, BV](out_t)
            core.vjp[target, BV](seed_t)
            ctx.enqueue_function[ckND](_lt[CD](core.grad_input_ptr["deter"]()), _lt[CD](_dp(gcd)), grid_dim=nbD, block_dim=TPB)
            ctx.enqueue_function[ckSC](_lt[CS](core.grad_input_ptr["stoch"]()), _lt[CS](_dp(gcs)), grid_dim=nbS, block_dim=TPB)
            enc.forward[target, BV](TileTensor(obt, row_major[BV, OBSD]()), output=tokscr_t)
            var gtok_t = TileTensor(core.grad_input_ptr["tokens"](), row_major[BV, TOK]())
            enc.vjp[target, BV](gtok_t, gobs_t)
        oe.step[target, Self.EncT](enc)
        ocore.step_graph[target](core)
        odec.step_graph[target](dec)
        orew.step_graph[target](rew)
        ocon.step_graph[target](con)
        ctx.synchronize()
        zc.free(); zs.free(); ho.free(); hcarry.free(); hones.free()
        zgd.free(); zgs.free()
        st.last_wm_loss = total


# ──────────────────────────────────────────────────────────────────────
# ParamSyncStep — copy core/prior params WMCoreGraph → WMImagineGraph.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct ParamSyncStep[
    DETER: Int, H: Int, STOCH: Int, CLASSES: Int, BLOCKS: Int, ACT: Int,
    TOKEN: Int,
](Movable & ImplicitlyDestructible):
    comptime CoreT = WMCoreGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN, SwishOp,
    ]
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT, SwishOp,
    ]

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def step[target: StaticString](
        mut self, mut core: Self.CoreT, mut imagine: Self.ImagT,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var names = List[String]()
        var ptrs = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        var lens = List[Int]()
        collect_params[target](core, names, ptrs, lens)
        apply_params[target](imagine, names, ptrs, lens, ctx=ctx)
        _ = names^; _ = ptrs^; _ = lens^


# ──────────────────────────────────────────────────────────────────────
# ACStep — imagination rollout + actor-critic loss. Trains value/policy;
# Polyak-updates slowvalue. Reads the start carry from state.cdeter[T].
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct ACStep[
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, HU: Int, VU: Int, PU: Int, BINS: Int,
    B: Int, T: Int, T_IMAG: Int,
](Movable & ImplicitlyDestructible):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime FEAT = Self.DETER + Self.SC
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT, SwishOp,
    ]
    comptime ValT = DreamerValue[Self.FEAT, Self.VU, Self.BINS, SwishOp]
    comptime PolT = DreamerPolicy[Self.FEAT, Self.PU, Self.ACT, SwishOp]
    comptime RewT = RewLossGraph[Self.DETER, Self.SC, Self.HU, Self.BINS, SwishOp]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU, SwishOp]
    var minstd: Scalar[DT]
    var maxstd: Scalar[DT]
    var lam: Scalar[DT]
    var actent: Scalar[DT]
    var slowreg: Scalar[DT]
    var slow_rate: Scalar[DT]
    var horizon: Scalar[DT]        # repval λ-return disc = 1 - 1/horizon
    var repval_scale: Scalar[DT]   # loss_scales.repval (reference 0.3)

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self(
            minstd=Scalar[DT](0.1), maxstd=Scalar[DT](1.0), lam=Scalar[DT](0.95),
            actent=Scalar[DT](3e-4), slowreg=Scalar[DT](1.0),
            slow_rate=Scalar[DT](0.02), horizon=Scalar[DT](333.0),
            repval_scale=Scalar[DT](0.3),
        )

    def step[target: StaticString](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, Self.T_IMAG],
        mut imagine: Self.ImagT,
        mut value: Self.ValT,
        mut slowvalue: Self.ValT,
        mut policy: Self.PolT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oval: DreamerOpt,
        mut opol: DreamerOpt,
        mut retnorm: PercentileNormalize,
        bins: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime if target == "cpu":
            self._ac_cpu[target](
                st, imagine, value, slowvalue, policy, rew, con,
                oval, opol, retnorm, bins,
            )
        else:
            self._ac_gpu[target](
                st, imagine, value, slowvalue, policy, rew, con,
                oval, opol, retnorm, bins,
            )

    def _ac_cpu[target: StaticString](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, Self.T_IMAG],
        mut imagine: Self.ImagT,
        mut value: Self.ValT,
        mut slowvalue: Self.ValT,
        mut policy: Self.PolT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oval: DreamerOpt,
        mut opol: DreamerOpt,
        mut retnorm: PercentileNormalize,
        bins: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSl = Self.BINS
        comptime TI = Self.T_IMAG
        var MINSTD = self.minstd
        var MAXSTD = self.maxstd
        # FIX (2): imagine from ALL B·T posterior carries (cdeter indices
        # 1..T; index 0 is the zero init) flattened to NS = T·B starts —
        # not just the final carry. Matches the reference (K = T).
        comptime NS = Self.T * Self.B
        var noise = st.noise                           # [TI*NS*ACT], shared w/ GPU
        var deter0 = st.cdeter + Self.B * D            # skip the zero-init
        var stoch0 = st.cstoch + Self.B * SCl
        var feats = _alloc(NS * TI * FEATl)
        var acts = _alloc(NS * TI * ACTD)
        var pmean = _alloc(NS * TI * ACTD)
        var pstd = _alloc(NS * TI * ACTD)
        var vlog = _alloc(NS * TI * BINSl)
        var svlog = _alloc(NS * TI * BINSl)
        var rewv = _alloc(NS * TI)
        var conv = _alloc(NS * TI)
        var cd = _alloc(NS * D)
        var cs = _alloc(NS * SCl)
        for i in range(NS * D):
            cd[i] = deter0[i]
        for i in range(NS * SCl):
            cs[i] = stoch0[i]
        var fb = _alloc(NS * FEATl)
        var pb = _alloc(NS * 2 * ACTD)
        var vb = _alloc(NS * BINSl)
        var svb = _alloc(NS * BINSl)
        var dummy1 = _alloc(NS * 1)
        for t in range(TI):
            for b in range(NS):
                for k in range(D):
                    fb[b * FEATl + k] = cd[b * D + k]
                for k in range(SCl):
                    fb[b * FEATl + D + k] = cs[b * SCl + k]
                for k in range(FEATl):
                    feats[(b * TI + t) * FEATl + k] = fb[b * FEATl + k]
            var ft = TileTensor(fb, row_major[NS, FEATl]())
            var pt = TileTensor(pb, row_major[NS, 2 * ACTD]())
            policy.forward[target, NS](ft, output=pt)
            for b in range(NS):
                for a in range(ACTD):
                    var mr = pb[b * 2 * ACTD + a]
                    var sr = pb[b * 2 * ACTD + ACTD + a]
                    pmean[(b * TI + t) * ACTD + a] = mr
                    pstd[(b * TI + t) * ACTD + a] = sr
                    var z = noise[(t * NS + b) * ACTD + a]
                    acts[(b * TI + t) * ACTD + a] = (
                        tanh(mr) + bounded_std(sr, MINSTD, MAXSTD) * z
                    )
            var ft2 = TileTensor(fb, row_major[NS, FEATl]())
            var vt = TileTensor(vb, row_major[NS, BINSl]())
            value.forward[target, NS](ft2, output=vt)
            var ft3 = TileTensor(fb, row_major[NS, FEATl]())
            var svt = TileTensor(svb, row_major[NS, BINSl]())
            slowvalue.forward[target, NS](ft3, output=svt)
            for b in range(NS):
                for c in range(BINSl):
                    vlog[(b * TI + t) * BINSl + c] = vb[b * BINSl + c]
                    svlog[(b * TI + t) * BINSl + c] = svb[b * BINSl + c]
            rew.set_input["nd", NS](TileTensor(cd, row_major[NS, D]()))
            rew.set_input["stoch_new", NS](TileTensor(cs, row_major[NS, SCl]()))
            rew.set_input["rtgt", NS](TileTensor(dummy1, row_major[NS, 1]()))
            var rlt = TileTensor(dummy1, row_major[NS, 1]())
            rew.forward[target, NS](rlt)
            var rew_logits = rew.node_out_ptr["rew"]()
            con.set_input["nd", NS](TileTensor(cd, row_major[NS, D]()))
            con.set_input["stoch_new", NS](TileTensor(cs, row_major[NS, SCl]()))
            con.set_input["ctgt", NS](TileTensor(dummy1, row_major[NS, 1]()))
            var clt = TileTensor(dummy1, row_major[NS, 1]())
            con.forward[target, NS](clt)
            var con_logit = con.node_out_ptr["con"]()
            for b in range(NS):
                rewv[b * TI + t] = twohot_pred[BINSl](rew_logits, b * BINSl, bins)
                conv[b * TI + t] = Scalar[DT](1.0) / (
                    Scalar[DT](1.0) + exp(-con_logit[b])
                )
            var at = _alloc(NS * ACTD)
            for b in range(NS):
                for a in range(ACTD):
                    at[b * ACTD + a] = acts[(b * TI + t) * ACTD + a]
            imagine.set_input["deter", NS](TileTensor(cd, row_major[NS, D]()))
            imagine.set_input["stoch", NS](TileTensor(cs, row_major[NS, SCl]()))
            imagine.set_input["action", NS](TileTensor(at, row_major[NS, ACTD]()))
            var fo = TileTensor(fb, row_major[NS, FEATl]())
            imagine.forward[target, NS](fo)
            var nd = imagine.node_out_ptr["nd"]()
            var sn = imagine.node_out_ptr["stoch_new"]()
            for i in range(NS * D):
                cd[i] = nd[i]
            for i in range(NS * SCl):
                cs[i] = sn[i]
            at.free()

        comptime TM1 = TI - 1
        var pol_loss = _alloc(NS * TM1)
        var val_loss = _alloc(NS * TM1)
        var ret = _alloc(NS * TM1)
        imag_loss_cpu[NS, TI, ACTD, BINSl](
            acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            retnorm, pol_loss, val_loss, ret,
        )
        var total: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            total += pol_loss[i] + val_loss[i]
        # log a per-element mean so the printed AC number stays comparable
        # across the B → B·T start-count change (DreamerOpt is scale-invariant
        # via RMS+AGC, so the cotangent below stays 1.0 — the gradient
        # direction is what matters, not the summed magnitude).
        total = total / Scalar[DT](NS * TM1)
        # ── diagnostics: is the policy moving / reward grounded / signal alive?
        var pma: Scalar[DT] = 0.0
        for i in range(NS * TI * ACTD):
            pma += pmean[i] if pmean[i] >= 0 else -pmean[i]
        st.dbg_pmean_abs = pma / Scalar[DT](NS * TI * ACTD)
        var rp: Scalar[DT] = 0.0
        for i in range(NS * TI):
            rp += rewv[i]
        st.dbg_rew_pred = rp / Scalar[DT](NS * TI)
        var rm: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            rm += ret[i]
        rm = rm / Scalar[DT](NS * TM1)
        st.dbg_ret_mean = rm
        var rv: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            var dd = ret[i] - rm
            rv += dd * dd
        st.dbg_ret_std = sqrt(rv / Scalar[DT](NS * TM1))
        var rscale = retnorm.stats()[1]
        var d_pol = _alloc(NS * TM1)
        var d_val = _alloc(NS * TM1)
        # mean-normalized cotangents (reference loss_scales: policy=1, value=1,
        # repval=0.3 as MEANS) so the imag value-loss and the repval value-loss
        # combine in the correct RATIO in the shared value gradient (RMS/AGC
        # normalize magnitude but preserve the relative mix).
        var inv_im = Scalar[DT](1.0) / Scalar[DT](NS * TM1)
        for i in range(NS * TM1):
            d_pol[i] = inv_im
            d_val[i] = inv_im
        var g_vlog = _alloc(NS * TI * BINSl)
        var g_pmean = _alloc(NS * TI * ACTD)
        var g_pstd = _alloc(NS * TI * ACTD)
        imag_loss_backward[NS, TI, ACTD, BINSl](
            acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            rscale, d_pol, d_val, g_vlog, g_pmean, g_pstd,
        )
        oval.zero_grad[target, Self.ValT](value)
        opol.zero_grad[target, Self.PolT](policy)
        var gfeat = _alloc(NS * FEATl)
        var vscr = _alloc(NS * BINSl)
        var pscr = _alloc(NS * 2 * ACTD)
        var polg = _alloc(NS * 2 * ACTD)
        for t in range(TI):
            var ftt = _alloc(NS * FEATl)
            for b in range(NS):
                for k in range(FEATl):
                    ftt[b * FEATl + k] = feats[(b * TI + t) * FEATl + k]
            var fvt = TileTensor(ftt, row_major[NS, FEATl]())
            var vot = TileTensor(vscr, row_major[NS, BINSl]())
            value.forward[target, NS](fvt, output=vot)
            var gv = _alloc(NS * BINSl)
            for b in range(NS):
                for c in range(BINSl):
                    gv[b * BINSl + c] = g_vlog[(b * TI + t) * BINSl + c]
            var gvt = TileTensor(gv, row_major[NS, BINSl]())
            var gft = TileTensor(gfeat, row_major[NS, FEATl]())
            value.vjp[target, NS](gvt, gft)
            var fpt = TileTensor(ftt, row_major[NS, FEATl]())
            var pot = TileTensor(pscr, row_major[NS, 2 * ACTD]())
            policy.forward[target, NS](fpt, output=pot)
            for b in range(NS):
                for a in range(ACTD):
                    polg[b * 2 * ACTD + a] = g_pmean[(b * TI + t) * ACTD + a]
                    polg[b * 2 * ACTD + ACTD + a] = g_pstd[(b * TI + t) * ACTD + a]
            var pgt = TileTensor(polg, row_major[NS, 2 * ACTD]())
            var gft2 = TileTensor(gfeat, row_major[NS, FEATl]())
            policy.vjp[target, NS](pgt, gft2)
            gv.free(); ftt.free()

        # ── repval: ground the value head on REAL replay transitions ──
        # (reference repval_loss=True, scale 0.3). Trains the value head ONLY
        # (replay features are sg'd — no grad to the WM). Accumulates into the
        # SAME value grad as the imag value-loss above → ONE oval.step (the
        # reference computes both in one backward). Without this, the value is
        # trained only on optimistic imagined returns and drifts/decouples.
        # Replay window features = the imagination START states (feats[:,0]);
        # bootstrap = the imagined return at those states (ret[:,0]). The
        # imagination starts are flattened time-major s=j·B+b; repl_loss wants
        # the replay window batch-major [b·T+j] (matching mb_rew/mb_dne), so we
        # transpose feats[:,0] and ret[:,0] into [B,T] here.
        comptime BT = Self.B * Self.T
        var feat_bt = _alloc(BT * FEATl)
        var boot_bt = _alloc(BT)
        var term_bt = _alloc(BT)
        for b in range(Self.B):
            for j in range(Self.T):
                var s = j * Self.B + b
                boot_bt[b * Self.T + j] = ret[s * TM1 + 0]
                term_bt[b * Self.T + j] = 0.0   # Pendulum: truncation, not term
                for k in range(FEATl):
                    feat_bt[(b * Self.T + j) * FEATl + k] = (
                        feats[(s * TI) * FEATl + k]
                    )
        var vlr = _alloc(BT * BINSl)
        var svlr = _alloc(BT * BINSl)
        var fbt_t = TileTensor(feat_bt, row_major[BT, FEATl]())
        var vlr_t = TileTensor(vlr, row_major[BT, BINSl]())
        value.forward[target, BT](fbt_t, output=vlr_t)        # sets value cache
        var fbt_t2 = TileTensor(feat_bt, row_major[BT, FEATl]())
        var svlr_t = TileTensor(svlr, row_major[BT, BINSl]())
        slowvalue.forward[target, BT](fbt_t2, output=svlr_t)
        var g_vlr = _alloc(BT * BINSl)
        var d_rep = _alloc(Self.B * TM1)
        var inv_rep = self.repval_scale / Scalar[DT](Self.B * TM1)
        for i in range(Self.B * TM1):
            d_rep[i] = inv_rep
        repl_loss_backward[Self.B, Self.T, BINSl](
            st.mb_dne, term_bt, st.mb_rew, boot_bt, vlr, svlr, bins,
            self.horizon, self.lam, self.slowreg, d_rep, g_vlr,
        )
        var g_vlr_t = TileTensor(g_vlr, row_major[BT, BINSl]())
        var grf = _alloc(BT * FEATl)
        var grf_t = TileTensor(grf, row_major[BT, FEATl]())
        value.vjp[target, BT](g_vlr_t, grf_t)   # accumulate into value.grad
        feat_bt.free(); boot_bt.free(); term_bt.free(); vlr.free(); svlr.free()
        g_vlr.free(); d_rep.free(); grf.free()

        oval.step[target, Self.ValT](value)
        opol.step[target, Self.PolT](policy)
        polyak_module[target, Self.ValT](value, slowvalue, self.slow_rate)

        feats.free(); acts.free(); pmean.free(); pstd.free(); vlog.free()
        svlog.free(); rewv.free(); conv.free(); cd.free(); cs.free()
        fb.free(); pb.free(); vb.free(); svb.free(); dummy1.free()
        pol_loss.free(); val_loss.free(); ret.free()
        d_pol.free(); d_val.free(); g_vlog.free(); g_pmean.free()
        g_pstd.free(); gfeat.free(); vscr.free(); pscr.free(); polg.free()
        st.last_ac_loss = total

    def _ac_gpu[target: StaticString](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, Self.T_IMAG],
        mut imagine: Self.ImagT,
        mut value: Self.ValT,
        mut slowvalue: Self.ValT,
        mut policy: Self.PolT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oval: DreamerOpt,
        mut opol: DreamerOpt,
        mut retnorm: PercentileNormalize,
        bins: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        # GPU hybrid: net forwards/vjps run on device; the small per-step
        # connective math (tanh+std sample, twohot expectation, sigmoid) and
        # the lambda-return imag_loss + repl_loss run on host via D2H/H2D of
        # small arrays. Mirrors _ac_cpu exactly (NS = T·B imagination starts,
        # mean-normalized cotangents, repval value-loss) → CPU↔GPU bit-match.
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSl = Self.BINS
        comptime TI = Self.T_IMAG
        # FIX (2): imagine from ALL NS = T·B posterior carries (not the final
        # carry B). NS is the rollout batch for every device op below.
        comptime NS = Self.T * Self.B
        var MINSTD = self.minstd
        var MAXSTD = self.maxstd
        var ctx = st.ctx.value()
        var noise = st.noise           # [TI*NS*ACT], shared w/ _ac_cpu

        comptime nbB = (NS + TPB - 1) // TPB
        comptime nbF = (NS * FEATl + TPB - 1) // TPB
        comptime nbD = (NS * D + TPB - 1) // TPB
        comptime nbS = (NS * SCl + TPB - 1) // TPB
        comptime nbBINS = (NS * BINSl + TPB - 1) // TPB
        comptime fck = _feat_concat_k[NS, D, SCl]
        comptime cpF = _bcopy[NS * FEATl]
        comptime cpD = _bcopy[NS * D]
        comptime cpS = _bcopy[NS * SCl]
        comptime cpBINS = _bcopy[NS * BINSl]
        comptime cpB1 = _bcopy[NS]

        # device working buffers (NS-wide)
        var cd = ctx.enqueue_create_buffer[DT](NS * D)
        var cs = ctx.enqueue_create_buffer[DT](NS * SCl)
        var fb = ctx.enqueue_create_buffer[DT](NS * FEATl)
        var pb = ctx.enqueue_create_buffer[DT](NS * 2 * ACTD)
        var vb = ctx.enqueue_create_buffer[DT](NS * BINSl)
        var svb = ctx.enqueue_create_buffer[DT](NS * BINSl)
        var at_d = ctx.enqueue_create_buffer[DT](NS * ACTD)
        var feats_d = ctx.enqueue_create_buffer[DT](NS * TI * FEATl)
        var dummy1 = ctx.enqueue_create_buffer[DT](NS)
        var rl_d = ctx.enqueue_create_buffer[DT](NS * BINSl)
        var cl_d = ctx.enqueue_create_buffer[DT](NS)
        var gv_d = ctx.enqueue_create_buffer[DT](NS * BINSl)
        var polg_d = ctx.enqueue_create_buffer[DT](NS * 2 * ACTD)
        var gfeat_d = ctx.enqueue_create_buffer[DT](NS * FEATl)
        var vscr = ctx.enqueue_create_buffer[DT](NS * BINSl)
        var pscr = ctx.enqueue_create_buffer[DT](NS * 2 * ACTD)

        # host arrays for the connective math + imag_loss
        var acts = _alloc(NS * TI * ACTD)
        var pmean = _alloc(NS * TI * ACTD)
        var pstd = _alloc(NS * TI * ACTD)
        var vlog = _alloc(NS * TI * BINSl)
        var svlog = _alloc(NS * TI * BINSl)
        var rewv = _alloc(NS * TI)
        var conv = _alloc(NS * TI)
        var hpb = _alloc(NS * 2 * ACTD)
        var hvb = _alloc(NS * BINSl)
        var hsvb = _alloc(NS * BINSl)
        var hrl = _alloc(NS * BINSl)
        var hcl = _alloc(NS)
        var hat = _alloc(NS * ACTD)

        # init carry cd/cs from ALL T posterior carries (d_cdeter indices
        # 1..T; index 0 is the zero init) flattened to NS = T·B (device copy).
        ctx.enqueue_function[cpD](
            _lt[NS * D](_dp(st.d_cdeter.value()) + Self.B * D),
            _lt[NS * D](_dp(cd)), grid_dim=nbD, block_dim=TPB,
        )
        ctx.enqueue_function[cpS](
            _lt[NS * SCl](_dp(st.d_cstoch.value()) + Self.B * SCl),
            _lt[NS * SCl](_dp(cs)), grid_dim=nbS, block_dim=TPB,
        )

        # ── imagination rollout ──
        for t in range(TI):
            ctx.enqueue_function[fck](
                _lt[NS * D](_dp(cd)), _lt[NS * SCl](_dp(cs)),
                _lt[NS * FEATl](_dp(fb)), grid_dim=nbB, block_dim=TPB,
            )
            ctx.enqueue_function[cpF](
                _lt[NS * FEATl](_dp(fb)),
                _lt[NS * FEATl](_dp(feats_d) + t * NS * FEATl),
                grid_dim=nbF, block_dim=TPB,
            )
            var ft = TileTensor(_dp(fb), row_major[NS, FEATl]())
            var pt = TileTensor(_dp(pb), row_major[NS, 2 * ACTD]())
            policy.forward[target, NS](ft, output=pt)
            ctx.synchronize(); ctx.enqueue_copy(hpb, pb); ctx.synchronize()
            for b in range(NS):
                for a in range(ACTD):
                    var mr = hpb[b * 2 * ACTD + a]
                    var sr = hpb[b * 2 * ACTD + ACTD + a]
                    pmean[(b * TI + t) * ACTD + a] = mr
                    pstd[(b * TI + t) * ACTD + a] = sr
                    var z = noise[(t * NS + b) * ACTD + a]
                    acts[(b * TI + t) * ACTD + a] = (
                        tanh(mr) + bounded_std(sr, MINSTD, MAXSTD) * z
                    )
            var ft2 = TileTensor(_dp(fb), row_major[NS, FEATl]())
            var vt = TileTensor(_dp(vb), row_major[NS, BINSl]())
            value.forward[target, NS](ft2, output=vt)
            var ft3 = TileTensor(_dp(fb), row_major[NS, FEATl]())
            var svt = TileTensor(_dp(svb), row_major[NS, BINSl]())
            slowvalue.forward[target, NS](ft3, output=svt)
            ctx.synchronize()
            ctx.enqueue_copy(hvb, vb); ctx.enqueue_copy(hsvb, svb)
            ctx.synchronize()
            for b in range(NS):
                for c in range(BINSl):
                    vlog[(b * TI + t) * BINSl + c] = hvb[b * BINSl + c]
                    svlog[(b * TI + t) * BINSl + c] = hsvb[b * BINSl + c]
            rew.set_input["nd", NS](TileTensor(_dp(cd), row_major[NS, D]()))
            rew.set_input["stoch_new", NS](TileTensor(_dp(cs), row_major[NS, SCl]()))
            rew.set_input["rtgt", NS](TileTensor(_dp(dummy1), row_major[NS, 1]()))
            var rlt = TileTensor(_dp(dummy1), row_major[NS, 1]())
            rew.forward[target, NS](rlt)
            ctx.enqueue_function[cpBINS](
                _lt[NS * BINSl](rew.node_out_ptr["rew"]()),
                _lt[NS * BINSl](_dp(rl_d)), grid_dim=nbBINS, block_dim=TPB,
            )
            con.set_input["nd", NS](TileTensor(_dp(cd), row_major[NS, D]()))
            con.set_input["stoch_new", NS](TileTensor(_dp(cs), row_major[NS, SCl]()))
            con.set_input["ctgt", NS](TileTensor(_dp(dummy1), row_major[NS, 1]()))
            var clt = TileTensor(_dp(dummy1), row_major[NS, 1]())
            con.forward[target, NS](clt)
            ctx.enqueue_function[cpB1](
                _lt[NS](con.node_out_ptr["con"]()),
                _lt[NS](_dp(cl_d)), grid_dim=nbB, block_dim=TPB,
            )
            ctx.synchronize()
            ctx.enqueue_copy(hrl, rl_d); ctx.enqueue_copy(hcl, cl_d)
            ctx.synchronize()
            for b in range(NS):
                rewv[b * TI + t] = twohot_pred[BINSl](hrl, b * BINSl, bins)
                conv[b * TI + t] = Scalar[DT](1.0) / (
                    Scalar[DT](1.0) + exp(-hcl[b])
                )
                for a in range(ACTD):
                    hat[b * ACTD + a] = acts[(b * TI + t) * ACTD + a]
            ctx.enqueue_copy(at_d, hat)
            imagine.set_input["deter", NS](TileTensor(_dp(cd), row_major[NS, D]()))
            imagine.set_input["stoch", NS](TileTensor(_dp(cs), row_major[NS, SCl]()))
            imagine.set_input["action", NS](TileTensor(_dp(at_d), row_major[NS, ACTD]()))
            var fo = TileTensor(_dp(fb), row_major[NS, FEATl]())
            imagine.forward[target, NS](fo)
            ctx.enqueue_function[cpD](
                _lt[NS * D](imagine.node_out_ptr["nd"]()),
                _lt[NS * D](_dp(cd)), grid_dim=nbD, block_dim=TPB,
            )
            ctx.enqueue_function[cpS](
                _lt[NS * SCl](imagine.node_out_ptr["stoch_new"]()),
                _lt[NS * SCl](_dp(cs)), grid_dim=nbS, block_dim=TPB,
            )
            ctx.synchronize()

        # ── lambda-return AC loss (host) ──
        comptime TM1 = TI - 1
        var pol_loss = _alloc(NS * TM1)
        var val_loss = _alloc(NS * TM1)
        var ret = _alloc(NS * TM1)
        imag_loss_cpu[NS, TI, ACTD, BINSl](
            acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            retnorm, pol_loss, val_loss, ret,
        )
        var total: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            total += pol_loss[i] + val_loss[i]
        total = total / Scalar[DT](NS * TM1)
        # ── diagnostics (match _ac_cpu) ──
        var pma: Scalar[DT] = 0.0
        for i in range(NS * TI * ACTD):
            pma += pmean[i] if pmean[i] >= 0 else -pmean[i]
        st.dbg_pmean_abs = pma / Scalar[DT](NS * TI * ACTD)
        var rp: Scalar[DT] = 0.0
        for i in range(NS * TI):
            rp += rewv[i]
        st.dbg_rew_pred = rp / Scalar[DT](NS * TI)
        var rm: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            rm += ret[i]
        rm = rm / Scalar[DT](NS * TM1)
        st.dbg_ret_mean = rm
        var rv: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            var dd = ret[i] - rm
            rv += dd * dd
        st.dbg_ret_std = sqrt(rv / Scalar[DT](NS * TM1))
        var rscale = retnorm.stats()[1]
        # mean-normalized cotangents (1/(NS·TM1)) — same RATIO as the repval
        # value-loss (0.3/(B·TM1)) so the two value-loss terms combine right.
        var d_pol = _alloc(NS * TM1)
        var d_val = _alloc(NS * TM1)
        var inv_im = Scalar[DT](1.0) / Scalar[DT](NS * TM1)
        for i in range(NS * TM1):
            d_pol[i] = inv_im
            d_val[i] = inv_im
        var g_vlog = _alloc(NS * TI * BINSl)
        var g_pmean = _alloc(NS * TI * ACTD)
        var g_pstd = _alloc(NS * TI * ACTD)
        imag_loss_backward[NS, TI, ACTD, BINSl](
            acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            rscale, d_pol, d_val, g_vlog, g_pmean, g_pstd,
        )

        # ── value / policy backward (device) ──
        oval.zero_grad[target, Self.ValT](value)
        opol.zero_grad[target, Self.PolT](policy)
        var hgv = _alloc(NS * BINSl)
        var hpolg = _alloc(NS * 2 * ACTD)
        for t in range(TI):
            var fvt = TileTensor(_dp(feats_d) + t * NS * FEATl, row_major[NS, FEATl]())
            var vot = TileTensor(_dp(vscr), row_major[NS, BINSl]())
            value.forward[target, NS](fvt, output=vot)
            for b in range(NS):
                for c in range(BINSl):
                    hgv[b * BINSl + c] = g_vlog[(b * TI + t) * BINSl + c]
            ctx.enqueue_copy(gv_d, hgv); ctx.synchronize()
            var gvt = TileTensor(_dp(gv_d), row_major[NS, BINSl]())
            var gft = TileTensor(_dp(gfeat_d), row_major[NS, FEATl]())
            value.vjp[target, NS](gvt, gft)
            var fpt = TileTensor(_dp(feats_d) + t * NS * FEATl, row_major[NS, FEATl]())
            var pot = TileTensor(_dp(pscr), row_major[NS, 2 * ACTD]())
            policy.forward[target, NS](fpt, output=pot)
            for b in range(NS):
                for a in range(ACTD):
                    hpolg[b * 2 * ACTD + a] = g_pmean[(b * TI + t) * ACTD + a]
                    hpolg[b * 2 * ACTD + ACTD + a] = g_pstd[(b * TI + t) * ACTD + a]
            ctx.enqueue_copy(polg_d, hpolg); ctx.synchronize()
            var pgt = TileTensor(_dp(polg_d), row_major[NS, 2 * ACTD]())
            var gft2 = TileTensor(_dp(gfeat_d), row_major[NS, FEATl]())
            policy.vjp[target, NS](pgt, gft2)

        # ── repval: ground the value head on REAL replay (device-hybrid) ──
        # Replay window features = imagination START states (feats_d t=0 block);
        # bootstrap = ret[:,0]. Transpose the time-major starts (s=j·B+b) to the
        # batch-major replay window [b·T+j] (matching mb_rew/mb_dne) on host,
        # forward value/slowvalue on device, repl_loss_backward on host, then
        # value.vjp accumulates into the SAME value grad before oval.step.
        comptime BT = Self.B * Self.T          # == NS
        var d_feat_bt = ctx.enqueue_create_buffer[DT](BT * FEATl)
        var d_vlr = ctx.enqueue_create_buffer[DT](BT * BINSl)
        var d_svlr = ctx.enqueue_create_buffer[DT](BT * BINSl)
        var d_g_vlr = ctx.enqueue_create_buffer[DT](BT * BINSl)
        var d_grf = ctx.enqueue_create_buffer[DT](BT * FEATl)
        # D2H sizes by the SOURCE DeviceBuffer → host dst must cover the WHOLE
        # feats_d; the imagination-START (t=0) block is its first NS*FEATl elems.
        var hfeat0 = _alloc(NS * TI * FEATl)
        ctx.synchronize(); ctx.enqueue_copy(hfeat0, feats_d); ctx.synchronize()
        var feat_bt = _alloc(BT * FEATl)
        var boot_bt = _alloc(BT)
        var term_bt = _alloc(BT)
        for b in range(Self.B):
            for j in range(Self.T):
                var s = j * Self.B + b
                boot_bt[b * Self.T + j] = ret[s * TM1 + 0]
                term_bt[b * Self.T + j] = 0.0   # Pendulum: truncation, not term
                for k in range(FEATl):
                    feat_bt[(b * Self.T + j) * FEATl + k] = hfeat0[s * FEATl + k]
        ctx.enqueue_copy(d_feat_bt, feat_bt); ctx.synchronize()
        var fbt_t = TileTensor(_dp(d_feat_bt), row_major[BT, FEATl]())
        var vlr_t = TileTensor(_dp(d_vlr), row_major[BT, BINSl]())
        value.forward[target, BT](fbt_t, output=vlr_t)      # sets value cache
        var fbt_t2 = TileTensor(_dp(d_feat_bt), row_major[BT, FEATl]())
        var svlr_t = TileTensor(_dp(d_svlr), row_major[BT, BINSl]())
        slowvalue.forward[target, BT](fbt_t2, output=svlr_t)
        var vlr = _alloc(BT * BINSl)
        var svlr = _alloc(BT * BINSl)
        ctx.synchronize()
        ctx.enqueue_copy(vlr, d_vlr); ctx.enqueue_copy(svlr, d_svlr)
        ctx.synchronize()
        var g_vlr = _alloc(BT * BINSl)
        var d_rep = _alloc(Self.B * TM1)
        var inv_rep = self.repval_scale / Scalar[DT](Self.B * TM1)
        for i in range(Self.B * TM1):
            d_rep[i] = inv_rep
        repl_loss_backward[Self.B, Self.T, BINSl](
            st.mb_dne, term_bt, st.mb_rew, boot_bt, vlr, svlr, bins,
            self.horizon, self.lam, self.slowreg, d_rep, g_vlr,
        )
        ctx.enqueue_copy(d_g_vlr, g_vlr); ctx.synchronize()
        var g_vlr_t = TileTensor(_dp(d_g_vlr), row_major[BT, BINSl]())
        var grf_t = TileTensor(_dp(d_grf), row_major[BT, FEATl]())
        value.vjp[target, BT](g_vlr_t, grf_t)   # accumulate into value.grad

        oval.step[target, Self.ValT](value)
        opol.step[target, Self.PolT](policy)
        polyak_module[target, Self.ValT](value, slowvalue, self.slow_rate, ctx=st.ctx)
        ctx.synchronize()

        acts.free(); pmean.free(); pstd.free(); vlog.free(); svlog.free()
        rewv.free(); conv.free(); hpb.free(); hvb.free(); hsvb.free()
        hrl.free(); hcl.free(); hat.free()
        pol_loss.free(); val_loss.free(); ret.free(); d_pol.free(); d_val.free()
        g_vlog.free(); g_pmean.free(); g_pstd.free(); hgv.free(); hpolg.free()
        hfeat0.free(); feat_bt.free(); boot_bt.free(); term_bt.free()
        vlr.free(); svlr.free(); g_vlr.free(); d_rep.free()
        st.last_ac_loss = total
