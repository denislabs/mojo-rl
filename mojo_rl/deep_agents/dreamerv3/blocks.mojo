"""DreamerV3 trainer blocks — SAC-style composable make/step units (storage).

Mirrors `deep_agents/tdmpc2/wm_step.mojo`: each block is a concrete struct with
`make[target](ctx)` + `step[target](mut state, mut <modules/opts>)`.
`target: StaticString` is comptime ("cpu"/"gpu"); `ctx: Optional[DeviceContext]`
is threaded at runtime via `DreamerState`. The CPU/GPU split inside each step is
a `comptime if target == "cpu": ... else: ...`.

Storage migration (Stage: DreamerV3 BPTT/imagination driver): inputs/scratch are
storage `Tensor`s (`.data` host List / `.dev` Optional[DeviceBuffer]). Persistent
scratch is allocated ONCE in `make[target]` via `_mk[target](n, ctx)` and
reused every step. Module nets (enc / value / policy heads) drive through the
storage `Module` surface (`forward[target,B](TensorRefs[1](in), out, ctx)` /
`vjp[target,B](TensorRefs[1](fin), go, TensorRefs[1](gi), ctx)`); the loss graphs
(WMCore/WMImagine/Dec/Rew/Con) own their params and drive through the storage
`ComputeGraph` surface (`set_input` / `forward[B,target]` / `vjp[B,target]` /
`node_output` / `grad_input`); DreamerOpt drives Modules via `step[target,M]` and
graphs via `begin_step(); graph.for_each_param[target](opt, ctx)`.

The Phase-1 host loss helpers (`imag_loss_*` / `repl_loss_backward` /
`twohot_pred` / `bounded_std` / `cat_sample`) are unchanged free functions taking
raw `UnsafePointer[Scalar[DT], MutAnyOrigin]`; CPU call sites pass
`rebind[...](tensor.data.unsafe_ptr())` (sanctioned host-pointer use).

GPU paths: the CPU branch is the convergence-gated path (DreamerV3 v1 trains on
CPU). The GPU branches of `_wm_gpu` / `_ac_gpu` are storage-port TODOs (the legacy
device kernels are not yet re-expressed on the storage surface) and raise.
"""

from std.memory import alloc
from std.math import tanh, exp, sqrt
from std.random import random_float64
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.deep_agents.dreamerv3.twohot import twohot_pred
from mojo_rl.deep_agents.dreamerv3.dists import bounded_std
from mojo_rl.deep_agents.dreamerv3.dists_discrete import cat_sample, UNIMIX
from mojo_rl.deep_agents.dreamerv3.normalize import PercentileNormalize
from mojo_rl.deep_agents.dreamerv3.repl_loss import repl_loss_backward
from mojo_rl.deep_agents.dreamerv3.imag_loss import (
    imag_loss_cpu, imag_loss_backward,
)
from mojo_rl.deep_agents.dreamerv3.param_sync import (
    collect_graph_params, apply_graph_params,
)
from mojo_rl.deep_agents.dreamerv3.polyak import polyak_module
from mojo_rl.deep_agents.dreamerv3.wm import (
    WMCoreGraph, WMImagineGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from mojo_rl.deep_agents.dreamerv3.nets import (
    DreamerEncoder, DreamerValue, DreamerPolicyHead,
)
from mojo_rl.nn.optimizer.dreamer_opt import DreamerOpt


@always_inline
def _hp(mut t: Tensor) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    """Sanctioned host-pointer view of a storage Tensor's CPU `data` — for the
    raw-pointer Phase-1 loss helpers (CPU only)."""
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](t.data.unsafe_ptr())


@always_inline
def _mk[target: StaticString](
    n: Int, ctx: Optional[DeviceContext]
) raises -> Tensor:
    """Scratch allocator for the DreamerV3 blocks. On CPU == `Tensor.make`. On
    GPU it allocates the device buffer AND sizes the host `.data` List, because
    the GPU WM/AC paths marshal the per-step reset masks / carries / loss reads
    through host `.data` (upload/download of small windows) — so the host side
    must be pre-sized to be indexed before the first download."""
    var t = Tensor.make[target](n, ctx)
    comptime if target == "gpu":
        t.ensure(n)  # size the host `.data` List (device buffer kept)
    return t^


# ── GPU marshalling kernels (kept for the GPU port; `def` not `fn`) ──────


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
    """Assemble the BPTT carry seed for one batch row."""
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


def _rowscale_k[B_: Int, W_: Int](
    src: LayoutTensor[DT, Layout.row_major(B_ * W_), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(B_ * W_), MutAnyOrigin],
    m: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < B_ * W_:
        dst[i] = rebind[Scalar[DT]](m[i // W_]) * rebind[Scalar[DT]](src[i])


def _rowscale_inplace_k[B_: Int, W_: Int](
    buf: LayoutTensor[DT, Layout.row_major(B_ * W_), MutAnyOrigin],
    m: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < B_ * W_:
        buf[i] = rebind[Scalar[DT]](m[i // W_]) * rebind[Scalar[DT]](buf[i])


# ──────────────────────────────────────────────────────────────────────
# DreamerState — cross-block shared buffers + ctx + inter-block scalars.
# Storage Tensors hold BOTH cpu `.data` and gpu `.dev` (one set per field).
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct DreamerState[
    OBS: Int, ACT: Int, DETER: Int, SC: Int, TOKEN: Int,
    B: Int, T: Int, T_IMAG: Int,
](Movable & ImplicitlyDeletable):
    var ctx: Optional[DeviceContext]
    # sampled batch (filled by the trainer from replay), batch-major.
    var mb_obs: Tensor   # [B,(T+1),OBS]
    var mb_act: Tensor   # [B,T,ACT]
    var mb_rew: Tensor   # [B,T]
    var mb_dne: Tensor   # [B,T]      transition is_terminal (cont target = 1-dne)
    var mb_fst: Tensor   # [B,T+1]    per-frame is_first (carry-reset mask)
    # RSSM carries (posterior carry sequence).
    var cdeter: Tensor   # [(T+1)*B*DETER]
    var cstoch: Tensor   # [(T+1)*B*SC]
    var toks: Tensor     # [T*B*TOKEN]
    var noise: Tensor    # [T_IMAG*T*B*ACT] (NS=T*B imag starts)
    var last_wm_loss: Scalar[DT]
    var last_ac_loss: Scalar[DT]
    # ── diagnostics (filled per train_step; see docs/runbook) ──
    var dbg_real_rew: Scalar[DT]
    var dbg_rew_pred: Scalar[DT]
    var dbg_ret_mean: Scalar[DT]
    var dbg_ret_std: Scalar[DT]
    var dbg_pmean_abs: Scalar[DT]
    # ── divergence probes ──
    var dbg_val_mean: Scalar[DT]
    var dbg_pstd: Scalar[DT]
    var dbg_rscale: Scalar[DT]
    # ── imagined continue-factor probe (conv = sigmoid(con_logit) over the
    #    imagined rollout). If the cont head never predicts termination these
    #    sit at ~disc (≈0.997) and the λ-return saturates → no actor signal. ──
    var dbg_con_mean: Scalar[DT]
    var dbg_con_min: Scalar[DT]
    # ── collapse probes: spread of value + latent feat over imagined states.
    #    val_std≈0 with healthy feat_std ⇒ value head collapsed to a constant
    #    (no advantage signal); feat_std≈0 ⇒ the latent representation collapsed. ──
    var dbg_val_std: Scalar[DT]
    var dbg_feat_std: Scalar[DT]
    # ── GPU time-major device minibatch + carries. On CPU these stay empty
    #    (length-0) Tensors. ──
    var d_obs: Tensor    # [T*B*OBS]
    var d_act: Tensor    # [T*B*ACT]
    var d_rew: Tensor    # [T*B]
    var d_cont: Tensor   # [T*B]
    var d_cdeter: Tensor # [(T+1)*B*DETER]
    var d_cstoch: Tensor # [(T+1)*B*SC]
    var d_toks: Tensor   # [T*B*TOKEN]

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self(
            ctx=ctx,
            mb_obs=Tensor.make["cpu"](Self.B * (Self.T + 1) * Self.OBS),
            mb_act=Tensor.make["cpu"](Self.B * Self.T * Self.ACT),
            mb_rew=Tensor.make["cpu"](Self.B * Self.T),
            mb_dne=Tensor.make["cpu"](Self.B * Self.T),
            mb_fst=Tensor.make["cpu"](Self.B * (Self.T + 1)),
            cdeter=Tensor.make["cpu"]((Self.T + 1) * Self.B * Self.DETER),
            cstoch=Tensor.make["cpu"]((Self.T + 1) * Self.B * Self.SC),
            toks=Tensor.make["cpu"](Self.T * Self.B * Self.TOKEN),
            noise=Tensor.make["cpu"](Self.T_IMAG * Self.T * Self.B * Self.ACT),
            last_wm_loss=Scalar[DT](0.0),
            last_ac_loss=Scalar[DT](0.0),
            dbg_real_rew=Scalar[DT](0.0),
            dbg_rew_pred=Scalar[DT](0.0),
            dbg_ret_mean=Scalar[DT](0.0),
            dbg_ret_std=Scalar[DT](0.0),
            dbg_pmean_abs=Scalar[DT](0.0),
            dbg_val_mean=Scalar[DT](0.0),
            dbg_pstd=Scalar[DT](0.0),
            dbg_rscale=Scalar[DT](0.0),
            dbg_con_mean=Scalar[DT](0.0),
            dbg_con_min=Scalar[DT](0.0),
            dbg_val_std=Scalar[DT](0.0),
            dbg_feat_std=Scalar[DT](0.0),
            d_obs=Tensor(), d_act=Tensor(), d_rew=Tensor(), d_cont=Tensor(),
            d_cdeter=Tensor(), d_cstoch=Tensor(), d_toks=Tensor(),
        )
        comptime if target == "gpu":
            var c = ctx.value()
            s.d_obs = Tensor.alloc_gpu(c, Self.T * Self.B * Self.OBS)
            s.d_act = Tensor.alloc_gpu(c, Self.T * Self.B * Self.ACT)
            s.d_rew = Tensor.alloc_gpu(c, Self.T * Self.B)
            s.d_cont = Tensor.alloc_gpu(c, Self.T * Self.B)
            s.d_cdeter = Tensor.alloc_gpu(c, (Self.T + 1) * Self.B * Self.DETER)
            s.d_cstoch = Tensor.alloc_gpu(c, (Self.T + 1) * Self.B * Self.SC)
            s.d_toks = Tensor.alloc_gpu(c, Self.T * Self.B * Self.TOKEN)
        return s^


# ──────────────────────────────────────────────────────────────────────
# WMStep — WM-BPTT over one sampled length-T window. Trains enc/core/dec/
# rew/con; fills state.cdeter / cstoch with the posterior carry sequence.
# ──────────────────────────────────────────────────────────────────────


struct WMStep[
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, BINS: Int, B: Int, T: Int,
](Movable & ImplicitlyDeletable):
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

    # Persistent scratch Tensors (allocated once in make, reused every step).
    var ob: Tensor       # [B*OBS] encoder input window
    var cin_d: Tensor    # [B*DETER] masked carry-deter input
    var cin_s: Tensor    # [B*SC]    masked carry-stoch input
    var at: Tensor       # [B*ACT]   masked prev-action
    var rtg: Tensor      # [B*OBS]   recon target
    var rwt: Tensor      # [B]       reward target
    var cnt: Tensor      # [B]       continue target
    var ndn: Tensor      # [B*DETER] next deter (head input)
    var snn: Tensor      # [B*SC]    next stoch (head input)
    var outbuf: Tensor   # [B*CARRY] core forward output
    var dl: Tensor       # [B]       head-loss readout
    var seed: Tensor     # [B*CARRY] core vjp seed
    var gcd: Tensor      # [B*DETER] carry grad accumulator (deter)
    var gcs: Tensor      # [B*SC]    carry grad accumulator (stoch)
    var ones1: Tensor    # [B]       loss cotangent (1.0)
    var tkscr: Tensor    # [B*TOKEN] enc recompute output
    var gtok: Tensor     # [B*TOKEN] token grad window
    var gobs: Tensor     # [B*OBS]   obs grad (discarded)

    def __init__(out self):
        self.dyn_scale = Scalar[DT](0.5)
        self.rep_scale = Scalar[DT](0.1)
        self.horizon = Scalar[DT](333.0)
        self.ob = Tensor()
        self.cin_d = Tensor()
        self.cin_s = Tensor()
        self.at = Tensor()
        self.rtg = Tensor()
        self.rwt = Tensor()
        self.cnt = Tensor()
        self.ndn = Tensor()
        self.snn = Tensor()
        self.outbuf = Tensor()
        self.dl = Tensor()
        self.seed = Tensor()
        self.gcd = Tensor()
        self.gcs = Tensor()
        self.ones1 = Tensor()
        self.tkscr = Tensor()
        self.gtok = Tensor()
        self.gobs = Tensor()

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime CARRYl = Self.CARRY
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime BV = Self.B
        var s = Self()
        # Finding 2: loss_scales dyn=0.5, rep=0.1 (paper Eq. 2).
        s.dyn_scale = Scalar[DT](0.5)
        s.rep_scale = Scalar[DT](0.1)
        s.horizon = Scalar[DT](333.0)
        s.ob = _mk[target](BV * OBSD, ctx)
        s.cin_d = _mk[target](BV * D, ctx)
        s.cin_s = _mk[target](BV * SCl, ctx)
        s.at = _mk[target](BV * ACTD, ctx)
        s.rtg = _mk[target](BV * OBSD, ctx)
        s.rwt = _mk[target](BV, ctx)
        s.cnt = _mk[target](BV, ctx)
        s.ndn = _mk[target](BV * D, ctx)
        s.snn = _mk[target](BV * SCl, ctx)
        s.outbuf = _mk[target](BV * CARRYl, ctx)
        s.dl = _mk[target](BV, ctx)
        s.seed = _mk[target](BV * CARRYl, ctx)
        s.gcd = _mk[target](BV * D, ctx)
        s.gcs = _mk[target](BV * SCl, ctx)
        s.ones1 = _mk[target](BV, ctx)
        s.tkscr = _mk[target](BV * TOK, ctx)
        s.gtok = _mk[target](BV * TOK, ctx)
        s.gobs = _mk[target](BV * OBSD, ctx)
        return s^

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
        # Finding 1: observe-step t produces the belief for obs_{t+1} using
        # prev-action a_t — token (+ recon target) = obs frame t+1.
        for t in range(Self.T):
            for b in range(Self.B):
                for k in range(OBSD):
                    self.ob.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
            enc.forward[target, Self.B](
                TensorRefs[1](self.ob), self.tkscr, None
            )
            var base = t * Self.B * TOK
            for i in range(Self.B * TOK):
                st.toks.data[base + i] = self.tkscr.data[i]
        for i in range(Self.B * D):
            st.cdeter.data[i] = 0.0
        for i in range(Self.B * SCl):
            st.cstoch.data[i] = 0.0
        var total: Scalar[DT] = 0.0
        # forward scan
        for t in range(Self.T):
            var dbase = t * Self.B * D
            var sbase = t * Self.B * SCl
            for b in range(Self.B):
                var keep = (
                    Scalar[DT](0.0) if st.mb_fst.data[b * (Self.T + 1) + t + 1] >= Scalar[DT](0.5)
                    else Scalar[DT](1.0)
                )
                for k in range(D):
                    self.cin_d.data[b * D + k] = keep * st.cdeter.data[dbase + b * D + k]
                for k in range(SCl):
                    self.cin_s.data[b * SCl + k] = keep * st.cstoch.data[sbase + b * SCl + k]
                for k in range(ACTD):
                    self.at.data[b * ACTD + k] = keep * st.mb_act.data[(b * Self.T + t) * ACTD + k]
            # tokens window for step t into a fresh Tensor for set_input
            for i in range(Self.B * TOK):
                self.tkscr.data[i] = st.toks.data[t * Self.B * TOK + i]
            core.set_input["deter", Self.B](self.cin_d, None)
            core.set_input["stoch", Self.B](self.cin_s, None)
            core.set_input["action", Self.B](self.at, None)
            core.set_input["tokens", Self.B](self.tkscr, None)
            core.forward[Self.B, target](self.outbuf, None)
            var ndbase = (t + 1) * Self.B * D
            var snbase = (t + 1) * Self.B * SCl
            for b in range(Self.B):
                for k in range(D):
                    st.cdeter.data[ndbase + b * D + k] = self.outbuf.data[b * CARRYl + 2 + k]
                for k in range(SCl):
                    st.cstoch.data[snbase + b * SCl + k] = self.outbuf.data[b * CARRYl + 2 + D + k]
                total += DYN * self.outbuf.data[b * CARRYl + 0] + REP * self.outbuf.data[b * CARRYl + 1]
            # head inputs (next carry) + targets
            for b in range(Self.B):
                for k in range(D):
                    self.ndn.data[b * D + k] = st.cdeter.data[ndbase + b * D + k]
                for k in range(SCl):
                    self.snn.data[b * SCl + k] = st.cstoch.data[snbase + b * SCl + k]
                for k in range(OBSD):
                    self.rtg.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
                self.rwt.data[b] = st.mb_rew.data[b * Self.T + t]
                self.cnt.data[b] = (Scalar[DT](1.0) - st.mb_dne.data[b * Self.T + t]) * (
                    Scalar[DT](1.0) - Scalar[DT](1.0) / self.horizon
                )
            dec.set_input["stoch_new", Self.B](self.snn, None)
            dec.set_input["nd", Self.B](self.ndn, None)
            dec.set_input["rtgt", Self.B](self.rtg, None)
            dec.forward[Self.B, target](self.dl, None)
            for b in range(Self.B):
                total += self.dl.data[b]
            rew.set_input["nd", Self.B](self.ndn, None)
            rew.set_input["stoch_new", Self.B](self.snn, None)
            rew.set_input["rtgt", Self.B](self.rwt, None)
            rew.forward[Self.B, target](self.dl, None)
            for b in range(Self.B):
                total += self.dl.data[b]
            con.set_input["nd", Self.B](self.ndn, None)
            con.set_input["stoch_new", Self.B](self.snn, None)
            con.set_input["ctgt", Self.B](self.cnt, None)
            con.forward[Self.B, target](self.dl, None)
            for b in range(Self.B):
                total += self.dl.data[b]

        # zero grads. enc is a Module → opt.zero_grad over the module; the loss
        # graphs OWN their params (a ComputeGraph is not a Module) → zero them via
        # the graph's own zero_grad.
        oe.zero_grad[target, M=Self.EncT](enc, None)
        core.zero_grad[target](None)
        dec.zero_grad[target](None)
        rew.zero_grad[target](None)
        con.zero_grad[target](None)
        for i in range(Self.B * D):
            self.gcd.data[i] = 0.0
        for i in range(Self.B * SCl):
            self.gcs.data[i] = 0.0
        for b in range(Self.B):
            self.ones1.data[b] = 1.0
        # backward scan
        for rev in range(Self.T):
            var t = Self.T - 1 - rev
            var dbase = t * Self.B * D
            var sbase = t * Self.B * SCl
            var ndbase = (t + 1) * Self.B * D
            var snbase = (t + 1) * Self.B * SCl
            for b in range(Self.B):
                var keep = (
                    Scalar[DT](0.0) if st.mb_fst.data[b * (Self.T + 1) + t + 1] >= Scalar[DT](0.5)
                    else Scalar[DT](1.0)
                )
                for k in range(D):
                    self.cin_d.data[b * D + k] = keep * st.cdeter.data[dbase + b * D + k]
                for k in range(SCl):
                    self.cin_s.data[b * SCl + k] = keep * st.cstoch.data[sbase + b * SCl + k]
                for k in range(ACTD):
                    self.at.data[b * ACTD + k] = keep * st.mb_act.data[(b * Self.T + t) * ACTD + k]
                for k in range(D):
                    self.ndn.data[b * D + k] = st.cdeter.data[ndbase + b * D + k]
                for k in range(SCl):
                    self.snn.data[b * SCl + k] = st.cstoch.data[snbase + b * SCl + k]
                for k in range(OBSD):
                    self.rtg.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
                self.rwt.data[b] = st.mb_rew.data[b * Self.T + t]
                self.cnt.data[b] = (Scalar[DT](1.0) - st.mb_dne.data[b * Self.T + t]) * (
                    Scalar[DT](1.0) - Scalar[DT](1.0) / self.horizon
                )
            # dec
            dec.set_input["stoch_new", Self.B](self.snn, None)
            dec.set_input["nd", Self.B](self.ndn, None)
            dec.set_input["rtgt", Self.B](self.rtg, None)
            dec.forward[Self.B, target](self.dl, None)
            dec.vjp[Self.B, target](self.ones1, None)
            # rew
            rew.set_input["nd", Self.B](self.ndn, None)
            rew.set_input["stoch_new", Self.B](self.snn, None)
            rew.set_input["rtgt", Self.B](self.rwt, None)
            rew.forward[Self.B, target](self.dl, None)
            rew.vjp[Self.B, target](self.ones1, None)
            # con
            con.set_input["nd", Self.B](self.ndn, None)
            con.set_input["stoch_new", Self.B](self.snn, None)
            con.set_input["ctgt", Self.B](self.cnt, None)
            con.forward[Self.B, target](self.dl, None)
            con.vjp[Self.B, target](self.ones1, None)
            # assemble the core vjp seed
            ref dnd = dec.grad_input["nd"]()
            ref dsn = dec.grad_input["stoch_new"]()
            ref rnd = rew.grad_input["nd"]()
            ref rsn = rew.grad_input["stoch_new"]()
            ref cnd = con.grad_input["nd"]()
            ref csn = con.grad_input["stoch_new"]()
            for b in range(Self.B):
                self.seed.data[b * CARRYl + 0] = DYN
                self.seed.data[b * CARRYl + 1] = REP
                for k in range(D):
                    self.seed.data[b * CARRYl + 2 + k] = (
                        self.gcd.data[b * D + k] + dnd.data[b * D + k]
                        + rnd.data[b * D + k] + cnd.data[b * D + k]
                    )
                for k in range(SCl):
                    self.seed.data[b * CARRYl + 2 + D + k] = (
                        self.gcs.data[b * SCl + k] + dsn.data[b * SCl + k]
                        + rsn.data[b * SCl + k] + csn.data[b * SCl + k]
                    )
            # tokens window for step t
            for i in range(Self.B * TOK):
                self.tkscr.data[i] = st.toks.data[t * Self.B * TOK + i]
            core.set_input["deter", Self.B](self.cin_d, None)
            core.set_input["stoch", Self.B](self.cin_s, None)
            core.set_input["action", Self.B](self.at, None)
            core.set_input["tokens", Self.B](self.tkscr, None)
            core.forward[Self.B, target](self.outbuf, None)
            core.vjp[Self.B, target](self.seed, None)
            ref gdt = core.grad_input["deter"]()
            ref gst = core.grad_input["stoch"]()
            # Finding 3: cut the BPTT carry gradient at an episode boundary.
            for b in range(Self.B):
                var keep = (
                    Scalar[DT](0.0) if st.mb_fst.data[b * (Self.T + 1) + t + 1] >= Scalar[DT](0.5)
                    else Scalar[DT](1.0)
                )
                for k in range(D):
                    self.gcd.data[b * D + k] = keep * gdt.data[b * D + k]
                for k in range(SCl):
                    self.gcs.data[b * SCl + k] = keep * gst.data[b * SCl + k]
            # encoder backward — re-encode obs frame t+1, vjp the token grad.
            ref gtok = core.grad_input["tokens"]()
            for i in range(Self.B * TOK):
                self.gtok.data[i] = gtok.data[i]
            for b in range(Self.B):
                for k in range(OBSD):
                    self.ob.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
            enc.forward[target, Self.B](TensorRefs[1](self.ob), self.tkscr, None)
            enc.vjp[target, Self.B](
                TensorRefs[1](self.ob), self.gtok, TensorRefs[1](self.gobs), None
            )
        # optimizer steps
        oe.step[target, M=Self.EncT](enc, None)
        ocore.begin_step()
        core.for_each_param[target](ocore, None)
        odec.begin_step()
        dec.for_each_param[target](odec, None)
        orew.begin_step()
        rew.for_each_param[target](orew, None)
        ocon.begin_step()
        con.for_each_param[target](ocon, None)
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
        # GPU WM-BPTT scan — storage port of the legacy `_wm_gpu` (device nets/
        # graphs run on `.dev`; the per-step reset masking, carry threading and
        # head-loss accumulation marshal through host `.data` via upload/download
        # of the SMALL scratch Tensors). Structurally mirrors `_wm_cpu` so the
        # CPU↔GPU parity test holds; the only erasure is the GPU kernel ABI.
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime CARRYl = Self.CARRY
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        var DYN = self.dyn_scale
        var REP = self.rep_scale
        var ctx = st.ctx.value()

        # ── encode tokens: enc(obs frame t+1) → toks[t] (host-staged) ──
        # `self.ob` is the encoder input window; fill on host then upload. The
        # token sequence is staged in host `st.toks.data` (CPU-resident); the
        # per-step window is re-uploaded into `self.tkscr` for the device core.
        for t in range(Self.T):
            for b in range(Self.B):
                for k in range(OBSD):
                    self.ob.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
            self.ob.upload(ctx)
            enc.forward[target, Self.B](TensorRefs[1](self.ob), self.tkscr, ctx)
            self.tkscr.download(ctx)
            var base = t * Self.B * TOK
            for i in range(Self.B * TOK):
                st.toks.data[base + i] = self.tkscr.data[i]

        # zero carries (host `.data` is authoritative for the GPU path).
        for i in range(Self.B * D):
            st.cdeter.data[i] = 0.0
        for i in range(Self.B * SCl):
            st.cstoch.data[i] = 0.0

        var total: Scalar[DT] = 0.0
        # ── forward scan ──
        for t in range(Self.T):
            var dbase = t * Self.B * D
            var sbase = t * Self.B * SCl
            for b in range(Self.B):
                var keep = (
                    Scalar[DT](0.0) if st.mb_fst.data[b * (Self.T + 1) + t + 1] >= Scalar[DT](0.5)
                    else Scalar[DT](1.0)
                )
                for k in range(D):
                    self.cin_d.data[b * D + k] = keep * st.cdeter.data[dbase + b * D + k]
                for k in range(SCl):
                    self.cin_s.data[b * SCl + k] = keep * st.cstoch.data[sbase + b * SCl + k]
                for k in range(ACTD):
                    self.at.data[b * ACTD + k] = keep * st.mb_act.data[(b * Self.T + t) * ACTD + k]
            for i in range(Self.B * TOK):
                self.tkscr.data[i] = st.toks.data[t * Self.B * TOK + i]
            self.cin_d.upload(ctx)
            self.cin_s.upload(ctx)
            self.at.upload(ctx)
            self.tkscr.upload(ctx)
            core.set_input["deter", Self.B](self.cin_d, ctx)
            core.set_input["stoch", Self.B](self.cin_s, ctx)
            core.set_input["action", Self.B](self.at, ctx)
            core.set_input["tokens", Self.B](self.tkscr, ctx)
            core.forward[Self.B, target](self.outbuf, ctx)
            self.outbuf.download(ctx)
            var ndbase = (t + 1) * Self.B * D
            var snbase = (t + 1) * Self.B * SCl
            for b in range(Self.B):
                for k in range(D):
                    st.cdeter.data[ndbase + b * D + k] = self.outbuf.data[b * CARRYl + 2 + k]
                for k in range(SCl):
                    st.cstoch.data[snbase + b * SCl + k] = self.outbuf.data[b * CARRYl + 2 + D + k]
                total += DYN * self.outbuf.data[b * CARRYl + 0] + REP * self.outbuf.data[b * CARRYl + 1]
            # head inputs (next carry) + targets — fill host, upload.
            for b in range(Self.B):
                for k in range(D):
                    self.ndn.data[b * D + k] = st.cdeter.data[ndbase + b * D + k]
                for k in range(SCl):
                    self.snn.data[b * SCl + k] = st.cstoch.data[snbase + b * SCl + k]
                for k in range(OBSD):
                    self.rtg.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
                self.rwt.data[b] = st.mb_rew.data[b * Self.T + t]
                self.cnt.data[b] = (Scalar[DT](1.0) - st.mb_dne.data[b * Self.T + t]) * (
                    Scalar[DT](1.0) - Scalar[DT](1.0) / self.horizon
                )
            self.ndn.upload(ctx)
            self.snn.upload(ctx)
            self.rtg.upload(ctx)
            self.rwt.upload(ctx)
            self.cnt.upload(ctx)
            dec.set_input["stoch_new", Self.B](self.snn, ctx)
            dec.set_input["nd", Self.B](self.ndn, ctx)
            dec.set_input["rtgt", Self.B](self.rtg, ctx)
            dec.forward[Self.B, target](self.dl, ctx)
            self.dl.download(ctx)
            for b in range(Self.B):
                total += self.dl.data[b]
            rew.set_input["nd", Self.B](self.ndn, ctx)
            rew.set_input["stoch_new", Self.B](self.snn, ctx)
            rew.set_input["rtgt", Self.B](self.rwt, ctx)
            rew.forward[Self.B, target](self.dl, ctx)
            self.dl.download(ctx)
            for b in range(Self.B):
                total += self.dl.data[b]
            con.set_input["nd", Self.B](self.ndn, ctx)
            con.set_input["stoch_new", Self.B](self.snn, ctx)
            con.set_input["ctgt", Self.B](self.cnt, ctx)
            con.forward[Self.B, target](self.dl, ctx)
            self.dl.download(ctx)
            for b in range(Self.B):
                total += self.dl.data[b]

        # zero grads (enc Module via opt; loss graphs own their params).
        oe.zero_grad[target, M=Self.EncT](enc, ctx)
        core.zero_grad[target](ctx)
        dec.zero_grad[target](ctx)
        rew.zero_grad[target](ctx)
        con.zero_grad[target](ctx)
        for i in range(Self.B * D):
            self.gcd.data[i] = 0.0
        for i in range(Self.B * SCl):
            self.gcs.data[i] = 0.0
        for b in range(Self.B):
            self.ones1.data[b] = 1.0
        self.ones1.upload(ctx)
        # ── backward scan ──
        for rev in range(Self.T):
            var t = Self.T - 1 - rev
            var dbase = t * Self.B * D
            var sbase = t * Self.B * SCl
            var ndbase = (t + 1) * Self.B * D
            var snbase = (t + 1) * Self.B * SCl
            for b in range(Self.B):
                var keep = (
                    Scalar[DT](0.0) if st.mb_fst.data[b * (Self.T + 1) + t + 1] >= Scalar[DT](0.5)
                    else Scalar[DT](1.0)
                )
                for k in range(D):
                    self.cin_d.data[b * D + k] = keep * st.cdeter.data[dbase + b * D + k]
                for k in range(SCl):
                    self.cin_s.data[b * SCl + k] = keep * st.cstoch.data[sbase + b * SCl + k]
                for k in range(ACTD):
                    self.at.data[b * ACTD + k] = keep * st.mb_act.data[(b * Self.T + t) * ACTD + k]
                for k in range(D):
                    self.ndn.data[b * D + k] = st.cdeter.data[ndbase + b * D + k]
                for k in range(SCl):
                    self.snn.data[b * SCl + k] = st.cstoch.data[snbase + b * SCl + k]
                for k in range(OBSD):
                    self.rtg.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
                self.rwt.data[b] = st.mb_rew.data[b * Self.T + t]
                self.cnt.data[b] = (Scalar[DT](1.0) - st.mb_dne.data[b * Self.T + t]) * (
                    Scalar[DT](1.0) - Scalar[DT](1.0) / self.horizon
                )
            self.cin_d.upload(ctx)
            self.cin_s.upload(ctx)
            self.at.upload(ctx)
            self.ndn.upload(ctx)
            self.snn.upload(ctx)
            self.rtg.upload(ctx)
            self.rwt.upload(ctx)
            self.cnt.upload(ctx)
            # dec
            dec.set_input["stoch_new", Self.B](self.snn, ctx)
            dec.set_input["nd", Self.B](self.ndn, ctx)
            dec.set_input["rtgt", Self.B](self.rtg, ctx)
            dec.forward[Self.B, target](self.dl, ctx)
            dec.vjp[Self.B, target](self.ones1, ctx)
            # rew
            rew.set_input["nd", Self.B](self.ndn, ctx)
            rew.set_input["stoch_new", Self.B](self.snn, ctx)
            rew.set_input["rtgt", Self.B](self.rwt, ctx)
            rew.forward[Self.B, target](self.dl, ctx)
            rew.vjp[Self.B, target](self.ones1, ctx)
            # con
            con.set_input["nd", Self.B](self.ndn, ctx)
            con.set_input["stoch_new", Self.B](self.snn, ctx)
            con.set_input["ctgt", Self.B](self.cnt, ctx)
            con.forward[Self.B, target](self.dl, ctx)
            con.vjp[Self.B, target](self.ones1, ctx)
            # assemble the core vjp seed on device via _seed_asm_k (reads the
            # head grad_inputs + carry-grad accumulators gcd/gcs, all device).
            self.gcd.upload(ctx)
            self.gcs.upload(ctx)
            comptime nbB = (Self.B + TPB - 1) // TPB
            ctx.enqueue_function[_seed_asm_k[Self.B, CARRYl, D, SCl]](
                self.seed.lt["gpu", Layout.row_major(Self.B * CARRYl)](),
                self.gcd.lt["gpu", Layout.row_major(Self.B * D)](),
                self.gcs.lt["gpu", Layout.row_major(Self.B * SCl)](),
                dec.grad_input["nd"]().lt["gpu", Layout.row_major(Self.B * D)](),
                rew.grad_input["nd"]().lt["gpu", Layout.row_major(Self.B * D)](),
                con.grad_input["nd"]().lt["gpu", Layout.row_major(Self.B * D)](),
                dec.grad_input["stoch_new"]().lt["gpu", Layout.row_major(Self.B * SCl)](),
                rew.grad_input["stoch_new"]().lt["gpu", Layout.row_major(Self.B * SCl)](),
                con.grad_input["stoch_new"]().lt["gpu", Layout.row_major(Self.B * SCl)](),
                DYN, REP, grid_dim=nbB, block_dim=TPB,
            )
            # tokens window for step t
            for i in range(Self.B * TOK):
                self.tkscr.data[i] = st.toks.data[t * Self.B * TOK + i]
            self.tkscr.upload(ctx)
            core.set_input["deter", Self.B](self.cin_d, ctx)
            core.set_input["stoch", Self.B](self.cin_s, ctx)
            core.set_input["action", Self.B](self.at, ctx)
            core.set_input["tokens", Self.B](self.tkscr, ctx)
            core.forward[Self.B, target](self.outbuf, ctx)
            core.vjp[Self.B, target](self.seed, ctx)
            # Finding 3: cut the BPTT carry gradient at an episode boundary —
            # row-scale the core grad_inputs by the keep mask into gcd/gcs.
            ref gdt = core.grad_input["deter"]()
            ref gst = core.grad_input["stoch"]()
            gdt.download(ctx)
            gst.download(ctx)
            for b in range(Self.B):
                var keep = (
                    Scalar[DT](0.0) if st.mb_fst.data[b * (Self.T + 1) + t + 1] >= Scalar[DT](0.5)
                    else Scalar[DT](1.0)
                )
                for k in range(D):
                    self.gcd.data[b * D + k] = keep * gdt.data[b * D + k]
                for k in range(SCl):
                    self.gcs.data[b * SCl + k] = keep * gst.data[b * SCl + k]
            # encoder backward — re-encode obs frame t+1, vjp the token grad.
            ref gtok = core.grad_input["tokens"]()
            ctx.enqueue_copy(self.gtok.dev.value(), gtok.dev.value())
            for b in range(Self.B):
                for k in range(OBSD):
                    self.ob.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
            self.ob.upload(ctx)
            enc.forward[target, Self.B](TensorRefs[1](self.ob), self.tkscr, ctx)
            enc.vjp[target, Self.B](
                TensorRefs[1](self.ob), self.gtok, TensorRefs[1](self.gobs), ctx
            )
        # optimizer steps
        oe.step[target, M=Self.EncT](enc, ctx)
        ocore.begin_step()
        core.for_each_param[target](ocore, ctx)
        odec.begin_step()
        dec.for_each_param[target](odec, ctx)
        orew.begin_step()
        rew.for_each_param[target](orew, ctx)
        ocon.begin_step()
        con.for_each_param[target](ocon, ctx)
        ctx.synchronize()
        st.last_wm_loss = total


# ──────────────────────────────────────────────────────────────────────
# ParamSyncStep — copy core/prior params WMCoreGraph → WMImagineGraph.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct ParamSyncStep[
    DETER: Int, H: Int, STOCH: Int, CLASSES: Int, BLOCKS: Int, ACT: Int,
    TOKEN: Int,
](Movable & ImplicitlyDeletable):
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
        var snap = collect_graph_params[target](core, ctx)
        apply_graph_params[target](imagine, snap, ctx=ctx)


# ──────────────────────────────────────────────────────────────────────
# ACStep — imagination rollout + actor-critic loss. Trains value/policy;
# Polyak-updates slowvalue. Reads the start carry from state.cdeter[T].
# ──────────────────────────────────────────────────────────────────────


struct ACStep[
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, HU: Int, VU: Int, PU: Int, BINS: Int,
    B: Int, T: Int, T_IMAG: Int, DISCRETE: Bool = False,
](Movable & ImplicitlyDeletable):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime FEAT = Self.DETER + Self.SC
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT, SwishOp,
    ]
    comptime ValT = DreamerValue[Self.FEAT, Self.VU, Self.BINS, SwishOp]
    comptime PolT = DreamerPolicyHead[
        Self.FEAT, Self.PU, Self.ACT, Self.DISCRETE, SwishOp
    ]
    comptime POUT = Self.ACT if Self.DISCRETE else 2 * Self.ACT
    comptime RewT = RewLossGraph[Self.DETER, Self.SC, Self.HU, Self.BINS, SwishOp]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU, SwishOp]
    var minstd: Scalar[DT]
    var maxstd: Scalar[DT]
    var lam: Scalar[DT]
    var actent: Scalar[DT]
    var slowreg: Scalar[DT]
    var slow_rate: Scalar[DT]
    var horizon: Scalar[DT]
    var repval_scale: Scalar[DT]
    var slowtar: Bool

    # Persistent scratch Tensors (NS = T*B imagination starts; BT = B*T == NS).
    var fb: Tensor       # [NS*FEAT] per-step feat
    var pb: Tensor       # [NS*POUT] policy output
    var vb: Tensor       # [NS*BINS] value output
    var svb: Tensor      # [NS*BINS] slowvalue output
    var cd: Tensor       # [NS*DETER] rollout carry deter
    var cs: Tensor       # [NS*SC]    rollout carry stoch
    var at: Tensor       # [NS*ACT]   imagined action
    var dummy1: Tensor   # [NS]       head-loss dummy
    var vscr: Tensor     # [NS*BINS]  value re-forward scratch
    var pscr: Tensor     # [NS*POUT]  policy re-forward scratch
    var ftt: Tensor      # [NS*FEAT]  per-step feat (backward)
    var gvt: Tensor      # [NS*BINS]  value grad-out window
    var gfeat: Tensor    # [NS*FEAT]  value/policy grad-in (discarded)
    var polg: Tensor     # [NS*POUT]  policy grad-out window
    var feat_bt: Tensor  # [BT*FEAT]  repval features
    var vlr: Tensor      # [BT*BINS]  repval value forward
    var svlr: Tensor     # [BT*BINS]  repval slowvalue forward
    var g_vlr: Tensor    # [BT*BINS]  repval value grad-out
    var grf: Tensor      # [BT*FEAT]  repval value grad-in (discarded)

    def __init__(out self):
        self.minstd = Scalar[DT](0.1)
        self.maxstd = Scalar[DT](1.0)
        self.lam = Scalar[DT](0.95)
        self.actent = Scalar[DT](3e-4)
        self.slowreg = Scalar[DT](1.0)
        self.slow_rate = Scalar[DT](0.02)
        self.horizon = Scalar[DT](333.0)
        self.repval_scale = Scalar[DT](0.3)
        self.slowtar = False
        self.fb = Tensor()
        self.pb = Tensor()
        self.vb = Tensor()
        self.svb = Tensor()
        self.cd = Tensor()
        self.cs = Tensor()
        self.at = Tensor()
        self.dummy1 = Tensor()
        self.vscr = Tensor()
        self.pscr = Tensor()
        self.ftt = Tensor()
        self.gvt = Tensor()
        self.gfeat = Tensor()
        self.polg = Tensor()
        self.feat_bt = Tensor()
        self.vlr = Tensor()
        self.svlr = Tensor()
        self.g_vlr = Tensor()
        self.grf = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        actent: Scalar[DT] = Scalar[DT](3e-4),
        slowtar: Bool = False,
    ) raises -> Self:
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSl = Self.BINS
        comptime POUTl = Self.POUT
        comptime NS = Self.T * Self.B
        comptime BT = Self.B * Self.T
        var s = Self()
        s.actent = actent
        s.slowtar = slowtar
        s.fb = _mk[target](NS * FEATl, ctx)
        s.pb = _mk[target](NS * POUTl, ctx)
        s.vb = _mk[target](NS * BINSl, ctx)
        s.svb = _mk[target](NS * BINSl, ctx)
        s.cd = _mk[target](NS * D, ctx)
        s.cs = _mk[target](NS * SCl, ctx)
        s.at = _mk[target](NS * ACTD, ctx)
        s.dummy1 = _mk[target](NS, ctx)
        s.vscr = _mk[target](NS * BINSl, ctx)
        s.pscr = _mk[target](NS * POUTl, ctx)
        s.ftt = _mk[target](NS * FEATl, ctx)
        s.gvt = _mk[target](NS * BINSl, ctx)
        s.gfeat = _mk[target](NS * FEATl, ctx)
        s.polg = _mk[target](NS * POUTl, ctx)
        s.feat_bt = _mk[target](BT * FEATl, ctx)
        s.vlr = _mk[target](BT * BINSl, ctx)
        s.svlr = _mk[target](BT * BINSl, ctx)
        s.g_vlr = _mk[target](BT * BINSl, ctx)
        s.grf = _mk[target](BT * FEATl, ctx)
        return s^

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
        comptime POUTl = Self.POUT
        var MINSTD = self.minstd
        var MAXSTD = self.maxstd
        # FIX (2): imagine from ALL B·T posterior carries (cdeter indices 1..T;
        # index 0 is the zero init) flattened to NS = T·B starts.
        comptime NS = Self.T * Self.B

        # host accumulator arrays (List → RAII, no leaks) for the loss helpers.
        var acts = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var pmean = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var pstd = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var vlog = List[Scalar[DT]](length=NS * TI * BINSl, fill=Scalar[DT](0))
        var svlog = List[Scalar[DT]](length=NS * TI * BINSl, fill=Scalar[DT](0))
        var rewv = List[Scalar[DT]](length=NS * TI, fill=Scalar[DT](0))
        var conv = List[Scalar[DT]](length=NS * TI, fill=Scalar[DT](0))
        var feats = List[Scalar[DT]](length=NS * TI * FEATl, fill=Scalar[DT](0))

        # init rollout carry cd/cs from posterior carries 1..T.
        for i in range(NS * D):
            self.cd.data[i] = st.cdeter.data[Self.B * D + i]
        for i in range(NS * SCl):
            self.cs.data[i] = st.cstoch.data[Self.B * SCl + i]

        for t in range(TI):
            for b in range(NS):
                for k in range(D):
                    self.fb.data[b * FEATl + k] = self.cd.data[b * D + k]
                for k in range(SCl):
                    self.fb.data[b * FEATl + D + k] = self.cs.data[b * SCl + k]
                for k in range(FEATl):
                    feats[(b * TI + t) * FEATl + k] = self.fb.data[b * FEATl + k]
            policy.forward[target, NS](TensorRefs[1](self.fb), self.pb, None)
            comptime if Self.DISCRETE:
                for b in range(NS):
                    for a in range(ACTD):
                        pmean[(b * TI + t) * ACTD + a] = self.pb.data[b * ACTD + a]
                        pstd[(b * TI + t) * ACTD + a] = 0.0
                    var z0 = st.noise.data[(t * NS + b) * ACTD + 0]
                    var u01 = (z0 + Scalar[DT](1.0)) * Scalar[DT](0.5)
                    var k = cat_sample[ACTD](_hp(self.pb), b * ACTD, UNIMIX, u01)
                    for a in range(ACTD):
                        acts[(b * TI + t) * ACTD + a] = (
                            Scalar[DT](1.0) if a == k else Scalar[DT](0.0)
                        )
            else:
                for b in range(NS):
                    for a in range(ACTD):
                        var mr = self.pb.data[b * 2 * ACTD + a]
                        var sr = self.pb.data[b * 2 * ACTD + ACTD + a]
                        pmean[(b * TI + t) * ACTD + a] = mr
                        pstd[(b * TI + t) * ACTD + a] = sr
                        var z = st.noise.data[(t * NS + b) * ACTD + a]
                        acts[(b * TI + t) * ACTD + a] = (
                            tanh(mr) + bounded_std(sr, MINSTD, MAXSTD) * z
                        )
            value.forward[target, NS](TensorRefs[1](self.fb), self.vb, None)
            slowvalue.forward[target, NS](TensorRefs[1](self.fb), self.svb, None)
            for b in range(NS):
                for c in range(BINSl):
                    vlog[(b * TI + t) * BINSl + c] = self.vb.data[b * BINSl + c]
                    svlog[(b * TI + t) * BINSl + c] = self.svb.data[b * BINSl + c]
            # rew head — read its logits from node_output["rew"].
            rew.set_input["nd", NS](self.cd, None)
            rew.set_input["stoch_new", NS](self.cs, None)
            rew.set_input["rtgt", NS](self.dummy1, None)
            rew.forward[NS, target](self.dummy1, None)
            ref rew_logits = rew.node_output["rew"]()
            con.set_input["nd", NS](self.cd, None)
            con.set_input["stoch_new", NS](self.cs, None)
            con.set_input["ctgt", NS](self.dummy1, None)
            con.forward[NS, target](self.dummy1, None)
            ref con_logit = con.node_output["con"]()
            for b in range(NS):
                rewv[b * TI + t] = twohot_pred[BINSl](
                    _hp(rew_logits), b * BINSl, bins
                )
                conv[b * TI + t] = Scalar[DT](1.0) / (
                    Scalar[DT](1.0) + exp(-con_logit.data[b])
                )
                for a in range(ACTD):
                    self.at.data[b * ACTD + a] = acts[(b * TI + t) * ACTD + a]
            imagine.set_input["deter", NS](self.cd, None)
            imagine.set_input["stoch", NS](self.cs, None)
            imagine.set_input["action", NS](self.at, None)
            imagine.forward[NS, target](self.fb, None)
            ref nd = imagine.node_output["nd"]()
            ref sn = imagine.node_output["stoch_new"]()
            for i in range(NS * D):
                self.cd.data[i] = nd.data[i]
            for i in range(NS * SCl):
                self.cs.data[i] = sn.data[i]

        comptime TM1 = TI - 1
        var pol_loss = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var val_loss = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var ret = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        imag_loss_cpu[NS, TI, ACTD, BINSl, Self.DISCRETE](
            acts.unsafe_ptr(), rewv.unsafe_ptr(), conv.unsafe_ptr(),
            vlog.unsafe_ptr(), svlog.unsafe_ptr(), pmean.unsafe_ptr(),
            pstd.unsafe_ptr(), bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            retnorm, pol_loss, val_loss,
            ret, self.slowtar,
        )
        var total: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            total += pol_loss[i] + val_loss[i]
        total = total / Scalar[DT](NS * TM1)
        # ── diagnostics ──
        var pma: Scalar[DT] = 0.0
        for i in range(NS * TI * ACTD):
            pma += pmean[i] if pmean[i] >= 0 else -pmean[i]
        st.dbg_pmean_abs = pma / Scalar[DT](NS * TI * ACTD)
        var rp: Scalar[DT] = 0.0
        for i in range(NS * TI):
            rp += rewv[i]
        st.dbg_rew_pred = rp / Scalar[DT](NS * TI)
        # imagined continue-factor (conv): mean + min over the rollout. min≈0.997
        # ⇒ the cont head never truncates → λ-return saturated → no actor signal.
        var cm: Scalar[DT] = 0.0
        var cmin: Scalar[DT] = conv[0]
        for i in range(NS * TI):
            cm += conv[i]
            if conv[i] < cmin:
                cmin = conv[i]
        st.dbg_con_mean = cm / Scalar[DT](NS * TI)
        st.dbg_con_min = cmin
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
        st.dbg_rscale = rscale
        var ps_acc: Scalar[DT] = 0.0
        comptime if not Self.DISCRETE:
            for i in range(NS * TI * ACTD):
                ps_acc += bounded_std(pstd[i], MINSTD, MAXSTD)
        st.dbg_pstd = ps_acc / Scalar[DT](NS * TI * ACTD)
        var vm_acc: Scalar[DT] = 0.0
        for b in range(NS):
            for t in range(TI):
                vm_acc += twohot_pred[BINSl](vlog.unsafe_ptr(), (b * TI + t) * BINSl, bins)
        var vmean = vm_acc / Scalar[DT](NS * TI)
        st.dbg_val_mean = vmean
        # value spread over imagined states (val_std≈0 ⇒ value head collapsed)
        var vv: Scalar[DT] = 0.0
        for b in range(NS):
            for t in range(TI):
                var dv = twohot_pred[BINSl](vlog.unsafe_ptr(), (b * TI + t) * BINSl, bins) - vmean
                vv += dv * dv
        st.dbg_val_std = sqrt(vv / Scalar[DT](NS * TI))
        # latent feat spread over imagined states (feat_std≈0 ⇒ latent collapsed)
        var fm: Scalar[DT] = 0.0
        for i in range(NS * TI * FEATl):
            fm += feats[i]
        fm = fm / Scalar[DT](NS * TI * FEATl)
        var fv: Scalar[DT] = 0.0
        for i in range(NS * TI * FEATl):
            var df = feats[i] - fm
            fv += df * df
        st.dbg_feat_std = sqrt(fv / Scalar[DT](NS * TI * FEATl))

        var d_pol = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var d_val = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var inv_im = Scalar[DT](1.0) / Scalar[DT](NS * TM1)
        for i in range(NS * TM1):
            d_pol[i] = inv_im
            d_val[i] = inv_im
        var g_vlog = List[Scalar[DT]](length=NS * TI * BINSl, fill=Scalar[DT](0))
        var g_pmean = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var g_pstd = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        imag_loss_backward[NS, TI, ACTD, BINSl, Self.DISCRETE](
            acts.unsafe_ptr(), rewv.unsafe_ptr(), conv.unsafe_ptr(),
            vlog.unsafe_ptr(), svlog.unsafe_ptr(), pmean.unsafe_ptr(),
            pstd.unsafe_ptr(), bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            rscale, d_pol.unsafe_ptr(), d_val.unsafe_ptr(),
            g_vlog.unsafe_ptr(), g_pmean.unsafe_ptr(), g_pstd.unsafe_ptr(),
            self.slowtar,
        )
        oval.zero_grad[target, M=Self.ValT](value, None)
        opol.zero_grad[target, M=Self.PolT](policy, None)
        for t in range(TI):
            for b in range(NS):
                for k in range(FEATl):
                    self.ftt.data[b * FEATl + k] = feats[(b * TI + t) * FEATl + k]
            value.forward[target, NS](TensorRefs[1](self.ftt), self.vscr, None)
            for b in range(NS):
                for c in range(BINSl):
                    self.gvt.data[b * BINSl + c] = g_vlog[(b * TI + t) * BINSl + c]
            value.vjp[target, NS](
                TensorRefs[1](self.ftt), self.gvt, TensorRefs[1](self.gfeat), None
            )
            policy.forward[target, NS](TensorRefs[1](self.ftt), self.pscr, None)
            comptime if Self.DISCRETE:
                for b in range(NS):
                    for a in range(ACTD):
                        self.polg.data[b * ACTD + a] = g_pmean[(b * TI + t) * ACTD + a]
            else:
                for b in range(NS):
                    for a in range(ACTD):
                        self.polg.data[b * 2 * ACTD + a] = g_pmean[(b * TI + t) * ACTD + a]
                        self.polg.data[b * 2 * ACTD + ACTD + a] = g_pstd[(b * TI + t) * ACTD + a]
            policy.vjp[target, NS](
                TensorRefs[1](self.ftt), self.polg, TensorRefs[1](self.gfeat), None
            )

        # ── repval: ground the value head on REAL replay transitions ──
        comptime BT = Self.B * Self.T
        var boot_bt = List[Scalar[DT]](length=BT, fill=Scalar[DT](0))
        var term_bt = List[Scalar[DT]](length=BT, fill=Scalar[DT](0))
        for b in range(Self.B):
            for j in range(Self.T):
                var s = j * Self.B + b
                boot_bt[b * Self.T + j] = ret[s * TM1 + 0]
                term_bt[b * Self.T + j] = 0.0   # Pendulum: truncation, not term
                for k in range(FEATl):
                    self.feat_bt.data[(b * Self.T + j) * FEATl + k] = (
                        feats[(s * TI) * FEATl + k]
                    )
        value.forward[target, BT](TensorRefs[1](self.feat_bt), self.vlr, None)
        slowvalue.forward[target, BT](TensorRefs[1](self.feat_bt), self.svlr, None)
        # repval runs over the REAL replay sequence (length Self.T), NOT the
        # imagination horizon: repl_loss_backward[Self.B, Self.T] emits a
        # cotangent shaped [Self.B, Self.T-1]. Size d_rep accordingly (TM1 above
        # is the imagination TM1 = T_IMAG-1; reusing it overflows when T_IMAG≠T).
        comptime TM1R = Self.T - 1
        var d_rep = List[Scalar[DT]](length=Self.B * TM1R, fill=Scalar[DT](0))
        var inv_rep = self.repval_scale / Scalar[DT](Self.B * TM1R)
        for i in range(Self.B * TM1R):
            d_rep[i] = inv_rep
        repl_loss_backward[Self.B, Self.T, BINSl](
            _hp(st.mb_dne), term_bt, _hp(st.mb_rew),
            boot_bt, _hp(self.vlr), _hp(self.svlr), bins,
            self.horizon, self.lam, self.slowreg, d_rep,
            _hp(self.g_vlr),
        )
        value.vjp[target, BT](
            TensorRefs[1](self.feat_bt), self.g_vlr, TensorRefs[1](self.grf), None
        )

        oval.step[target, M=Self.ValT](value, None)
        opol.step[target, M=Self.PolT](policy, None)
        polyak_module[target, Self.ValT](value, slowvalue, self.slow_rate)
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
        # GPU imagination-AC — storage port of the legacy `_ac_gpu`. The device
        # nets (policy/value/slowvalue) + loss graphs (rew/con/imagine) run on
        # `.dev`; the per-step connective math (tanh+std sample, twohot, sigmoid)
        # and the λ-return imag_loss / repl_loss run on HOST via download/upload
        # of the small scratch Tensors. Mirrors `_ac_cpu` exactly → CPU↔GPU
        # parity. (DISCRETE GPU AC is unported — guarded below.)
        comptime assert not Self.DISCRETE, (
            "discrete (categorical) GPU AC not ported — use train_target='cpu' "
            "for discrete-action envs; GPU discrete is a follow-up parity step."
        )
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSl = Self.BINS
        comptime TI = Self.T_IMAG
        comptime POUTl = Self.POUT
        var MINSTD = self.minstd
        var MAXSTD = self.maxstd
        comptime NS = Self.T * Self.B
        var ctx = st.ctx.value()

        # host accumulator arrays (List → RAII) for the loss helpers.
        var acts = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var pmean = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var pstd = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var vlog = List[Scalar[DT]](length=NS * TI * BINSl, fill=Scalar[DT](0))
        var svlog = List[Scalar[DT]](length=NS * TI * BINSl, fill=Scalar[DT](0))
        var rewv = List[Scalar[DT]](length=NS * TI, fill=Scalar[DT](0))
        var conv = List[Scalar[DT]](length=NS * TI, fill=Scalar[DT](0))
        var feats = List[Scalar[DT]](length=NS * TI * FEATl, fill=Scalar[DT](0))

        # init rollout carry cd/cs from posterior carries 1..T. cdeter/cstoch
        # host `.data` is authoritative (filled by _wm_gpu's download); fill the
        # NS rollout carry on host then upload.
        for i in range(NS * D):
            self.cd.data[i] = st.cdeter.data[Self.B * D + i]
        for i in range(NS * SCl):
            self.cs.data[i] = st.cstoch.data[Self.B * SCl + i]
        self.cd.upload(ctx)
        self.cs.upload(ctx)

        comptime nbB = (NS + TPB - 1) // TPB
        for t in range(TI):
            # feat = concat([cd, cs]) on device via _feat_concat_k.
            ctx.enqueue_function[_feat_concat_k[NS, D, SCl]](
                self.cd.lt["gpu", Layout.row_major(NS * D)](),
                self.cs.lt["gpu", Layout.row_major(NS * SCl)](),
                self.fb.lt["gpu", Layout.row_major(NS * FEATl)](),
                grid_dim=nbB, block_dim=TPB,
            )
            self.fb.download(ctx)
            for b in range(NS):
                for k in range(FEATl):
                    feats[(b * TI + t) * FEATl + k] = self.fb.data[b * FEATl + k]
            policy.forward[target, NS](TensorRefs[1](self.fb), self.pb, ctx)
            self.pb.download(ctx)
            for b in range(NS):
                for a in range(ACTD):
                    var mr = self.pb.data[b * 2 * ACTD + a]
                    var sr = self.pb.data[b * 2 * ACTD + ACTD + a]
                    pmean[(b * TI + t) * ACTD + a] = mr
                    pstd[(b * TI + t) * ACTD + a] = sr
                    var z = st.noise.data[(t * NS + b) * ACTD + a]
                    acts[(b * TI + t) * ACTD + a] = (
                        tanh(mr) + bounded_std(sr, MINSTD, MAXSTD) * z
                    )
            value.forward[target, NS](TensorRefs[1](self.fb), self.vb, ctx)
            slowvalue.forward[target, NS](TensorRefs[1](self.fb), self.svb, ctx)
            self.vb.download(ctx)
            self.svb.download(ctx)
            for b in range(NS):
                for c in range(BINSl):
                    vlog[(b * TI + t) * BINSl + c] = self.vb.data[b * BINSl + c]
                    svlog[(b * TI + t) * BINSl + c] = self.svb.data[b * BINSl + c]
            # rew head — read its logits from node_output["rew"].
            rew.set_input["nd", NS](self.cd, ctx)
            rew.set_input["stoch_new", NS](self.cs, ctx)
            rew.set_input["rtgt", NS](self.dummy1, ctx)
            rew.forward[NS, target](self.dummy1, ctx)
            ref rew_logits = rew.node_output["rew"]()
            rew_logits.download(ctx)
            con.set_input["nd", NS](self.cd, ctx)
            con.set_input["stoch_new", NS](self.cs, ctx)
            con.set_input["ctgt", NS](self.dummy1, ctx)
            con.forward[NS, target](self.dummy1, ctx)
            ref con_logit = con.node_output["con"]()
            con_logit.download(ctx)
            for b in range(NS):
                rewv[b * TI + t] = twohot_pred[BINSl](
                    _hp(rew_logits), b * BINSl, bins
                )
                conv[b * TI + t] = Scalar[DT](1.0) / (
                    Scalar[DT](1.0) + exp(-con_logit.data[b])
                )
                for a in range(ACTD):
                    self.at.data[b * ACTD + a] = acts[(b * TI + t) * ACTD + a]
            self.at.upload(ctx)
            imagine.set_input["deter", NS](self.cd, ctx)
            imagine.set_input["stoch", NS](self.cs, ctx)
            imagine.set_input["action", NS](self.at, ctx)
            imagine.forward[NS, target](self.fb, ctx)
            ref nd = imagine.node_output["nd"]()
            ref sn = imagine.node_output["stoch_new"]()
            ctx.enqueue_copy(self.cd.dev.value(), nd.dev.value())
            ctx.enqueue_copy(self.cs.dev.value(), sn.dev.value())

        comptime TM1 = TI - 1
        var pol_loss = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var val_loss = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var ret = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        imag_loss_cpu[NS, TI, ACTD, BINSl, Self.DISCRETE](
            acts.unsafe_ptr(), rewv.unsafe_ptr(), conv.unsafe_ptr(),
            vlog.unsafe_ptr(), svlog.unsafe_ptr(), pmean.unsafe_ptr(),
            pstd.unsafe_ptr(), bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            retnorm, pol_loss, val_loss,
            ret, self.slowtar,
        )
        var total: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            total += pol_loss[i] + val_loss[i]
        total = total / Scalar[DT](NS * TM1)
        # ── diagnostics ──
        var pma: Scalar[DT] = 0.0
        for i in range(NS * TI * ACTD):
            pma += pmean[i] if pmean[i] >= 0 else -pmean[i]
        st.dbg_pmean_abs = pma / Scalar[DT](NS * TI * ACTD)
        var rp: Scalar[DT] = 0.0
        for i in range(NS * TI):
            rp += rewv[i]
        st.dbg_rew_pred = rp / Scalar[DT](NS * TI)
        # imagined continue-factor (conv): mean + min over the rollout. min≈0.997
        # ⇒ the cont head never truncates → λ-return saturated → no actor signal.
        var cm: Scalar[DT] = 0.0
        var cmin: Scalar[DT] = conv[0]
        for i in range(NS * TI):
            cm += conv[i]
            if conv[i] < cmin:
                cmin = conv[i]
        st.dbg_con_mean = cm / Scalar[DT](NS * TI)
        st.dbg_con_min = cmin
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
        st.dbg_rscale = rscale
        var ps_acc: Scalar[DT] = 0.0
        comptime if not Self.DISCRETE:
            for i in range(NS * TI * ACTD):
                ps_acc += bounded_std(pstd[i], MINSTD, MAXSTD)
        st.dbg_pstd = ps_acc / Scalar[DT](NS * TI * ACTD)
        var vm_acc: Scalar[DT] = 0.0
        for b in range(NS):
            for t in range(TI):
                vm_acc += twohot_pred[BINSl](vlog.unsafe_ptr(), (b * TI + t) * BINSl, bins)
        var vmean = vm_acc / Scalar[DT](NS * TI)
        st.dbg_val_mean = vmean
        # value spread over imagined states (val_std≈0 ⇒ value head collapsed)
        var vv: Scalar[DT] = 0.0
        for b in range(NS):
            for t in range(TI):
                var dv = twohot_pred[BINSl](vlog.unsafe_ptr(), (b * TI + t) * BINSl, bins) - vmean
                vv += dv * dv
        st.dbg_val_std = sqrt(vv / Scalar[DT](NS * TI))
        # latent feat spread over imagined states (feat_std≈0 ⇒ latent collapsed)
        var fm: Scalar[DT] = 0.0
        for i in range(NS * TI * FEATl):
            fm += feats[i]
        fm = fm / Scalar[DT](NS * TI * FEATl)
        var fv: Scalar[DT] = 0.0
        for i in range(NS * TI * FEATl):
            var df = feats[i] - fm
            fv += df * df
        st.dbg_feat_std = sqrt(fv / Scalar[DT](NS * TI * FEATl))

        var d_pol = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var d_val = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var inv_im = Scalar[DT](1.0) / Scalar[DT](NS * TM1)
        for i in range(NS * TM1):
            d_pol[i] = inv_im
            d_val[i] = inv_im
        var g_vlog = List[Scalar[DT]](length=NS * TI * BINSl, fill=Scalar[DT](0))
        var g_pmean = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var g_pstd = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        imag_loss_backward[NS, TI, ACTD, BINSl, Self.DISCRETE](
            acts.unsafe_ptr(), rewv.unsafe_ptr(), conv.unsafe_ptr(),
            vlog.unsafe_ptr(), svlog.unsafe_ptr(), pmean.unsafe_ptr(),
            pstd.unsafe_ptr(), bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            rscale, d_pol.unsafe_ptr(), d_val.unsafe_ptr(),
            g_vlog.unsafe_ptr(), g_pmean.unsafe_ptr(), g_pstd.unsafe_ptr(),
            self.slowtar,
        )
        oval.zero_grad[target, M=Self.ValT](value, ctx)
        opol.zero_grad[target, M=Self.PolT](policy, ctx)
        for t in range(TI):
            for b in range(NS):
                for k in range(FEATl):
                    self.ftt.data[b * FEATl + k] = feats[(b * TI + t) * FEATl + k]
            self.ftt.upload(ctx)
            value.forward[target, NS](TensorRefs[1](self.ftt), self.vscr, ctx)
            for b in range(NS):
                for c in range(BINSl):
                    self.gvt.data[b * BINSl + c] = g_vlog[(b * TI + t) * BINSl + c]
            self.gvt.upload(ctx)
            value.vjp[target, NS](
                TensorRefs[1](self.ftt), self.gvt, TensorRefs[1](self.gfeat), ctx
            )
            policy.forward[target, NS](TensorRefs[1](self.ftt), self.pscr, ctx)
            for b in range(NS):
                for a in range(ACTD):
                    self.polg.data[b * 2 * ACTD + a] = g_pmean[(b * TI + t) * ACTD + a]
                    self.polg.data[b * 2 * ACTD + ACTD + a] = g_pstd[(b * TI + t) * ACTD + a]
            self.polg.upload(ctx)
            policy.vjp[target, NS](
                TensorRefs[1](self.ftt), self.polg, TensorRefs[1](self.gfeat), ctx
            )

        # ── repval: ground the value head on REAL replay transitions ──
        comptime BT = Self.B * Self.T
        var boot_bt = List[Scalar[DT]](length=BT, fill=Scalar[DT](0))
        var term_bt = List[Scalar[DT]](length=BT, fill=Scalar[DT](0))
        for b in range(Self.B):
            for j in range(Self.T):
                var s = j * Self.B + b
                boot_bt[b * Self.T + j] = ret[s * TM1 + 0]
                term_bt[b * Self.T + j] = 0.0   # Pendulum: truncation, not term
                for k in range(FEATl):
                    self.feat_bt.data[(b * Self.T + j) * FEATl + k] = (
                        feats[(s * TI) * FEATl + k]
                    )
        self.feat_bt.upload(ctx)
        value.forward[target, BT](TensorRefs[1](self.feat_bt), self.vlr, ctx)
        slowvalue.forward[target, BT](TensorRefs[1](self.feat_bt), self.svlr, ctx)
        self.vlr.download(ctx)
        self.svlr.download(ctx)
        # repval runs over the REAL replay sequence (length Self.T), NOT the
        # imagination horizon: repl_loss_backward[Self.B, Self.T] emits a
        # cotangent shaped [Self.B, Self.T-1]. Size d_rep accordingly (TM1 above
        # is the imagination TM1 = T_IMAG-1; reusing it overflows when T_IMAG≠T).
        comptime TM1R = Self.T - 1
        var d_rep = List[Scalar[DT]](length=Self.B * TM1R, fill=Scalar[DT](0))
        var inv_rep = self.repval_scale / Scalar[DT](Self.B * TM1R)
        for i in range(Self.B * TM1R):
            d_rep[i] = inv_rep
        repl_loss_backward[Self.B, Self.T, BINSl](
            _hp(st.mb_dne), term_bt, _hp(st.mb_rew),
            boot_bt, _hp(self.vlr), _hp(self.svlr), bins,
            self.horizon, self.lam, self.slowreg, d_rep,
            _hp(self.g_vlr),
        )
        self.g_vlr.upload(ctx)
        value.vjp[target, BT](
            TensorRefs[1](self.feat_bt), self.g_vlr, TensorRefs[1](self.grf), ctx
        )

        oval.step[target, M=Self.ValT](value, ctx)
        opol.step[target, M=Self.PolT](policy, ctx)
        polyak_module[target, Self.ValT](value, slowvalue, self.slow_rate, ctx=st.ctx)
        ctx.synchronize()
        st.last_ac_loss = total
