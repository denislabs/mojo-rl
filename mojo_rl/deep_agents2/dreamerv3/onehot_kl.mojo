"""OneHotKL[STOCH, CLASSES] — DreamerV3 dynamics/representation KL.

Ports `references/dreamerv3-main/dreamerv3/rssm.py:RSSM.loss` together
with the categorical KL of `embodied/jax/outs.py:Categorical(unimix)` and
the `Agg(sum over STOCH)` wrapper:

    p   = unimix(softmax(logits))   where  unimix(p) = (1-u)·p + u/C
    KL(self‖other)_s = Σ_c p_self[s,c]·(log p_self[s,c] − log p_other[s,c])
    dyn = max(Σ_s KL(sg(post)‖prior)_s, free_nats)     # grad → prior only
    rep = max(Σ_s KL(post‖sg(prior))_s, free_nats)     # grad → post  only

Forward `dyn` and `rep` are numerically identical (stop-gradient is the
identity in forward); they differ only in which side the gradient flows
to. The free-nats `max` gates the gradient: when `Σ_s KL < free_nats` the
loss is the constant `free_nats` and the whole row's gradient is zero.

`grad_prior` receives ONLY the dyn contribution (rep stop-grads the prior);
`grad_post` receives ONLY the rep contribution (dyn stop-grads the post).
Both are WRITTEN (`=`), not accumulated — the caller (RSSM loss graph)
owns any cross-path accumulation.

Inputs/outputs are flat row-major pointers: logits `[BATCH·STOCH·CLASSES]`
(group `(b,s)` spans `CLASSES` contiguous lanes), `dyn`/`rep`/`d_dyn`/
`d_rep` `[BATCH]`.

CPU-only at landing — Pendulum v1 trains the world model on CPU. A GPU
port is gated on GPU world-model training (same rationale as the GRUCell
GPU stub).
"""

from std.math import exp, log
from std.memory import alloc
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for


@always_inline
def _dlt[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


# ── OneHotKLLoss GPU kernels. NG=B·STOCH groups of CLASSES lanes; NN=B·SC.
#    fwd1 (per group): softmax+unimix(post/prior) → caches + per-group KL.
#    fwd2 (per batch): Σ_s KL → free-nats clamp → out[B,2] + active[B].
#    bwd (per group): asymmetric dyn→prior / rep→post grads.  ────────────
def _kl_fwd1_kernel[NG: Int, C: Int](
    post: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    prior: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    smpo: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    ppo: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    smpr: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    ppr: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    klpart: LayoutTensor[DT, Layout.row_major(NG), MutAnyOrigin],
    u: Scalar[DT],
):
    var g = Int(global_idx.x)
    if g < NG:
        var base = g * C
        var uni = u / Scalar[DT](C)
        var omu = Scalar[DT](1.0) - u
        # post softmax + mix
        var zmax = rebind[Scalar[DT]](post[base])
        for c in range(1, C):
            var v = rebind[Scalar[DT]](post[base + c])
            if v > zmax:
                zmax = v
        var ss: Scalar[DT] = 0.0
        for c in range(C):
            var e = exp(rebind[Scalar[DT]](post[base + c]) - zmax)
            smpo[base + c] = e
            ss += e
        var inv = Scalar[DT](1.0) / ss
        for c in range(C):
            var sm = rebind[Scalar[DT]](smpo[base + c]) * inv
            smpo[base + c] = sm
            ppo[base + c] = omu * sm + uni
        # prior softmax + mix
        var zmx2 = rebind[Scalar[DT]](prior[base])
        for c in range(1, C):
            var v = rebind[Scalar[DT]](prior[base + c])
            if v > zmx2:
                zmx2 = v
        var ss2: Scalar[DT] = 0.0
        for c in range(C):
            var e = exp(rebind[Scalar[DT]](prior[base + c]) - zmx2)
            smpr[base + c] = e
            ss2 += e
        var inv2 = Scalar[DT](1.0) / ss2
        for c in range(C):
            var sm = rebind[Scalar[DT]](smpr[base + c]) * inv2
            smpr[base + c] = sm
            ppr[base + c] = omu * sm + uni
        # group KL
        var kl: Scalar[DT] = 0.0
        for c in range(C):
            var pp = rebind[Scalar[DT]](ppo[base + c])
            kl += pp * (log(pp) - log(rebind[Scalar[DT]](ppr[base + c])))
        klpart[g] = kl


def _kl_fwd2_kernel[B: Int, STOCH: Int](
    klpart: LayoutTensor[DT, Layout.row_major(B * STOCH), MutAnyOrigin],
    outb: LayoutTensor[DT, Layout.row_major(B * 2), MutAnyOrigin],
    active: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    free_nats: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b < B:
        var ksum: Scalar[DT] = 0.0
        for s in range(STOCH):
            ksum += rebind[Scalar[DT]](klpart[b * STOCH + s])
        var clamped = ksum if ksum > free_nats else free_nats
        outb[b * 2] = clamped
        outb[b * 2 + 1] = clamped
        active[b] = Scalar[DT](1.0) if ksum > free_nats else Scalar[DT](0.0)


def _kl_bwd_kernel[NG: Int, C: Int, STOCH: Int](
    go: LayoutTensor[DT, Layout.row_major(NG // STOCH * 2), MutAnyOrigin],
    active: LayoutTensor[DT, Layout.row_major(NG // STOCH), MutAnyOrigin],
    smpo: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    ppo: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    smpr: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    ppr: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    g_post: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    g_prior: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    one_m_u: Scalar[DT],
):
    var g = Int(global_idx.x)
    if g < NG:
        var b = g // STOCH
        var base = g * C
        var act = rebind[Scalar[DT]](active[b])
        var wd = rebind[Scalar[DT]](go[b * 2]) * act
        var wr = rebind[Scalar[DT]](go[b * 2 + 1]) * act
        var S: Scalar[DT] = 0.0
        var W: Scalar[DT] = 0.0
        for c in range(C):
            S += (rebind[Scalar[DT]](ppo[base + c]) / rebind[Scalar[DT]](ppr[base + c])) * rebind[Scalar[DT]](smpr[base + c])
            W += rebind[Scalar[DT]](smpo[base + c]) * (
                log(rebind[Scalar[DT]](ppo[base + c])) - log(rebind[Scalar[DT]](ppr[base + c]))
            )
        for j in range(C):
            var a_over_p = rebind[Scalar[DT]](ppo[base + j]) / rebind[Scalar[DT]](ppr[base + j])
            g_prior[base + j] = wd * (-one_m_u * rebind[Scalar[DT]](smpr[base + j]) * (a_over_p - S))
            var lr_diff = log(rebind[Scalar[DT]](ppo[base + j])) - log(rebind[Scalar[DT]](ppr[base + j]))
            g_post[base + j] = wr * (one_m_u * rebind[Scalar[DT]](smpo[base + j]) * (lr_diff - W))


struct OneHotKL[STOCH: Int, CLASSES: Int](Movable & ImplicitlyDestructible):
    comptime GROUP = Self.STOCH * Self.CLASSES

    var unimix: Scalar[DT]
    var free_nats: Scalar[DT]

    # Caches (CPU), lazily grown to BATCH·GROUP / BATCH.
    var sm_post: List[Scalar[DT]]    # pre-mix softmax(post)
    var sm_prior: List[Scalar[DT]]
    var p_post: List[Scalar[DT]]     # unimix-mixed probs
    var p_prior: List[Scalar[DT]]
    var active: List[Scalar[DT]]     # [BATCH] 1.0 if not clamped
    var cache_n: Int

    def __init__(out self):
        self.unimix = Scalar[DT](0.01)
        self.free_nats = Scalar[DT](1.0)
        self.sm_post = List[Scalar[DT]]()
        self.sm_prior = List[Scalar[DT]]()
        self.p_post = List[Scalar[DT]]()
        self.p_prior = List[Scalar[DT]]()
        self.active = List[Scalar[DT]]()
        self.cache_n = 0

    @staticmethod
    def make(
        unimix: Scalar[DT] = Scalar[DT](0.01),
        free_nats: Scalar[DT] = Scalar[DT](1.0),
    ) -> Self:
        var k = Self()
        k.unimix = unimix
        k.free_nats = free_nats
        return k^

    def _ensure_cache(mut self, batch: Int):
        if self.cache_n < batch:
            self.sm_post = List[Scalar[DT]](
                length=batch * Self.GROUP, fill=Scalar[DT](0.0)
            )
            self.sm_prior = List[Scalar[DT]](
                length=batch * Self.GROUP, fill=Scalar[DT](0.0)
            )
            self.p_post = List[Scalar[DT]](
                length=batch * Self.GROUP, fill=Scalar[DT](0.0)
            )
            self.p_prior = List[Scalar[DT]](
                length=batch * Self.GROUP, fill=Scalar[DT](0.0)
            )
            self.active = List[Scalar[DT]](length=batch, fill=Scalar[DT](0.0))
            self.cache_n = batch

    @staticmethod
    def _softmax_mix(
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],
        base: Int,
        u: Scalar[DT],
        mut sm_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mut p_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """softmax + unimix mix for one (b,s) group of CLASSES lanes."""
        var zmax = z[base]
        for c in range(1, Self.CLASSES):
            if z[base + c] > zmax:
                zmax = z[base + c]
        var ssum: Scalar[DT] = 0.0
        for c in range(Self.CLASSES):
            var e = exp(z[base + c] - zmax)
            sm_out[base + c] = e
            ssum += e
        var inv = Scalar[DT](1.0) / ssum
        var uni = u / Scalar[DT](Self.CLASSES)
        var one_m_u = Scalar[DT](1.0) - u
        for c in range(Self.CLASSES):
            var sm = sm_out[base + c] * inv
            sm_out[base + c] = sm
            p_out[base + c] = one_m_u * sm + uni

    def forward[
        BATCH: Int
    ](
        mut self,
        post_logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
        prior_logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mut dyn_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mut rep_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        self._ensure_cache(BATCH)
        var smpo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.sm_post.unsafe_ptr()
        )
        var smpr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.sm_prior.unsafe_ptr()
        )
        var ppo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.p_post.unsafe_ptr()
        )
        var ppr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.p_prior.unsafe_ptr()
        )
        for b in range(BATCH):
            var kl_sum: Scalar[DT] = 0.0
            for s in range(Self.STOCH):
                var base = (b * Self.STOCH + s) * Self.CLASSES
                Self._softmax_mix(post_logits, base, self.unimix, smpo, ppo)
                Self._softmax_mix(prior_logits, base, self.unimix, smpr, ppr)
                for c in range(Self.CLASSES):
                    var pp = ppo[base + c]
                    kl_sum += pp * (log(pp) - log(ppr[base + c]))
            var clamped = kl_sum if kl_sum > self.free_nats else self.free_nats
            dyn_out[b] = clamped
            rep_out[b] = clamped
            self.active[b] = (
                Scalar[DT](1.0) if kl_sum > self.free_nats else Scalar[DT](0.0)
            )

    def backward[
        BATCH: Int
    ](
        mut self,
        d_dyn: UnsafePointer[Scalar[DT], MutAnyOrigin],
        d_rep: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mut grad_post: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mut grad_prior: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var smpo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.sm_post.unsafe_ptr()
        )
        var smpr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.sm_prior.unsafe_ptr()
        )
        var ppo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.p_post.unsafe_ptr()
        )
        var ppr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.p_prior.unsafe_ptr()
        )
        var one_m_u = Scalar[DT](1.0) - self.unimix
        for b in range(BATCH):
            var act = self.active[b]
            var wd = d_dyn[b] * act
            var wr = d_rep[b] * act
            for s in range(Self.STOCH):
                var base = (b * Self.STOCH + s) * Self.CLASSES
                # ── dyn → grad_prior ──
                # S = Σ_c (p_post[c]/p_prior[c])·sm_prior[c]
                var S: Scalar[DT] = 0.0
                for c in range(Self.CLASSES):
                    S += (ppo[base + c] / ppr[base + c]) * smpr[base + c]
                # ── rep → grad_post ──
                # W = Σ_c sm_post[c]·(log p_post[c] − log p_prior[c])
                var W: Scalar[DT] = 0.0
                for c in range(Self.CLASSES):
                    W += smpo[base + c] * (
                        log(ppo[base + c]) - log(ppr[base + c])
                    )
                for j in range(Self.CLASSES):
                    # prior grad (dyn term).
                    var a_over_p = ppo[base + j] / ppr[base + j]
                    var gpr = -one_m_u * smpr[base + j] * (a_over_p - S)
                    grad_prior[base + j] = wd * gpr
                    # post grad (rep term).
                    var lr_diff = log(ppo[base + j]) - log(ppr[base + j])
                    var gpo = one_m_u * smpo[base + j] * (lr_diff - W)
                    grad_post[base + j] = wr * gpo


# ──────────────────────────────────────────────────────────────────────
# OneHotKLLoss — OneHotKL wrapped as an nn2 graph Module (ARITY=2).
#   inputs: post[B, S·C], prior[B, S·C]   output: [B, 2] = [dyn, rep]
# The asymmetric sg (dyn→prior, rep→post) lives INSIDE the op's vjp; the
# ComputeGraph just routes the two output cotangents back to the two inputs
# and accumulates them with everything else (e.g. at the shared new_deter).
# ──────────────────────────────────────────────────────────────────────


struct OneHotKLLoss[STOCH: Int, CLASSES: Int](Module):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.SC)
    comptime OUT_DIM = 2

    @staticmethod
    def display_label() -> String:
        return String("OneHotKL")

    var kl: OneHotKL[Self.STOCH, Self.CLASSES]
    # GPU caches: softmax/mixed probs [B·SC], active [B], per-group KL [B·STOCH]
    var _smpo: Optional[DeviceBuffer[DT]]
    var _smpr: Optional[DeviceBuffer[DT]]
    var _ppo: Optional[DeviceBuffer[DT]]
    var _ppr: Optional[DeviceBuffer[DT]]
    var _act: Optional[DeviceBuffer[DT]]
    var _klpart: Optional[DeviceBuffer[DT]]
    var _dev_n: Int
    var ts: TargetStorage

    def __init__(out self):
        self.kl = OneHotKL[Self.STOCH, Self.CLASSES]()
        self._smpo = None
        self._smpr = None
        self._ppo = None
        self._ppr = None
        self._act = None
        self._klpart = None
        self._dev_n = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "OneHotKLLoss: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.kl = OneHotKL[Self.STOCH, Self.CLASSES].make(
            Scalar[DT](0.01), Scalar[DT](1.0)
        )
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("OneHotKLLoss.make[gpu]: ctx required")
            m.ts = TargetStorage.make_gpu(ctx.value())
        return m^

    def _ensure_dev(mut self, nn: Int, ng: Int, b: Int) raises:
        if self._dev_n < nn:
            var ctx = self.ts.ctx.value()
            self._smpo = ctx.enqueue_create_buffer[DT](nn)
            self._smpr = ctx.enqueue_create_buffer[DT](nn)
            self._ppo = ctx.enqueue_create_buffer[DT](nn)
            self._ppr = ctx.enqueue_create_buffer[DT](nn)
            self._act = ctx.enqueue_create_buffer[DT](b)
            self._klpart = ctx.enqueue_create_buffer[DT](ng)
            self._dev_n = nn

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["OneHotKLLoss", target](self.ts.target_tag)
        var post = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            inputs.tile[0, BATCH, Self.SC]().ptr
        )
        var prior = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            inputs.tile[1, BATCH, Self.SC]().ptr
        )
        var o = typed_view_mut[BATCH, 2](output).ptr
        comptime if target == "cpu":
            var dyn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
            var rep: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
            self.kl.forward[BATCH](post, prior, dyn, rep)
            for b in range(BATCH):
                o[b * 2] = dyn[b]
                o[b * 2 + 1] = rep[b]
            dyn.free(); rep.free()
        else:
            comptime NN = BATCH * Self.SC
            comptime NG = BATCH * Self.STOCH
            self._ensure_dev(NN, NG, BATCH)
            var ctx = self.ts.ctx.value()
            var smpo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._smpo.value().unsafe_ptr())
            var smpr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._smpr.value().unsafe_ptr())
            var ppo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._ppo.value().unsafe_ptr())
            var ppr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._ppr.value().unsafe_ptr())
            var actp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._act.value().unsafe_ptr())
            var klp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._klpart.value().unsafe_ptr())
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            comptime nb1 = (NG + TPB - 1) // TPB
            comptime k1 = _kl_fwd1_kernel[NG, Self.CLASSES]
            ctx.enqueue_function[k1](
                _dlt[NN](post), _dlt[NN](prior), _dlt[NN](smpo), _dlt[NN](ppo),
                _dlt[NN](smpr), _dlt[NN](ppr), _dlt[NG](klp), self.kl.unimix,
                grid_dim=nb1, block_dim=TPB,
            )
            comptime nb2 = (BATCH + TPB - 1) // TPB
            comptime k2 = _kl_fwd2_kernel[BATCH, Self.STOCH]
            ctx.enqueue_function[k2](
                _dlt[NG](klp), _dlt[BATCH * 2](op), _dlt[BATCH](actp),
                self.kl.free_nats, grid_dim=nb2, block_dim=TPB,
            )

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        var go = typed_view[BATCH, 2](grad_output).ptr
        var g_post = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_inputs.tile[0, BATCH, Self.SC]().ptr
        )
        var g_prior = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_inputs.tile[1, BATCH, Self.SC]().ptr
        )
        comptime if target == "cpu":
            var d_dyn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
            var d_rep: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
            for b in range(BATCH):
                d_dyn[b] = go[b * 2]
                d_rep[b] = go[b * 2 + 1]
            self.kl.backward[BATCH](d_dyn, d_rep, g_post, g_prior)
            d_dyn.free(); d_rep.free()
        else:
            comptime NN = BATCH * Self.SC
            comptime NG = BATCH * Self.STOCH
            var ctx = self.ts.ctx.value()
            var smpo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._smpo.value().unsafe_ptr())
            var smpr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._smpr.value().unsafe_ptr())
            var ppo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._ppo.value().unsafe_ptr())
            var ppr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._ppr.value().unsafe_ptr())
            var actp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self._act.value().unsafe_ptr())
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
            var one_m_u = Scalar[DT](1.0) - self.kl.unimix
            comptime nb = (NG + TPB - 1) // TPB
            comptime kb = _kl_bwd_kernel[NG, Self.CLASSES, Self.STOCH]
            ctx.enqueue_function[kb](
                _dlt[BATCH * 2](gop), _dlt[BATCH](actp),
                _dlt[NN](smpo), _dlt[NN](ppo), _dlt[NN](smpr), _dlt[NN](ppr),
                _dlt[NN](g_post), _dlt[NN](g_prior), one_m_u,
                grid_dim=nb, block_dim=TPB,
            )
