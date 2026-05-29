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
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for


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

    var kl: OneHotKL[Self.STOCH, Self.CLASSES]
    var ts: TargetStorage

    def __init__(out self):
        self.kl = OneHotKL[Self.STOCH, Self.CLASSES]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "OneHotKLLoss: spike CPU-only"
        var m = Self()
        m.kl = OneHotKL[Self.STOCH, Self.CLASSES].make(
            Scalar[DT](0.01), Scalar[DT](1.0)
        )
        m.ts = TargetStorage.make_cpu()
        return m^

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["OneHotKLLoss", target](self.ts.target_tag)
        var post = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, Self.SC](inputs[0]).ptr
        )
        var prior = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, Self.SC](inputs[1]).ptr
        )
        var o = typed_view_mut[BATCH, 2](output).ptr
        var dyn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        var rep: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        self.kl.forward[BATCH](post, prior, dyn, rep)
        for b in range(BATCH):
            o[b * 2] = dyn[b]
            o[b * 2 + 1] = rep[b]
        dyn.free(); rep.free()

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var go = typed_view[BATCH, 2](grad_output).ptr
        var d_dyn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        var d_rep: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        for b in range(BATCH):
            d_dyn[b] = go[b * 2]
            d_rep[b] = go[b * 2 + 1]
        var g_post = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view_mut[BATCH, Self.SC](grad_inputs[0]).ptr
        )
        var g_prior = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view_mut[BATCH, Self.SC](grad_inputs[1]).ptr
        )
        self.kl.backward[BATCH](d_dyn, d_rep, g_post, g_prior)
        d_dyn.free(); d_rep.free()
