"""LeWMTrainer — config-driven offline JEPA trainer over LeWMLossGraph.

Owns the loss graph + one Adam (graph overloads) + scratch. `train_step`
runs zero_grad → set_input → forward → mean-reduce → seed 1/B → vjp →
Adam.step, returning the batch-mean loss. `collapse_probes` computes the
representation-collapse diagnostics off the `emb` node (var_min over latent
dims, mean |off-diagonal correlation|). `save_params` / `load_params`
persist the graph's parameters (Adam state not persisted — eval/MPC only
needs weights; resume is a follow-up).

Parameterized directly by dims + BATCH + train_target; presets are type
aliases.

GPU path (Phase E). The graph IO buffers (loss output [BATCH,1], grad seed
[BATCH,1]) are `Scratch` fields → device buffers on GPU, host lists on CPU.
The backward seed is the constant `1/BATCH`, written once at `make` (no
per-step host work — the SAC capturability discipline). On GPU `train_step`
device-reduces the per-sample loss into a `(Σmean, count)` accumulator and
returns a `0` sentinel; the driver drains it at flush cadence via
`read_loss_accum` / `reset_loss_accum`. `collapse_probes` D2H-copies the
`emb` node once (a diagnostic, off the capture hot loop). Checkpoint
visitors wrap each param's device pointer in a non-owning `DeviceBuffer`
for D2H (save) / H2D (load).
"""

from std.collections import Dict
from std.memory import alloc
from std.gpu import thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major
from std.math import sqrt

from ...nn2.constants import DT, TPB_REDUCE
from ...nn2.core import ParamVisitor
from ...nn2.core.amp import AMPPolicy, NoAMP
from ...nn2.core.scratch import Scratch
from ...nn2.core.scratch_walkers import init_scratch_auto
from ...nn2.core.target_storage import TargetStorage, assert_tag_for
from ...nn2.initializer import Kaiming
from ...nn2.optimizer.adam import Adam
from .loss_graph import LeWMLossGraph

from mojo_rl.deep_agents2.loss.seed_grad_inv_batch import seed_grad_inv_batch


# ── GPU loss reduction: acc[0] += mean(src); acc[1] += 1 ───────────────
# Single-block `block.sum` over the [BATCH] per-sample loss. No per-step
# D2H — the driver drains the accumulator at flush cadence.
def _reduce_mean_acc_kernel[BATCH: Int](
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    acc: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        my_sum += src[k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0] = acc[0] + total[0] / Scalar[DT](BATCH)
        acc[1] = acc[1] + Scalar[DT](1.0)


# ── checkpoint visitors (params only, in for_each_param order) ─────────
# `ctx` is None on CPU (direct host deref) and Some on GPU (wrap each
# param's device pointer in a non-owning DeviceBuffer for the transfer).
struct _SaveVisitor(ParamVisitor):
    var vals: List[Scalar[DT]]
    var ctx: Optional[DeviceContext]

    def __init__(out self, ctx: Optional[DeviceContext] = None):
        self.vals = List[Scalar[DT]]()
        self.ctx = ctx

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        if self.ctx:
            var c = self.ctx.value()
            var dev = DeviceBuffer[DT](c, p, n_elems, owning=False)
            var host = List[Scalar[DT]](length=n_elems, fill=Scalar[DT](0.0))
            c.enqueue_copy(host.unsafe_ptr(), dev)
            c.synchronize()
            for i in range(n_elems):
                self.vals.append(host[i])
        else:
            for i in range(n_elems):
                self.vals.append(p[i])


struct _LoadVisitor(ParamVisitor):
    var vals: List[Scalar[DT]]
    var idx: Int
    var ctx: Optional[DeviceContext]

    def __init__(
        out self, var vals: List[Scalar[DT]],
        ctx: Optional[DeviceContext] = None,
    ):
        self.vals = vals^
        self.idx = 0
        self.ctx = ctx

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        if self.ctx:
            var c = self.ctx.value()
            var host = List[Scalar[DT]](length=n_elems, fill=Scalar[DT](0.0))
            for i in range(n_elems):
                host[i] = self.vals[self.idx]
                self.idx += 1
            var dev = DeviceBuffer[DT](c, p, n_elems, owning=False)
            c.enqueue_copy(dev, host.unsafe_ptr())
            c.synchronize()
        else:
            for i in range(n_elems):
                p[i] = self.vals[self.idx]
                self.idx += 1


# Named export: record each param's name → values (CPU direct / GPU D2H).
# Feeds LeWMPredictor.sync_from_named for the MPC predictor-from-latents path.
struct _NamedExportVisitor(ParamVisitor):
    var d: UnsafePointer[Dict[String, List[Scalar[DT]]], MutAnyOrigin]
    var ctx: Optional[DeviceContext]

    def __init__(
        out self,
        d: UnsafePointer[Dict[String, List[Scalar[DT]]], MutAnyOrigin],
        ctx: Optional[DeviceContext] = None,
    ):
        self.d = d
        self.ctx = ctx

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var vals = List[Scalar[DT]](length=n_elems, fill=Scalar[DT](0.0))
        if self.ctx:
            var c = self.ctx.value()
            var dev = DeviceBuffer[DT](c, p, n_elems, owning=False)
            c.enqueue_copy(vals.unsafe_ptr(), dev)
            c.synchronize()
        else:
            for i in range(n_elems):
                vals[i] = p[i]
        self.d[][name] = vals^


struct LeWMTrainer[
    IN_CH: Int, IMG: Int, PATCH: Int, HIDDEN: Int, ENC_HEADS: Int,
    ENC_LAYERS: Int, EMB: Int, ENC_PROJ_H: Int, ENC_FF_MULT: Int,
    T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, N_PREDS: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
    PRED_PROJ_H: Int, SIG_PROJ: Int, SIG_KNOTS: Int,
    BATCH: Int, train_target: StaticString = "cpu",
    PRED_DIM_HEAD: Int = 0,
](Movable & ImplicitlyDestructible):
    # PRED_DIM_HEAD 0 ⇒ standard EMB/PRED_HEADS attention; >0 ⇒ the paper's
    # expanded predictor attention (e.g. 16 heads × 64 = 1024 inner > EMB).
    # Added last (after train_target) so existing positional call sites are
    # unchanged. Bit-identical at the default 0.
    comptime LG = LeWMLossGraph[
        Self.IN_CH, Self.IMG, Self.PATCH, Self.HIDDEN, Self.ENC_HEADS,
        Self.ENC_LAYERS, Self.EMB, Self.ENC_PROJ_H, Self.ENC_FF_MULT,
        Self.T, Self.ACT, Self.SMOOTHED, Self.AE_MLP,
        Self.H, Self.N_PREDS, Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
        Self.PRED_PROJ_H, Self.SIG_PROJ, Self.SIG_KNOTS, Self.PRED_DIM_HEAD,
    ]
    comptime PIX = Self.T * Self.IN_CH * Self.IMG * Self.IMG
    comptime ACTIN = Self.T * Self.ACT
    comptime TE = Self.T * Self.EMB

    var graph: Self.LG
    var opt: Adam
    # Graph IO scratch: device buffer on GPU, host list on CPU.
    var _loss_out: Scratch["lewm_loss_out", Self.BATCH]
    var _grad_seed: Scratch["lewm_grad_seed", Self.BATCH]
    # GPU-only `(Σmean, count)` loss accumulator (drained at flush).
    var _loss_acc_dev: Optional[DeviceBuffer[DT]]
    # Host staging: probe emb (B·TE) + eval-loss D2H (BATCH). Always host.
    var emb_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var loss_host: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.LG()
        self.opt = Adam()
        self._loss_out = Scratch["lewm_loss_out", Self.BATCH]()
        self._grad_seed = Scratch["lewm_grad_seed", Self.BATCH]()
        self._loss_acc_dev = None
        self.emb_buf = alloc[Scalar[DT]](Self.BATCH * Self.TE)
        self.loss_host = alloc[Scalar[DT]](Self.BATCH)
        self.ts = TargetStorage.make_uninit()

    def __del__(deinit self):
        self.emb_buf.free()
        self.loss_host.free()

    @staticmethod
    def make(
        lam: Scalar[DT] = 0.09,
        lr: Scalar[DT] = 1e-3,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var t = Self()
        t.graph = Self.LG.make[target = Self.train_target, INIT=Kaiming](
            ctx=ctx
        )
        t.graph.set_node_attr["sig_s", "multiplier"](lam)
        t.opt = Adam.make_graph[Self.train_target](t.graph, ctx=ctx)
        t.opt.lr = lr
        t.ts = TargetStorage.make[Self.train_target](ctx=ctx)
        init_scratch_auto[Self, Self.train_target](t, ctx)
        # The backward seed for a mean-over-batch loss is the constant
        # 1/BATCH in every slot — write it once (nothing in the step
        # mutates it, so this stays out of the per-step / capturable path).
        seed_grad_inv_batch[Self.train_target, Self.BATCH](
            t._grad_seed.target_ptr[Self.train_target](), ctx=ctx
        )
        comptime if Self.train_target == "gpu":
            var c = ctx.value()
            var acc = c.enqueue_create_buffer[DT](2)
            acc.enqueue_fill(0.0)
            t._loss_acc_dev = acc^
        return t^

    def train_step[
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        pix: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        act: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises -> Scalar[DT]:
        assert_tag_for["LeWMTrainer", Self.train_target](self.ts.target_tag)
        self.opt.zero_grad_graph[Self.train_target](self.graph)
        self.graph.set_input["pixels", Self.BATCH](pix)
        self.graph.set_input["actions", Self.BATCH](act)

        var loss_p = self._loss_out.target_ptr[Self.train_target]()
        var loss_t = TileTensor(loss_p, row_major[Self.BATCH, 1]())
        self.graph.forward[Self.train_target, Self.BATCH, POLICY](loss_t)

        var m: Scalar[DT] = 0.0
        comptime if Self.train_target == "cpu":
            for b in range(Self.BATCH):
                m += loss_p[b]
            m /= Scalar[DT](Self.BATCH)
        else:
            # Device reduce-accumulate; the driver drains `_loss_acc_dev`
            # at flush cadence. `m` stays a 0 sentinel (no per-step D2H).
            var ctx = self.ts.ctx.value()
            comptime red = _reduce_mean_acc_kernel[Self.BATCH]
            ctx.enqueue_function[red](
                loss_p, self._loss_acc_dev.value().unsafe_ptr(),
                grid_dim=1, block_dim=TPB_REDUCE,
            )

        var gseed_p = self._grad_seed.target_ptr[Self.train_target]()
        var gseed_t = TileTensor(gseed_p, row_major[Self.BATCH, 1]())
        self.graph.vjp[Self.train_target, Self.BATCH, POLICY](gseed_t)
        self.opt.step_graph[Self.train_target](self.graph)
        return m

    def eval_loss[
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        pix: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        act: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises -> Scalar[DT]:
        """Forward-only batch-mean loss (no grad / no optimizer step)."""
        assert_tag_for["LeWMTrainer", Self.train_target](self.ts.target_tag)
        self.graph.set_input["pixels", Self.BATCH](pix)
        self.graph.set_input["actions", Self.BATCH](act)
        var loss_p = self._loss_out.target_ptr[Self.train_target]()
        var loss_t = TileTensor(loss_p, row_major[Self.BATCH, 1]())
        self.graph.forward[Self.train_target, Self.BATCH, POLICY](loss_t)

        var m: Scalar[DT] = 0.0
        comptime if Self.train_target == "cpu":
            for b in range(Self.BATCH):
                m += loss_p[b]
        else:
            # eval is off the capture hot loop — D2H the [BATCH] vector once.
            var ctx = self.ts.ctx.value()
            var dev = DeviceBuffer[DT](ctx, loss_p, Self.BATCH, owning=False)
            ctx.enqueue_copy(self.loss_host, dev)
            ctx.synchronize()
            for b in range(Self.BATCH):
                m += self.loss_host[b]
        return m / Scalar[DT](Self.BATCH)

    def forward_into[
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        pix: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        act: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        pred_host: UnsafePointer[Scalar[DT], MutAnyOrigin],
        tgt_host: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Forward-only readout for eval/planning: run the graph over
        (pix, act) and copy the predicted latents (`pred` node) and the
        encoded target latents (`tgt` node) — both (BATCH, H·EMB) — to the
        caller's host buffers. `tgt` is action-independent (the encoded
        real future), so it's the fixed goal a planner scores against;
        `pred` is the action-conditioned prediction. No grad / no step."""
        assert_tag_for["LeWMTrainer", Self.train_target](self.ts.target_tag)
        comptime HE = Self.H * Self.EMB
        self.graph.set_input["pixels", Self.BATCH](pix)
        self.graph.set_input["actions", Self.BATCH](act)
        var loss_p = self._loss_out.target_ptr[Self.train_target]()
        var loss_t = TileTensor(loss_p, row_major[Self.BATCH, 1]())
        self.graph.forward[Self.train_target, Self.BATCH, POLICY](loss_t)
        var pred_src = self.graph.node_out_ptr["pred"]()
        var tgt_src = self.graph.node_out_ptr["tgt"]()
        comptime N = Self.BATCH * HE
        comptime if Self.train_target == "cpu":
            for i in range(N):
                pred_host[i] = pred_src[i]
                tgt_host[i] = tgt_src[i]
        else:
            var ctx = self.ts.ctx.value()
            var pred_dev = DeviceBuffer[DT](ctx, pred_src, N, owning=False)
            var tgt_dev = DeviceBuffer[DT](ctx, tgt_src, N, owning=False)
            ctx.enqueue_copy(pred_host, pred_dev)
            ctx.enqueue_copy(tgt_host, tgt_dev)
            ctx.synchronize()

    def read_node_into[
        name: StaticString,
    ](
        mut self,
        host: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n: Int,
    ) raises:
        """Copy the named graph node's output (n elements) to a host buffer.
        Valid after a forward (`forward_into`/`train_step`/`eval_loss`) has
        populated the node buffers. Used by the MPC path to read `emb`."""
        var src = self.graph.node_out_ptr[name]()
        comptime if Self.train_target == "cpu":
            for i in range(n):
                host[i] = src[i]
        else:
            var ctx = self.ts.ctx.value()
            var dev = DeviceBuffer[DT](ctx, src, n, owning=False)
            ctx.enqueue_copy(host, dev)
            ctx.synchronize()

    def export_named_params(mut self) raises -> Dict[String, List[Scalar[DT]]]:
        """Snapshot all graph params as a name→values dict (CPU/GPU). Feeds
        `LeWMPredictor.sync_from_named` so the MPC predictor shares the
        trained encoder-free weights (matched by name)."""
        var d = Dict[String, List[Scalar[DT]]]()
        var v = _NamedExportVisitor(UnsafePointer(to=d), ctx=self.ts.ctx)
        self.graph.for_each_param[Self.train_target, _NamedExportVisitor]("", v)
        _ = v^
        return d^

    def reset_loss_accum(mut self) raises:
        """Zero the device `(Σmean, count)` loss accumulator (GPU, flush)."""
        comptime if Self.train_target == "gpu":
            self._loss_acc_dev.value().enqueue_fill(0.0)

    def read_loss_accum(mut self) raises -> Scalar[DT]:
        """D2H the device loss accumulator once and return the window mean
        (Σmean / count). 0 if no steps. GPU only — CPU `train_step` already
        returns the per-step loss."""
        comptime if Self.train_target == "gpu":
            var ctx = self.ts.ctx.value()
            var h = ctx.enqueue_create_host_buffer[DT](2)
            ctx.enqueue_copy(h, self._loss_acc_dev.value())
            ctx.synchronize()
            var s = h.unsafe_ptr()[0]
            var n = h.unsafe_ptr()[1]
            if n == Scalar[DT](0.0):
                return Scalar[DT](0.0)
            return s / n
        else:
            return Scalar[DT](0.0)

    def collapse_probes(mut self) raises -> Tuple[Scalar[DT], Scalar[DT]]:
        """(var_min, gram_off) over the last forward's `emb`, viewed as
        BATCH·T samples of EMB latent dims. Healthy: var_min > 0.1,
        gram_off < 0.5 (legacy thresholds)."""
        var emb_src = self.graph.node_out_ptr["emb"]()
        comptime ns = Self.BATCH * Self.T
        comptime D = Self.EMB
        comptime if Self.train_target == "cpu":
            for i in range(ns * D):
                self.emb_buf[i] = emb_src[i]
        else:
            # D2H the emb node output once (diagnostic, off the hot loop).
            var ctx = self.ts.ctx.value()
            var dev = DeviceBuffer[DT](ctx, emb_src, ns * D, owning=False)
            ctx.enqueue_copy(self.emb_buf, dev)
            ctx.synchronize()

        # per-dim mean + variance
        var mean = alloc[Scalar[DT]](D)
        var std = alloc[Scalar[DT]](D)
        var var_min = Scalar[DT](1e30)
        for d in range(D):
            var s: Scalar[DT] = 0.0
            for r in range(ns):
                s += self.emb_buf[r * D + d]
            var mu = s / Scalar[DT](ns)
            mean[d] = mu
            var v: Scalar[DT] = 0.0
            for r in range(ns):
                var df = self.emb_buf[r * D + d] - mu
                v += df * df
            v /= Scalar[DT](ns)
            std[d] = sqrt(v + Scalar[DT](1e-8))
            if v < var_min:
                var_min = v

        # mean |off-diagonal correlation|
        var acc: Scalar[DT] = 0.0
        var cnt: Int = 0
        for i in range(D):
            for j in range(D):
                if i == j:
                    continue
                var c: Scalar[DT] = 0.0
                for r in range(ns):
                    c += (
                        (self.emb_buf[r * D + i] - mean[i])
                        * (self.emb_buf[r * D + j] - mean[j])
                    )
                c /= Scalar[DT](ns)
                acc += (c / (std[i] * std[j])).__abs__()
                cnt += 1
        var gram_off = acc / Scalar[DT](cnt)
        mean.free()
        std.free()
        return (var_min, gram_off)

    def save_params(mut self, path: String) raises:
        var v = _SaveVisitor(ctx=self.ts.ctx)
        self.graph.for_each_param[Self.train_target, _SaveVisitor]("", v)
        var s = String()
        s += String(len(v.vals)) + "\n"
        for i in range(len(v.vals)):
            s += String(Float64(v.vals[i])) + "\n"
        with open(path, "w") as f:
            f.write(s)

    def load_params(mut self, path: String) raises:
        var content: String
        with open(path, "r") as f:
            content = f.read()
        var lines = content.split("\n")
        var n = Int(lines[0])
        var vals = List[Scalar[DT]]()
        for i in range(n):
            vals.append(Scalar[DT](Float64(lines[i + 1])))
        var v = _LoadVisitor(vals^, ctx=self.ts.ctx)
        self.graph.for_each_param[Self.train_target, _LoadVisitor]("", v)
