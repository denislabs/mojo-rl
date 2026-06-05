"""LeWMTrainer — config-driven offline JEPA trainer over LeWMLossGraph.

Owns the loss graph + one Adam (graph overloads) + scratch. `train_step`
runs zero_grad → set_input → forward → mean-reduce → seed 1/B → vjp →
Adam.step, returning the batch-mean loss. `collapse_probes` computes the
representation-collapse diagnostics off the `emb` node (var_min over latent
dims, mean |off-diagonal correlation|). `save_params` / `load_params`
persist the graph's parameters (Adam state not persisted — eval/MPC only
needs weights; resume is a follow-up).

Parameterized directly by dims + BATCH + train_target; presets are type
aliases. GPU branches mirror CPU (validated at scale on NVIDIA, Phase E).
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major
from std.math import sqrt

from ...nn2.constants import DT
from ...nn2.initializer import Kaiming
from ...nn2.optimizer.adam import Adam
from ...nn2.core import ParamVisitor
from .loss_graph import LeWMLossGraph


# ── checkpoint visitors (params only, in for_each_param order) ─────────
struct _SaveVisitor(ParamVisitor):
    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

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
        for i in range(n_elems):
            self.vals.append(p[i])


struct _LoadVisitor(ParamVisitor):
    var vals: List[Scalar[DT]]
    var idx: Int

    def __init__(out self, var vals: List[Scalar[DT]]):
        self.vals = vals^
        self.idx = 0

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
        for i in range(n_elems):
            p[i] = self.vals[self.idx]
            self.idx += 1


struct LeWMTrainer[
    IN_CH: Int, IMG: Int, PATCH: Int, HIDDEN: Int, ENC_HEADS: Int,
    ENC_LAYERS: Int, EMB: Int, ENC_PROJ_H: Int, ENC_FF_MULT: Int,
    T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, N_PREDS: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
    PRED_PROJ_H: Int, SIG_PROJ: Int, SIG_KNOTS: Int,
    BATCH: Int, train_target: StaticString = "cpu",
](Movable & ImplicitlyDestructible):
    comptime LG = LeWMLossGraph[
        Self.IN_CH, Self.IMG, Self.PATCH, Self.HIDDEN, Self.ENC_HEADS,
        Self.ENC_LAYERS, Self.EMB, Self.ENC_PROJ_H, Self.ENC_FF_MULT,
        Self.T, Self.ACT, Self.SMOOTHED, Self.AE_MLP,
        Self.H, Self.N_PREDS, Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
        Self.PRED_PROJ_H, Self.SIG_PROJ, Self.SIG_KNOTS,
    ]
    comptime PIX = Self.T * Self.IN_CH * Self.IMG * Self.IMG
    comptime ACTIN = Self.T * Self.ACT
    comptime TE = Self.T * Self.EMB

    var graph: Self.LG
    var opt: Adam
    var loss_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var gseed_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var emb_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.graph = Self.LG()
        self.opt = Adam()
        self.loss_buf = alloc[Scalar[DT]](Self.BATCH)
        self.gseed_buf = alloc[Scalar[DT]](Self.BATCH)
        self.emb_buf = alloc[Scalar[DT]](Self.BATCH * Self.TE)

    def __del__(deinit self):
        self.loss_buf.free()
        self.gseed_buf.free()
        self.emb_buf.free()

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
        for b in range(Self.BATCH):
            t.gseed_buf[b] = Scalar[DT](1.0 / Float64(Self.BATCH))
        return t^

    def train_step(
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
        comptime if Self.train_target != "cpu":
            raise Error(
                "LeWMTrainer.train_step: only CPU wired so far (Phase E: GPU)"
            )
        self.opt.zero_grad_graph[Self.train_target](self.graph)
        self.graph.set_input["pixels", Self.BATCH](pix)
        self.graph.set_input["actions", Self.BATCH](act)
        var loss_t = TileTensor(self.loss_buf, row_major[Self.BATCH, 1]())
        self.graph.forward[Self.train_target, Self.BATCH](loss_t)
        var m: Scalar[DT] = 0.0
        for b in range(Self.BATCH):
            m += self.loss_buf[b]
        m /= Scalar[DT](Self.BATCH)
        var gseed_t = TileTensor(self.gseed_buf, row_major[Self.BATCH, 1]())
        self.graph.vjp[Self.train_target, Self.BATCH](gseed_t)
        self.opt.step_graph[Self.train_target](self.graph)
        return m

    def eval_loss(
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
        self.graph.set_input["pixels", Self.BATCH](pix)
        self.graph.set_input["actions", Self.BATCH](act)
        var loss_t = TileTensor(self.loss_buf, row_major[Self.BATCH, 1]())
        self.graph.forward[Self.train_target, Self.BATCH](loss_t)
        var m: Scalar[DT] = 0.0
        for b in range(Self.BATCH):
            m += self.loss_buf[b]
        return m / Scalar[DT](Self.BATCH)

    def collapse_probes(mut self) raises -> Tuple[Scalar[DT], Scalar[DT]]:
        """(var_min, gram_off) over the last forward's `emb`, viewed as
        BATCH·T samples of EMB latent dims. Healthy: var_min > 0.1,
        gram_off < 0.5 (legacy thresholds)."""
        var emb_src = self.graph.node_out_ptr["emb"]()
        comptime ns = Self.BATCH * Self.T
        comptime D = Self.EMB
        for i in range(ns * D):
            self.emb_buf[i] = emb_src[i]

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
        var v = _SaveVisitor()
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
        var v = _LoadVisitor(vals^)
        self.graph.for_each_param[Self.train_target, _LoadVisitor]("", v)
