"""LeWMDecoderTrainer — trains the reconstruction probe on frozen `emb`.

Owns `LeWMDecoderLossGraph` + one Adam (graph overloads) + IO scratch.
`train_step(emb, tgt)` runs zero_grad → set_input → forward → mean-reduce →
seed 1/B → vjp → Adam.step on the decoder params only (the encoder is
external and frozen — `emb` is fed as data). `recon_into` reads the `recon`
node (patch space) for visualization. Mirrors `LeWMTrainer`'s GPU discipline
(device loss accumulator drained at flush; constant grad seed written once).

Parameterized by the decoder dims + the per-frame BATCH (= B·T frames) +
train_target. The caller derives N_Q / PATCH_PX from (C, IMG, PATCH_D).
"""

from std.memory import alloc
from std.gpu import thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ...nn2.constants import DT, TPB_REDUCE
from ...nn2.core import ParamVisitor
from ...nn2.core.amp import AMPPolicy, NoAMP
from ...nn2.core.scratch import Scratch
from ...nn2.core.scratch_walkers import init_scratch_auto
from ...nn2.core.target_storage import TargetStorage, assert_tag_for
from ...nn2.initializer import Kaiming
from ...nn2.optimizer.adam import Adam
from .decoder import LeWMDecoderLossGraph

from mojo_rl.deep_agents2.loss.seed_grad_inv_batch import seed_grad_inv_batch


def _dec_reduce_mean_acc_kernel[BATCH: Int](
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


struct LeWMDecoderTrainer[
    EMB: Int, HID: Int, N_Q: Int, PATCH_PX: Int, FF: Int, N_LAYERS: Int,
    BATCH: Int, train_target: StaticString = "cpu",
](Movable & ImplicitlyDestructible):
    comptime DG = LeWMDecoderLossGraph[
        Self.EMB, Self.HID, Self.N_Q, Self.PATCH_PX, Self.FF, Self.N_LAYERS
    ]
    comptime RECON = Self.N_Q * Self.PATCH_PX

    var graph: Self.DG
    var opt: Adam
    var _loss_out: Scratch["dec_loss_out", Self.BATCH]
    var _grad_seed: Scratch["dec_grad_seed", Self.BATCH]
    var _loss_acc_dev: Optional[DeviceBuffer[DT]]
    var loss_host: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.DG()
        self.opt = Adam()
        self._loss_out = Scratch["dec_loss_out", Self.BATCH]()
        self._grad_seed = Scratch["dec_grad_seed", Self.BATCH]()
        self._loss_acc_dev = None
        self.loss_host = alloc[Scalar[DT]](Self.BATCH)
        self.ts = TargetStorage.make_uninit()

    def __del__(deinit self):
        self.loss_host.free()

    @staticmethod
    def make(
        lr: Scalar[DT] = 1e-3,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var t = Self()
        t.graph = Self.DG.make[target = Self.train_target, INIT=Kaiming](
            ctx=ctx
        )
        t.opt = Adam.make_graph[Self.train_target](t.graph, ctx=ctx)
        t.opt.lr = lr
        t.ts = TargetStorage.make[Self.train_target](ctx=ctx)
        init_scratch_auto[Self, Self.train_target](t, ctx)
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
        emb: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        tgt: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises -> Scalar[DT]:
        assert_tag_for["LeWMDecoderTrainer", Self.train_target](
            self.ts.target_tag
        )
        self.opt.zero_grad_graph[Self.train_target](self.graph)
        self.graph.set_input["emb", Self.BATCH](emb)
        self.graph.set_input["tgt", Self.BATCH](tgt)

        var loss_p = self._loss_out.target_ptr[Self.train_target]()
        var loss_t = TileTensor(loss_p, row_major[Self.BATCH, 1]())
        self.graph.forward[Self.train_target, Self.BATCH, POLICY](loss_t)

        var m: Scalar[DT] = 0.0
        comptime if Self.train_target == "cpu":
            for b in range(Self.BATCH):
                m += loss_p[b]
            m /= Scalar[DT](Self.BATCH)
        else:
            var ctx = self.ts.ctx.value()
            comptime red = _dec_reduce_mean_acc_kernel[Self.BATCH]
            ctx.enqueue_function[red](
                loss_p, self._loss_acc_dev.value().unsafe_ptr(),
                grid_dim=1, block_dim=TPB_REDUCE,
            )

        var gseed_p = self._grad_seed.target_ptr[Self.train_target]()
        var gseed_t = TileTensor(gseed_p, row_major[Self.BATCH, 1]())
        self.graph.vjp[Self.train_target, Self.BATCH, POLICY](gseed_t)
        self.opt.step_graph[Self.train_target](self.graph)
        return m

    def recon_into[
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        emb: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        recon_host: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Forward-only readout of the `recon` node (B, N_Q·PATCH_PX, patch
        space) into a host buffer. Caller un-patchifies for display."""
        assert_tag_for["LeWMDecoderTrainer", Self.train_target](
            self.ts.target_tag
        )
        # tgt input must be set for the graph forward; feed zeros (the loss
        # node is computed but ignored — we only read `recon`).
        self.graph.set_input["emb", Self.BATCH](emb)
        # reuse the emb-driven forward; tgt left as last value is fine since
        # we don't read loss. But set it to a valid buffer to be safe: the
        # recon node does not depend on tgt.
        var loss_p = self._loss_out.target_ptr[Self.train_target]()
        var loss_t = TileTensor(loss_p, row_major[Self.BATCH, 1]())
        self.graph.forward[Self.train_target, Self.BATCH, POLICY](loss_t)
        var src = self.graph.node_out_ptr["recon"]()
        comptime N = Self.BATCH * Self.RECON
        comptime if Self.train_target == "cpu":
            for i in range(N):
                recon_host[i] = src[i]
        else:
            var ctx = self.ts.ctx.value()
            var dev = DeviceBuffer[DT](ctx, src, N, owning=False)
            ctx.enqueue_copy(recon_host, dev)
            ctx.synchronize()

    def read_loss_accum(mut self) raises -> Scalar[DT]:
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

    def reset_loss_accum(mut self) raises:
        comptime if Self.train_target == "gpu":
            self._loss_acc_dev.value().enqueue_fill(0.0)

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
