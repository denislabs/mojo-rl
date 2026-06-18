"""Trainer — minimal supervised classification trainer (storage surface).

Holds a `MODEL` (Module producing logits [BATCH, NC]), an `Adam`, and a
`CrossEntropyLoss[NC]`. `train_epoch` runs the batch loop (forward → CE loss
accumulate → CE vjp → model.vjp → opt.step) over a flat dataset; `eval_top1`
runs forward + argmax accuracy. CPU + GPU (per-batch upload on GPU).

Flat-dataset convention (the example builds these): `train_x` = [N·IN] row-major
images, `train_y` = [N·NC] one-hot labels; `test_x` = [N·IN], `test_labels` =
[N] integer class ids (as DT). Topology must match what was trained.
"""

from std.gpu.host import DeviceContext
from std.memory import memcpy

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from mojo_rl.nn.core.initializer import Initializer
from ..optimizer.adam import Adam
from ..loss.cross_entropy import CrossEntropyLoss


struct Trainer[
    MODEL: Module, NC: Int, IN: Int, BATCH: Int, target: StaticString
](Movable & ImplicitlyDeletable):
    var model: Self.MODEL
    var opt: Adam
    var loss: CrossEntropyLoss[Self.NC]
    var batch_x: Tensor
    var batch_y: Tensor
    var logits: Tensor
    var grad: Tensor
    var gi: Tensor

    def __init__(out self):
        self.model = Self.MODEL()
        self.opt = Adam()
        self.loss = CrossEntropyLoss[Self.NC]()
        self.batch_x = Tensor()
        self.batch_y = Tensor()
        self.logits = Tensor()
        self.grad = Tensor()
        self.gi = Tensor()

    @staticmethod
    def make[
        INIT: Initializer
    ](ctx: Optional[DeviceContext] = None, lr: Scalar[DT] = 1e-3) raises -> Self:
        """`ctx` required on GPU (the caller owns it and reuses it for
        train/eval so all device buffers share one context)."""
        var t = Self()
        comptime if Self.target == "cpu":
            t.model = Self.MODEL.make_cpu()
            t.loss = CrossEntropyLoss[Self.NC].make_cpu()
            t.model.reinit["cpu", INIT](None)
        else:
            var c = ctx.value()
            t.model = Self.MODEL.make_gpu(c)
            t.loss = CrossEntropyLoss[Self.NC].make_gpu(c)
            t.model.reinit["gpu", INIT](ctx)
        t.opt = Adam(lr=lr)
        t.batch_x = Tensor.alloc(Self.BATCH * Self.IN)
        t.batch_y = Tensor.alloc(Self.BATCH * Self.NC)
        return t^

    def train_epoch[
        N_TRAIN: Int
    ](
        mut self,
        ref train_x: List[Scalar[DT]],
        ref train_y: List[Scalar[DT]],
        ctx: Optional[DeviceContext],
    ) raises -> Scalar[DT]:
        comptime n_batches = N_TRAIN // Self.BATCH
        self.loss.reset_accum[Self.target]()
        for nb in range(n_batches):
            var x0 = nb * Self.BATCH * Self.IN
            var y0 = nb * Self.BATCH * Self.NC
            memcpy(dest=self.batch_x.data.unsafe_ptr(), src=train_x.unsafe_ptr() + x0, count=Self.BATCH * Self.IN)
            memcpy(dest=self.batch_y.data.unsafe_ptr(), src=train_y.unsafe_ptr() + y0, count=Self.BATCH * Self.NC)
            comptime if Self.target == "gpu":
                self.batch_x.upload(ctx.value())
                self.batch_y.upload(ctx.value())
            self.model.zero_grad[Self.target](ctx)
            self.model.forward[Self.target, Self.BATCH](
                TensorRefs[Self.MODEL.ARITY].of1(self.batch_x), self.logits, ctx
            )
            self.loss.forward_accumulate[Self.target, Self.BATCH](
                self.logits, self.batch_y, ctx
            )
            self.loss.vjp[Self.target, Self.BATCH](
                self.logits, self.batch_y, self.grad, ctx
            )
            self.model.vjp[Self.target, Self.BATCH](
                TensorRefs[Self.MODEL.ARITY].of1(self.batch_x), self.grad,
                TensorRefs[Self.MODEL.ARITY].of1(self.gi), ctx,
            )
            self.opt.step[Self.target](self.model, ctx)
        return self.loss.read_accum[Self.target](ctx)

    def eval_top1[
        N_TEST: Int
    ](
        mut self,
        ref test_x: List[Scalar[DT]],
        ref test_labels: List[Int32],
        ctx: Optional[DeviceContext],
    ) raises -> Float64:
        comptime n_batches = N_TEST // Self.BATCH
        var correct = 0
        for nb in range(n_batches):
            var x0 = nb * Self.BATCH * Self.IN
            memcpy(dest=self.batch_x.data.unsafe_ptr(), src=test_x.unsafe_ptr() + x0, count=Self.BATCH * Self.IN)
            comptime if Self.target == "gpu":
                self.batch_x.upload(ctx.value())
            self.model.forward[Self.target, Self.BATCH](
                TensorRefs[Self.MODEL.ARITY].of1(self.batch_x), self.logits, ctx
            )
            comptime if Self.target == "gpu":
                self.logits.download(ctx.value())
            for b in range(Self.BATCH):
                var best = 0
                var bestv = self.logits.data[b * Self.NC]
                for c in range(1, Self.NC):
                    var v = self.logits.data[b * Self.NC + c]
                    if v > bestv:
                        bestv = v
                        best = c
                if best == Int(test_labels[nb * Self.BATCH + b]):
                    correct += 1
        return Float64(correct) / Float64(n_batches * Self.BATCH)
