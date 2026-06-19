"""Trainer — minimal supervised classification trainer (storage surface).

Holds a `MODEL` (Module producing logits [BATCH, NC]), an `Adam`, and a
`CrossEntropyLoss[NC]`. `train_epoch` runs the batch loop (forward → CE loss
accumulate → CE vjp → model.vjp → opt.step) over a flat dataset; `eval_top1`
runs forward + argmax accuracy.

Data path (matches legacy "resident dataset + slice", no per-batch transfer):
- GPU: the whole dataset is uploaded to device ONCE (lazily, on first use), then
  each batch input is a zero-copy `create_sub_buffer` VIEW into that resident
  buffer (no per-batch H2D). The owned `batch_x/batch_y` just carry the view.
- CPU: the host `List` is already resident; each batch is a cheap `memcpy` slice
  (a zero-copy view would need an offset threaded through `forward`, not worth it
  — the copy is a few % of CPU wall time).

Flat-dataset convention (the example builds these): `train_x` = [N·IN] row-major
images, `train_y` = [N·NC] one-hot labels; `test_x` = [N·IN], `test_labels` =
[N] integer class ids. Topology must match what was trained.
"""

from std.gpu.host import DeviceContext
from std.memory import memcpy

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
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
    # GPU-resident dataset (uploaded once; batches are sub-buffer views).
    var ds_x: Tensor
    var ds_y: Tensor
    var ds_tx: Tensor
    var _up_train: Bool
    var _up_test: Bool

    def __init__(out self):
        self.model = Self.MODEL()
        self.opt = Adam()
        self.loss = CrossEntropyLoss[Self.NC]()
        self.batch_x = Tensor()
        self.batch_y = Tensor()
        self.logits = Tensor()
        self.grad = Tensor()
        self.gi = Tensor()
        self.ds_x = Tensor()
        self.ds_y = Tensor()
        self.ds_tx = Tensor()
        self._up_train = False
        self._up_test = False

    @staticmethod
    def make[
        INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None, lr: Scalar[DT] = 1e-3
    ) raises -> Self:
        """`ctx` required on GPU (the caller owns it and reuses it for
        train/eval so all device buffers share one context)."""
        var t = Self()
        # One factory: allocates + initializes the model with INIT at construction.
        t.model = Self.MODEL.make[Self.target, INIT](ctx)
        comptime if Self.target == "cpu":
            t.loss = CrossEntropyLoss[Self.NC].make_cpu()
            t.batch_x = Tensor.alloc(Self.BATCH * Self.IN)
            t.batch_y = Tensor.alloc(Self.BATCH * Self.NC)
        else:
            t.loss = CrossEntropyLoss[Self.NC].make_gpu(ctx.value())
            # batch_x/batch_y carry sub-buffer views on GPU — no owned slab.
        t.opt = Adam(lr=lr)
        return t^

    @staticmethod
    def _upload_dataset(
        mut dst: Tensor, ref src: List[Scalar[DT]], n: Int, ctx: DeviceContext
    ) raises:
        """Host List → resident device buffer (one-time, no per-batch H2D)."""
        dst.dev = ctx.enqueue_create_buffer[DT](n)
        var hb = ctx.enqueue_create_host_buffer[DT](n)
        ctx.synchronize()
        memcpy(dest=hb.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
        ctx.enqueue_copy(dst.dev.value(), hb)
        ctx.synchronize()
        dst.n = n

    def _slice_train(mut self, x0: Int, y0: Int) raises:
        """Point batch_x/batch_y at GPU sub-buffer views of the resident set."""
        self.batch_x.dev = Optional(
            self.ds_x.dev.value().create_sub_buffer[DT](
                x0, Self.BATCH * Self.IN
            )
        )
        self.batch_x.n = Self.BATCH * Self.IN
        self.batch_y.dev = Optional(
            self.ds_y.dev.value().create_sub_buffer[DT](
                y0, Self.BATCH * Self.NC
            )
        )
        self.batch_y.n = Self.BATCH * Self.NC

    def train_epoch[
        N_TRAIN: Int
    ](
        mut self,
        ref train_x: List[Scalar[DT]],
        ref train_y: List[Scalar[DT]],
        ctx: Optional[DeviceContext],
    ) raises -> Scalar[DT]:
        comptime n_batches = N_TRAIN // Self.BATCH
        comptime if Self.target == "gpu":
            if not self._up_train:
                Self._upload_dataset(
                    self.ds_x, train_x, N_TRAIN * Self.IN, ctx.value()
                )
                Self._upload_dataset(
                    self.ds_y, train_y, N_TRAIN * Self.NC, ctx.value()
                )
                self._up_train = True
        self.loss.reset_accum[Self.target]()
        for nb in range(n_batches):
            var x0 = nb * Self.BATCH * Self.IN
            var y0 = nb * Self.BATCH * Self.NC
            comptime if Self.target == "cpu":
                memcpy(
                    dest=self.batch_x.data.unsafe_ptr(),
                    src=train_x.unsafe_ptr() + x0,
                    count=Self.BATCH * Self.IN,
                )
                memcpy(
                    dest=self.batch_y.data.unsafe_ptr(),
                    src=train_y.unsafe_ptr() + y0,
                    count=Self.BATCH * Self.NC,
                )
            else:
                self._slice_train(x0, y0)
            self.model.zero_grad[Self.target](ctx)
            self.model.forward[Self.target, Self.BATCH](
                TensorRefs[Self.MODEL.ARITY](self.batch_x), self.logits, ctx
            )
            self.loss.forward_accumulate[Self.target, Self.BATCH](
                self.logits, self.batch_y, ctx
            )
            self.loss.vjp[Self.target, Self.BATCH](
                self.logits, self.batch_y, self.grad, ctx
            )
            self.model.vjp[Self.target, Self.BATCH](
                TensorRefs[Self.MODEL.ARITY](self.batch_x),
                self.grad,
                TensorRefs[Self.MODEL.ARITY](self.gi),
                ctx,
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
        comptime if Self.target == "gpu":
            if not self._up_test:
                Self._upload_dataset(
                    self.ds_tx, test_x, N_TEST * Self.IN, ctx.value()
                )
                self._up_test = True
        var correct = 0
        for nb in range(n_batches):
            var x0 = nb * Self.BATCH * Self.IN
            comptime if Self.target == "cpu":
                memcpy(
                    dest=self.batch_x.data.unsafe_ptr(),
                    src=test_x.unsafe_ptr() + x0,
                    count=Self.BATCH * Self.IN,
                )
            else:
                self.batch_x.dev = Optional(
                    self.ds_tx.dev.value().create_sub_buffer[DT](
                        x0, Self.BATCH * Self.IN
                    )
                )
                self.batch_x.n = Self.BATCH * Self.IN
            self.model.forward[Self.target, Self.BATCH](
                TensorRefs[Self.MODEL.ARITY](self.batch_x), self.logits, ctx
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
