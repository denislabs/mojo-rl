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
from std.time import perf_counter_ns
from layout import Layout

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..optimizer.adam import Adam
from ..optimizer.optimizer import Optimizer
from ..loss.cross_entropy import CrossEntropyLoss
from .shuffle_kernels import (
    init_identity_indices_kernel,
    fisher_yates_shuffle_kernel,
    increment_seed_kernel,
    gather_rows_kernel,
)
from .augmenter import Augmenter, IdentityAugmenter
from ..optimizer.lr_scheduler import Scheduler, ConstantSchedule


@fieldwise_init
struct TrainResult(Movable & ImplicitlyDeletable):
    """Per-epoch training history returned by `Trainer.train_gpu`."""

    var epoch_train_loss: List[Float64]
    var epoch_test_top1: List[Float64]
    var epoch_train_s: List[Float64]
    var epoch_eval_s: List[Float64]

    @staticmethod
    def empty() -> Self:
        return Self(
            epoch_train_loss=List[Float64](),
            epoch_test_top1=List[Float64](),
            epoch_train_s=List[Float64](),
            epoch_eval_s=List[Float64](),
        )


struct Trainer[
    MODEL: Module, NC: Int, IN: Int, BATCH: Int, target: StaticString,
    POLICY: AMPPolicy = NoAMP,
    OPT: Optimizer = Adam,
](Movable & ImplicitlyDeletable):
    var model: Self.MODEL
    var opt: Self.OPT
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
        self.opt = Self.OPT()
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
        t.opt = Self.OPT()
        t.opt.set_lr(lr)
        # Engage the optimizer's contiguous-arena mode (GPU single-kernel step);
        # a NO-OP on CPU. Must come AFTER the model is made + initialized — it
        # rebinds the model's param buffers to arena slices.
        t.opt.adopt[Self.target](t.model, ctx)
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
            self.opt.zero_grad[Self.target](self.model, ctx)
            self.model.forward[Self.target, Self.BATCH, POLICY=Self.POLICY](
                TensorRefs[Self.MODEL.ARITY](self.batch_x), self.logits, ctx
            )
            self.loss.forward_accumulate[Self.target, Self.BATCH](
                self.logits, self.batch_y, ctx
            )
            self.loss.vjp[Self.target, Self.BATCH](
                self.logits, self.batch_y, self.grad, ctx
            )
            self.model.vjp[Self.target, Self.BATCH, POLICY=Self.POLICY](
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
            self.model.forward[Self.target, Self.BATCH, POLICY=Self.POLICY](
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

    def train_gpu[
        N_TRAIN: Int,
        N_TEST: Int,
        AUGMENTER: Augmenter = IdentityAugmenter,
        SCHEDULER: Scheduler = ConstantSchedule,
    ](
        mut self,
        ref train_x: List[Scalar[DT]],
        ref train_y: List[Scalar[DT]],
        ref test_x: List[Scalar[DT]],
        ref test_labels: List[Int32],
        ctx: Optional[DeviceContext],
        epochs: Int = 1,
        print_progress: Bool = True,
        shuffle: Bool = False,
        rng_seed: UInt64 = 42,
        aug_seed: UInt64 = 1000,
    ) raises -> TrainResult:
        """Whole-dataset GPU training run with per-epoch top-1 eval — the
        one-call convenience over `train_epoch`/`eval_top1`. Uploads train+test
        ONCE (resident), then each epoch trains over all batches and evaluates.

        `shuffle=True` runs an on-device Fisher-Yates permutation of the train
        set each epoch (device-resident indices/seed, gather into owned batch
        buffers — capture-friendly, no host involvement); `shuffle=False` slices
        contiguous sub-buffer views (zero-copy).

        `AUGMENTER` (default identity): when not a no-op, an augmentation buffer
        is allocated once and `AUGMENTER.augment` FULLY rewrites it from the raw
        resident train set each epoch; training then reads the augmented buffer
        (the labels are never augmented). `SCHEDULER` (default constant): when not
        constant, the optimizer LR is set to `base_lr * lr_scale_at(epoch,
        epochs)` each epoch (base_lr = the LR on the optimizer before the call).

        BatchNorm is toggled to train mode before each epoch and eval mode before
        the test pass via `set_attr["training"]` (no-op for nets without BN)."""
        comptime assert (
            Self.target == "gpu"
        ), "Trainer.train_gpu requires target='gpu'"
        comptime assert (
            N_TRAIN % Self.BATCH == 0
        ), "train_gpu: N_TRAIN must be divisible by BATCH"
        comptime assert (
            N_TEST % Self.BATCH == 0
        ), "train_gpu: N_TEST must be divisible by BATCH"
        comptime n_batches = N_TRAIN // Self.BATCH
        comptime blocks_init = (N_TRAIN + TPB - 1) // TPB
        comptime blocks_gx = (Self.BATCH * Self.IN + TPB - 1) // TPB
        comptime blocks_gy = (Self.BATCH * Self.NC + TPB - 1) // TPB
        comptime USE_AUG = not AUGMENTER.IS_NOOP
        comptime USE_SCHED = not SCHEDULER.IS_CONSTANT
        var c = ctx.value()
        var result = TrainResult.empty()

        # Resident train set (once). Test is uploaded lazily by eval_top1.
        if not self._up_train:
            Self._upload_dataset(self.ds_x, train_x, N_TRAIN * Self.IN, c)
            Self._upload_dataset(self.ds_y, train_y, N_TRAIN * Self.NC, c)
            self._up_train = True

        # Augmentation working buffer (X only; rewritten from raw ds_x each
        # epoch). Labels (ds_y) are never augmented.
        var aug = Tensor()
        comptime if USE_AUG:
            aug = Tensor.alloc_gpu(c, N_TRAIN * Self.IN)

        # LR schedule: capture the caller-set base LR; per-epoch LR is set to
        # base_lr * scale. Constant default leaves the LR exactly as set.
        var base_lr = self.opt.get_lr()

        # Shuffle scratch (device-resident; allocated once, reused per epoch).
        # Owned gather targets `gx`/`gy` — NOT self.batch_x (which eval_top1
        # repoints at a test sub-buffer, so reusing it as a gather target would
        # corrupt the test set).
        var indices = TensorImpl[DType.int32]()
        var seed = TensorImpl[DType.uint64]()
        var gx = Tensor()
        var gy = Tensor()
        if shuffle:
            indices = TensorImpl[DType.int32].alloc_gpu(c, N_TRAIN)
            c.enqueue_function[init_identity_indices_kernel[N_TRAIN]](
                indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                grid_dim=blocks_init,
                block_dim=TPB,
            )
            seed.ensure(1)
            seed.data[0] = rng_seed
            seed.n = 1
            seed.upload(c)
            gx = Tensor.alloc_gpu(c, Self.BATCH * Self.IN)
            gy = Tensor.alloc_gpu(c, Self.BATCH * Self.NC)

        for epoch in range(epochs):
            var t0 = perf_counter_ns()
            comptime if USE_SCHED:
                self.opt.set_lr(
                    base_lr
                    * Scalar[DT](SCHEDULER.lr_scale_at(epoch, epochs))
                )
            self.loss.reset_accum["gpu"]()
            self.model.set_attr["training"](Scalar[DT](1.0))
            # (Re)build the augmented training set from the raw resident set.
            comptime if USE_AUG:
                AUGMENTER.augment[N_TRAIN, Self.IN, DT](
                    c,
                    aug.lt["gpu", Layout.row_major(N_TRAIN, Self.IN)](),
                    self.ds_x.lt["gpu", Layout.row_major(N_TRAIN, Self.IN)](),
                    epoch,
                    aug_seed,
                )
            if shuffle:
                c.enqueue_function[fisher_yates_shuffle_kernel[N_TRAIN]](
                    indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                    seed.lt["gpu", Layout.row_major(1)](),
                    grid_dim=1,
                    block_dim=1,
                )
                c.enqueue_function[increment_seed_kernel](
                    seed.lt["gpu", Layout.row_major(1)](),
                    grid_dim=1,
                    block_dim=1,
                )
            for nb in range(n_batches):
                if shuffle:
                    var off = nb * Self.BATCH
                    # Gather X from the augmented set (if any) else raw ds_x.
                    comptime if USE_AUG:
                        c.enqueue_function[
                            gather_rows_kernel[N_TRAIN, Self.BATCH, Self.IN, DT]
                        ](
                            gx.lt[
                                "gpu", Layout.row_major(Self.BATCH, Self.IN)
                            ](),
                            aug.lt["gpu", Layout.row_major(N_TRAIN, Self.IN)](),
                            indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                            off,
                            grid_dim=blocks_gx,
                            block_dim=TPB,
                        )
                    else:
                        c.enqueue_function[
                            gather_rows_kernel[N_TRAIN, Self.BATCH, Self.IN, DT]
                        ](
                            gx.lt[
                                "gpu", Layout.row_major(Self.BATCH, Self.IN)
                            ](),
                            self.ds_x.lt[
                                "gpu", Layout.row_major(N_TRAIN, Self.IN)
                            ](),
                            indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                            off,
                            grid_dim=blocks_gx,
                            block_dim=TPB,
                        )
                    c.enqueue_function[
                        gather_rows_kernel[N_TRAIN, Self.BATCH, Self.NC, DT]
                    ](
                        gy.lt["gpu", Layout.row_major(Self.BATCH, Self.NC)](),
                        self.ds_y.lt[
                            "gpu", Layout.row_major(N_TRAIN, Self.NC)
                        ](),
                        indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                        off,
                        grid_dim=blocks_gy,
                        block_dim=TPB,
                    )
                    self.opt.zero_grad["gpu"](self.model, ctx)
                    self.model.forward[
                        "gpu", Self.BATCH, POLICY = Self.POLICY
                    ](TensorRefs[Self.MODEL.ARITY](gx), self.logits, ctx)
                    self.loss.forward_accumulate["gpu", Self.BATCH](
                        self.logits, gy, ctx
                    )
                    self.loss.vjp["gpu", Self.BATCH](
                        self.logits, gy, self.grad, ctx
                    )
                    self.model.vjp["gpu", Self.BATCH, POLICY = Self.POLICY](
                        TensorRefs[Self.MODEL.ARITY](gx),
                        self.grad,
                        TensorRefs[Self.MODEL.ARITY](self.gi),
                        ctx,
                    )
                    self.opt.step["gpu"](self.model, ctx)
                else:
                    var x0 = nb * Self.BATCH * Self.IN
                    var y0 = nb * Self.BATCH * Self.NC
                    # Sub-buffer view of X from the augmented set else raw ds_x.
                    comptime if USE_AUG:
                        self.batch_x.dev = Optional(
                            aug.dev.value().create_sub_buffer[DT](
                                x0, Self.BATCH * Self.IN
                            )
                        )
                    else:
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
                    self.opt.zero_grad["gpu"](self.model, ctx)
                    self.model.forward[
                        "gpu", Self.BATCH, POLICY = Self.POLICY
                    ](
                        TensorRefs[Self.MODEL.ARITY](self.batch_x),
                        self.logits,
                        ctx,
                    )
                    self.loss.forward_accumulate["gpu", Self.BATCH](
                        self.logits, self.batch_y, ctx
                    )
                    self.loss.vjp["gpu", Self.BATCH](
                        self.logits, self.batch_y, self.grad, ctx
                    )
                    self.model.vjp["gpu", Self.BATCH, POLICY = Self.POLICY](
                        TensorRefs[Self.MODEL.ARITY](self.batch_x),
                        self.grad,
                        TensorRefs[Self.MODEL.ARITY](self.gi),
                        ctx,
                    )
                    self.opt.step["gpu"](self.model, ctx)
            var avg = Float64(self.loss.read_accum["gpu"](ctx))
            c.synchronize()
            var t1 = perf_counter_ns()
            self.model.set_attr["training"](Scalar[DT](0.0))
            var top1 = self.eval_top1[N_TEST](test_x, test_labels, ctx)
            var t2 = perf_counter_ns()
            result.epoch_train_loss.append(avg)
            result.epoch_test_top1.append(top1)
            result.epoch_train_s.append(Float64(t1 - t0) / 1e9)
            result.epoch_eval_s.append(Float64(t2 - t1) / 1e9)
            if print_progress:
                print(
                    "epoch " + String(epoch)
                    + " | train_loss=" + String(avg)
                    + " | test_top1=" + String(top1 * 100.0) + "%"
                    + " | train=" + String(Float64(t1 - t0) / 1e9) + "s"
                    + " | eval=" + String(Float64(t2 - t1) / 1e9) + "s"
                )
        return result^
