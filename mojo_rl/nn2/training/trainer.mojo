"""Trainer[NET, OPT, LOSS, BATCH, target] — owns IO buffers and runs the
standard supervised forward / backward / step loop.

`target` is a comptime struct param: a trainer's identity is tied to
one device for its lifetime. Internally, Trainer threads `Self.target`
to each method call on net / optim / loss.

Two API surfaces:

  - **Per-step** (`train_step` / `predict`): caller supplies per-batch
    host pointers each iteration. Useful for small/interactive
    workloads, RL-style training, etc. Trainer handles upload internally.

  - **Whole-dataset** (`train_gpu`, GPU only): caller uploads the entire
    training + test sets to device once, Trainer slices by pointer offset
    per batch internally. No host copies in the inner loop.

Construction:

  - `Trainer[NET, OPT, LOSS, BATCH, target].make[INIT](ctx?)` is the
    one-call factory. Builds net + optim + loss internally via
    `NET.make[target, INIT]`, `LOSS.make[target]`, `OPT.make[target]`.
    No `type_of`, no `^` at the user level.

  - `Trainer[...].make_from(net, optim, loss_fn, ctx?)` accepts
    pre-built components — for cases where the user wants custom weight
    init or special construction.
"""

from std.memory import alloc
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Module, Optimizer, Loss, Initializer, AMPPolicy, NoAMP
from .augmenter import Augmenter, IdentityAugmenter
from .lr_scheduler import Scheduler, ConstantSchedule
from .shuffle_kernels import (
    init_identity_indices_kernel,
    fisher_yates_shuffle_kernel,
    increment_seed_kernel,
    gather_rows_kernel,
)


# ──────────────────────────────────────────────────────────────────────────
# TrainResult — per-epoch metrics
# ──────────────────────────────────────────────────────────────────────────


@fieldwise_init
struct TrainResult(Movable & ImplicitlyDestructible):
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


@fieldwise_init
struct Trainer[
    NET: Module,
    OPT: Optimizer,
    LOSS: Loss,
    BATCH: Int,
    target: StaticString = "cpu",
    POLICY: AMPPolicy = NoAMP,
](Movable & ImplicitlyDestructible):
    comptime IN_DIM = Self.NET.IN_DIMS[0]
    comptime OUT_DIM = Self.NET.OUT_DIM

    var net: Self.NET
    var optim: Self.OPT
    var loss_fn: Self.LOSS

    # CPU side (used when target=="cpu" — length-1 stubs otherwise).
    var input_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var target_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var output_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var grad_out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var grad_in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]

    # GPU side (Some when target=="gpu", None when "cpu").
    var input_dev: Optional[DeviceBuffer[DT]]
    var target_dev: Optional[DeviceBuffer[DT]]
    var output_dev: Optional[DeviceBuffer[DT]]
    var grad_out_dev: Optional[DeviceBuffer[DT]]
    var grad_in_dev: Optional[DeviceBuffer[DT]]
    var input_host: Optional[HostBuffer[DT]]
    var target_host: Optional[HostBuffer[DT]]
    var output_host: Optional[HostBuffer[DT]]
    var ctx: Optional[DeviceContext]

    # ------------------------------------------------------------------
    # Factories — `make[INIT]` builds net/optim/loss internally. The
    # `make_from(...)` overload accepts pre-built components.
    # ------------------------------------------------------------------

    @staticmethod
    def make[INIT: Initializer]() raises -> Self:
        """CPU one-call factory. Builds net + optim + loss internally."""
        comptime assert (
            Self.target == "cpu"
        ), "Trainer.make[INIT]() is CPU-only; pass ctx for GPU"
        var net = Self.NET.make[Self.target, INIT]()
        var loss = Self.LOSS.make[Self.target]()
        var optim = Self.OPT.make[Self.target](net)
        return Self.make_from(net^, optim^, loss^)

    @staticmethod
    def make[INIT: Initializer](ctx: DeviceContext) raises -> Self:
        """GPU one-call factory."""
        comptime assert (
            Self.target == "gpu"
        ), "Trainer.make[INIT](ctx) requires target='gpu'"
        var net = Self.NET.make[Self.target, INIT](ctx)
        var loss = Self.LOSS.make[Self.target](ctx)
        var optim = Self.OPT.make[Self.target](net, ctx)
        return Self.make_from(net^, optim^, loss^, ctx)

    @staticmethod
    def make_from(
        var net: Self.NET,
        var optim: Self.OPT,
        var loss_fn: Self.LOSS,
    ) raises -> Self:
        """CPU factory from pre-built components."""
        comptime assert (
            Self.target == "cpu"
        ), "Trainer.make_from(net, optim, loss_fn) is CPU-only"
        comptime assert (
            Self.LOSS.OUT_DIM == Self.NET.OUT_DIM
        ), "Trainer: loss N_CLASSES must equal net OUT_DIM"
        var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            Self.BATCH * Self.IN_DIM
        )
        var tg_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            Self.BATCH * Self.OUT_DIM
        )
        var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](Self.BATCH * Self.OUT_DIM)
        var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            Self.BATCH * Self.OUT_DIM
        )
        var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            Self.BATCH * Self.IN_DIM
        )
        return Self(
            net=net^,
            optim=optim^,
            loss_fn=loss_fn^,
            input_buf=in_buf,
            target_buf=tg_buf,
            output_buf=out_buf,
            grad_out_buf=go_buf,
            grad_in_buf=gi_buf,
            input_dev=None,
            target_dev=None,
            output_dev=None,
            grad_out_dev=None,
            grad_in_dev=None,
            input_host=None,
            target_host=None,
            output_host=None,
            ctx=None,
        )

    @staticmethod
    def make_from(
        var net: Self.NET,
        var optim: Self.OPT,
        var loss_fn: Self.LOSS,
        ctx: DeviceContext,
    ) raises -> Self:
        """GPU factory from pre-built components."""
        comptime assert (
            Self.target == "gpu"
        ), "Trainer.make_from(net, optim, loss_fn, ctx) requires target='gpu'"
        comptime assert (
            Self.LOSS.OUT_DIM == Self.NET.OUT_DIM
        ), "Trainer: loss N_CLASSES must equal net OUT_DIM"
        var in_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.IN_DIM)
        var tg_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OUT_DIM)
        var out_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OUT_DIM)
        var go_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OUT_DIM)
        var gi_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.IN_DIM)
        var in_host = ctx.enqueue_create_host_buffer[DT](
            Self.BATCH * Self.IN_DIM
        )
        var tg_host = ctx.enqueue_create_host_buffer[DT](
            Self.BATCH * Self.OUT_DIM
        )
        var out_host = ctx.enqueue_create_host_buffer[DT](
            Self.BATCH * Self.OUT_DIM
        )
        ctx.synchronize()
        var stub_in: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](1)
        var stub_tg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](1)
        var stub_out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](1)
        var stub_go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](1)
        var stub_gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](1)
        return Self(
            net=net^,
            optim=optim^,
            loss_fn=loss_fn^,
            input_buf=stub_in,
            target_buf=stub_tg,
            output_buf=stub_out,
            grad_out_buf=stub_go,
            grad_in_buf=stub_gi,
            input_dev=in_dev^,
            target_dev=tg_dev^,
            output_dev=out_dev^,
            grad_out_dev=go_dev^,
            grad_in_dev=gi_dev^,
            input_host=in_host^,
            target_host=tg_host^,
            output_host=out_host^,
            ctx=ctx,
        )

    def __del__(deinit self):
        self.input_buf.free()
        self.target_buf.free()
        self.output_buf.free()
        self.grad_out_buf.free()
        self.grad_in_buf.free()

    # ------------------------------------------------------------------
    # Pipeline core — called by every train_step variant.
    # ------------------------------------------------------------------

    def _train_step_views(
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        targets: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises -> Scalar[DT]:
        comptime assert input.flat_rank == 2, "input must be rank-2"
        comptime assert targets.flat_rank == 2, "targets must be rank-2"
        comptime if Self.target == "cpu":
            # MutAnyOrigin laundering: trait variadics on the unified
            # Module require origin=MutAnyOrigin. `*_buf` fields are
            # already `UnsafePointer[Scalar[DT], MutAnyOrigin]` (see
            # struct decl); only `input` needs rebinding.
            var input_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            var input_my = TileTensor(
                input_p, row_major[Self.BATCH, Self.IN_DIM]()
            )
            var output = TileTensor(
                self.output_buf, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            var grad_out = TileTensor(
                self.grad_out_buf, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            var grad_in = TileTensor(
                self.grad_in_buf, row_major[Self.BATCH, Self.IN_DIM]()
            )
            self.optim.zero_grad[Self.target](self.net)
            self.net.forward[Self.target, Self.BATCH, POLICY=Self.POLICY](
                input_my, output=output
            )
            var L = self.loss_fn.forward[
                Self.target, Self.BATCH, POLICY=Self.POLICY
            ](output, targets)
            self.loss_fn.vjp[Self.target, Self.BATCH, POLICY=Self.POLICY](
                targets, grad_out
            )
            self.net.vjp[Self.target, Self.BATCH, POLICY=Self.POLICY](
                grad_out, grad_in
            )
            self.optim.step[Self.target](self.net)
            return L
        else:
            # MutAnyOrigin laundering — see train_step's else-branch comment.
            var out_ptr: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.output_dev.value().unsafe_ptr()
            var go_ptr: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.grad_out_dev.value().unsafe_ptr()
            var gi_ptr: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.grad_in_dev.value().unsafe_ptr()
            var in_ptr_my = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            var input_my = TileTensor(
                in_ptr_my, row_major[Self.BATCH, Self.IN_DIM]()
            )
            var output = TileTensor(
                out_ptr, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            var grad_out = TileTensor(
                go_ptr, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            var grad_in = TileTensor(
                gi_ptr, row_major[Self.BATCH, Self.IN_DIM]()
            )
            self.optim.zero_grad[Self.target](self.net)
            self.net.forward[Self.target, Self.BATCH, POLICY=Self.POLICY](
                input_my, output=output
            )
            var L = self.loss_fn.forward[
                Self.target, Self.BATCH, POLICY=Self.POLICY
            ](output, targets)
            self.loss_fn.vjp[Self.target, Self.BATCH, POLICY=Self.POLICY](
                targets, grad_out
            )
            self.net.vjp[Self.target, Self.BATCH, POLICY=Self.POLICY](
                grad_out, grad_in
            )
            self.optim.step[Self.target](self.net)
            return L

    # ------------------------------------------------------------------
    # Per-step API.
    # ------------------------------------------------------------------

    def train_step(
        mut self,
        input_host_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        target_host_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        comptime if Self.target == "cpu":
            for k in range(Self.BATCH * Self.IN_DIM):
                self.input_buf[k] = input_host_ptr[k]
            for k in range(Self.BATCH * Self.OUT_DIM):
                self.target_buf[k] = target_host_ptr[k]
            var input = TileTensor(
                self.input_buf, row_major[Self.BATCH, Self.IN_DIM]()
            )
            var targets = TileTensor(
                self.target_buf, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            return self._train_step_views(input, targets)
        else:
            var ctx = self.ctx.value()
            var in_host_buf: HostBuffer[DT] = self.input_host.value()
            var tg_host_buf: HostBuffer[DT] = self.target_host.value()
            for k in range(Self.BATCH * Self.IN_DIM):
                in_host_buf.unsafe_ptr()[k] = input_host_ptr[k]
            for k in range(Self.BATCH * Self.OUT_DIM):
                tg_host_buf.unsafe_ptr()[k] = target_host_ptr[k]
            ctx.enqueue_copy(self.input_dev.value(), in_host_buf)
            ctx.enqueue_copy(self.target_dev.value(), tg_host_buf)
            # Launder pointers through MutAnyOrigin so Mojo's aliasing
            # analyzer doesn't see `self.input_dev` and `self.target_dev`
            # (different fields, different buffers) as overlapping `self.*`.
            var in_ptr: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.input_dev.value().unsafe_ptr()
            var tg_ptr: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.target_dev.value().unsafe_ptr()
            var input = TileTensor(in_ptr, row_major[Self.BATCH, Self.IN_DIM]())
            var targets = TileTensor(
                tg_ptr, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            return self._train_step_views(input, targets)

    def predict(
        mut self,
        input_host_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        output_host_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime if Self.target == "cpu":
            for k in range(Self.BATCH * Self.IN_DIM):
                self.input_buf[k] = input_host_ptr[k]
            var input = TileTensor(
                self.input_buf, row_major[Self.BATCH, Self.IN_DIM]()
            )
            var output = TileTensor(
                self.output_buf, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            self.net.forward[Self.target, Self.BATCH, POLICY=Self.POLICY](
                input, output=output
            )
            for k in range(Self.BATCH * Self.OUT_DIM):
                output_host_ptr[k] = self.output_buf[k]
        else:
            var ctx = self.ctx.value()
            var in_host_buf: HostBuffer[DT] = self.input_host.value()
            var out_host_buf: HostBuffer[DT] = self.output_host.value()
            for k in range(Self.BATCH * Self.IN_DIM):
                in_host_buf.unsafe_ptr()[k] = input_host_ptr[k]
            ctx.enqueue_copy(self.input_dev.value(), in_host_buf)
            var in_ptr: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.input_dev.value().unsafe_ptr()
            var out_ptr: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.output_dev.value().unsafe_ptr()
            var input = TileTensor(in_ptr, row_major[Self.BATCH, Self.IN_DIM]())
            var output = TileTensor(
                out_ptr, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            self.net.forward[Self.target, Self.BATCH, POLICY=Self.POLICY](
                input, output=output
            )
            ctx.enqueue_copy(out_host_buf, self.output_dev.value())
            ctx.synchronize()
            for k in range(Self.BATCH * Self.OUT_DIM):
                output_host_ptr[k] = out_host_buf.unsafe_ptr()[k]

    # ------------------------------------------------------------------
    # Whole-dataset GPU training.
    # ------------------------------------------------------------------

    def train_gpu[
        N_TRAIN: Int,
    ](
        mut self,
        train_x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        train_y: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        epochs: Int = 1,
        print_progress: Bool = True,
        shuffle: Bool = False,
        rng_seed: UInt64 = 42,
    ) raises -> TrainResult:
        comptime assert (
            Self.target == "gpu"
        ), "Trainer.train_gpu requires target='gpu'"
        comptime assert (
            N_TRAIN % Self.BATCH == 0
        ), "Trainer.train_gpu: N_TRAIN must be divisible by BATCH"
        comptime N_BATCHES = N_TRAIN // Self.BATCH
        comptime BLOCKS_INIT = (N_TRAIN + TPB - 1) // TPB
        comptime BLOCKS_GATHER_X = (
            Self.BATCH * Self.IN_DIM + TPB - 1
        ) // TPB
        comptime BLOCKS_GATHER_Y = (
            Self.BATCH * Self.OUT_DIM + TPB - 1
        ) // TPB

        var result = TrainResult.empty()
        var ctx = self.ctx.value()
        var x_base = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            train_x.ptr
        )
        var y_base = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            train_y.ptr
        )

        # Shuffle scratch (device-only; allocated once, reused across epochs).
        var indices_dev: Optional[DeviceBuffer[DType.int32]] = None
        var seed_dev: Optional[DeviceBuffer[DType.uint64]] = None
        var shuf_x_dev: Optional[DeviceBuffer[DT]] = None
        var shuf_y_dev: Optional[DeviceBuffer[DT]] = None
        if shuffle:
            var idx = ctx.enqueue_create_buffer[DType.int32](N_TRAIN)
            var seed = ctx.enqueue_create_buffer[DType.uint64](1)
            var sx = ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.IN_DIM
            )
            var sy = ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.OUT_DIM
            )
            var seed_host = ctx.enqueue_create_host_buffer[DType.uint64](1)
            seed_host.unsafe_ptr()[0] = rng_seed
            ctx.enqueue_copy(seed, seed_host)
            ctx.synchronize()
            var idx_t = LayoutTensor[
                DType.int32, Layout.row_major(N_TRAIN), MutAnyOrigin
            ](idx.unsafe_ptr())
            ctx.enqueue_function[init_identity_indices_kernel[N_TRAIN]](
                idx_t,
                grid_dim=(BLOCKS_INIT,),
                block_dim=(TPB,),
            )
            indices_dev = idx^
            seed_dev = seed^
            shuf_x_dev = sx^
            shuf_y_dev = sy^

        for epoch in range(epochs):
            var t0 = perf_counter_ns()
            var epoch_loss: Scalar[DT] = 0.0
            if shuffle:
                var idx_t = LayoutTensor[
                    DType.int32, Layout.row_major(N_TRAIN), MutAnyOrigin
                ](indices_dev.value().unsafe_ptr())
                var seed_t = LayoutTensor[
                    DType.uint64, Layout.row_major(1), MutAnyOrigin
                ](seed_dev.value().unsafe_ptr())
                ctx.enqueue_function[fisher_yates_shuffle_kernel[N_TRAIN]](
                    idx_t, seed_t, grid_dim=(1,), block_dim=(1,)
                )
                ctx.enqueue_function[increment_seed_kernel](
                    seed_t, grid_dim=(1,), block_dim=(1,)
                )
            for b in range(N_BATCHES):
                if shuffle:
                    var full_x_t = LayoutTensor[
                        DT,
                        Layout.row_major(N_TRAIN, Self.IN_DIM),
                        MutAnyOrigin,
                    ](x_base)
                    var full_y_t = LayoutTensor[
                        DT,
                        Layout.row_major(N_TRAIN, Self.OUT_DIM),
                        MutAnyOrigin,
                    ](y_base)
                    var idx_t = LayoutTensor[
                        DType.int32, Layout.row_major(N_TRAIN), MutAnyOrigin
                    ](indices_dev.value().unsafe_ptr())
                    var sx_p = shuf_x_dev.value().unsafe_ptr()
                    var sy_p = shuf_y_dev.value().unsafe_ptr()
                    var shuf_x_t = LayoutTensor[
                        DT,
                        Layout.row_major(Self.BATCH, Self.IN_DIM),
                        MutAnyOrigin,
                    ](sx_p)
                    var shuf_y_t = LayoutTensor[
                        DT,
                        Layout.row_major(Self.BATCH, Self.OUT_DIM),
                        MutAnyOrigin,
                    ](sy_p)
                    var offset = b * Self.BATCH
                    ctx.enqueue_function[
                        gather_rows_kernel[
                            N_TRAIN, Self.BATCH, Self.IN_DIM, DT
                        ]
                    ](
                        shuf_x_t,
                        full_x_t,
                        idx_t,
                        offset,
                        grid_dim=(BLOCKS_GATHER_X,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        gather_rows_kernel[
                            N_TRAIN, Self.BATCH, Self.OUT_DIM, DT
                        ]
                    ](
                        shuf_y_t,
                        full_y_t,
                        idx_t,
                        offset,
                        grid_dim=(BLOCKS_GATHER_Y,),
                        block_dim=(TPB,),
                    )
                    var input = TileTensor(
                        sx_p, row_major[Self.BATCH, Self.IN_DIM]()
                    )
                    var targets = TileTensor(
                        sy_p, row_major[Self.BATCH, Self.OUT_DIM]()
                    )
                    epoch_loss += self._train_step_views(input, targets)
                else:
                    var x_ptr = x_base + b * Self.BATCH * Self.IN_DIM
                    var y_ptr = y_base + b * Self.BATCH * Self.OUT_DIM
                    var input = TileTensor(
                        x_ptr, row_major[Self.BATCH, Self.IN_DIM]()
                    )
                    var targets = TileTensor(
                        y_ptr, row_major[Self.BATCH, Self.OUT_DIM]()
                    )
                    epoch_loss += self._train_step_views(input, targets)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            var train_s = Float64(t1 - t0) / 1e9
            var avg = Float64(epoch_loss / Scalar[DT](N_BATCHES))
            result.epoch_train_loss.append(avg)
            result.epoch_train_s.append(train_s)
            result.epoch_eval_s.append(0.0)
            if print_progress:
                print(
                    "epoch "
                    + String(epoch)
                    + " | train_loss="
                    + String(avg)
                    + " | train="
                    + String(train_s)
                    + "s"
                )
        return result^

    def train_gpu[
        N_TRAIN: Int,
        N_TEST: Int,
        AUGMENTER: Augmenter = IdentityAugmenter,
        SCHEDULER: Scheduler = ConstantSchedule,
    ](
        mut self,
        train_x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        train_y: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        test_x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        test_y_labels: UnsafePointer[Int32, MutAnyOrigin],
        epochs: Int = 1,
        print_progress: Bool = True,
        shuffle: Bool = False,
        rng_seed: UInt64 = 42,
        aug_seed: UInt64 = 1000,
    ) raises -> TrainResult:
        """Whole-dataset GPU training with per-epoch eval.

        BatchNorm is toggled to train mode before each epoch and eval mode
        before the per-epoch test pass via `net.set_attr["training"]`
        (no-op for nets without BN). When `AUGMENTER` is not the identity,
        an augmentation buffer is allocated once and `AUGMENTER.augment`
        re-fills it from `train_x` each epoch before the mini-batch loop;
        training then reads the augmented buffer. When `SCHEDULER` is not
        constant, the optimizer LR is set to `base_lr * SCHEDULER.lr_scale_at
        (epoch, epochs)` each epoch (base_lr = the LR set on `optim` before
        the call); the constant default leaves the LR untouched.
        """
        comptime assert (
            Self.target == "gpu"
        ), "Trainer.train_gpu requires target='gpu'"
        comptime assert (
            N_TRAIN % Self.BATCH == 0
        ), "Trainer.train_gpu: N_TRAIN must be divisible by BATCH"
        comptime assert (
            N_TEST % Self.BATCH == 0
        ), "Trainer.train_gpu: N_TEST must be divisible by BATCH"
        comptime N_BATCHES_TRAIN = N_TRAIN // Self.BATCH
        comptime BLOCKS_INIT = (N_TRAIN + TPB - 1) // TPB
        comptime BLOCKS_GATHER_X = (
            Self.BATCH * Self.IN_DIM + TPB - 1
        ) // TPB
        comptime BLOCKS_GATHER_Y = (
            Self.BATCH * Self.OUT_DIM + TPB - 1
        ) // TPB

        var result = TrainResult.empty()
        var ctx = self.ctx.value()
        var raw_x = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            train_x.ptr
        )
        var y_base = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            train_y.ptr
        )

        # Augmentation buffer (allocated once, refilled per epoch). Identity
        # → skip it and train on `train_x` directly. A real augmenter fully
        # rewrites this buffer from `train_x` each epoch (no pre-copy).
        comptime USE_AUG = not AUGMENTER.IS_NOOP
        var aug_dev: Optional[DeviceBuffer[DT]] = None
        var x_base = raw_x
        comptime if USE_AUG:
            aug_dev = ctx.enqueue_create_buffer[DT](N_TRAIN * Self.IN_DIM)
            x_base = aug_dev.value().unsafe_ptr()

        # LR schedule: capture the caller-set base LR; per epoch the LR is
        # set to base_lr * SCHEDULER.lr_scale_at(epoch, epochs). Skipped
        # entirely for the constant schedule (LR left exactly as set).
        comptime USE_SCHED = not SCHEDULER.IS_CONSTANT
        var base_lr: Scalar[DT] = 0.0
        comptime if USE_SCHED:
            base_lr = self.optim.get_lr()

        # Shuffle scratch (device-only; allocated once, reused across epochs).
        var indices_dev: Optional[DeviceBuffer[DType.int32]] = None
        var seed_dev: Optional[DeviceBuffer[DType.uint64]] = None
        var shuf_x_dev: Optional[DeviceBuffer[DT]] = None
        var shuf_y_dev: Optional[DeviceBuffer[DT]] = None
        if shuffle:
            var idx = ctx.enqueue_create_buffer[DType.int32](N_TRAIN)
            var seed = ctx.enqueue_create_buffer[DType.uint64](1)
            var sx = ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.IN_DIM
            )
            var sy = ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.OUT_DIM
            )
            var seed_host = ctx.enqueue_create_host_buffer[DType.uint64](1)
            seed_host.unsafe_ptr()[0] = rng_seed
            ctx.enqueue_copy(seed, seed_host)
            ctx.synchronize()
            var idx_t = LayoutTensor[
                DType.int32, Layout.row_major(N_TRAIN), MutAnyOrigin
            ](idx.unsafe_ptr())
            ctx.enqueue_function[init_identity_indices_kernel[N_TRAIN]](
                idx_t,
                grid_dim=(BLOCKS_INIT,),
                block_dim=(TPB,),
            )
            indices_dev = idx^
            seed_dev = seed^
            shuf_x_dev = sx^
            shuf_y_dev = sy^

        for epoch in range(epochs):
            var t0 = perf_counter_ns()
            var epoch_loss: Scalar[DT] = 0.0
            if shuffle:
                var idx_t = LayoutTensor[
                    DType.int32, Layout.row_major(N_TRAIN), MutAnyOrigin
                ](indices_dev.value().unsafe_ptr())
                var seed_t = LayoutTensor[
                    DType.uint64, Layout.row_major(1), MutAnyOrigin
                ](seed_dev.value().unsafe_ptr())
                ctx.enqueue_function[fisher_yates_shuffle_kernel[N_TRAIN]](
                    idx_t, seed_t, grid_dim=(1,), block_dim=(1,)
                )
                ctx.enqueue_function[increment_seed_kernel](
                    seed_t, grid_dim=(1,), block_dim=(1,)
                )
            # LR schedule for this epoch.
            comptime if USE_SCHED:
                self.optim.set_lr(
                    base_lr * Scalar[DT](SCHEDULER.lr_scale_at(epoch, epochs))
                )
            # BN → train mode; (re)build the augmented training set.
            self.net.set_attr["training"](Scalar[DT](1.0))
            comptime if USE_AUG:
                var raw_lt = LayoutTensor[
                    DT, Layout.row_major(N_TRAIN, Self.IN_DIM), MutAnyOrigin
                ](raw_x)
                var aug_lt = LayoutTensor[
                    DT, Layout.row_major(N_TRAIN, Self.IN_DIM), MutAnyOrigin
                ](aug_dev.value().unsafe_ptr())
                AUGMENTER.augment[N_TRAIN, Self.IN_DIM, DT](
                    ctx, aug_lt, raw_lt, epoch, aug_seed
                )
            for b in range(N_BATCHES_TRAIN):
                if shuffle:
                    var full_x_t = LayoutTensor[
                        DT,
                        Layout.row_major(N_TRAIN, Self.IN_DIM),
                        MutAnyOrigin,
                    ](x_base)
                    var full_y_t = LayoutTensor[
                        DT,
                        Layout.row_major(N_TRAIN, Self.OUT_DIM),
                        MutAnyOrigin,
                    ](y_base)
                    var idx_t = LayoutTensor[
                        DType.int32, Layout.row_major(N_TRAIN), MutAnyOrigin
                    ](indices_dev.value().unsafe_ptr())
                    var sx_p = shuf_x_dev.value().unsafe_ptr()
                    var sy_p = shuf_y_dev.value().unsafe_ptr()
                    var shuf_x_t = LayoutTensor[
                        DT,
                        Layout.row_major(Self.BATCH, Self.IN_DIM),
                        MutAnyOrigin,
                    ](sx_p)
                    var shuf_y_t = LayoutTensor[
                        DT,
                        Layout.row_major(Self.BATCH, Self.OUT_DIM),
                        MutAnyOrigin,
                    ](sy_p)
                    var offset = b * Self.BATCH
                    ctx.enqueue_function[
                        gather_rows_kernel[
                            N_TRAIN, Self.BATCH, Self.IN_DIM, DT
                        ]
                    ](
                        shuf_x_t,
                        full_x_t,
                        idx_t,
                        offset,
                        grid_dim=(BLOCKS_GATHER_X,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        gather_rows_kernel[
                            N_TRAIN, Self.BATCH, Self.OUT_DIM, DT
                        ]
                    ](
                        shuf_y_t,
                        full_y_t,
                        idx_t,
                        offset,
                        grid_dim=(BLOCKS_GATHER_Y,),
                        block_dim=(TPB,),
                    )
                    var input = TileTensor(
                        sx_p, row_major[Self.BATCH, Self.IN_DIM]()
                    )
                    var targets = TileTensor(
                        sy_p, row_major[Self.BATCH, Self.OUT_DIM]()
                    )
                    epoch_loss += self._train_step_views(input, targets)
                else:
                    var x_ptr = x_base + b * Self.BATCH * Self.IN_DIM
                    var y_ptr = y_base + b * Self.BATCH * Self.OUT_DIM
                    var input = TileTensor(
                        x_ptr, row_major[Self.BATCH, Self.IN_DIM]()
                    )
                    var targets = TileTensor(
                        y_ptr, row_major[Self.BATCH, Self.OUT_DIM]()
                    )
                    epoch_loss += self._train_step_views(input, targets)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            var train_s = Float64(t1 - t0) / 1e9

            self.net.set_attr["training"](Scalar[DT](0.0))  # BN → eval mode
            var top1 = self.eval_top1_gpu[N_TEST](test_x, test_y_labels)
            var t2 = perf_counter_ns()
            var eval_s = Float64(t2 - t1) / 1e9

            var avg = Float64(epoch_loss / Scalar[DT](N_BATCHES_TRAIN))
            result.epoch_train_loss.append(avg)
            result.epoch_test_top1.append(top1)
            result.epoch_train_s.append(train_s)
            result.epoch_eval_s.append(eval_s)
            if print_progress:
                print(
                    "epoch "
                    + String(epoch)
                    + " | train_loss="
                    + String(avg)
                    + " | test_top1="
                    + String(top1 * 100.0)
                    + "%"
                    + " | train="
                    + String(train_s)
                    + "s"
                    + " | eval="
                    + String(eval_s)
                    + "s"
                )
        return result^

    def eval_top1_gpu[
        N_TEST: Int,
    ](
        mut self,
        test_x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        test_y_labels: UnsafePointer[Int32, MutAnyOrigin],
    ) raises -> Float64:
        comptime assert (
            Self.target == "gpu"
        ), "Trainer.eval_top1_gpu requires target='gpu'"
        comptime assert (
            N_TEST % Self.BATCH == 0
        ), "Trainer.eval_top1_gpu: N_TEST must be divisible by BATCH"
        comptime N_BATCHES = N_TEST // Self.BATCH

        var ctx = self.ctx.value()
        var out_host: HostBuffer[DT] = self.output_host.value()
        var x_base = test_x.ptr
        var n_correct: Int = 0
        for b in range(N_BATCHES):
            var x_ptr_my = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                x_base + b * Self.BATCH * Self.IN_DIM
            )
            var out_ptr_my = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.output_dev.value().unsafe_ptr()
            )
            var input = TileTensor(x_ptr_my, row_major[Self.BATCH, Self.IN_DIM]())
            var output = TileTensor(
                out_ptr_my, row_major[Self.BATCH, Self.OUT_DIM](),
            )
            self.net.forward[Self.target, Self.BATCH, POLICY=Self.POLICY](
                input, output=output
            )
            ctx.enqueue_copy(out_host, self.output_dev.value())
            ctx.synchronize()
            for k in range(Self.BATCH):
                var best_c: Int = 0
                var best_v: Scalar[DT] = out_host.unsafe_ptr()[
                    k * Self.OUT_DIM + 0
                ]
                for c in range(1, Self.OUT_DIM):
                    var v = out_host.unsafe_ptr()[k * Self.OUT_DIM + c]
                    if v > best_v:
                        best_v = v
                        best_c = c
                if best_c == Int(test_y_labels[b * Self.BATCH + k]):
                    n_correct += 1
        return Float64(n_correct) / Float64(N_TEST)
