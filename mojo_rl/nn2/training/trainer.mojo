"""Trainer[NET, OPT, LOSS, BATCH, target] — owns IO buffers and runs the
standard supervised forward / backward / step loop.

Phase 2.4: `target` is a comptime struct param on Trainer (per user
direction — a trainer's identity is tied to one target for its lifetime).
Internally, Trainer threads `Self.target` to each method call on
net/optim/loss.

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
from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import Module, Optimizer, Loss, Initializer


# ──────────────────────────────────────────────────────────────────────────
# TrainResult — per-epoch metrics
# ──────────────────────────────────────────────────────────────────────────


@fieldwise_init
struct TrainResult(Movable & ImplicitlyDestructible):
    var epoch_train_loss: List[Float64]
    var epoch_test_top1:  List[Float64]
    var epoch_train_s:    List[Float64]
    var epoch_eval_s:     List[Float64]

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
](Movable & ImplicitlyDestructible):
    comptime IN_DIM = Self.NET.IN_DIM
    comptime OUT_DIM = Self.NET.OUT_DIM

    var net: Self.NET
    var optim: Self.OPT
    var loss_fn: Self.LOSS

    # CPU side (used when target=="cpu" — length-1 stubs otherwise).
    var input_buf:    UnsafePointer[Scalar[DT], MutAnyOrigin]
    var target_buf:   UnsafePointer[Scalar[DT], MutAnyOrigin]
    var output_buf:   UnsafePointer[Scalar[DT], MutAnyOrigin]
    var grad_out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var grad_in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin]

    # GPU side (Some when target=="gpu", None when "cpu").
    var input_dev:    Optional[DeviceBuffer[DT]]
    var target_dev:   Optional[DeviceBuffer[DT]]
    var output_dev:   Optional[DeviceBuffer[DT]]
    var grad_out_dev: Optional[DeviceBuffer[DT]]
    var grad_in_dev:  Optional[DeviceBuffer[DT]]
    var input_host:   Optional[HostBuffer[DT]]
    var target_host:  Optional[HostBuffer[DT]]
    var output_host:  Optional[HostBuffer[DT]]
    var ctx:          Optional[DeviceContext]

    # ------------------------------------------------------------------
    # Factories — `make[INIT]` builds net/optim/loss internally. The
    # `make_from(...)` overload accepts pre-built components.
    # ------------------------------------------------------------------

    @staticmethod
    def make[INIT: Initializer]() raises -> Self:
        """CPU one-call factory. Builds net + optim + loss internally."""
        comptime assert Self.target == "cpu", (
            "Trainer.make[INIT]() is CPU-only; pass ctx for GPU"
        )
        var net   = Self.NET.make[Self.target, INIT]()
        var loss  = Self.LOSS.make[Self.target]()
        var optim = Self.OPT.make[Self.target](net)
        return Self.make_from(net^, optim^, loss^)

    @staticmethod
    def make[INIT: Initializer](ctx: DeviceContext) raises -> Self:
        """GPU one-call factory."""
        comptime assert Self.target == "gpu", (
            "Trainer.make[INIT](ctx) requires target='gpu'"
        )
        var net   = Self.NET.make[Self.target, INIT](ctx)
        var loss  = Self.LOSS.make[Self.target](ctx)
        var optim = Self.OPT.make[Self.target](net, ctx)
        return Self.make_from(net^, optim^, loss^, ctx)

    @staticmethod
    def make_from(
        var net: Self.NET,
        var optim: Self.OPT,
        var loss_fn: Self.LOSS,
    ) raises -> Self:
        """CPU factory from pre-built components."""
        comptime assert Self.target == "cpu", (
            "Trainer.make_from(net, optim, loss_fn) is CPU-only"
        )
        comptime assert Self.LOSS.OUT_DIM == Self.NET.OUT_DIM, (
            "Trainer: loss N_CLASSES must equal net OUT_DIM"
        )
        var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](Self.BATCH * Self.IN_DIM)
        var tg_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](Self.BATCH * Self.OUT_DIM)
        var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](Self.BATCH * Self.OUT_DIM)
        var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](Self.BATCH * Self.OUT_DIM)
        var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](Self.BATCH * Self.IN_DIM)
        return Self(
            net=net^, optim=optim^, loss_fn=loss_fn^,
            input_buf=in_buf, target_buf=tg_buf, output_buf=out_buf,
            grad_out_buf=go_buf, grad_in_buf=gi_buf,
            input_dev=None, target_dev=None, output_dev=None,
            grad_out_dev=None, grad_in_dev=None,
            input_host=None, target_host=None, output_host=None,
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
        comptime assert Self.target == "gpu", (
            "Trainer.make_from(net, optim, loss_fn, ctx) requires target='gpu'"
        )
        comptime assert Self.LOSS.OUT_DIM == Self.NET.OUT_DIM, (
            "Trainer: loss N_CLASSES must equal net OUT_DIM"
        )
        var in_dev   = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.IN_DIM)
        var tg_dev   = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OUT_DIM)
        var out_dev  = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OUT_DIM)
        var go_dev   = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OUT_DIM)
        var gi_dev   = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.IN_DIM)
        var in_host  = ctx.enqueue_create_host_buffer[DT](Self.BATCH * Self.IN_DIM)
        var tg_host  = ctx.enqueue_create_host_buffer[DT](Self.BATCH * Self.OUT_DIM)
        var out_host = ctx.enqueue_create_host_buffer[DT](Self.BATCH * Self.OUT_DIM)
        ctx.synchronize()
        var stub_in:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
        var stub_tg:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
        var stub_out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
        var stub_go:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
        var stub_gi:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
        return Self(
            net=net^, optim=optim^, loss_fn=loss_fn^,
            input_buf=stub_in, target_buf=stub_tg, output_buf=stub_out,
            grad_out_buf=stub_go, grad_in_buf=stub_gi,
            input_dev=in_dev^, target_dev=tg_dev^, output_dev=out_dev^,
            grad_out_dev=go_dev^, grad_in_dev=gi_dev^,
            input_host=in_host^, target_host=tg_host^, output_host=out_host^,
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

    def _train_step_views[
        LIN: TensorLayout,
        LT: TensorLayout,
        OIN: MutOrigin,
        OT: MutOrigin,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        targets: TileTensor[DT, LT, OT],
    ) raises -> Scalar[DT]:
        comptime assert input.flat_rank   == 2, "input must be rank-2"
        comptime assert targets.flat_rank == 2, "targets must be rank-2"
        comptime if Self.target == "cpu":
            var output   = TileTensor(self.output_buf,   row_major[Self.BATCH, Self.OUT_DIM]())
            var grad_out = TileTensor(self.grad_out_buf, row_major[Self.BATCH, Self.OUT_DIM]())
            var grad_in  = TileTensor(self.grad_in_buf,  row_major[Self.BATCH, Self.IN_DIM]())
            self.optim.zero_grad[Self.target](self.net)
            self.net.forward[Self.target, Self.BATCH](input, output)
            var L = self.loss_fn.forward[Self.target, Self.BATCH](output, targets)
            self.loss_fn.backward[Self.target, Self.BATCH](targets, grad_out)
            self.net.backward[Self.target, Self.BATCH](grad_out, grad_in)
            self.optim.step[Self.target](self.net)
            return L
        else:
            var output   = TileTensor(self.output_dev.value(),   row_major[Self.BATCH, Self.OUT_DIM]())
            var grad_out = TileTensor(self.grad_out_dev.value(), row_major[Self.BATCH, Self.OUT_DIM]())
            var grad_in  = TileTensor(self.grad_in_dev.value(),  row_major[Self.BATCH, Self.IN_DIM]())
            self.optim.zero_grad[Self.target](self.net)
            self.net.forward[Self.target, Self.BATCH](input, output)
            var L = self.loss_fn.forward[Self.target, Self.BATCH](output, targets)
            self.loss_fn.backward[Self.target, Self.BATCH](targets, grad_out)
            self.net.backward[Self.target, Self.BATCH](grad_out, grad_in)
            self.optim.step[Self.target](self.net)
            return L

    # ------------------------------------------------------------------
    # Per-step API.
    # ------------------------------------------------------------------

    def train_step(
        mut self,
        input_host_ptr:  UnsafePointer[Scalar[DT], MutAnyOrigin],
        target_host_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        comptime if Self.target == "cpu":
            for k in range(Self.BATCH * Self.IN_DIM):
                self.input_buf[k] = input_host_ptr[k]
            for k in range(Self.BATCH * Self.OUT_DIM):
                self.target_buf[k] = target_host_ptr[k]
            var input   = TileTensor(self.input_buf,  row_major[Self.BATCH, Self.IN_DIM]())
            var targets = TileTensor(self.target_buf, row_major[Self.BATCH, Self.OUT_DIM]())
            return self._train_step_views(input, targets)
        else:
            var ctx = self.ctx.value()
            var in_host_buf: HostBuffer[DT] = self.input_host.value()
            var tg_host_buf: HostBuffer[DT] = self.target_host.value()
            for k in range(Self.BATCH * Self.IN_DIM):
                in_host_buf.unsafe_ptr()[k] = input_host_ptr[k]
            for k in range(Self.BATCH * Self.OUT_DIM):
                tg_host_buf.unsafe_ptr()[k] = target_host_ptr[k]
            ctx.enqueue_copy(self.input_dev.value(),  in_host_buf)
            ctx.enqueue_copy(self.target_dev.value(), tg_host_buf)
            var input   = TileTensor(self.input_dev.value(),  row_major[Self.BATCH, Self.IN_DIM]())
            var targets = TileTensor(self.target_dev.value(), row_major[Self.BATCH, Self.OUT_DIM]())
            return self._train_step_views(input, targets)

    def predict(
        mut self,
        input_host_ptr:  UnsafePointer[Scalar[DT], MutAnyOrigin],
        output_host_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime if Self.target == "cpu":
            for k in range(Self.BATCH * Self.IN_DIM):
                self.input_buf[k] = input_host_ptr[k]
            var input  = TileTensor(self.input_buf,  row_major[Self.BATCH, Self.IN_DIM]())
            var output = TileTensor(self.output_buf, row_major[Self.BATCH, Self.OUT_DIM]())
            self.net.forward[Self.target, Self.BATCH](input, output)
            for k in range(Self.BATCH * Self.OUT_DIM):
                output_host_ptr[k] = self.output_buf[k]
        else:
            var ctx = self.ctx.value()
            var in_host_buf:  HostBuffer[DT] = self.input_host.value()
            var out_host_buf: HostBuffer[DT] = self.output_host.value()
            for k in range(Self.BATCH * Self.IN_DIM):
                in_host_buf.unsafe_ptr()[k] = input_host_ptr[k]
            ctx.enqueue_copy(self.input_dev.value(), in_host_buf)
            var input  = TileTensor(self.input_dev.value(),  row_major[Self.BATCH, Self.IN_DIM]())
            var output = TileTensor(self.output_dev.value(), row_major[Self.BATCH, Self.OUT_DIM]())
            self.net.forward[Self.target, Self.BATCH](input, output)
            ctx.enqueue_copy(out_host_buf, self.output_dev.value())
            ctx.synchronize()
            for k in range(Self.BATCH * Self.OUT_DIM):
                output_host_ptr[k] = out_host_buf.unsafe_ptr()[k]

    # ------------------------------------------------------------------
    # Whole-dataset GPU training.
    # ------------------------------------------------------------------

    def train_gpu[
        N_TRAIN: Int,
        LXT: TensorLayout, LYT: TensorLayout,
        OXT: MutOrigin, OYT: MutOrigin,
    ](
        mut self,
        train_x: TileTensor[DT, LXT, OXT],
        train_y: TileTensor[DT, LYT, OYT],
        epochs: Int = 1,
        print_progress: Bool = True,
    ) raises -> TrainResult:
        comptime assert Self.target == "gpu", (
            "Trainer.train_gpu requires target='gpu'"
        )
        comptime assert N_TRAIN % Self.BATCH == 0, (
            "Trainer.train_gpu: N_TRAIN must be divisible by BATCH"
        )
        comptime N_BATCHES = N_TRAIN // Self.BATCH

        var result = TrainResult.empty()
        var ctx = self.ctx.value()
        var x_base = train_x.ptr
        var y_base = train_y.ptr

        for epoch in range(epochs):
            var t0 = perf_counter_ns()
            var epoch_loss: Scalar[DT] = 0.0
            for b in range(N_BATCHES):
                var x_ptr = x_base + b * Self.BATCH * Self.IN_DIM
                var y_ptr = y_base + b * Self.BATCH * Self.OUT_DIM
                var input   = TileTensor(x_ptr, row_major[Self.BATCH, Self.IN_DIM]())
                var targets = TileTensor(y_ptr, row_major[Self.BATCH, Self.OUT_DIM]())
                epoch_loss += self._train_step_views(input, targets)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            var train_s = Float64(t1 - t0) / 1e9
            var avg = Float64(epoch_loss / Scalar[DT](N_BATCHES))
            result.epoch_train_loss.append(avg)
            result.epoch_train_s.append(train_s)
            result.epoch_eval_s.append(0.0)
            if print_progress:
                print("epoch " + String(epoch)
                    + " | train_loss=" + String(avg)
                    + " | train=" + String(train_s) + "s")
        return result^

    def train_gpu[
        N_TRAIN: Int, N_TEST: Int,
        LXT: TensorLayout, LYT: TensorLayout, LX2: TensorLayout,
        OXT: MutOrigin, OYT: MutOrigin, OX2: MutOrigin,
    ](
        mut self,
        train_x: TileTensor[DT, LXT, OXT],
        train_y: TileTensor[DT, LYT, OYT],
        test_x:  TileTensor[DT, LX2, OX2],
        test_y_labels: UnsafePointer[Int32, MutAnyOrigin],
        epochs: Int = 1,
        print_progress: Bool = True,
    ) raises -> TrainResult:
        comptime assert Self.target == "gpu", (
            "Trainer.train_gpu requires target='gpu'"
        )
        comptime assert N_TRAIN % Self.BATCH == 0, (
            "Trainer.train_gpu: N_TRAIN must be divisible by BATCH"
        )
        comptime assert N_TEST % Self.BATCH == 0, (
            "Trainer.train_gpu: N_TEST must be divisible by BATCH"
        )
        comptime N_BATCHES_TRAIN = N_TRAIN // Self.BATCH

        var result = TrainResult.empty()
        var ctx = self.ctx.value()
        var x_base = train_x.ptr
        var y_base = train_y.ptr

        for epoch in range(epochs):
            var t0 = perf_counter_ns()
            var epoch_loss: Scalar[DT] = 0.0
            for b in range(N_BATCHES_TRAIN):
                var x_ptr = x_base + b * Self.BATCH * Self.IN_DIM
                var y_ptr = y_base + b * Self.BATCH * Self.OUT_DIM
                var input   = TileTensor(x_ptr, row_major[Self.BATCH, Self.IN_DIM]())
                var targets = TileTensor(y_ptr, row_major[Self.BATCH, Self.OUT_DIM]())
                epoch_loss += self._train_step_views(input, targets)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            var train_s = Float64(t1 - t0) / 1e9

            var top1 = self.eval_top1_gpu[N_TEST](test_x, test_y_labels)
            var t2 = perf_counter_ns()
            var eval_s = Float64(t2 - t1) / 1e9

            var avg = Float64(epoch_loss / Scalar[DT](N_BATCHES_TRAIN))
            result.epoch_train_loss.append(avg)
            result.epoch_test_top1.append(top1)
            result.epoch_train_s.append(train_s)
            result.epoch_eval_s.append(eval_s)
            if print_progress:
                print("epoch " + String(epoch)
                    + " | train_loss=" + String(avg)
                    + " | test_top1=" + String(top1 * 100.0) + "%"
                    + " | train=" + String(train_s) + "s"
                    + " | eval=" + String(eval_s) + "s")
        return result^

    def eval_top1_gpu[
        N_TEST: Int,
        LX: TensorLayout,
        OX: MutOrigin,
    ](
        mut self,
        test_x: TileTensor[DT, LX, OX],
        test_y_labels: UnsafePointer[Int32, MutAnyOrigin],
    ) raises -> Float64:
        comptime assert Self.target == "gpu", (
            "Trainer.eval_top1_gpu requires target='gpu'"
        )
        comptime assert N_TEST % Self.BATCH == 0, (
            "Trainer.eval_top1_gpu: N_TEST must be divisible by BATCH"
        )
        comptime N_BATCHES = N_TEST // Self.BATCH

        var ctx = self.ctx.value()
        var out_host: HostBuffer[DT] = self.output_host.value()
        var x_base = test_x.ptr
        var n_correct: Int = 0
        for b in range(N_BATCHES):
            var x_ptr = x_base + b * Self.BATCH * Self.IN_DIM
            var input  = TileTensor(x_ptr, row_major[Self.BATCH, Self.IN_DIM]())
            var output = TileTensor(self.output_dev.value(), row_major[Self.BATCH, Self.OUT_DIM]())
            self.net.forward[Self.target, Self.BATCH](input, output)
            ctx.enqueue_copy(out_host, self.output_dev.value())
            ctx.synchronize()
            for k in range(Self.BATCH):
                var best_c: Int = 0
                var best_v: Scalar[DT] = out_host.unsafe_ptr()[k * Self.OUT_DIM + 0]
                for c in range(1, Self.OUT_DIM):
                    var v = out_host.unsafe_ptr()[k * Self.OUT_DIM + c]
                    if v > best_v:
                        best_v = v
                        best_c = c
                if best_c == Int(test_y_labels[b * Self.BATCH + k]):
                    n_correct += 1
        return Float64(n_correct) / Float64(N_TEST)
