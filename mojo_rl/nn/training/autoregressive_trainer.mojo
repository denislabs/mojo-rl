"""AutoregressiveTrainer — a task-specialized driver for next-token LM
training, on top of the generic per-step `Trainer`.

The generic `Trainer` packages the *step* (forward → loss → vjp → clip →
optimizer). `Trainer.train_gpu` packages a whole *run* — but only for the
fixed-dataset / epoch / argmax-classification regime (the MLP case). An
autoregressive LM has a different regime: streaming random windows over a
token corpus, a cosine LR schedule, per-token cross-entropy eval, and text
generation. This driver packages *that* regime so the example collapses to

    var artr = AutoregressiveTrainer[NET, OPT, VOCAB, SEQ, BATCH].make_from(
        trainer^, tok^, split^, base_lr, warmup_iters, total_iters,
    )
    # (model-construction surgery on artr.trainer.net here, if any)
    artr.fit(eval_every=250, n_val_windows=256)
    print(artr.generate("ROMEO:", 200, temperature=0.8))

It wraps the regime, not the model: the model / optimizer / loss live in the
inner `Trainer`, and `fit()` delegates each step to `Trainer.train_step`.
Any causal sequence model using the one-hot `[SEQ·VOCAB]` in / logits
`[SEQ·VOCAB]` out convention works (char- or BPE-token, weight-tied or not).
Model-specific init surgery (e.g. the GPT's c_proj scaling / weight-tie
wiring) stays at the call site — this driver doesn't know about it.

GPU-only for now (eval + generation use device buffers). A recurrent model
(LSTM) can't use it until its recurrence is wrapped behind `Module.forward`/
`vjp` — `train_step` drives those, which `LSTMCell` (step-API) doesn't
implement.
"""

from std.math import exp, cos
from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module, Optimizer
from ..core.module import mptr
from ..loss import SequenceCrossEntropyLoss
from ..datasets import CharTokenizer, DatasetSplit, make_batch
from .trainer import Trainer


@fieldwise_init
struct AutoregressiveTrainer[
    NET: Module,
    OPT: Optimizer,
    VOCAB: Int,
    SEQ: Int,
    BATCH: Int,
    target: StaticString = "gpu",
](Movable):
    comptime IN_DIM = Self.SEQ * Self.VOCAB
    comptime OUT_DIM = Self.SEQ * Self.VOCAB
    comptime LOSS = SequenceCrossEntropyLoss[Self.SEQ, Self.VOCAB]
    comptime TRAINER = Trainer[
        Self.NET, Self.OPT, Self.LOSS, Self.BATCH, target = Self.target
    ]

    var trainer: Self.TRAINER
    var tok: CharTokenizer
    var train_ids: List[Int]
    var val_ids: List[Int]
    # Cosine-with-warmup LR schedule (per-iter scale on base_lr).
    var base_lr: Scalar[DT]
    var warmup_iters: Int
    var total_iters: Int
    var min_lr_scale: Float64

    # ----- Factory --------------------------------------------------------

    @staticmethod
    def make_from(
        var trainer: Self.TRAINER,
        var tok: CharTokenizer,
        var split: DatasetSplit,
        base_lr: Scalar[DT],
        warmup_iters: Int,
        total_iters: Int,
        min_lr_scale: Float64 = 0.1,
    ) raises -> Self:
        """Wrap a built `Trainer` + corpus + LR schedule. The model is
        expected to already be in its final home inside `trainer`; do any
        model-construction surgery (tie wiring, scaled init) on
        `result.trainer.net` AFTER this call (the net won't move again)."""
        comptime assert (
            Self.target == "gpu"
        ), "AutoregressiveTrainer is GPU-only for now (eval/gen use device)"
        return Self(
            trainer^, tok^, split.train.copy(), split.val.copy(),
            base_lr, warmup_iters, total_iters, min_lr_scale,
        )

    # ----- Schedule + one-hot helpers ------------------------------------

    def _lr_scale(self, it: Int) -> Scalar[DT]:
        """Linear warmup then cosine decay to `min_lr_scale`."""
        if it < self.warmup_iters:
            return Scalar[DT](Float64(it + 1) / Float64(self.warmup_iters))
        var denom = self.total_iters - self.warmup_iters
        if denom < 1:
            denom = 1
        var prog = Float64(it - self.warmup_iters) / Float64(denom)
        if prog > 1.0:
            prog = 1.0
        var c = 0.5 * (1.0 + cos(3.14159265358979 * prog))
        return Scalar[DT](
            self.min_lr_scale + (1.0 - self.min_lr_scale) * c
        )

    def _one_hot_into(
        self,
        dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ids: List[Int],
        n_rows: Int,
    ):
        """Flat one-hot `[n_rows, SEQ·VOCAB]` from token ids `[n_rows·SEQ]`."""
        for i in range(n_rows * Self.IN_DIM):
            dst[i] = 0.0
        for r in range(n_rows):
            for t in range(Self.SEQ):
                var tid = ids[r * Self.SEQ + t]
                if tid >= 0 and tid < Self.VOCAB:
                    dst[r * Self.IN_DIM + t * Self.VOCAB + tid] = 1.0

    # ----- Eval (per-token CE over device-resident val windows) ----------

    def _eval_loss(
        mut self,
        val_in: UnsafePointer[Scalar[DT], MutAnyOrigin],
        val_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
        out_d: DeviceBuffer[DT],
        n_batches: Int,
    ) raises -> Float64:
        self.trainer.net.set_attr["training"](Scalar[DT](0.0))  # dropout off
        var total: Float64 = 0.0
        for vb in range(n_batches):
            var in_tt = TileTensor(
                val_in + vb * Self.BATCH * Self.IN_DIM,
                row_major[Self.BATCH, Self.IN_DIM](),
            )
            var out_tt = TileTensor(
                mptr(out_d.unsafe_ptr()),
                row_major[Self.BATCH, Self.OUT_DIM](),
            )
            self.trainer.net.forward[Self.target, Self.BATCH](
                in_tt, output=out_tt
            )
            var tgt_tt = TileTensor(
                val_tgt + vb * Self.BATCH * Self.OUT_DIM,
                row_major[Self.BATCH, Self.OUT_DIM](),
            )
            total += Float64(
                self.trainer.loss_fn.forward[Self.target, Self.BATCH](
                    out_tt, tgt_tt
                )
            )
        self.trainer.net.set_attr["training"](Scalar[DT](1.0))
        return total / Float64(n_batches)

    def _upload_windows(
        self,
        ids: List[Int],
        n_windows: Int,
        mut in_d: DeviceBuffer[DT],
        mut tgt_d: DeviceBuffer[DT],
        mut tgt_ids: List[Int],
    ) raises:
        """Sample `n_windows` windows from `ids`, one-hot input+target on the
        host, upload into the caller's `in_d`/`tgt_d` buffers (reassigned to
        correctly-sized allocations). Also returns the raw target ids in
        `tgt_ids` (for top-1)."""
        var ctx = self.trainer.ctx.value()
        var mb = make_batch(ids, n_windows, Self.SEQ)
        var in_h = ctx.enqueue_create_host_buffer[DT](n_windows * Self.IN_DIM)
        var tgt_h = ctx.enqueue_create_host_buffer[DT](
            n_windows * Self.OUT_DIM
        )
        ctx.synchronize()
        self._one_hot_into(mptr(in_h.unsafe_ptr()), mb.inputs, n_windows)
        self._one_hot_into(mptr(tgt_h.unsafe_ptr()), mb.targets, n_windows)
        var ind = ctx.enqueue_create_buffer[DT](n_windows * Self.IN_DIM)
        var tgd = ctx.enqueue_create_buffer[DT](n_windows * Self.OUT_DIM)
        ctx.enqueue_copy(ind, in_h)
        ctx.enqueue_copy(tgd, tgt_h)
        ctx.synchronize()
        in_d = ind^
        tgt_d = tgd^
        tgt_ids = mb.targets.copy()

    def val_loss(mut self, n_windows: Int) raises -> Float64:
        """Mean per-token val CE (nats) over `n_windows` freshly sampled
        windows. `n_windows` must be a multiple of BATCH."""
        if n_windows % Self.BATCH != 0:
            raise Error("val_loss: n_windows must be a multiple of BATCH")
        var ctx = self.trainer.ctx.value()
        var in_d = ctx.enqueue_create_buffer[DT](1)
        var tgt_d = ctx.enqueue_create_buffer[DT](1)
        var tgt_ids = List[Int]()
        self._upload_windows(self.val_ids, n_windows, in_d, tgt_d, tgt_ids)
        var out_d = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OUT_DIM)
        return self._eval_loss(
            mptr(in_d.unsafe_ptr()), mptr(tgt_d.unsafe_ptr()),
            out_d, n_windows // Self.BATCH,
        )

    def val_top1(mut self, n_windows: Int) raises -> Float64:
        """Per-token top-1 argmax accuracy over `n_windows` val windows —
        a diagnostic that the loss is consistent with good next-token
        prediction (not artifactually low)."""
        if n_windows % Self.BATCH != 0:
            raise Error("val_top1: n_windows must be a multiple of BATCH")
        var ctx = self.trainer.ctx.value()
        var in_d = ctx.enqueue_create_buffer[DT](1)
        var tgt_d = ctx.enqueue_create_buffer[DT](1)
        var tgt_ids = List[Int]()
        self._upload_windows(self.val_ids, n_windows, in_d, tgt_d, tgt_ids)
        var in_p = mptr(in_d.unsafe_ptr())
        var out_d = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OUT_DIM)
        var out_h = ctx.enqueue_create_host_buffer[DT](
            Self.BATCH * Self.OUT_DIM
        )
        self.trainer.net.set_attr["training"](Scalar[DT](0.0))
        var correct: Int = 0
        var count: Int = 0
        var oh = out_h.unsafe_ptr()
        var n_batches = n_windows // Self.BATCH
        for vb in range(n_batches):
            var in_tt = TileTensor(
                in_p + vb * Self.BATCH * Self.IN_DIM,
                row_major[Self.BATCH, Self.IN_DIM](),
            )
            var out_tt = TileTensor(
                mptr(out_d.unsafe_ptr()),
                row_major[Self.BATCH, Self.OUT_DIM](),
            )
            self.trainer.net.forward[Self.target, Self.BATCH](
                in_tt, output=out_tt
            )
            ctx.enqueue_copy(out_h, out_d)
            ctx.synchronize()
            for b in range(Self.BATCH):
                for t in range(Self.SEQ):
                    var row = b * Self.OUT_DIM + t * Self.VOCAB
                    var best_v = Float64(oh[row])
                    var best_i = 0
                    for v in range(1, Self.VOCAB):
                        var x = Float64(oh[row + v])
                        if x > best_v:
                            best_v = x
                            best_i = v
                    if best_i == tgt_ids[vb * Self.BATCH * Self.SEQ
                                         + b * Self.SEQ + t]:
                        correct += 1
                    count += 1
        self.trainer.net.set_attr["training"](Scalar[DT](1.0))
        return Float64(correct) / Float64(count)

    # ----- The packaged training run -------------------------------------

    def fit(
        mut self,
        eval_every: Int = 0,
        n_val_windows: Int = 0,
        print_progress: Bool = True,
    ) raises -> Float64:
        """Run `total_iters` of next-token training: each iter samples a
        fresh random window batch, applies the cosine-warmup LR, and runs
        one packaged `train_step`. When `eval_every > 0` and `n_val_windows
        > 0`, reports per-token val CE on a fixed pre-sampled window set every
        `eval_every` iters. Returns the final val loss (or the last train
        loss if eval is off)."""
        var ctx = self.trainer.ctx.value()
        var do_eval = eval_every > 0 and n_val_windows > 0
        if do_eval and n_val_windows % Self.BATCH != 0:
            raise Error("fit: n_val_windows must be a multiple of BATCH")

        # Pre-sample the val windows ONCE so the val curve is comparable
        # across iters.
        var val_in_d = ctx.enqueue_create_buffer[DT](1)
        var val_tgt_d = ctx.enqueue_create_buffer[DT](1)
        var val_tgt_ids = List[Int]()
        var n_val_batches = 0
        if do_eval:
            self._upload_windows(
                self.val_ids, n_val_windows, val_in_d, val_tgt_d, val_tgt_ids
            )
            n_val_batches = n_val_windows // Self.BATCH
        var eval_out_d = ctx.enqueue_create_buffer[DT](
            Self.BATCH * Self.OUT_DIM
        )
        var val_in_p = mptr(val_in_d.unsafe_ptr())
        var val_tgt_p = mptr(val_tgt_d.unsafe_ptr())

        # Reused train staging (host one-hot Lists; train_step owns the H2D).
        var in_list = List[Scalar[DT]](
            length=Self.BATCH * Self.IN_DIM, fill=0.0
        )
        var tgt_list = List[Scalar[DT]](
            length=Self.BATCH * Self.OUT_DIM, fill=0.0
        )
        var in_lp = mptr(in_list.unsafe_ptr())
        var tgt_lp = mptr(tgt_list.unsafe_ptr())

        var last: Float64 = 0.0
        for it in range(self.total_iters):
            self.trainer.optim.set_lr(self.base_lr * self._lr_scale(it))
            var mb = make_batch(self.train_ids, Self.BATCH, Self.SEQ)
            self._one_hot_into(in_lp, mb.inputs, Self.BATCH)
            self._one_hot_into(tgt_lp, mb.targets, Self.BATCH)
            var tl = Float64(self.trainer.train_step(in_list, tgt_list))
            last = tl
            if do_eval and (
                (it + 1) % eval_every == 0 or (it + 1) == self.total_iters
            ):
                var v = self._eval_loss(
                    val_in_p, val_tgt_p, eval_out_d, n_val_batches
                )
                last = v
                if print_progress:
                    print(
                        "  iter " + String(it + 1) + "/"
                        + String(self.total_iters)
                        + "  train=" + String(Float32(tl))
                        + "  val=" + String(Float32(v))
                        + "  lr_scale=" + String(self._lr_scale(it))
                    )
        return last

    # ----- Sampling / generation -----------------------------------------

    def _sample_token(
        self,
        row: UnsafePointer[Scalar[DT], MutAnyOrigin],
        temperature: Float64,
        top_k: Int,
    ) -> Int:
        """nanoGPT-style: greedy if `temperature <= 0`, else top-k softmax."""
        if temperature <= 0.0:
            var bv = Float64(row[0])
            var bi = 0
            for v in range(1, Self.VOCAB):
                if Float64(row[v]) > bv:
                    bv = Float64(row[v])
                    bi = v
            return bi
        var inv_t = 1.0 / temperature
        var scaled = List[Float64](capacity=Self.VOCAB)
        for v in range(Self.VOCAB):
            scaled.append(Float64(row[v]) * inv_t)
        var keep = List[Bool](capacity=Self.VOCAB)
        if top_k > 0 and top_k < Self.VOCAB:
            for _ in range(Self.VOCAB):
                keep.append(False)
            var work = List[Float64](capacity=Self.VOCAB)
            for v in range(Self.VOCAB):
                work.append(scaled[v])
            for _ in range(top_k):
                var bv: Float64 = -1e30
                var bi = 0
                for v in range(Self.VOCAB):
                    if work[v] > bv:
                        bv = work[v]
                        bi = v
                keep[bi] = True
                work[bi] = -1e30
        else:
            for _ in range(Self.VOCAB):
                keep.append(True)
        var m: Float64 = -1e30
        for v in range(Self.VOCAB):
            if keep[v] and scaled[v] > m:
                m = scaled[v]
        var se: Float64 = 0.0
        var exps = List[Float64](capacity=Self.VOCAB)
        for v in range(Self.VOCAB):
            if keep[v]:
                var e = exp(scaled[v] - m)
                exps.append(e)
                se += e
            else:
                exps.append(0.0)
        var u = random_float64(0.0, 1.0) * se
        var acc = 0.0
        for v in range(Self.VOCAB):
            acc += exps[v]
            if u < acc:
                return v
        return Self.VOCAB - 1

    def generate(
        mut self,
        prompt: String,
        n_tokens: Int,
        temperature: Float64 = 0.0,
        top_k: Int = 0,
        pad_id: Int = 0,
    ) raises -> String:
        """Autoregressively continue `prompt` for `n_tokens`. Front-anchored
        window: the last min(n_have, SEQ) ids sit at positions 0.., logits
        read at `n_eff - 1` (causal → the tail pad is invisible). Greedy when
        `temperature <= 0`, else top-k softmax sampling."""
        var ctx = self.trainer.ctx.value()
        self.trainer.net.set_attr["training"](Scalar[DT](0.0))  # eval mode
        var all_ids = self.tok.encode(prompt)
        var prompt_len = len(all_ids)
        if prompt_len == 0:
            raise Error("generate: empty prompt")

        var inp_h = ctx.enqueue_create_host_buffer[DT](Self.IN_DIM)
        var inp_d = ctx.enqueue_create_buffer[DT](Self.IN_DIM)
        var out_d = ctx.enqueue_create_buffer[DT](Self.OUT_DIM)
        var out_h = ctx.enqueue_create_host_buffer[DT](Self.OUT_DIM)
        ctx.synchronize()

        for _gen in range(n_tokens):
            for i in range(Self.IN_DIM):
                inp_h.unsafe_ptr()[i] = 0.0
            var n_have = len(all_ids)
            var n_eff = n_have if n_have <= Self.SEQ else Self.SEQ
            var first = 0 if n_have <= Self.SEQ else n_have - Self.SEQ
            for t in range(Self.SEQ):
                var tid = all_ids[first + t] if t < n_eff else pad_id
                if tid >= 0 and tid < Self.VOCAB:
                    inp_h.unsafe_ptr()[t * Self.VOCAB + tid] = 1.0
            ctx.enqueue_copy(inp_d, inp_h)

            var inp_t = TileTensor(
                mptr(inp_d.unsafe_ptr()), row_major[1, Self.IN_DIM]()
            )
            var out_t = TileTensor(
                mptr(out_d.unsafe_ptr()), row_major[1, Self.OUT_DIM]()
            )
            self.trainer.net.forward[Self.target, 1](inp_t, output=out_t)
            ctx.enqueue_copy(out_h, out_d)
            ctx.synchronize()

            var read_pos = n_eff - 1
            var row_ptr = mptr(out_h.unsafe_ptr()) + read_pos * Self.VOCAB
            all_ids.append(self._sample_token(row_ptr, temperature, top_k))

        var gen = List[Int](capacity=n_tokens)
        for i in range(prompt_len, len(all_ids)):
            gen.append(all_ids[i])
        self.trainer.net.set_attr["training"](Scalar[DT](1.0))
        return self.tok.decode(gen)
