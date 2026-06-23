"""AutoregressiveTrainer — a task-specialized driver for next-token LM
training on the **storage** surface (CPU `List` / GPU `DeviceBuffer` `Tensor`s,
`TensorRefs` input packs).

The storage `Trainer` (`trainer.mojo`) is classification-specialized
(`CrossEntropyLoss[NC]` + `train_epoch`/`eval_top1` over a flat resident
dataset), so — unlike the legacy `AutoregressiveTrainer`, which *wrapped* a
generic per-step `Trainer` — this driver is **self-contained**: it owns the
`net` / `opt` / `SequenceCrossEntropyLoss` / scratch `Tensor`s and runs its own
per-step loop (forward → SeqCE loss → vjp → grad-clip → optimizer step) over
streaming random windows, with a cosine-with-warmup LR schedule, per-token CE
eval, and text generation. The example collapses to

    var artr = AutoregressiveTrainer[NET, OPT, VOCAB, SEQ, BATCH].make_from(
        net^, opt^, tok^, split^, ctx, base_lr, warmup_iters, total_iters,
        grad_clip=1.0,
    )
    # model-construction surgery on artr.net here (scaled-init / tie wiring)
    artr.fit(eval_every=250, n_val_windows=256)
    print(artr.generate("ROMEO:", 200, temperature=0.8))

It wraps the *regime*, not the model: the model / optimizer / loss live as
fields, and `make_from` engages the optimizer's contiguous arena (`opt.adopt`)
once `net` is in its final home. Any causal sequence model using the one-hot
`[SEQ·VOCAB]` in / logits `[SEQ·VOCAB]` out convention works (char- or BPE-
token, weight-tied or not). Model-specific init surgery (e.g. the GPT's c_proj
scaling / weight-tie wiring) stays at the call site — this driver doesn't know
about it.

GPU-only for now (eval + generation use device buffers). A recurrent model
(LSTM) can't use it until its recurrence is wrapped behind `Module.forward`/
`vjp` — the per-step loop drives those.
"""

from std.math import exp, cos
from std.random import random_float64
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.module import Module
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..optimizer.optimizer import Optimizer
from ..optimizer.adam import Adam
from ..loss.sequence_cross_entropy import SequenceCrossEntropyLoss
from mojo_rl.nn.datasets import CharTokenizer, DatasetSplit, make_batch


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

    var net: Self.NET
    var opt: Self.OPT
    var loss: Self.LOSS
    var ctx: DeviceContext
    # Reused per-step staging (host one-hot Lists + device buffers, lazy).
    var in_t: Tensor
    var tgt_t: Tensor
    var logits: Tensor
    var grad: Tensor
    var gi: Tensor
    var tok: CharTokenizer
    var train_ids: List[Int]
    var val_ids: List[Int]
    # Cosine-with-warmup LR schedule (per-iter scale on base_lr).
    var base_lr: Scalar[DT]
    var warmup_iters: Int
    var total_iters: Int
    var min_lr_scale: Float64
    var grad_clip: Scalar[DT]

    def __init__(
        out self,
        var net: Self.NET,
        var opt: Self.OPT,
        var loss: Self.LOSS,
        ctx: DeviceContext,
        var tok: CharTokenizer,
        var train_ids: List[Int],
        var val_ids: List[Int],
        base_lr: Scalar[DT],
        warmup_iters: Int,
        total_iters: Int,
        min_lr_scale: Float64,
        grad_clip: Scalar[DT],
    ):
        self.net = net^
        self.opt = opt^
        self.loss = loss^
        self.ctx = ctx
        self.in_t = Tensor()
        self.tgt_t = Tensor()
        self.logits = Tensor()
        self.grad = Tensor()
        self.gi = Tensor()
        self.tok = tok^
        self.train_ids = train_ids^
        self.val_ids = val_ids^
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr_scale = min_lr_scale
        self.grad_clip = grad_clip

    # ----- Factory --------------------------------------------------------

    @staticmethod
    def make_from(
        var net: Self.NET,
        var opt: Self.OPT,
        var tok: CharTokenizer,
        var split: DatasetSplit,
        ctx: DeviceContext,
        base_lr: Scalar[DT],
        warmup_iters: Int,
        total_iters: Int,
        min_lr_scale: Float64 = 0.1,
        grad_clip: Scalar[DT] = 0.0,
    ) raises -> Self:
        """Wrap a built `net` + `opt` + corpus + LR schedule. The model is
        expected to already be in its final home; do any model-construction
        surgery (tie wiring, scaled init) on `result.net` AFTER this call (the
        net won't move again). Engages the optimizer's contiguous arena
        (`opt.adopt`) here so the GPU step is a single grouped kernel and
        `clip_grads` runs on the arena (capture-safe, no per-param D2H)."""
        comptime assert (
            Self.target == "gpu"
        ), "AutoregressiveTrainer is GPU-only for now (eval/gen use device)"
        var loss = Self.LOSS.make_gpu(ctx)
        var s = Self(
            net^, opt^, loss^, ctx, tok^,
            split.train.copy(), split.val.copy(),
            base_lr, warmup_iters, total_iters, min_lr_scale, grad_clip,
        )
        # Arena adopt AFTER net is in its final home (self.net): rebinds the
        # model's param buffers to arena slices in place, so any subsequent
        # tie-wiring (which points the head at the embedding `Tensor` cells)
        # and scaled-init surgery operate on the arena-backed buffers.
        s.opt.adopt[Self.target](s.net, Optional(s.ctx))
        return s^

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

    @staticmethod
    def _upload_onehot(
        ctx: DeviceContext, mut t: Tensor, ids: List[Int], n_rows: Int
    ) raises:
        """Host one-hot `[n_rows, SEQ·VOCAB]` from token ids `[n_rows·SEQ]`,
        then H2D upload into `t` (device buffer reallocated to fit). Static
        (takes `ctx` by value) so call sites can pass `self.in_t`/`self.tgt_t`
        as `mut t` without aliasing the whole `self`."""
        var total = n_rows * Self.IN_DIM
        t.ensure(total)
        for i in range(total):
            t.data[i] = Scalar[DT](0)
        for r in range(n_rows):
            for tt in range(Self.SEQ):
                var tid = ids[r * Self.SEQ + tt]
                if tid >= 0 and tid < Self.VOCAB:
                    t.data[r * Self.IN_DIM + tt * Self.VOCAB + tid] = (
                        Scalar[DT](1)
                    )
        t.n = total
        t.upload(ctx)

    # ----- Eval (per-token CE / top-1 over freshly sampled val windows) ---

    def _eval_loss_ids(
        mut self, in_ids: List[Int], tgt_ids: List[Int], n_batches: Int
    ) raises -> Float64:
        """Mean per-token CE (nats) over `n_batches` batches of pre-sampled
        window ids (`[n_batches·BATCH·SEQ]` flat)."""
        self.net.set_attr["training"](Scalar[DT](0.0))  # dropout off
        var total: Float64 = 0.0
        var stride = Self.BATCH * Self.SEQ
        for vb in range(n_batches):
            var bi = List[Int](capacity=stride)
            var bt = List[Int](capacity=stride)
            for k in range(stride):
                bi.append(in_ids[vb * stride + k])
                bt.append(tgt_ids[vb * stride + k])
            Self._upload_onehot(self.ctx, self.in_t, bi, Self.BATCH)
            Self._upload_onehot(self.ctx, self.tgt_t, bt, Self.BATCH)
            self.net.forward[Self.target, Self.BATCH](
                TensorRefs[Self.NET.ARITY](self.in_t),
                self.logits,
                Optional(self.ctx),
            )
            total += Float64(
                self.loss.forward[Self.target, Self.BATCH](
                    self.logits, self.tgt_t, Optional(self.ctx)
                )
            )
        self.net.set_attr["training"](Scalar[DT](1.0))
        return total / Float64(n_batches)

    def val_loss(mut self, n_windows: Int) raises -> Float64:
        """Mean per-token val CE (nats) over `n_windows` freshly sampled
        windows. `n_windows` must be a multiple of BATCH."""
        if n_windows % Self.BATCH != 0:
            raise Error("val_loss: n_windows must be a multiple of BATCH")
        var mb = make_batch(self.val_ids, n_windows, Self.SEQ)
        return self._eval_loss_ids(
            mb.inputs, mb.targets, n_windows // Self.BATCH
        )

    def val_top1(mut self, n_windows: Int) raises -> Float64:
        """Per-token top-1 argmax accuracy over `n_windows` val windows — a
        diagnostic that the loss is consistent with good next-token prediction
        (not artifactually low)."""
        if n_windows % Self.BATCH != 0:
            raise Error("val_top1: n_windows must be a multiple of BATCH")
        var mb = make_batch(self.val_ids, n_windows, Self.SEQ)
        self.net.set_attr["training"](Scalar[DT](0.0))
        var correct: Int = 0
        var count: Int = 0
        var stride = Self.BATCH * Self.SEQ
        var n_batches = n_windows // Self.BATCH
        for vb in range(n_batches):
            var bi = List[Int](capacity=stride)
            for k in range(stride):
                bi.append(mb.inputs[vb * stride + k])
            Self._upload_onehot(self.ctx, self.in_t, bi, Self.BATCH)
            self.net.forward[Self.target, Self.BATCH](
                TensorRefs[Self.NET.ARITY](self.in_t),
                self.logits,
                Optional(self.ctx),
            )
            self.logits.download(self.ctx)
            for b in range(Self.BATCH):
                for t in range(Self.SEQ):
                    var row = b * Self.OUT_DIM + t * Self.VOCAB
                    var best_v = Float64(self.logits.data[row])
                    var best_i = 0
                    for v in range(1, Self.VOCAB):
                        var x = Float64(self.logits.data[row + v])
                        if x > best_v:
                            best_v = x
                            best_i = v
                    if best_i == mb.targets[vb * stride + b * Self.SEQ + t]:
                        correct += 1
                    count += 1
        self.net.set_attr["training"](Scalar[DT](1.0))
        return Float64(correct) / Float64(count)

    # ----- One packaged training step ------------------------------------

    def _train_step(mut self, it: Int) raises -> Float64:
        """Sample a fresh random window batch, apply the cosine-warmup LR, and
        run one forward → SeqCE → vjp → grad-clip → optimizer step. Returns the
        (pre-step) train CE for this batch."""
        self.opt.set_lr(self.base_lr * self._lr_scale(it))
        var mb = make_batch(self.train_ids, Self.BATCH, Self.SEQ)
        Self._upload_onehot(self.ctx, self.in_t, mb.inputs, Self.BATCH)
        Self._upload_onehot(self.ctx, self.tgt_t, mb.targets, Self.BATCH)
        self.opt.zero_grad[Self.target](self.net, Optional(self.ctx))
        self.net.forward[Self.target, Self.BATCH](
            TensorRefs[Self.NET.ARITY](self.in_t),
            self.logits,
            Optional(self.ctx),
        )
        var tl = Float64(
            self.loss.forward[Self.target, Self.BATCH](
                self.logits, self.tgt_t, Optional(self.ctx)
            )
        )
        self.loss.vjp[Self.target, Self.BATCH](
            self.logits, self.tgt_t, self.grad, Optional(self.ctx)
        )
        self.net.vjp[Self.target, Self.BATCH](
            TensorRefs[Self.NET.ARITY](self.in_t),
            self.grad,
            TensorRefs[Self.NET.ARITY](self.gi),
            Optional(self.ctx),
        )
        if self.grad_clip > Scalar[DT](0.0):
            _ = self.opt.clip_grads[Self.target](
                self.net, self.grad_clip, Optional(self.ctx)
            )
        self.opt.step[Self.target](self.net, Optional(self.ctx))
        return tl

    # ----- The packaged training run -------------------------------------

    def fit(
        mut self,
        eval_every: Int = 0,
        n_val_windows: Int = 0,
        print_progress: Bool = True,
    ) raises -> Float64:
        """Run `total_iters` of next-token training. When `eval_every > 0` and
        `n_val_windows > 0`, reports per-token val CE on a fixed pre-sampled
        window set every `eval_every` iters. Returns the final val loss (or the
        last train loss if eval is off)."""
        var do_eval = eval_every > 0 and n_val_windows > 0
        if do_eval and n_val_windows % Self.BATCH != 0:
            raise Error("fit: n_val_windows must be a multiple of BATCH")
        # Pre-sample the val windows ONCE so the val curve is comparable across
        # iters.
        var val_mb = make_batch(
            self.val_ids, n_val_windows if do_eval else Self.BATCH, Self.SEQ
        )
        var n_val_batches = n_val_windows // Self.BATCH

        var last: Float64 = 0.0
        for it in range(self.total_iters):
            var tl = self._train_step(it)
            last = tl
            if do_eval and (
                (it + 1) % eval_every == 0 or (it + 1) == self.total_iters
            ):
                var v = self._eval_loss_ids(
                    val_mb.inputs, val_mb.targets, n_val_batches
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

    @staticmethod
    def _sample_token(
        ref logits: Tensor, base: Int, temperature: Float64, top_k: Int
    ) -> Int:
        """nanoGPT-style: greedy if `temperature <= 0`, else top-k softmax over
        `logits.data[base : base+VOCAB]`."""
        if temperature <= 0.0:
            var bv = Float64(logits.data[base])
            var bi = 0
            for v in range(1, Self.VOCAB):
                if Float64(logits.data[base + v]) > bv:
                    bv = Float64(logits.data[base + v])
                    bi = v
            return bi
        var inv_t = 1.0 / temperature
        var scaled = List[Float64](capacity=Self.VOCAB)
        for v in range(Self.VOCAB):
            scaled.append(Float64(logits.data[base + v]) * inv_t)
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
        window: the last min(n_have, SEQ) ids sit at positions 0.., logits read
        at `n_eff - 1` (causal → the tail pad is invisible). Greedy when
        `temperature <= 0`, else top-k softmax sampling."""
        self.net.set_attr["training"](Scalar[DT](0.0))  # eval mode
        var all_ids = self.tok.encode(prompt)
        var prompt_len = len(all_ids)
        if prompt_len == 0:
            raise Error("generate: empty prompt")

        # Dedicated B=1 scratch — the reused `self.in_t`/`self.logits` carry
        # BATCH-sized device buffers from training; a B=1 forward would leave a
        # stale length vs the reallocated buffer (Metal copy overflow).
        var gen_in = Tensor()
        var gen_out = Tensor()
        for _gen in range(n_tokens):
            var n_have = len(all_ids)
            var n_eff = n_have if n_have <= Self.SEQ else Self.SEQ
            var first = 0 if n_have <= Self.SEQ else n_have - Self.SEQ
            var win = List[Int](capacity=Self.SEQ)
            for t in range(Self.SEQ):
                win.append(all_ids[first + t] if t < n_eff else pad_id)
            Self._upload_onehot(self.ctx, gen_in, win, 1)
            self.net.forward[Self.target, 1](
                TensorRefs[Self.NET.ARITY](gen_in),
                gen_out,
                Optional(self.ctx),
            )
            gen_out.download(self.ctx)
            var read_pos = n_eff - 1
            all_ids.append(
                Self._sample_token(
                    gen_out, read_pos * Self.VOCAB, temperature, top_k
                )
            )

        var gen = List[Int](capacity=n_tokens)
        for i in range(prompt_len, len(all_ids)):
            gen.append(all_ids[i])
        self.net.set_attr["training"](Scalar[DT](1.0))
        return self.tok.decode(gen)
