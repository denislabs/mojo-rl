"""Char-LSTM on TinyShakespeare (CPU).

Port of `examples/nn/lstm/lstm_tinyshakespeare_training.mojo` to nn2 —
the LSTM analog of the conv/resnet examples (Karpathy's char-rnn).

Recurrence uses the nn2 `LSTMCell` step API (the caller owns the (h, c)
state and a per-timestep cache and runs the BPTT loop). The projection
and loss are a plain nn2 `Linear[HIDDEN, VOCAB]` + `CrossEntropyLoss`
applied over the flattened [SEQ·BATCH, …] batch — so per-token CE,
projection forward/backward, and `Adam` all go through stock nn2.
`Adam` trains BOTH the cell and the projection.

Pipeline per training step:
  1. forward T steps: cell.step_forward, stash each h_t row into
     H_all[SEQ·BATCH, HIDDEN] and the cache into cache_buf[t].
  2. logits = proj(H_all);  loss = CE(logits, targets_onehot).
  3. CE.vjp → dlogits;  proj.vjp → dH_all  (per-token dL/dh_t).
  4. BPTT t=SEQ-1..0: dh_t = dH_all[t] + dh_recur, dc_t = dc_recur;
     cell.step_backward accumulates param grads, threads dh/dc back.
  5. Adam on cell + proj (each Adam clips its own grad-norm to CLIP_NORM
     internally via `max_grad_norm`).

Run:
    pixi run mojo run -I . examples/nn2/lstm/lstm_tinyshakespeare_training.mojo
"""

from std.memory import alloc, memset
from std.random import seed, random_float64
from std.math import log
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.lstm_cell import LSTMCell
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.datasets import (
    CharTokenizer,
    load_text,
    train_val_split,
    make_batch,
    to_one_hot,
)


comptime VOCAB = 65
comptime HIDDEN = 128
comptime SEQ = 32
comptime BATCH = 16
comptime NROWS = SEQ * BATCH  # flattened rows for proj + CE
comptime LR: Scalar[DT] = 0.003
comptime N_STEPS = 400
comptime EVAL_EVERY = 50
comptime CLIP_NORM: Scalar[DT] = 5.0

comptime Cell = LSTMCell[VOCAB, HIDDEN]
comptime Proj = Linear[HIDDEN, VOCAB]


def _row(t: Int, b: Int) -> Int:
    # Row of the flattened [SEQ·BATCH] batch for (timestep t, sample b).
    return t * BATCH + b


def train_step(
    mut cell: Cell,
    mut proj: Proj,
    mut ce: CrossEntropyLoss[VOCAB],
    inp_oh: List[Scalar[DT]],  # [BATCH, SEQ*VOCAB]
    target_ids: List[Int],  # [BATCH*SEQ], target_ids[b*SEQ + t]
) raises -> Scalar[DT]:
    cell.zero_grad["cpu"]()
    proj.zero_grad["cpu"]()

    var h_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        (SEQ + 1) * BATCH * HIDDEN
    )
    var c_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        (SEQ + 1) * BATCH * HIDDEN
    )
    var cache_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        SEQ * BATCH * Cell.CACHE_SIZE
    )
    var x_step: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * VOCAB
    )
    var H_all: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        NROWS * HIDDEN
    )
    var logits: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        NROWS * VOCAB
    )
    var targets: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        NROWS * VOCAB
    )
    var dlogits: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        NROWS * VOCAB
    )
    var dH_all: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        NROWS * HIDDEN
    )
    memset(h_buf, 0, (SEQ + 1) * BATCH * HIDDEN)
    memset(c_buf, 0, (SEQ + 1) * BATCH * HIDDEN)
    memset(targets, 0, NROWS * VOCAB)

    # ---- forward over time, collecting per-row hidden states ----
    for t in range(SEQ):
        for b in range(BATCH):
            for v in range(VOCAB):
                x_step[b * VOCAB + v] = inp_oh[b * SEQ * VOCAB + t * VOCAB + v]
        var x_t = TileTensor(x_step, row_major[BATCH, VOCAB]())
        var hp = TileTensor(
            h_buf + t * BATCH * HIDDEN, row_major[BATCH, HIDDEN]()
        )
        var cp = TileTensor(
            c_buf + t * BATCH * HIDDEN, row_major[BATCH, HIDDEN]()
        )
        var ht = TileTensor(
            h_buf + (t + 1) * BATCH * HIDDEN, row_major[BATCH, HIDDEN]()
        )
        var ct = TileTensor(
            c_buf + (t + 1) * BATCH * HIDDEN, row_major[BATCH, HIDDEN]()
        )
        var cc = TileTensor(
            cache_buf + t * BATCH * Cell.CACHE_SIZE,
            row_major[BATCH, Cell.CACHE_SIZE](),
        )
        cell.step_forward["cpu", BATCH](x_t, hp, cp, ht, ct, cc)
        # Scatter h_t rows into the flattened batch + set target one-hot.
        var ht_p = h_buf + (t + 1) * BATCH * HIDDEN
        for b in range(BATCH):
            var r = _row(t, b)
            for j in range(HIDDEN):
                H_all[r * HIDDEN + j] = ht_p[b * HIDDEN + j]
            targets[r * VOCAB + target_ids[b * SEQ + t]] = 1.0

    # ---- projection + per-token CE ----
    var H_all_t = TileTensor(H_all, row_major[NROWS, HIDDEN]())
    var logits_t = TileTensor(logits, row_major[NROWS, VOCAB]())
    var targets_t = TileTensor(targets, row_major[NROWS, VOCAB]())
    proj.forward["cpu", NROWS](H_all_t, output=logits_t)
    var loss = ce.forward["cpu", NROWS](logits_t, targets_t)

    # ---- backward: CE → proj → per-row dH_all ----
    var dlogits_t = TileTensor(dlogits, row_major[NROWS, VOCAB]())
    var dH_all_t = TileTensor(dH_all, row_major[NROWS, HIDDEN]())
    ce.vjp["cpu", NROWS](targets_t, dlogits_t)
    proj.vjp["cpu", NROWS](dlogits_t, dH_all_t)

    # ---- BPTT through the LSTM ----
    var dh_t: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * HIDDEN
    )
    var dc_t: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * HIDDEN
    )
    var dh_recur: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * HIDDEN
    )
    var dc_recur: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * HIDDEN
    )
    var dx_unused: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * VOCAB
    )
    var dh_prev: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * HIDDEN
    )
    var dc_prev: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * HIDDEN
    )
    memset(dh_recur, 0, BATCH * HIDDEN)
    memset(dc_recur, 0, BATCH * HIDDEN)

    for tt in range(SEQ):
        var t = SEQ - 1 - tt
        for b in range(BATCH):
            var r = _row(t, b)
            for j in range(HIDDEN):
                dh_t[b * HIDDEN + j] = (
                    dH_all[r * HIDDEN + j] + dh_recur[b * HIDDEN + j]
                )
                dc_t[b * HIDDEN + j] = dc_recur[b * HIDDEN + j]
        for b in range(BATCH):
            for v in range(VOCAB):
                x_step[b * VOCAB + v] = inp_oh[b * SEQ * VOCAB + t * VOCAB + v]

        var dh_tt = TileTensor(dh_t, row_major[BATCH, HIDDEN]())
        var dc_tt = TileTensor(dc_t, row_major[BATCH, HIDDEN]())
        var x_t = TileTensor(x_step, row_major[BATCH, VOCAB]())
        var hp = TileTensor(
            h_buf + t * BATCH * HIDDEN, row_major[BATCH, HIDDEN]()
        )
        var cp = TileTensor(
            c_buf + t * BATCH * HIDDEN, row_major[BATCH, HIDDEN]()
        )
        var cc = TileTensor(
            cache_buf + t * BATCH * Cell.CACHE_SIZE,
            row_major[BATCH, Cell.CACHE_SIZE](),
        )
        var dx_tt = TileTensor(dx_unused, row_major[BATCH, VOCAB]())
        var dhp_tt = TileTensor(dh_prev, row_major[BATCH, HIDDEN]())
        var dcp_tt = TileTensor(dc_prev, row_major[BATCH, HIDDEN]())
        cell.step_backward["cpu", BATCH](
            dh_tt, dc_tt, x_t, hp, cp, cc, dx_tt, dhp_tt, dcp_tt
        )
        for i in range(BATCH * HIDDEN):
            dh_recur[i] = dh_prev[i]
            dc_recur[i] = dc_prev[i]

    h_buf.free()
    c_buf.free()
    cache_buf.free()
    x_step.free()
    H_all.free()
    logits.free()
    targets.free()
    dlogits.free()
    dH_all.free()
    dh_t.free()
    dc_t.free()
    dh_recur.free()
    dc_recur.free()
    dx_unused.free()
    dh_prev.free()
    dc_prev.free()
    return loss


def eval_loss(
    mut cell: Cell,
    mut proj: Proj,
    mut ce: CrossEntropyLoss[VOCAB],
    val_ids: List[Int],
    n_batches: Int,
) raises -> Float64:
    var total: Float64 = 0.0
    for _ in range(n_batches):
        var batch = make_batch(val_ids, BATCH, SEQ)
        var inp_oh = to_one_hot(batch.inputs, VOCAB, BATCH, SEQ)

        var h_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            BATCH * HIDDEN
        )
        var c_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            BATCH * HIDDEN
        )
        var h_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            BATCH * HIDDEN
        )
        var c_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            BATCH * HIDDEN
        )
        var x_step: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            BATCH * VOCAB
        )
        var H_all: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            NROWS * HIDDEN
        )
        var logits: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            NROWS * VOCAB
        )
        var targets: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](NROWS * VOCAB)
        memset(h_buf, 0, BATCH * HIDDEN)
        memset(c_buf, 0, BATCH * HIDDEN)
        memset(targets, 0, NROWS * VOCAB)

        for t in range(SEQ):
            for b in range(BATCH):
                for v in range(VOCAB):
                    x_step[b * VOCAB + v] = inp_oh[
                        b * SEQ * VOCAB + t * VOCAB + v
                    ]
            var x_t = TileTensor(x_step, row_major[BATCH, VOCAB]())
            var hp = TileTensor(h_buf, row_major[BATCH, HIDDEN]())
            var cp = TileTensor(c_buf, row_major[BATCH, HIDDEN]())
            var ht = TileTensor(h_new, row_major[BATCH, HIDDEN]())
            var ct = TileTensor(c_new, row_major[BATCH, HIDDEN]())
            cell.step_forward_no_cache["cpu", BATCH](x_t, hp, cp, ht, ct)
            for b in range(BATCH):
                var r = _row(t, b)
                for j in range(HIDDEN):
                    H_all[r * HIDDEN + j] = h_new[b * HIDDEN + j]
                targets[r * VOCAB + batch.targets[b * SEQ + t]] = 1.0
            for i in range(BATCH * HIDDEN):
                h_buf[i] = h_new[i]
                c_buf[i] = c_new[i]

        var H_all_t = TileTensor(H_all, row_major[NROWS, HIDDEN]())
        var logits_t = TileTensor(logits, row_major[NROWS, VOCAB]())
        var targets_t = TileTensor(targets, row_major[NROWS, VOCAB]())
        proj.forward["cpu", NROWS](H_all_t, output=logits_t)
        total += Float64(ce.forward["cpu", NROWS](logits_t, targets_t))

        h_buf.free()
        c_buf.free()
        h_new.free()
        c_new.free()
        x_step.free()
        H_all.free()
        logits.free()
        targets.free()
    return total / Float64(n_batches)


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn2 TinyShakespeare char-LSTM (CPU)")
    print("=" * 70)
    print(
        "  vocab="
        + String(VOCAB)
        + " hidden="
        + String(HIDDEN)
        + " seq="
        + String(SEQ)
        + " batch="
        + String(BATCH)
    )

    var text = load_text()
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error("vocab mismatch: got " + String(tok.vocab_size))
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)
    print(
        "  tokens="
        + String(len(ids))
        + " train="
        + String(len(split.train))
        + " val="
        + String(len(split.val))
    )

    var cell = Cell.make[target="cpu", INIT=Xavier]()
    var proj = Proj.make[target="cpu", INIT=Xavier]()
    var ce = CrossEntropyLoss[VOCAB].make["cpu"]()
    # Grad clipping is native to the optimizer now (`Adam.max_grad_norm`):
    # each Adam clips its own module's grad-norm to CLIP_NORM before the step.
    # NB this is per-module (cell, proj clipped independently), not one global
    # norm across both — the cell grads dominate, so it behaves nearly the same.
    var adam_cell = Adam.make["cpu", M=Cell](cell)
    adam_cell.lr = LR
    adam_cell.max_grad_norm = CLIP_NORM
    var adam_proj = Adam.make["cpu", M=Proj](proj)
    adam_proj.lr = LR
    adam_proj.max_grad_norm = CLIP_NORM

    var val0 = eval_loss(cell, proj, ce, split.val, 4)
    print(
        "\n[step 0] val_loss="
        + String(val0)
        + "  (random ≈ ln(V)="
        + String(log(Float64(VOCAB)))
        + ")"
    )

    var run: Scalar[DT] = 0.0
    var cnt = 0
    for step in range(N_STEPS):
        var batch = make_batch(split.train, BATCH, SEQ)
        var inp_oh = to_one_hot(batch.inputs, VOCAB, BATCH, SEQ)
        var loss = train_step(cell, proj, ce, inp_oh, batch.targets)
        adam_cell.step["cpu"](cell)  # clips cell grad-norm internally
        adam_proj.step["cpu"](proj)  # clips proj grad-norm internally
        run += loss
        cnt += 1
        if (step + 1) % EVAL_EVERY == 0:
            var v = eval_loss(cell, proj, ce, split.val, 4)
            print(
                "[step "
                + String(step + 1)
                + "] train_loss="
                + String(run / Scalar[DT](cnt))
                + " val_loss="
                + String(v)
            )
            run = 0.0
            cnt = 0

    var valf = eval_loss(cell, proj, ce, split.val, 4)
    print(
        "\n[final] val_loss=" + String(valf) + " (start " + String(val0) + ")"
    )
    if valf < val0 - 0.5:
        print("  PASS: validation loss dropped > 0.5 nats")
    else:
        print("  WARN: little improvement — raise N_STEPS or tune LR")
    print("=" * 70)
