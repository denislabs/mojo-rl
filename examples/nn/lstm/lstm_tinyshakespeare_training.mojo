"""TinyShakespeare char-LSTM training (CPU smoke test).

Validates `LSTMCell` end-to-end on real data: char-level language modeling
on Shakespeare. This is the LSTM analog of the GPT TinyShakespeare smoke
test at `examples/nn/transformer/gpt_tinyshakespeare_training.mojo`, and
the LSTM analog of CIFAR for CNNs — the canonical "does this layer
actually learn?" experiment, originally Karpathy's char-rnn (2015).

Architecture (small, CPU-friendly):
    one-hot[VOCAB] --> LSTMCell[VOCAB, HIDDEN] --> Linear[HIDDEN, VOCAB] --> CE
                                  |
                          (h_t, c_t threaded over SEQ time steps,
                           h_0=c_0=0 at sequence start)

The LSTM input is the one-hot character vector — no separate embedding
layer. The first matmul inside LSTMCell (W_ih, [VOCAB, 4*HIDDEN]) acts as
a learned input embedding for each character.

Backward is BPTT: starting from t=SEQ-1 down to t=0, threading dh/dc
back through `LSTMCell.step_backward`. Param grads ACCUMULATE across all
T steps (the multi-call accumulation hazard from
project_autodiff_multicall_accumulation.md — `LSTMCell.step_backward`
is designed for this).

Goal: validation loss should drop from ~ln(65)≈4.17 (random) to roughly
2.0–2.5 in 200 steps with a 64-dim hidden cell. Greedy/temperature
sampling at the end gives a qualitative "does it look text-like?" check.

Run:
    pixi run mojo run -I . examples/nn/lstm/lstm_tinyshakespeare_training.mojo
"""

from std.random import seed, random_float64
from std.math import cos, log, exp, sqrt
from std.memory import alloc, memset, UnsafePointer

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import LSTMCell
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.datasets import (
    CharTokenizer,
    load_text,
    train_val_split,
    make_batch,
    to_one_hot,
)


# =============================================================================
# Hyperparameters (tiny config for CPU smoke)
# =============================================================================
comptime VOCAB = 65
comptime HIDDEN = 128
comptime SEQ = 64
comptime BATCH = 8

comptime LR_LSTM = 3e-3
comptime LR_PROJ = 3e-3
comptime BETA1 = 0.9
comptime BETA2 = 0.999
comptime EPS = 1e-8

comptime N_STEPS = 1000
comptime WARMUP_STEPS = 20
comptime PRINT_EVERY = 20
comptime EVAL_EVERY = 50
comptime EVAL_BATCHES = 4

comptime GRAD_CLIP_NORM = 5.0  # Karpathy char-rnn default.


# =============================================================================
# Aliases
# =============================================================================
comptime Cell = LSTMCell[VOCAB, HIDDEN]
comptime LSTM_PS = Cell.PARAM_SIZE
comptime LSTM_CS = Cell.CACHE_SIZE
comptime PROJ_W = HIDDEN * VOCAB
comptime PROJ_B = VOCAB
comptime PROJ_PS = PROJ_W + PROJ_B


# =============================================================================
# LR schedule: linear warmup then cosine decay to 10% of peak
# =============================================================================
def lr_scale(step: Int, warmup: Int, total: Int) -> Float64:
    if step < warmup:
        return Float64(step + 1) / Float64(warmup)
    var progress = Float64(step - warmup) / Float64(max(1, total - warmup))
    if progress > 1.0:
        progress = 1.0
    var c = 0.5 * (1.0 + cos(progress * 3.141592653589793))
    return 0.1 + 0.9 * c


# =============================================================================
# Output projection (Linear[HIDDEN, VOCAB]) — managed inline.
#
# Layout: params = [W_proj (HIDDEN * VOCAB, row-major) | b_proj (VOCAB)].
# =============================================================================
@always_inline
def proj_forward(
    h: LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    p: LayoutTensor[dtype, Layout.row_major(PROJ_PS), MutAnyOrigin],
    mut logits: LayoutTensor[
        dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
    ],
):
    for b in range(BATCH):
        for v in range(VOCAB):
            var acc = Float64(0.0)
            for j in range(HIDDEN):
                acc += Float64(rebind[Scalar[dtype]](h[b, j])) * Float64(
                    rebind[Scalar[dtype]](p[j * VOCAB + v])
                )
            acc += Float64(rebind[Scalar[dtype]](p[PROJ_W + v]))
            logits[b, v] = Scalar[dtype](acc)


@always_inline
def proj_backward(
    h: LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    p: LayoutTensor[dtype, Layout.row_major(PROJ_PS), MutAnyOrigin],
    dlogits: LayoutTensor[dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin],
    mut dh: LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    mut dp: LayoutTensor[dtype, Layout.row_major(PROJ_PS), MutAnyOrigin],
):
    """Compute dh += dlogits @ W^T, accumulate dW and db."""
    # dh[b, j] += sum_v dlogits[b, v] * W[j, v]
    for b in range(BATCH):
        for j in range(HIDDEN):
            var acc = Float64(0.0)
            for v in range(VOCAB):
                acc += Float64(rebind[Scalar[dtype]](dlogits[b, v])) * Float64(
                    rebind[Scalar[dtype]](p[j * VOCAB + v])
                )
            var prev = Float64(rebind[Scalar[dtype]](dh[b, j]))
            dh[b, j] = Scalar[dtype](prev + acc)

    # dW[j, v] += sum_b h[b, j] * dlogits[b, v]
    for j in range(HIDDEN):
        for v in range(VOCAB):
            var acc = Float64(0.0)
            for b in range(BATCH):
                acc += Float64(rebind[Scalar[dtype]](h[b, j])) * Float64(
                    rebind[Scalar[dtype]](dlogits[b, v])
                )
            var prev = Float64(rebind[Scalar[dtype]](dp[j * VOCAB + v]))
            dp[j * VOCAB + v] = Scalar[dtype](prev + acc)

    # db[v] += sum_b dlogits[b, v]
    for v in range(VOCAB):
        var acc = Float64(0.0)
        for b in range(BATCH):
            acc += Float64(rebind[Scalar[dtype]](dlogits[b, v]))
        var prev = Float64(rebind[Scalar[dtype]](dp[PROJ_W + v]))
        dp[PROJ_W + v] = Scalar[dtype](prev + acc)


# =============================================================================
# CE forward + backward (per-step), averaged over BATCH * SEQ at the end
# =============================================================================
@always_inline
def ce_forward_step(
    logits: LayoutTensor[dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin],
    target_ids: List[Int],
    seq_offset: Int,
) -> Float64:
    """CE loss summed over BATCH at this time step (no normalization)."""
    var s = Float64(0.0)
    for b in range(BATCH):
        var max_v = Float64(rebind[Scalar[dtype]](logits[b, 0]))
        for v in range(1, VOCAB):
            var x = Float64(rebind[Scalar[dtype]](logits[b, v]))
            if x > max_v:
                max_v = x
        var sum_exp = Float64(0.0)
        for v in range(VOCAB):
            sum_exp += exp(Float64(rebind[Scalar[dtype]](logits[b, v])) - max_v)
        var lse = max_v + log(sum_exp)
        var tid = target_ids[b * SEQ + seq_offset]
        var lp = Float64(rebind[Scalar[dtype]](logits[b, tid])) - lse
        s += -lp
    return s


@always_inline
def ce_backward_step(
    logits: LayoutTensor[dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin],
    target_ids: List[Int],
    seq_offset: Int,
    mut dlogits: LayoutTensor[
        dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
    ],
    norm: Float64,
):
    """dlogits = (softmax(logits) - one_hot(target)) / norm."""
    for b in range(BATCH):
        var max_v = Float64(rebind[Scalar[dtype]](logits[b, 0]))
        for v in range(1, VOCAB):
            var x = Float64(rebind[Scalar[dtype]](logits[b, v]))
            if x > max_v:
                max_v = x
        var sum_exp = Float64(0.0)
        for v in range(VOCAB):
            sum_exp += exp(Float64(rebind[Scalar[dtype]](logits[b, v])) - max_v)
        var tid = target_ids[b * SEQ + seq_offset]
        for v in range(VOCAB):
            var sm = (
                exp(Float64(rebind[Scalar[dtype]](logits[b, v])) - max_v)
                / sum_exp
            )
            var t = Float64(1.0) if v == tid else Float64(0.0)
            dlogits[b, v] = Scalar[dtype]((sm - t) / norm)


# =============================================================================
# Adam state (per-buffer, flat layout)
# =============================================================================
struct AdamState(Movable):
    """Plain Adam (m, v) state for a flat parameter buffer.

    Lives on CPU as two `[PS]`-sized List buffers. `step_num` is host-side.
    """

    var m: List[Scalar[dtype]]
    var v: List[Scalar[dtype]]
    var step_num: Int

    def __init__(out self, ps: Int):
        self.m = List[Scalar[dtype]](capacity=ps)
        self.v = List[Scalar[dtype]](capacity=ps)
        for _ in range(ps):
            self.m.append(0)
            self.v.append(0)
        self.step_num = 0

    def __init__(out self, *, deinit take: Self):
        self.m = take.m^
        self.v = take.v^
        self.step_num = take.step_num


def adam_step(
    mut params: List[Scalar[dtype]],
    grads: List[Scalar[dtype]],
    mut st: AdamState,
    base_lr: Float64,
    lr_mul: Float64,
):
    """In-place Adam update on flat List buffers."""
    st.step_num += 1
    var bc1 = 1.0 - (BETA1**st.step_num)
    var bc2 = 1.0 - (BETA2**st.step_num)
    var lr = base_lr * lr_mul
    for i in range(len(params)):
        var g = Float64(grads[i])
        var m = Float64(st.m[i])
        var v = Float64(st.v[i])
        m = BETA1 * m + (1.0 - BETA1) * g
        v = BETA2 * v + (1.0 - BETA2) * g * g
        st.m[i] = Scalar[dtype](m)
        st.v[i] = Scalar[dtype](v)
        var m_hat = m / bc1
        var v_hat = v / bc2
        params[i] = Scalar[dtype](
            Float64(params[i]) - lr * m_hat / (sqrt(v_hat) + EPS)
        )


# =============================================================================
# Gradient clipping (global norm)
# =============================================================================
def clip_grads_global_norm(
    mut g_lstm: List[Scalar[dtype]],
    mut g_proj: List[Scalar[dtype]],
    max_norm: Float64,
) -> Float64:
    """Karpathy-style global L2 clip across both grad buffers."""
    var sq = Float64(0.0)
    for i in range(len(g_lstm)):
        var v = Float64(g_lstm[i])
        sq += v * v
    for i in range(len(g_proj)):
        var v = Float64(g_proj[i])
        sq += v * v
    var nrm = sqrt(sq)
    if nrm > max_norm:
        var scale = max_norm / nrm
        for i in range(len(g_lstm)):
            g_lstm[i] = Scalar[dtype](Float64(g_lstm[i]) * scale)
        for i in range(len(g_proj)):
            g_proj[i] = Scalar[dtype](Float64(g_proj[i]) * scale)
    return nrm


# =============================================================================
# Forward + backward over one minibatch (full BPTT)
# Returns per-token CE loss (in nats).
# =============================================================================
def train_step(
    mut lstm_params: List[Scalar[dtype]],
    mut proj_params: List[Scalar[dtype]],
    mut lstm_grads: List[Scalar[dtype]],
    mut proj_grads: List[Scalar[dtype]],
    inp_oh: List[Scalar[dtype]],  # [BATCH * SEQ * VOCAB]
    target_ids: List[Int],  # [BATCH * SEQ]
) raises -> Float64:
    # Per-step buffers
    var h_buf = alloc[Scalar[dtype]]((SEQ + 1) * BATCH * HIDDEN)
    var c_buf = alloc[Scalar[dtype]]((SEQ + 1) * BATCH * HIDDEN)
    var cache_buf = alloc[Scalar[dtype]](SEQ * BATCH * LSTM_CS)
    var logits_buf = alloc[Scalar[dtype]](SEQ * BATCH * VOCAB)
    memset(h_buf, 0, (SEQ + 1) * BATCH * HIDDEN)
    memset(c_buf, 0, (SEQ + 1) * BATCH * HIDDEN)
    memset(cache_buf, 0, SEQ * BATCH * LSTM_CS)
    memset(logits_buf, 0, SEQ * BATCH * VOCAB)

    var lstm_p_ptr = lstm_params.unsafe_ptr()
    var proj_p_ptr = proj_params.unsafe_ptr()
    var lstm_p = LayoutTensor[dtype, Layout.row_major(LSTM_PS), MutAnyOrigin](
        lstm_p_ptr
    )
    var proj_p = LayoutTensor[dtype, Layout.row_major(PROJ_PS), MutAnyOrigin](
        proj_p_ptr
    )

    # ---- forward: gather x_t, run LSTM step, projection, accumulate loss ----
    # Note: to_one_hot returns flat [BATCH, SEQ * VOCAB] row-major. The
    # [BATCH, VOCAB] slice at a given time t is NOT contiguous (samples
    # are strided by SEQ*VOCAB), so we explicitly gather it into a
    # contiguous buffer per step.
    var total_loss_sum = Float64(0.0)
    var x_step_buf = alloc[Scalar[dtype]](BATCH * VOCAB)
    for t in range(SEQ):
        # Gather one-hot inputs at time t into a contiguous [BATCH, VOCAB] buffer.
        for b in range(BATCH):
            for v in range(VOCAB):
                x_step_buf[b * VOCAB + v] = inp_oh[
                    b * SEQ * VOCAB + t * VOCAB + v
                ]
        var x_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
        ](x_step_buf)
        var hp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](h_buf + t * BATCH * HIDDEN)
        var cp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](c_buf + t * BATCH * HIDDEN)
        var ht_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](h_buf + (t + 1) * BATCH * HIDDEN)
        var ct_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](c_buf + (t + 1) * BATCH * HIDDEN)
        var cc_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LSTM_CS), MutAnyOrigin
        ](cache_buf + t * BATCH * LSTM_CS)

        Cell.step_forward[BATCH](x_t, hp_t, cp_t, lstm_p, ht_t, ct_t, cc_t)

        # Projection: logits[t] = h_t @ W_proj + b_proj
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
        ](logits_buf + t * BATCH * VOCAB)
        proj_forward(ht_t, proj_p, logits_t)

        total_loss_sum += ce_forward_step(logits_t, target_ids, t)

    var ntoken = Float64(BATCH * SEQ)
    var avg_loss = total_loss_sum / ntoken

    # ---------------- Backward over time ----------------
    # Zero grad buffers first
    for i in range(len(lstm_grads)):
        lstm_grads[i] = 0
    for i in range(len(proj_grads)):
        proj_grads[i] = 0

    var lstm_g_ptr = lstm_grads.unsafe_ptr()
    var proj_g_ptr = proj_grads.unsafe_ptr()
    var lstm_g = LayoutTensor[dtype, Layout.row_major(LSTM_PS), MutAnyOrigin](
        lstm_g_ptr
    )
    var proj_g = LayoutTensor[dtype, Layout.row_major(PROJ_PS), MutAnyOrigin](
        proj_g_ptr
    )

    var dlogits_buf = alloc[Scalar[dtype]](BATCH * VOCAB)
    var dh_state = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var dc_state = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var dx_unused = alloc[Scalar[dtype]](BATCH * VOCAB)
    var dh_prev = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var dc_prev = alloc[Scalar[dtype]](BATCH * HIDDEN)
    memset(dh_state, 0, BATCH * HIDDEN)
    memset(dc_state, 0, BATCH * HIDDEN)

    for tt in range(SEQ):
        var t = SEQ - 1 - tt

        var dlogits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
        ](dlogits_buf)
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
        ](logits_buf + t * BATCH * VOCAB)
        ce_backward_step(logits_t, target_ids, t, dlogits_t, ntoken)

        # Projection backward — adds dh contribution from current step's logits
        # to dh_state (which has accumulated dh from t+1's LSTM step).
        var ht_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](h_buf + (t + 1) * BATCH * HIDDEN)
        var dh_state_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](dh_state)
        proj_backward(ht_t, proj_p, dlogits_t, dh_state_t, proj_g)

        # LSTM step backward — produces dh_prev, dc_prev for the previous step.
        # Re-gather x_t (one-hot at time t).
        for b in range(BATCH):
            for v in range(VOCAB):
                x_step_buf[b * VOCAB + v] = inp_oh[
                    b * SEQ * VOCAB + t * VOCAB + v
                ]
        var x_t_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
        ](x_step_buf)
        var hp_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](h_buf + t * BATCH * HIDDEN)
        var cp_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](c_buf + t * BATCH * HIDDEN)
        var cc_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, LSTM_CS), MutAnyOrigin
        ](cache_buf + t * BATCH * LSTM_CS)
        var dh_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](dh_state)
        var dc_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](dc_state)
        var dx_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
        ](dx_unused)
        var dhp_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](dh_prev)
        var dcp_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
        ](dc_prev)

        Cell.step_backward[BATCH](
            dh_view,
            dc_view,
            x_t_view,
            hp_view,
            cp_view,
            lstm_p,
            cc_view,
            dx_view,
            dhp_view,
            dcp_view,
            lstm_g,
        )
        # Thread dh, dc back as next backward step's incoming gradient.
        for i in range(BATCH * HIDDEN):
            dh_state[i] = dh_prev[i]
            dc_state[i] = dc_prev[i]

    h_buf.free()
    c_buf.free()
    cache_buf.free()
    logits_buf.free()
    x_step_buf.free()
    dlogits_buf.free()
    dh_state.free()
    dc_state.free()
    dx_unused.free()
    dh_prev.free()
    dc_prev.free()

    return avg_loss


# =============================================================================
# Validation loss (forward only)
# =============================================================================
def eval_loss(
    mut lstm_params: List[Scalar[dtype]],
    mut proj_params: List[Scalar[dtype]],
    val_ids: List[Int],
    n_batches: Int,
) raises -> Float64:
    var lstm_p = LayoutTensor[dtype, Layout.row_major(LSTM_PS), MutAnyOrigin](
        lstm_params.unsafe_ptr()
    )
    var proj_p = LayoutTensor[dtype, Layout.row_major(PROJ_PS), MutAnyOrigin](
        proj_params.unsafe_ptr()
    )

    var total = Float64(0.0)
    for _ in range(n_batches):
        var batch = make_batch(val_ids, BATCH, SEQ)
        var inp_oh = to_one_hot(batch.inputs, VOCAB, BATCH, SEQ)

        # Same forward as training, no backward.
        var h_buf = alloc[Scalar[dtype]]((SEQ + 1) * BATCH * HIDDEN)
        var c_buf = alloc[Scalar[dtype]]((SEQ + 1) * BATCH * HIDDEN)
        memset(h_buf, 0, (SEQ + 1) * BATCH * HIDDEN)
        memset(c_buf, 0, (SEQ + 1) * BATCH * HIDDEN)

        var x_buf = alloc[Scalar[dtype]](BATCH * VOCAB)
        var logits_buf = alloc[Scalar[dtype]](BATCH * VOCAB)

        var batch_loss = Float64(0.0)
        for t in range(SEQ):
            for b in range(BATCH):
                for v in range(VOCAB):
                    x_buf[b * VOCAB + v] = inp_oh[
                        b * SEQ * VOCAB + t * VOCAB + v
                    ]
            var x_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
            ](x_buf)
            var hp_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
            ](h_buf + t * BATCH * HIDDEN)
            var cp_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
            ](c_buf + t * BATCH * HIDDEN)
            var ht_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
            ](h_buf + (t + 1) * BATCH * HIDDEN)
            var ct_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
            ](c_buf + (t + 1) * BATCH * HIDDEN)

            Cell.step_forward_no_cache[BATCH](
                x_t, hp_t, cp_t, lstm_p, ht_t, ct_t
            )
            var logits_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
            ](logits_buf)
            proj_forward(ht_t, proj_p, logits_t)
            batch_loss += ce_forward_step(logits_t, batch.targets, t)

        total += batch_loss / Float64(BATCH * SEQ)

        h_buf.free()
        c_buf.free()
        x_buf.free()
        logits_buf.free()

    return total / Float64(n_batches)


# =============================================================================
# Sampling — feed one char at a time, advance (h, c)
# =============================================================================
def sample_categorical(
    logits_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    temperature: Float64,
) -> Int:
    if temperature <= 0.0:
        var best_v = Float64(logits_row[0])
        var best_idx = 0
        for v in range(1, VOCAB):
            var x = Float64(logits_row[v])
            if x > best_v:
                best_v = x
                best_idx = v
        return best_idx

    var inv_t = 1.0 / temperature
    var max_l = Float64(logits_row[0]) * inv_t
    for v in range(1, VOCAB):
        var x = Float64(logits_row[v]) * inv_t
        if x > max_l:
            max_l = x
    var sum_exp = 0.0
    var exps = List[Float64](capacity=VOCAB)
    for v in range(VOCAB):
        var e = exp(Float64(logits_row[v]) * inv_t - max_l)
        exps.append(e)
        sum_exp += e
    var u = random_float64(0.0, 1.0) * sum_exp
    var acc = 0.0
    for v in range(VOCAB):
        acc += exps[v]
        if u < acc:
            return v
    return VOCAB - 1


def generate_text(
    mut lstm_params: List[Scalar[dtype]],
    mut proj_params: List[Scalar[dtype]],
    tok: CharTokenizer,
    prompt: String,
    n_tokens: Int,
    temperature: Float64 = 0.8,
) raises -> String:
    var lstm_p = LayoutTensor[dtype, Layout.row_major(LSTM_PS), MutAnyOrigin](
        lstm_params.unsafe_ptr()
    )
    var proj_p = LayoutTensor[dtype, Layout.row_major(PROJ_PS), MutAnyOrigin](
        proj_params.unsafe_ptr()
    )

    # Single-batch state (BATCH=1 for sampling).
    var h_buf = alloc[Scalar[dtype]](HIDDEN)
    var c_buf = alloc[Scalar[dtype]](HIDDEN)
    var h_new = alloc[Scalar[dtype]](HIDDEN)
    var c_new = alloc[Scalar[dtype]](HIDDEN)
    memset(h_buf, 0, HIDDEN)
    memset(c_buf, 0, HIDDEN)

    var x_buf = alloc[Scalar[dtype]](VOCAB)
    var logits_buf = alloc[Scalar[dtype]](VOCAB)

    var ids = tok.encode(prompt)
    if len(ids) == 0:
        raise Error("generate_text: prompt is empty after tokenization")

    @parameter
    @always_inline
    def _step(tid: Int) raises:
        memset(x_buf, 0, VOCAB)
        x_buf[tid] = Scalar[dtype](1.0)

        var x_t = LayoutTensor[dtype, Layout.row_major(1, VOCAB), MutAnyOrigin](
            x_buf
        )
        var hp_t = LayoutTensor[
            dtype, Layout.row_major(1, HIDDEN), MutAnyOrigin
        ](h_buf)
        var cp_t = LayoutTensor[
            dtype, Layout.row_major(1, HIDDEN), MutAnyOrigin
        ](c_buf)
        var ht_t = LayoutTensor[
            dtype, Layout.row_major(1, HIDDEN), MutAnyOrigin
        ](h_new)
        var ct_t = LayoutTensor[
            dtype, Layout.row_major(1, HIDDEN), MutAnyOrigin
        ](c_new)
        Cell.step_forward_no_cache[1](x_t, hp_t, cp_t, lstm_p, ht_t, ct_t)

        # Swap h_buf/h_new pointers conceptually — copy back.
        for j in range(HIDDEN):
            h_buf[j] = h_new[j]
            c_buf[j] = c_new[j]

    # Warm up state with the prompt.
    for i in range(len(ids)):
        _step(ids[i])

    # Generate.
    var generated = List[Int](capacity=n_tokens)
    var last = ids[len(ids) - 1]
    for _ in range(n_tokens):
        # Run one more step for the next-token prediction:
        # produce logits from current h and the previous token last.
        # We already advanced h with `last` in the loop entry, so logits
        # come from the projection of the most recent h.
        var ht_view = LayoutTensor[
            dtype, Layout.row_major(1, HIDDEN), MutAnyOrigin
        ](h_buf)
        var lo_view = LayoutTensor[
            dtype, Layout.row_major(1, VOCAB), MutAnyOrigin
        ](logits_buf)
        # proj_forward expects BATCH=BATCH, but we use BATCH=1 here. Inline:
        for v in range(VOCAB):
            var acc = Float64(0.0)
            for j in range(HIDDEN):
                acc += Float64(rebind[Scalar[dtype]](ht_view[0, j])) * Float64(
                    rebind[Scalar[dtype]](proj_p[j * VOCAB + v])
                )
            acc += Float64(rebind[Scalar[dtype]](proj_p[PROJ_W + v]))
            lo_view[0, v] = Scalar[dtype](acc)

        var next_id = sample_categorical(logits_buf, temperature)
        generated.append(next_id)
        # Step forward with the newly sampled token to update (h, c).
        _step(next_id)
        last = next_id

    h_buf.free()
    c_buf.free()
    h_new.free()
    c_new.free()
    x_buf.free()
    logits_buf.free()

    return tok.decode(generated)


# =============================================================================
# Driver
# =============================================================================
def main() raises:
    seed(42)

    print("=" * 70)
    print("TinyShakespeare char-LSTM training (CPU smoke)")
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
    print(
        "  lr_lstm="
        + String(LR_LSTM)
        + " lr_proj="
        + String(LR_PROJ)
        + " grad_clip="
        + String(GRAD_CLIP_NORM)
    )
    print("  steps=" + String(N_STEPS) + " warmup=" + String(WARMUP_STEPS))
    print(
        "  LSTM_PS="
        + String(LSTM_PS)
        + " PROJ_PS="
        + String(PROJ_PS)
        + " (total="
        + String(LSTM_PS + PROJ_PS)
        + ")"
    )

    # ---------- Load data ----------
    print("\n[data] loading TinyShakespeare...")
    var text = load_text()
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error(
            "vocab mismatch: tokenizer found "
            + String(tok.vocab_size)
            + " unique chars, expected VOCAB="
            + String(VOCAB)
        )
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)
    print(
        "  total tokens="
        + String(len(ids))
        + " train="
        + String(len(split.train))
        + " val="
        + String(len(split.val))
    )

    # ---------- Initialize ----------
    var lstm_params = List[Scalar[dtype]](capacity=LSTM_PS)
    for _ in range(LSTM_PS):
        lstm_params.append(0)
    var lstm_grads = List[Scalar[dtype]](capacity=LSTM_PS)
    for _ in range(LSTM_PS):
        lstm_grads.append(0)
    var proj_params = List[Scalar[dtype]](capacity=PROJ_PS)
    for _ in range(PROJ_PS):
        proj_params.append(0)
    var proj_grads = List[Scalar[dtype]](capacity=PROJ_PS)
    for _ in range(PROJ_PS):
        proj_grads.append(0)

    var lstm_p_view = LayoutTensor[
        dtype, Layout.row_major(LSTM_PS), MutAnyOrigin
    ](lstm_params.unsafe_ptr())
    Cell.initialize_params[Xavier[]](lstm_p_view)

    # Output projection: small Xavier init manually
    var lim = sqrt(6.0 / Float64(HIDDEN + VOCAB))
    for j in range(HIDDEN):
        for v in range(VOCAB):
            var u = random_float64(-1.0, 1.0)
            proj_params[j * VOCAB + v] = Scalar[dtype](u * lim)
    for v in range(VOCAB):
        proj_params[PROJ_W + v] = Scalar[dtype](0.0)

    var lstm_adam = AdamState(LSTM_PS)
    var proj_adam = AdamState(PROJ_PS)

    # ---------- Initial val ----------
    var val_init = eval_loss(lstm_params, proj_params, split.val, EVAL_BATCHES)
    print(
        "\n[step 0] initial val loss="
        + String(val_init)
        + "  (random ≈ ln(V)="
        + String(log(Float64(VOCAB)))
        + ")"
    )

    # ---------- Training loop ----------
    var loss_running = Float64(0.0)
    var loss_count = 0
    for step in range(N_STEPS):
        var s = lr_scale(step, WARMUP_STEPS, N_STEPS)

        var batch = make_batch(split.train, BATCH, SEQ)
        var inp_oh = to_one_hot(batch.inputs, VOCAB, BATCH, SEQ)

        var loss = train_step(
            lstm_params,
            proj_params,
            lstm_grads,
            proj_grads,
            inp_oh,
            batch.targets,
        )

        var pre_clip_norm = clip_grads_global_norm(
            lstm_grads, proj_grads, GRAD_CLIP_NORM
        )

        adam_step(lstm_params, lstm_grads, lstm_adam, LR_LSTM, s)
        adam_step(proj_params, proj_grads, proj_adam, LR_PROJ, s)

        loss_running += loss
        loss_count += 1

        if (step + 1) % PRINT_EVERY == 0:
            var avg = loss_running / Float64(loss_count)
            print(
                "[step "
                + String(step + 1)
                + "] train_loss="
                + String(avg)
                + " grad_norm="
                + String(pre_clip_norm)
                + " lr_scale="
                + String(s)
            )
            loss_running = 0.0
            loss_count = 0

        if (step + 1) % EVAL_EVERY == 0:
            var v = eval_loss(lstm_params, proj_params, split.val, EVAL_BATCHES)
            print("           val_loss=" + String(v))

    # ---------- Final eval ----------
    var val_final = eval_loss(lstm_params, proj_params, split.val, EVAL_BATCHES)
    print(
        "\n[final] val_loss="
        + String(val_final)
        + " (start "
        + String(val_init)
        + ")"
    )
    if val_final < val_init - 0.5:
        print("  PASS: validation loss decreased by > 0.5 nats")
    else:
        print(
            "  WARN: val loss did not improve substantially — "
            + "increase N_STEPS or check learning rate"
        )

    # ---------- Sampling ----------
    var prompt = String("ROMEO:")
    print("\n[sample] prompt = " + repr(prompt))

    print("\n[sample] greedy (T=0.0):")
    var greedy = generate_text(lstm_params, proj_params, tok, prompt, 200, 0.0)
    print(prompt + greedy)

    print("\n[sample] temperature (T=0.8):")
    var temp_s = generate_text(lstm_params, proj_params, tok, prompt, 200, 0.8)
    print(prompt + temp_s)

    print("\n" + "=" * 70)
