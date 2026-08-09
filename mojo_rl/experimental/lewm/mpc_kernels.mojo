"""LeWM MPC latent-rollout data-movement kernels (ported from legacy).

The autoregressive MPC shot rolls the predictor forward in LATENT space:
the encoder runs once (start + goal latents), then for each horizon step we
slide an H-token window out of the rolling latent buffer, predict the next
H latents under the action window, and append the last predicted token.
These are the verbatim legacy `kernels.mojo` movement ops, retargeted to
nn (DT + target-dispatched CPU/GPU). Latent space is tiny (B·ROLL_T·EMB),
so the GPU kernels are flat 1-D grids.

  emb_seq:  (B, ROLL_T·EMB)  rolling latent buffer, ROLL_T = H + horizon
  latent_ctx: (B, H·EMB)     the predictor's context input (window [k,k+H))
  actions_buf: (B, T·EMB-actions) (B, T·ACT) — ActionEmbedder is Tokenwise[T]
"""

from std.gpu import global_idx, thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, row_major

from ...nn.constants import DT, TPB, TPB_REDUCE


# ── replicate_start: emb_seq[:, 0:H, :] = emb_start; rest 0 ─────────────
def _replicate_kernel[BATCH: Int, H: Int, EMB: Int, ROLL_T: Int](
    emb_start: LayoutTensor[DT, Layout.row_major(BATCH * EMB), MutAnyOrigin],
    emb_seq: LayoutTensor[DT, Layout.row_major(BATCH * ROLL_T * EMB), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * ROLL_T * EMB:
        return
    var b = idx // (ROLL_T * EMB)
    var rem = idx % (ROLL_T * EMB)
    var p = rem // EMB
    var d = rem % EMB
    if p < H:
        emb_seq[idx] = emb_start[b * EMB + d]
    else:
        emb_seq[idx] = Scalar[DT](0.0)


def mpc_replicate_start[
    target: StaticString, BATCH: Int, H: Int, EMB: Int, ROLL_T: Int,
](
    emb_start: Pointer[Scalar[DT], MutAnyOrigin],
    emb_seq: Pointer[Scalar[DT], MutAnyOrigin],
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime if target == "cpu":
        for b in range(BATCH):
            for p in range(ROLL_T):
                for d in range(EMB):
                    var v = emb_start[unsafe_offset=b * EMB + d] if p < H else Scalar[DT](0.0)
                    emb_seq[unsafe_offset=b * ROLL_T * EMB + p * EMB + d] = v
    else:
        var c = ctx.value()
        comptime N = BATCH * ROLL_T * EMB
        c.enqueue_function[_replicate_kernel[BATCH, H, EMB, ROLL_T]](
            LayoutTensor[DT, Layout.row_major(BATCH * EMB), MutAnyOrigin](emb_start),
            LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](emb_seq),
            grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )


# ── slide latent context: latent_ctx = emb_seq[:, k:k+H, :] (B, H·EMB) ──
def _slide_ctx_kernel[BATCH: Int, H: Int, EMB: Int, ROLL_T: Int](
    emb_seq: LayoutTensor[DT, Layout.row_major(BATCH * ROLL_T * EMB), MutAnyOrigin],
    latent_ctx: LayoutTensor[DT, Layout.row_major(BATCH * H * EMB), MutAnyOrigin],
    k_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var k = Int(k_arg)
    var idx = Int(global_idx.x)
    if idx >= BATCH * H * EMB:
        return
    var b = idx // (H * EMB)
    var rem = idx % (H * EMB)
    var p = rem // EMB
    var d = rem % EMB
    latent_ctx[idx] = emb_seq[b * ROLL_T * EMB + (k + p) * EMB + d]


def mpc_slide_latent_ctx[
    target: StaticString, BATCH: Int, H: Int, EMB: Int, ROLL_T: Int,
](
    emb_seq: Pointer[Scalar[DT], MutAnyOrigin],
    latent_ctx: Pointer[Scalar[DT], MutAnyOrigin],
    k: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime if target == "cpu":
        for b in range(BATCH):
            for p in range(H):
                for d in range(EMB):
                    latent_ctx[unsafe_offset=b * H * EMB + p * EMB + d] = emb_seq[unsafe_offset=
                        b * ROLL_T * EMB + (k + p) * EMB + d
                    ]
    else:
        var c = ctx.value()
        comptime N = BATCH * H * EMB
        c.enqueue_function[_slide_ctx_kernel[BATCH, H, EMB, ROLL_T]](
            LayoutTensor[DT, Layout.row_major(BATCH * ROLL_T * EMB), MutAnyOrigin](emb_seq),
            LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](latent_ctx),
            Int64(k), grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )


# ── slide actions: actions_buf[:, 0:H] = plan[:, k:k+H]; rest 0 (B,T·ACT)
def _slide_act_kernel[BATCH: Int, T: Int, H: Int, ACT: Int, NEEDED: Int](
    plan: LayoutTensor[DT, Layout.row_major(BATCH * NEEDED * ACT), MutAnyOrigin],
    actions_buf: LayoutTensor[DT, Layout.row_major(BATCH * T * ACT), MutAnyOrigin],
    k_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var k = Int(k_arg)
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * ACT:
        return
    var b = idx // (T * ACT)
    var rem = idx % (T * ACT)
    var p = rem // ACT
    var a = rem % ACT
    if p < H:
        actions_buf[idx] = plan[b * NEEDED * ACT + (k + p) * ACT + a]
    else:
        actions_buf[idx] = Scalar[DT](0.0)


def mpc_slide_actions[
    target: StaticString, BATCH: Int, T: Int, H: Int, ACT: Int, NEEDED: Int,
](
    plan: Pointer[Scalar[DT], MutAnyOrigin],
    actions_buf: Pointer[Scalar[DT], MutAnyOrigin],
    k: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime if target == "cpu":
        for b in range(BATCH):
            for p in range(T):
                for a in range(ACT):
                    var v = (
                        plan[unsafe_offset=b * NEEDED * ACT + (k + p) * ACT + a]
                        if p < H else Scalar[DT](0.0)
                    )
                    actions_buf[unsafe_offset=b * T * ACT + p * ACT + a] = v
    else:
        var c = ctx.value()
        comptime N = BATCH * T * ACT
        c.enqueue_function[_slide_act_kernel[BATCH, T, H, ACT, NEEDED]](
            LayoutTensor[DT, Layout.row_major(BATCH * NEEDED * ACT), MutAnyOrigin](plan),
            LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](actions_buf),
            Int64(k), grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )


# ── store pred[:, H-1, :] → emb_seq[:, k+H, :] ─────────────────────────
def _store_kernel[BATCH: Int, H: Int, EMB: Int, ROLL_T: Int](
    pred: LayoutTensor[DT, Layout.row_major(BATCH * H * EMB), MutAnyOrigin],
    emb_seq: LayoutTensor[DT, Layout.row_major(BATCH * ROLL_T * EMB), MutAnyOrigin],
    k_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var k = Int(k_arg)
    var idx = Int(global_idx.x)
    if idx >= BATCH * EMB:
        return
    var b = idx // EMB
    var d = idx % EMB
    emb_seq[b * ROLL_T * EMB + (k + H) * EMB + d] = pred[
        b * H * EMB + (H - 1) * EMB + d
    ]


def mpc_store_pred_last[
    target: StaticString, BATCH: Int, H: Int, EMB: Int, ROLL_T: Int,
](
    pred: Pointer[Scalar[DT], MutAnyOrigin],
    emb_seq: Pointer[Scalar[DT], MutAnyOrigin],
    k: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime if target == "cpu":
        for b in range(BATCH):
            for d in range(EMB):
                emb_seq[unsafe_offset=b * ROLL_T * EMB + (k + H) * EMB + d] = pred[unsafe_offset=
                    b * H * EMB + (H - 1) * EMB + d
                ]
    else:
        var c = ctx.value()
        comptime N = BATCH * EMB
        c.enqueue_function[_store_kernel[BATCH, H, EMB, ROLL_T]](
            LayoutTensor[DT, Layout.row_major(BATCH * H * EMB), MutAnyOrigin](pred),
            LayoutTensor[DT, Layout.row_major(BATCH * ROLL_T * EMB), MutAnyOrigin](emb_seq),
            Int64(k), grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )


# ── score: mean MSE(emb_seq[:, goal_pos, :], emb_goal) over B·EMB ──────
def _score_kernel[BATCH: Int, EMB: Int, ROLL_T: Int](
    emb_seq: LayoutTensor[DT, Layout.row_major(BATCH * ROLL_T * EMB), MutAnyOrigin],
    emb_goal: LayoutTensor[DT, Layout.row_major(BATCH * EMB), MutAnyOrigin],
    score_out: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    goal_pos_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var goal_pos = Int(goal_pos_arg)
    var t = Int(thread_idx.x)
    var my: Scalar[DT] = 0.0
    var k = t
    while k < BATCH * EMB:
        var b = k // EMB
        var d = k % EMB
        var ev = rebind[Scalar[DT]](
            emb_seq[b * ROLL_T * EMB + goal_pos * EMB + d]
        )
        var gv = rebind[Scalar[DT]](emb_goal[k])
        var diff = ev - gv
        my += diff * diff
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my)
    if t == 0:
        score_out[0] = total[0]


def mpc_score[
    target: StaticString, BATCH: Int, EMB: Int, ROLL_T: Int,
](
    emb_seq: Pointer[Scalar[DT], MutAnyOrigin],
    emb_goal: Pointer[Scalar[DT], MutAnyOrigin],
    goal_pos: Int,
    ctx: Optional[DeviceContext] = None,
) raises -> Float64:
    """Mean squared error between the rolled latent at `goal_pos` and the
    goal latent, averaged over B·EMB. Lower is better."""
    comptime if target == "cpu":
        var s: Float64 = 0.0
        for b in range(BATCH):
            for d in range(EMB):
                var diff = Float64(
                    emb_seq[unsafe_offset=b * ROLL_T * EMB + goal_pos * EMB + d]
                    - emb_goal[unsafe_offset=b * EMB + d]
                )
                s += diff * diff
        return s / Float64(BATCH * EMB)
    else:
        var c = ctx.value()
        var sd = c.enqueue_create_buffer[DT](1)
        c.enqueue_function[_score_kernel[BATCH, EMB, ROLL_T]](
            LayoutTensor[DT, Layout.row_major(BATCH * ROLL_T * EMB), MutAnyOrigin](emb_seq),
            LayoutTensor[DT, Layout.row_major(BATCH * EMB), MutAnyOrigin](emb_goal),
            LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin](
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](sd.unsafe_ptr())
            ),
            Int64(goal_pos), grid_dim=1, block_dim=TPB_REDUCE,
        )
        var hb = c.enqueue_create_host_buffer[DT](1)
        c.enqueue_copy(hb, sd)
        c.synchronize()
        return Float64(hb.unsafe_ptr()[unsafe_offset=0]) / Float64(BATCH * EMB)
