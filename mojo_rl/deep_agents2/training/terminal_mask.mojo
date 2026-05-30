"""Terminal-bootstrap mask shared by the SAC/DDPG/TD3 target-y blocks.

Each target-y graph computes only the BOOTSTRAP term (`γ·soft_v` for SAC,
`γ·Q'` for DDPG, `γ·min(Q1',Q2')` for TD3) into `mb_y`. `apply_terminal_mask`
then writes the full TD target in-place:

    mb_y[b] = r[b] + (1 − term[b]) · bootstrap[b]

dropping the bootstrap on natural termination (`term=1`) and keeping it on
time-limit truncation (`term=0`) — CleanRL semantics. For envs that never
terminate naturally (`term ≡ 0`) this reduces to `r + bootstrap`,
bit-identical to the previous in-graph `Add(r, bootstrap)` (multiply by an
exact 1.0 introduces no rounding).

Forward-only (target, never differentiated). Capture-safe (elementwise,
fixed buffers, no host work on the GPU path).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT, TPB


def _mask_bootstrap_kernel[N: Int](
    r: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    term: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    y: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var nonterm = Scalar[DT](1.0) - rebind[Scalar[DT]](term[i])
        y[i] = rebind[Scalar[DT]](r[i]) + nonterm * rebind[Scalar[DT]](y[i])


def apply_terminal_mask[
    target: StaticString, N: Int
](
    ctx: Optional[DeviceContext],
    mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_term_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    """In-place `mb_y[b] = r[b] + (1 − term[b]) · mb_y[b]` over N samples.

    `mb_y` enters holding the bootstrap term and exits holding the full TD
    target. `ctx` is required (and used) only on the GPU path."""
    comptime if target == "cpu":
        for b in range(N):
            var nonterm = Scalar[DT](1.0) - mb_term_ptr[b]
            mb_y_ptr[b] = mb_r_ptr[b] + nonterm * mb_y_ptr[b]
    else:
        comptime layout = Layout.row_major(N)
        var r_lt = LayoutTensor[DT, layout, MutAnyOrigin](mb_r_ptr)
        var t_lt = LayoutTensor[DT, layout, MutAnyOrigin](mb_term_ptr)
        var y_lt = LayoutTensor[DT, layout, MutAnyOrigin](mb_y_ptr)
        comptime n_blocks = (N + TPB - 1) // TPB
        ctx.value().enqueue_function[_mask_bootstrap_kernel[N]](
            r_lt, t_lt, y_lt, grid_dim=n_blocks, block_dim=TPB,
        )
