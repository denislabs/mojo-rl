"""Off-policy critic-update helpers — reusable across SAC / TD3 / DDPG.

Phase 9B.4. Extracted from `SACTrainer` so the upcoming DDPG/TD3 trainers
can import a generic surface instead of duplicating the boilerplate.

Three free functions, all `CRITIC: Module`-generic:

  `concat_sa[OBS, ACT, B]`
      Pack (obs, act) → sa side-by-side. Used both for the critic
      target-y compute (next-state action) and the critic update
      (replay action).

  `critic_update_step[CRITIC, BATCH, SA_DIM]`
      One full single-critic MSE step: `zero_grad → critic.forward
      → mse.forward → mse.backward → critic.backward → opt.step`.
      Returns the scalar loss (for logging). DDPG uses this directly
      once (single critic). SAC + TD3 call it twice (twin critics).

  `twin_critic_update_step[CRITIC, BATCH, OBS, ACT]`
      Convenience wrapper: `concat_sa` + two `critic_update_step` calls
      against the shared target `mb_y`. Returns the sum of both losses.
      The most common pattern in off-policy continuous algorithms.

Free functions (not methods) so the Mojo nightly aliasing analyzer
doesn't have to prove that `self.pair1.online` and `self.critic1_opt`
don't alias when both come from a `mut self` receiver. Caller passes
each `mut` arg distinctly. See memory:
`feedback_mojo_tile_tensor_generic_origin`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.mse import MSELoss


def concat_sa[
    OBS: Int, ACT: Int, B: Int
](
    obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
    act: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_sa: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """`sa[b, :OBS] = obs[b]; sa[b, OBS:] = act[b]`. CPU only."""
    comptime SA = OBS + ACT
    for b in range(B):
        for d in range(OBS):
            out_sa[b * SA + d] = obs[b * OBS + d]
        for j in range(ACT):
            out_sa[b * SA + OBS + j] = act[b * ACT + j]


def _concat_sa_kernel[
    OBS: Int, ACT: Int, B: Int
](
    obs: LayoutTensor[DT, Layout.row_major(B, OBS), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(B, ACT), MutAnyOrigin],
    out_sa: LayoutTensor[DT, Layout.row_major(B, OBS + ACT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    comptime SA = OBS + ACT
    var total = B * SA
    if idx < total:
        var b = idx // SA
        var d = idx % SA
        if d < OBS:
            out_sa[b, d] = rebind[Scalar[DT]](obs[b, d])
        else:
            out_sa[b, d] = rebind[Scalar[DT]](act[b, d - OBS])


def concat_sa_gpu[
    OBS: Int, ACT: Int, B: Int
](
    ctx: DeviceContext,
    obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
    act: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_sa: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    """GPU `concat_sa` — one thread per [b, d] over the full SA shape."""
    comptime SA = OBS + ACT
    var obs_lt = LayoutTensor[DT, Layout.row_major(B, OBS), MutAnyOrigin](obs)
    var act_lt = LayoutTensor[DT, Layout.row_major(B, ACT), MutAnyOrigin](act)
    var out_lt = LayoutTensor[DT, Layout.row_major(B, SA), MutAnyOrigin](out_sa)
    comptime TPB = 128
    comptime total = B * SA
    comptime n_blocks = (total + TPB - 1) // TPB
    comptime kernel = _concat_sa_kernel[OBS, ACT, B]
    ctx.enqueue_function[kernel](
        obs_lt,
        act_lt,
        out_lt,
        grid_dim=n_blocks,
        block_dim=TPB,
    )


def critic_update_step[
    CRITIC: Module,
    BATCH: Int,
    SA_DIM: Int,
](
    mut critic: CRITIC,
    mut opt: Adam,
    mut mse_loss: MSELoss[1],
    mb_sa_t: TileTensor[
        dtype=DT,
        address_space=AddressSpace.GENERIC,
        element_size=1,
        origin=MutAnyOrigin,
        ...,
    ],
    mb_y_t: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    mb_q: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_q: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_sa: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises -> Scalar[DT]:
    """One critic update against a precomputed target `mb_y_t`. Returns
    the scalar MSE loss (for logging). Caller owns scratch (`mb_q`,
    `mb_grad_q`, `mb_grad_sa`). Steps: `zero_grad → critic.forward
    → mse.forward → mse.backward → critic.backward → opt.step`.

    CPU only.
    """
    var mb_q_t = TileTensor(mb_q, row_major[BATCH, 1]())
    opt.zero_grad["cpu", M=CRITIC](critic)
    critic.forward["cpu", BATCH](mb_sa_t, output=mb_q_t)
    var loss = mse_loss.forward["cpu", BATCH](mb_q_t, mb_y_t)
    var mb_grad_q_t = TileTensor(mb_grad_q, row_major[BATCH, 1]())
    mse_loss.vjp["cpu", BATCH](mb_y_t, mb_grad_q_t)
    var mb_grad_sa_t = TileTensor(mb_grad_sa, row_major[BATCH, SA_DIM]())
    critic.vjp["cpu", BATCH](mb_grad_q_t, mb_grad_sa_t)
    opt.step["cpu", M=CRITIC](critic)
    return loss


def twin_critic_update_step[
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](
    mut critic1: CRITIC,
    mut critic1_opt: Adam,
    mut critic2: CRITIC,
    mut critic2_opt: Adam,
    mut mse_loss: MSELoss[1],
    mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_a_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_sa_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_y_t: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    mb_q1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_q2_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_q1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_q2_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_sa1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_sa2_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises -> Scalar[DT]:
    """Twin-critic update (SAC / TD3 pattern): pack `(s, a) → sa`, then
    one `critic_update_step` per critic against the shared target.
    Returns the sum of both losses (for logging)."""
    comptime SA = OBS + ACT
    concat_sa[OBS, ACT, BATCH](mb_s_ptr, mb_a_ptr, mb_sa_ptr)
    var mb_sa_t = TileTensor(mb_sa_ptr, row_major[BATCH, SA]())
    var loss1 = critic_update_step[CRITIC, BATCH, SA](
        critic1,
        critic1_opt,
        mse_loss,
        mb_sa_t,
        mb_y_t,
        mb_q1_ptr,
        mb_grad_q1_ptr,
        mb_grad_sa1_ptr,
    )
    var loss2 = critic_update_step[CRITIC, BATCH, SA](
        critic2,
        critic2_opt,
        mse_loss,
        mb_sa_t,
        mb_y_t,
        mb_q2_ptr,
        mb_grad_q2_ptr,
        mb_grad_sa2_ptr,
    )
    return loss1 + loss2
