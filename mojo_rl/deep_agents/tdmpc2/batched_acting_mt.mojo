"""Batched (N_ENVS at a time) action selection for MULTI-TASK TD-MPC2.

The multi-task sibling of `batched_acting.mojo`. Two differences, both from the
task embedding:

  * every net input is a CONCATENATION — the encoder eats `[obs | tem]` and the
    policy eats `[z | tem]`, where `tem` is the current task's embedding row;
  * the action is masked per task (`action_mask[task]`), because a task with
    fewer actuators than `MAX_ACT` must emit exactly zero on the padding
    columns rather than whatever the shared policy head produced there.

The caller gathers `tem` (the driver owns the `TaskEmbedding` table) and passes
it in as `[N_ENVS, TASK_EMB]` — one row per lane. All lanes in a call share a
task, which is what the segment-alternating driver guarantees; gathering N
identical rows keeps the concat a plain elementwise copy with no broadcast.
"""

from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext
from std.math import tanh
from std.gpu import global_idx

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.call import call_forward
from mojo_rl.nn.primitives.rsample import RSample

from mojo_rl.deep_agents.training.blocks.action_kernels import (
    offpolicy_copy2d_kernel,
)


def mt_concat2_kernel[
    N: Int, D1: Int, D2: Int
](
    a: LayoutTensor[DT, Layout.row_major(N, D1), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(N, D2), MutAnyOrigin],
    out_t: LayoutTensor[DT, Layout.row_major(N, D1 + D2), MutAnyOrigin],
):
    """`out[n] = a[n] ++ b[n]` — the `[obs | tem]` / `[z | tem]` join."""
    var i = Int(global_idx.x)
    var total = N * (D1 + D2)
    if i >= total:
        return
    var n = i // (D1 + D2)
    var d = i % (D1 + D2)
    if d < D1:
        out_t[n, d] = rebind[Scalar[DT]](a[n, d])
    else:
        out_t[n, d] = rebind[Scalar[DT]](b[n, d - D1])


def mt_mask_clamp_kernel[
    N: Int, ACT: Int, ALP: Int
](
    alp: LayoutTensor[DT, Layout.row_major(N, ALP), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    action_out: LayoutTensor[DT, Layout.row_major(N, ACT), MutAnyOrigin],
    action_scale: Scalar[DT],
):
    """`action[n,j] = clamp(alp[n,j], ±scale) · mask[j]`.

    `alp`'s trailing column is the rsample log-prob and is dropped. The mask is
    the CURRENT task's row of `action_mask`, uploaded once per segment: columns
    beyond a task's real action dim are zeroed so a padded task cannot drive
    actuators it does not have."""
    var i = Int(global_idx.x)
    var total = N * ACT
    if i >= total:
        return
    var n = i // ACT
    var j = i % ACT
    var a = rebind[Scalar[DT]](alp[n, j])
    if a > action_scale:
        a = action_scale
    elif a < -action_scale:
        a = -action_scale
    action_out[n, j] = a * rebind[Scalar[DT]](mask[j])


def mt_greedy_mask_kernel[
    N: Int, ACT: Int, POL: Int
](
    pio: LayoutTensor[DT, Layout.row_major(N, POL), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    action_out: LayoutTensor[DT, Layout.row_major(N, ACT), MutAnyOrigin],
    action_scale: Scalar[DT],
):
    """`action[n,j] = tanh(pio[n,j])·scale·mask[j]` — the deterministic head,
    matching the single-obs `select_action(explore=False)`."""
    var i = Int(global_idx.x)
    var total = N * ACT
    if i >= total:
        return
    var n = i // ACT
    var j = i % ACT
    action_out[n, j] = (
        tanh(rebind[Scalar[DT]](pio[n, j]))
        * action_scale
        * rebind[Scalar[DT]](mask[j])
    )


def tdmpc2_mt_select_action_batched[
    ENC_M: Module,
    POL_M: Module,
    target: StaticString,
    N_ENVS: Int,
    MAX_OBS: Int,
    ACT: Int,
    LATENT: Int,
    EMB: Int,
](
    mut encoder: ENC_M,
    mut policy: POL_M,
    mut rs: RSample[ACT],
    mut ob_scr: Tensor,      # [N, MAX_OBS]
    mut ein_scr: Tensor,     # [N, MAX_OBS + EMB]
    mut z_scr: Tensor,       # [N, LATENT]
    mut pin_scr: Tensor,     # [N, LATENT + EMB]
    mut pio_scr: Tensor,     # [N, 2*ACT]
    mut alp_scr: Tensor,     # [N, ACT+1]
    mut tem: Tensor,         # [N, EMB]  — caller-gathered task embedding
    mut mask_scr: Tensor,    # [ACT]     — the current task's action mask
    obs: LayoutTensor[DT, Layout.row_major(N_ENVS, MAX_OBS), MutAnyOrigin],
    action: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    action_scale: Scalar[DT],
    explore: Bool,
    ctx: Optional[DeviceContext],
) raises:
    """`a = mask · π([encode([obs|tem]) | tem])` for N envs, into `action`.

    Scratch is caller-owned and sized once — same reason as the single-task
    helper: this runs per env-step and per-call device allocation is the
    hot-loop footgun.
    """
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime AOBS = MAX_OBS + EMB
    comptime PIN = LATENT + EMB
    comptime POL = 2 * ACT

    comptime if target == "cpu":
        ob_scr.ensure(N_ENVS * MAX_OBS)
        ein_scr.ensure(N_ENVS * AOBS)
        z_scr.ensure(N_ENVS * LATENT)
        pin_scr.ensure(N_ENVS * PIN)
        pio_scr.ensure(N_ENVS * POL)
        alp_scr.ensure(N_ENVS * (ACT + 1))
        for e in range(N_ENVS):
            for d in range(MAX_OBS):
                ein_scr.data[e * AOBS + d] = rebind[Scalar[DT]](obs[e, d])
            for k in range(EMB):
                ein_scr.data[e * AOBS + MAX_OBS + k] = tem.data[e * EMB + k]
        call_forward[target, N_ENVS](
            encoder, TensorRefs[ENC_M.ARITY](ein_scr), z_scr, ctx
        )
        for e in range(N_ENVS):
            for k in range(LATENT):
                pin_scr.data[e * PIN + k] = z_scr.data[e * LATENT + k]
            for k in range(EMB):
                pin_scr.data[e * PIN + LATENT + k] = tem.data[e * EMB + k]
        call_forward[target, N_ENVS](
            policy, TensorRefs[POL_M.ARITY](pin_scr), pio_scr, ctx
        )
        if explore:
            call_forward[target, N_ENVS](
                rs, TensorRefs[1](pio_scr), alp_scr, ctx
            )
            for e in range(N_ENVS):
                for j in range(ACT):
                    var a = alp_scr.data[e * (ACT + 1) + j]
                    if a > action_scale:
                        a = action_scale
                    elif a < -action_scale:
                        a = -action_scale
                    action[e, j] = a * mask_scr.data[j]
        else:
            for e in range(N_ENVS):
                for j in range(ACT):
                    action[e, j] = (
                        tanh(pio_scr.data[e * POL + j])
                        * action_scale
                        * mask_scr.data[j]
                    )
    else:
        var c = ctx.value()
        ob_scr.ensure_gpu(c, N_ENVS * MAX_OBS)
        ein_scr.ensure_gpu(c, N_ENVS * AOBS)
        z_scr.ensure_gpu(c, N_ENVS * LATENT)
        pin_scr.ensure_gpu(c, N_ENVS * PIN)
        pio_scr.ensure_gpu(c, N_ENVS * POL)
        alp_scr.ensure_gpu(c, N_ENVS * (ACT + 1))

        comptime tot_obs = N_ENVS * MAX_OBS
        c.enqueue_function[offpolicy_copy2d_kernel[N_ENVS, MAX_OBS]](
            obs,
            ob_scr.lt["gpu", Layout.row_major(N_ENVS, MAX_OBS)](),
            grid_dim=(tot_obs + TPB - 1) // TPB,
            block_dim=TPB,
        )
        comptime tot_ein = N_ENVS * AOBS
        c.enqueue_function[mt_concat2_kernel[N_ENVS, MAX_OBS, EMB]](
            ob_scr.lt["gpu", Layout.row_major(N_ENVS, MAX_OBS)](),
            tem.lt["gpu", Layout.row_major(N_ENVS, EMB)](),
            ein_scr.lt["gpu", Layout.row_major(N_ENVS, AOBS)](),
            grid_dim=(tot_ein + TPB - 1) // TPB,
            block_dim=TPB,
        )
        call_forward[target, N_ENVS](
            encoder, TensorRefs[ENC_M.ARITY](ein_scr), z_scr, ctx
        )
        comptime tot_pin = N_ENVS * PIN
        c.enqueue_function[mt_concat2_kernel[N_ENVS, LATENT, EMB]](
            z_scr.lt["gpu", Layout.row_major(N_ENVS, LATENT)](),
            tem.lt["gpu", Layout.row_major(N_ENVS, EMB)](),
            pin_scr.lt["gpu", Layout.row_major(N_ENVS, PIN)](),
            grid_dim=(tot_pin + TPB - 1) // TPB,
            block_dim=TPB,
        )
        call_forward[target, N_ENVS](
            policy, TensorRefs[POL_M.ARITY](pin_scr), pio_scr, ctx
        )

        comptime tot_act = N_ENVS * ACT
        if explore:
            call_forward[target, N_ENVS](
                rs, TensorRefs[1](pio_scr), alp_scr, ctx
            )
            c.enqueue_function[mt_mask_clamp_kernel[N_ENVS, ACT, ACT + 1]](
                alp_scr.lt["gpu", Layout.row_major(N_ENVS, ACT + 1)](),
                mask_scr.lt["gpu", Layout.row_major(ACT)](),
                action,
                action_scale,
                grid_dim=(tot_act + TPB - 1) // TPB,
                block_dim=TPB,
            )
        else:
            c.enqueue_function[mt_greedy_mask_kernel[N_ENVS, ACT, POL]](
                pio_scr.lt["gpu", Layout.row_major(N_ENVS, POL)](),
                mask_scr.lt["gpu", Layout.row_major(ACT)](),
                action,
                action_scale,
                grid_dim=(tot_act + TPB - 1) // TPB,
                block_dim=TPB,
            )
