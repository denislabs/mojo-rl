"""LeWM checkpoint per-network helpers.

Provides the two primitives the LeWMGPUState save/load methods use:

- `_write_gpu_net_sections[MODEL, OPT](state, ctx, prefix) -> String`:
  sync device->host then emit `{prefix}params:`,
  `{prefix}optimizer_state:`, and (when applicable) `{prefix}model_state:`
  sections suitable for concatenation into a multi-network file.
- `_read_gpu_net_sections[MODEL, OPT](mut state, content, ctx, prefix)`:
  the symmetric load — reads the prefixed sections, fills host mirrors,
  enqueues uploads (caller synchronizes once at end-of-load).

The trainer-level orchestration (walking the 4 shared networks +
3*DEPTH-fold cond_block stack) lives as methods directly on
`LeWMGPUState` in `offline_trainer.mojo`, because Mojo nightly requires
field access on a typed self rather than on a duck-typed comptime param.
"""

from std.gpu.host import DeviceContext

from ...nn.model import Model
from ...nn.optimizer import Optimizer
from ...nn.training import GPUNetworkState
from ...nn.constants import dtype
from ...nn.checkpoint import (
    write_float_section_ptr,
    read_float_section_list,
)


# =============================================================================
# Per-network helpers (used by both Pong and PushT LeWMGPUState specializations)
# =============================================================================


def _write_gpu_net_sections[
    MODEL: Model, OPTIMIZER: Optimizer
](
    gpu_state: GPUNetworkState[MODEL, OPTIMIZER, dtype],
    ctx: DeviceContext,
    prefix: String,
) raises -> String:
    """Sync one GPUNetworkState device->host then emit prefixed sections.

    Returns a string with ``{prefix}params:``,
    ``{prefix}optimizer_state:``, and (when ``MODEL.STATE_SIZE > 0``)
    ``{prefix}model_state:`` sections, suitable for concatenation into a
    larger checkpoint file.

    Synchronizes the device queue before reading the host mirror — caller
    can write the returned string immediately.
    """
    ctx.enqueue_copy(gpu_state.params_host, gpu_state.params_buf)
    ctx.enqueue_copy(gpu_state.opt_state_host, gpu_state.opt_state_buf)
    comptime if MODEL.STATE_SIZE > 0:
        ctx.enqueue_copy(
            gpu_state.model_state_host, gpu_state.model_state_buf
        )
    ctx.synchronize()

    var content = write_float_section_ptr(
        prefix + "params:",
        gpu_state.params_host.unsafe_ptr(),
        gpu_state.PARAM_SIZE,
    )
    content += write_float_section_ptr(
        prefix + "optimizer_state:",
        gpu_state.opt_state_host.unsafe_ptr(),
        gpu_state.OPT_STATE_SIZE,
    )
    comptime if MODEL.STATE_SIZE > 0:
        content += write_float_section_ptr(
            prefix + "model_state:",
            gpu_state.model_state_host.unsafe_ptr(),
            gpu_state.MODEL_STATE_SIZE,
        )
    return content^


def _read_gpu_net_sections[
    MODEL: Model, OPTIMIZER: Optimizer
](
    mut gpu_state: GPUNetworkState[MODEL, OPTIMIZER, dtype],
    content: String,
    ctx: DeviceContext,
    prefix: String,
) raises:
    """Read prefixed sections and upload host->device.

    Counterpart of ``_write_gpu_net_sections``. Reads
    ``{prefix}params:``, ``{prefix}optimizer_state:`` and (if applicable)
    ``{prefix}model_state:`` from `content`, copies into the host
    mirrors, then enqueues uploads to device. Caller is responsible for
    synchronizing the queue once all networks are loaded (one global sync
    at end-of-load is cheaper than per-network).
    """
    var loaded_params = read_float_section_list[dtype](
        content, prefix + "params:", gpu_state.PARAM_SIZE
    )
    for i in range(gpu_state.PARAM_SIZE):
        gpu_state.params_host[i] = loaded_params[i]
    ctx.enqueue_copy(gpu_state.params_buf, gpu_state.params_host)

    var loaded_opt = read_float_section_list[dtype](
        content, prefix + "optimizer_state:", gpu_state.OPT_STATE_SIZE
    )
    for i in range(gpu_state.OPT_STATE_SIZE):
        gpu_state.opt_state_host[i] = loaded_opt[i]
    ctx.enqueue_copy(gpu_state.opt_state_buf, gpu_state.opt_state_host)

    comptime if MODEL.STATE_SIZE > 0:
        var loaded_ms = read_float_section_list[dtype](
            content, prefix + "model_state:", gpu_state.MODEL_STATE_SIZE
        )
        for i in range(gpu_state.MODEL_STATE_SIZE):
            gpu_state.model_state_host[i] = loaded_ms[i]
        ctx.enqueue_copy(
            gpu_state.model_state_buf, gpu_state.model_state_host
        )


