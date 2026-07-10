"""Smoke gate: Swimmer (FLUID) through the GPU-batched fields facade.

Swimmer is the only env with fluid forces (<option density=4000 viscosity=0.1>).
The batched facade used to RAISE at construction ("fluid not ported"); Stage A
wired compute_fluid_forces_fields into the integrators' passive seam, so the
guard was removed and Swimmer now runs on Phyics3dBatchedEnvFields. This checks:
  * construction + reset + a short step loop + a selective reset run clean, and
  * the state stays finite (fluid drag is dissipative — velocities must not blow
    up) and non-trivial (the swimmer actually moves).

Swimmer is hinge-only (no free joint) so Newton stays per-env on Apple — no
heavy cooperative/blocked kernel. RK4 is Swimmer's native integrator.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_batched_env_fields_swimmer.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_batched_env_fields import Phyics3dBatchedEnvFields
from mojo_rl.envs.swimmer.swimmer_xml import SwimmerModel
from mojo_rl.envs.swimmer.swimmer_config import SwimmerConfig

comptime N = 4
comptime OBS_DIM = SwimmerModel.OBS_DIM
comptime N_STEPS = 12
comptime EnvT = Phyics3dBatchedEnvFields[SwimmerModel, SwimmerConfig, N]


def main() raises:
    print("=== Swimmer FLUID batched-facade smoke, N =", N, "===")
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    print("  constructed ok (fluid guard removed)")

    env.reset_batch[N](ctx, UInt64(7))
    ctx.synchronize()
    print("  reset_batch ok")

    for s in range(N_STEPS):
        env.step_batch[N](ctx, UInt64(s + 1))
    ctx.synchronize()
    print("  stepped", N_STEPS, "steps ok")

    # Obs must be finite (fluid drag is dissipative) and non-trivial.
    var h_obs = ctx.enqueue_create_host_buffer[DT](N * OBS_DIM)
    ctx.enqueue_copy(h_obs, env._obs)
    ctx.synchronize()
    var max_mag = Float64(0)
    for i in range(N * OBS_DIM):
        var v = Float64(h_obs[i])
        if v != v or abs(v) > 1e6:
            raise Error("non-finite / exploded obs at " + String(i))
        if abs(v) > max_mag:
            max_mag = abs(v)
    print("  obs finite; max |obs| =", max_mag)
    if max_mag < 1e-9:
        raise Error("obs is all ~0 — swimmer vacuous")

    # qvel must stay finite (fluid drag dissipative).
    env.d.qvel.download(ctx)
    for i in range(N * SwimmerModel.NV):
        var v = Float64(env.d.qvel.data[i])
        if v != v or abs(v) > 1e6:
            raise Error("non-finite / exploded qvel at " + String(i))

    # Selective reset path (fluid lanes reset + FK-fields).
    env.selective_reset_batch[N](ctx, UInt64(99))
    ctx.synchronize()
    print("  selective_reset_batch ok")
    print("test_batched_env_fields_swimmer: ALL PASS")
