"""NVIDIA localization probe for the humanoid batched-facade training crash.

The humanoid single-integrator probe (test_humanoid_fields_sync_probe) passes
on NVIDIA — the physics step is clean. So the training crash is in the batched
FACADE's own kernels (hooks, slab<->fields sync, cfrc_ext/cvel, extract_obs).
This constructs the batched facade and steps it with step_batch[DEBUG=True],
which ctx.synchronize()+prints after each kernel group. The last
"[step_batch] N ..." printed before a crash names the faulting group.

Run: MODULAR_DEBUG=device-sync-mode pixi run -e nvidia mojo run -I . \
        tests/physics3d/test_humanoid_batched_facade_probe.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_batched_env_fields import Phyics3dBatchedEnvFields
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel
from mojo_rl.envs.humanoid.humanoid_config import HumanoidConfig

comptime N = 2
comptime E = Phyics3dBatchedEnvFields[
    HumanoidModel, HumanoidConfig, N, TERMINATE_ON_UNHEALTHY=True
]


def main() raises:
    print("=== Humanoid BATCHED-FACADE step probe (DEBUG sync) ===")
    print("INTEGRATOR =", HumanoidConfig.INTEGRATOR)
    var ctx = DeviceContext()
    var env = E(ctx)
    print("constructed ok")
    env.reset_batch[N](ctx, UInt64(0))
    ctx.synchronize()
    print("reset_batch ok")
    for s in range(3):
        print("--- step", s, "---")
        env._step_impl[N, True](ctx, UInt64(s + 1))
    print("=== BATCHED FACADE STEPS OK ===")
