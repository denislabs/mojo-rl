"""P5 smoke: SAC trains against Phyics3dEnv[Hopper] — a CONTACT
locomotion env entirely on the fields path (per-stage RK4 contact + limit
solving). Integration smoke, not convergence: 2k CPU steps, asserts the
loop completes, episodes terminate via the health check, and rewards flow
(hopper pays an alive bonus, so returns must be positive).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_env_fields_hopper_smoke.mojo
"""

from std.random import seed
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.hopper import HopperModel, HopperConfig

comptime EnvT = Phyics3dEnv[
    HopperModel, HopperConfig, DT, TERMINATE_ON_UNHEALTHY=True
]
comptime OBS_DIM = EnvT.OBS_DIM  # 11
comptime ACT_DIM = EnvT.ACTION_DIM  # 3
comptime NUM_STEPS = 2_000


def main() raises:
    seed(11)
    print("--- SAC smoke on Phyics3dEnv[Hopper] (contacts, CPU) ---")
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    var agent = SAC["cpu", OBS_DIM, ACT_DIM, 32, 10_000, 64](
        window_size=20,
        initial_episode_fill=0.0,
    )
    _ = agent.train_single[EnvT](
        env,
        NUM_STEPS,
        print_every=1_000,
        verbose=True,
    )
    print(
        "  episodes:", agent.ep_count(),
        " mean return (last 20):", agent.mean_return(),
    )
    if agent.ep_count() < 3:
        raise Error("too few episodes — termination plumbing broken?")
    var mr = Float64(agent.mean_return())
    if not (mr > 0.0 and mr < 10_000.0):
        raise Error("mean return not sane — reward plumbing broken?")
    print("test_env_fields_hopper_smoke: ALL PASS")
