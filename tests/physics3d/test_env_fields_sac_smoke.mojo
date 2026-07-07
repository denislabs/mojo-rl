"""P5 smoke: deep SAC trains against Phyics3dEnvFields end-to-end.

The point is INTEGRATION, not convergence (Apple = smoke only): the
`deep_agents.sac` facade + single-env off-policy driver run a real training
loop (replay, episode accounting via was_terminated, actor/critic updates)
against the fields-path env for 2k steps on InvertedPendulum. Asserts the
loop completes, episodes are counted, and the mean return is a sane
positive number (IP pays +1 per surviving step, so even a random policy
earns > 0).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_env_fields_sac_smoke.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC
from mojo_rl.envs.phyics3d_env_fields import Phyics3dEnvFields
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_xml import (
    InvertedPendulumModel,
)
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_config import (
    InvertedPendulumConfig,
)

comptime EnvT = Phyics3dEnvFields[
    InvertedPendulumModel,
    InvertedPendulumConfig,
    DT,
    TERMINATE_ON_UNHEALTHY=True,
]
comptime OBS_DIM = EnvT.OBS_DIM  # 4
comptime ACT_DIM = EnvT.ACTION_DIM  # 1
comptime HIDDEN = 64
comptime BATCH = 32
comptime REPLAY_CAPACITY = 10_000
comptime NUM_STEPS = 2_000


def main() raises:
    seed(7)
    print("--- SAC smoke on Phyics3dEnvFields[InvertedPendulum] (CPU) ---")
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    var agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN](
        window_size=20,
        initial_episode_fill=0.0,
    )
    _ = agent.train_single[EnvT](
        env,
        NUM_STEPS,
        print_every=500,
        verbose=True,
    )
    print(
        "  episodes:", agent.ep_count(),
        " mean return (last 20):", agent.mean_return(),
    )
    if agent.ep_count() < 5:
        raise Error("too few episodes — termination/reset plumbing broken?")
    var mr = Float64(agent.mean_return())
    if not (mr > 0.0 and mr <= 1000.0):
        raise Error("mean return not sane — reward plumbing broken?")
    print("test_env_fields_sac_smoke: ALL PASS")
