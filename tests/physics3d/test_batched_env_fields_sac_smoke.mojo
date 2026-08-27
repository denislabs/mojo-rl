"""Driver smoke: SAC (GPU) trains against Phyics3dBatchedEnv[Hopper]
through the real batched off-policy driver (`run_offpolicy_train_batched`
via the `SAC[...]` facade) — the NVIDIA-relevant training path, with the
physics entirely on the per-field tensor GPU path.

Integration, not convergence (Apple = smoke only): 1.5k driver steps at
N_ENVS=4, asserts the loop completes, episodes terminate via the health
check (selective GPU resets), and rewards flow (hopper pays an alive
bonus, so returns must be positive).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_batched_env_fields_sac_smoke.mojo
"""

from std.random import seed
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.hopper import HopperModel, HopperConfig

comptime N_ENVS = 4
comptime EnvT = Phyics3dBatchedEnv[
    HopperModel, HopperConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=True
]
comptime OBS_DIM = EnvT.OBS_DIM  # 11
comptime ACT_DIM = EnvT.ACT_DIM  # 3
comptime NUM_STEPS = 1_500


def main() raises:
    seed(11)
    print("--- SAC (GPU) smoke on Phyics3dBatchedEnv[Hopper] ---")
    with DeviceContext() as ctx:
        var agent = SAC["gpu", OBS_DIM, ACT_DIM, 64, 20_000, 64](
            ctx=ctx,
            learning_starts=400,
            window_size=20,
            initial_episode_fill=0.0,
        )
        var env = EnvT(ctx)
        # USE_ENV_CUDA_GRAPH=False: the driver's inline env-capture path
        # (marked TEMP DIAGNOSTIC in driver_offpolicy.mojo) no-ops capture
        # AND replay on Apple, freezing the env after iteration 0 — the
        # known pre-existing Apple failure mode of the GPU batched tests.
        # Capture is an NVIDIA-only win anyway.
        _ = agent.train[EnvT, N_ENVS=N_ENVS, USE_ENV_CUDA_GRAPH=False](
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=500,
            verbose=True,
        )
        print(
            "  episodes:", agent.ep_count(),
            " mean return (last 20):", agent.mean_return(),
        )
        if agent.ep_count() < 3:
            raise Error("too few episodes — GPU termination/reset broken?")
        var mr = Float64(agent.mean_return())
        if not (mr > 0.0 and mr < 10_000.0):
            raise Error("mean return not sane — reward plumbing broken?")
    print("test_batched_env_fields_sac_smoke: ALL PASS")
