"""Construction smoke for the new Design-F presets (SAC/DDPG/TD3/PPO/A2C/MBPO).

Builds each capitalized preset on the CPU target and runs a greedy/act
probe where cheap — enough to typecheck the whole config → agent type
graph and confirm the fused default nets instantiate.
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.sac import SAC
from mojo_rl.deep_agents2.ddpg import DDPG
from mojo_rl.deep_agents2.td3 import TD3
from mojo_rl.deep_agents2.ppo import PPO
from mojo_rl.deep_agents2.ppo_discrete import PPODiscrete
from mojo_rl.deep_agents2.a2c import A2C, A2CDiscrete
from mojo_rl.deep_agents2.mbpo import MBPO


def main() raises:
    print("=== new config-preset construction smoke (CPU) ===")

    comptime OBS = 4
    comptime ACT = 2
    comptime N_ACT = 2
    comptime ROLLOUT = 8
    comptime MB = 8
    comptime EPOCHS = 2

    var sac = SAC["cpu", OBS, ACT, 32, 1024]()
    _ = sac
    print("  SAC          built OK")

    var ddpg = DDPG["cpu", OBS, ACT, 32, 1024]()
    _ = ddpg
    print("  DDPG         built OK")

    var td3 = TD3["cpu", OBS, ACT, 32, 1024]()
    _ = td3
    print("  TD3          built OK")

    var ppo = PPO["cpu", OBS, ACT, ROLLOUT, MB, EPOCHS]()
    _ = ppo
    print("  PPO          built OK")

    var ppod = PPODiscrete["cpu", OBS, N_ACT, ROLLOUT, MB, EPOCHS]()
    _ = ppod
    print("  PPODiscrete  built OK")

    var a2c = A2C["cpu", OBS, ACT, ROLLOUT]()
    _ = a2c
    print("  A2C          built OK")

    var a2cd = A2CDiscrete["cpu", OBS, N_ACT, ROLLOUT]()
    _ = a2cd
    print("  A2CDiscrete  built OK")

    var mbpo = MBPO["cpu", OBS, ACT, 32, 1024, 1024]()
    _ = mbpo
    print("  MBPO         built OK")

    print("ALL PRESETS CONSTRUCTED")
