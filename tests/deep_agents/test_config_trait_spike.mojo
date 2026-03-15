"""Test: can we define a Config trait with comptime Model fields?"""

from mojo_rl.nn.model import Model, Linear, LinearReLU, LinearTanh, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import Network


trait OffPolicyConfig:
    comptime obs_dim: Int
    comptime action_dim: Int
    comptime batch_size: Int
    comptime buffer_capacity: Int
    comptime ActorModel: Model
    comptime CriticModel: Model
    comptime ActorOpt: Optimizer
    comptime CriticOpt: Optimizer
    comptime NUM_CRITICS: Int


struct DDPGCfg[
    OBS: Int, ACT: Int, HIDDEN: Int = 64,
    cap: Int = 1000, bs: Int = 32,
](OffPolicyConfig):
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.bs
    comptime buffer_capacity: Int = Self.cap
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.ACT],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[Self.OBS + Self.ACT, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[0.001]
    comptime CriticOpt = Adam[0.001]
    comptime NUM_CRITICS: Int = 1


struct TD3Cfg[
    OBS: Int, ACT: Int, HIDDEN: Int = 64,
    cap: Int = 1000, bs: Int = 32,
](OffPolicyConfig):
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.bs
    comptime buffer_capacity: Int = Self.cap
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.ACT],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[Self.OBS + Self.ACT, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[0.001]
    comptime CriticOpt = Adam[0.001]
    comptime NUM_CRITICS: Int = 2  # Twin critics!


struct TestAgent[Config: OffPolicyConfig]:
    comptime OBS = Self.Config.obs_dim
    comptime ACTIONS = Self.Config.action_dim
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[Self.Config.CriticModel, Self.Config.CriticOpt]

    fn __init__(out self):
        pass

    fn info(self):
        print("OBS:", Self.OBS, "ACTIONS:", Self.ACTIONS)
        print("Actor params:", Self.Config.ActorModel.PARAM_SIZE)
        print("Critic params:", Self.Config.CriticModel.PARAM_SIZE)
        print("NUM_CRITICS:", Self.Config.NUM_CRITICS)


fn main():
    print("=== Config Trait Spike ===")

    print("\nDDPG Config:")
    var ddpg = TestAgent[DDPGCfg[3, 1]]()
    ddpg.info()

    print("\nTD3 Config:")
    var td3 = TestAgent[TD3Cfg[3, 1]]()
    td3.info()

    print("\n=== OK ===")
