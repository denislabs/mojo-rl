from deep_agents.gpu import A2CAgent
from envs import CartPoleEnv
from random import seed
from gpu.host import DeviceContext


fn main() raises:
    seed(42)
    print("GPU A2C on CartPole")
    print()

    with DeviceContext() as ctx:
        # CartPoleEnv implements both CPU and GPU traits
        _ = A2CAgent.train[CartPoleEnv](ctx, num_updates=100, verbose=True)
