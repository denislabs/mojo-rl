from deep_agents.gpu import A2CAgent
from envs import GPUCartPole
from random import seed
from gpu.host import DeviceContext


fn main() raises:
    seed(42)
    print("GPU A2C on CartPole")
    print()

    with DeviceContext() as ctx:
        # Pass environment type explicitly - no instance needed
        _ = A2CAgent.train[GPUCartPole](ctx, num_updates=100, verbose=True)
