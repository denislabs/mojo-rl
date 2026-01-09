from deep_agents.gpu import train_a2c
from random import seed
from gpu.host import DeviceContext


fn main() raises:
    seed(42)
    print("GPU A2C on CartPole")
    print()

    with DeviceContext() as ctx:
        _ = train_a2c(ctx, num_updates=100, verbose=True)
