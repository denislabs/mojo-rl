"""Test composable GPU RL framework."""

from deep_rl.gpu.env_trait import GPUEnv, gpu_random_range
from deep_rl.gpu.envs.cartpole import GPUCartPole


def main():
    print("Testing GPUCartPole")
    print("OBS_DIM:", GPUCartPole.OBS_DIM)
    print("NUM_ACTIONS:", GPUCartPole.NUM_ACTIONS)
    print("STATE_SIZE:", GPUCartPole.STATE_SIZE)
