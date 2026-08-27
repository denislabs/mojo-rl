"""C51 storage GPU smoke — single-env CartPole on a gpu-target trainer.

Exercises the GPU path: H2D obs bridge + online forward + D2H expected-Q argmax,
the on-device categorical projection / gather-slice / CE / scatter kernels, the
distributional diag kernel (`_c51_diag_kernel`) + device-resident accumulators.
Asserts learning beats random.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_c51_gpu_smoke.mojo
"""

from std.random import seed
from max.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.c51.trainer import C51Trainer
from mojo_rl.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train, run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.envs.cartpole import CartPoleEnv

comptime OBS = 4
comptime NA = 2
comptime N_ATOMS = 51
comptime H = 64
comptime BATCH = 32
comptime CAP = 20_000

comptime C51QNet = Sequential[
    Linear[OBS, H], ReLU[H], Linear[H, H], ReLU[H], Linear[H, NA * N_ATOMS],
]


def main() raises:
    print("=== C51 storage GPU smoke (CartPole) ===")
    seed(42)
    with DeviceContext() as ctx:
        var trainer = C51Trainer[
            "gpu", UniformSampleGpuStep[OBS, 1, BATCH, CAP], C51QNet,
            N_ATOMS=N_ATOMS, NUM_ACTIONS=NA,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](1e-4),
            epsilon_min=Scalar[DT](0.05),
            learning_starts=1_000,
            target_update_freq=1000,
            v_min=Scalar[DT](0.0),
            v_max=Scalar[DT](100.0),
        )
        var env = CartPoleEnv[DT]()
        _ = run_offpolicy_discrete_train(
            trainer, env, 14_000, ctx=ctx, print_every=5000, verbose=True,
        )
        var eval_env = CartPoleEnv[DT]()
        var ret = run_offpolicy_discrete_eval(
            trainer, eval_env, 5, max_steps_per_episode=200, verbose=False,
        )
        print("  eval mean return =", ret, " (random ~22)")
        assert_true(ret > Scalar[DT](60.0), "C51 GPU did not learn (eval <= 60)")
    print("=== PASSED ===")
