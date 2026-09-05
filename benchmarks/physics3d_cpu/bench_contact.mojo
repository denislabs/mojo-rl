"""Contact-rich models, CPU step vs MuJoCo — see `harness.mojo`.

    pixi run mojo build -I . -I benchmarks benchmarks/physics3d_cpu/bench_contact.mojo -o <bin>
    <bin> <sawyer_reach|dog_stand|humanoid_cmu> [warmup] [steps] [rounds]

All three XMLs step with Euler in MuJoCo (Sawyer says so; dog and humanoid_CMU
say nothing and get the default). These are the rows where the constraint
solver, not collision, has a chance to be the gap.
"""

from std.sys import argv

from mojo_rl.envs.metaworld import SawyerReachModel, SawyerReachConfig
from mojo_rl.envs.dm_control.dog import DMDogStandWalkModel, DMDogStandConfig
from mojo_rl.envs.dm_control.humanoid_cmu import (
    DMHumanoidCMUModel,
    DMHumanoidCMUConfig,
    WALK_SPEED,
)

from physics3d_cpu.harness import bench

comptime DT = DType.float32


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: bench_contact <model> [warmup] [steps] [rounds]")
        return
    var name = String(args[1])
    var warmup = Int(atol(String(args[2]))) if len(args) > 2 else 2000
    var steps = Int(atol(String(args[3]))) if len(args) > 3 else 20000
    var rounds = Int(atol(String(args[4]))) if len(args) > 4 else 1
    if name == "sawyer_reach":
        bench[SawyerReachModel, SawyerReachConfig, DT, True](name, warmup, steps, rounds)
    elif name == "dog_stand":
        bench[DMDogStandWalkModel, DMDogStandConfig, DT, True](name, warmup, steps, rounds)
    elif name == "humanoid_cmu":
        bench[DMHumanoidCMUModel, DMHumanoidCMUConfig[WALK_SPEED], DT, True](
            name, warmup, steps, rounds
        )
    else:
        print("unknown model:", name)
