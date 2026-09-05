"""Gym MuJoCo models, CPU step vs MuJoCo — see `harness.mojo`.

    pixi run mojo build -I . -I benchmarks benchmarks/physics3d_cpu/bench_gym.mojo -o <bin>
    <bin> <walker2d|hopper|humanoid|half_cheetah|ant> [warmup] [steps] [rounds]

Integrators follow each XML's `<option integrator>`: RK4 for walker2d, hopper,
humanoid and ant; Euler (the default) for half_cheetah.
"""

from std.sys import argv

from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.walker2d.walker2d_config import Walker2dConfig
from mojo_rl.envs.hopper.hopper_xml import HopperModel
from mojo_rl.envs.hopper.hopper_config import HopperConfig
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel
from mojo_rl.envs.humanoid.humanoid_config import HumanoidConfig
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from mojo_rl.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from mojo_rl.envs.ant.ant_xml import AntModel
from mojo_rl.envs.ant.ant_config import AntConfig

from physics3d_cpu.harness import bench

comptime DT = DType.float32


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: bench_gym <model> [warmup] [steps] [rounds]")
        return
    var name = String(args[1])
    var warmup = Int(atol(String(args[2]))) if len(args) > 2 else 2000
    var steps = Int(atol(String(args[3]))) if len(args) > 3 else 20000
    var rounds = Int(atol(String(args[4]))) if len(args) > 4 else 1
    if name == "walker2d":
        bench[Walker2dModel, Walker2dConfig, DT, False](name, warmup, steps, rounds)
    elif name == "hopper":
        bench[HopperModel, HopperConfig, DT, False](name, warmup, steps, rounds)
    elif name == "humanoid":
        bench[HumanoidModel, HumanoidConfig, DT, False](name, warmup, steps, rounds)
    elif name == "half_cheetah":
        bench[HalfCheetahModel, HalfCheetahConfig, DT, True](name, warmup, steps, rounds)
    elif name == "ant":
        bench[AntModel, AntConfig, DT, False](name, warmup, steps, rounds)
    else:
        print("unknown model:", name)
