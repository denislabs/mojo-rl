"""SO-ARM101 and the parked-prop scenes, CPU step vs MuJoCo — see `harness.mojo`.

    pixi run mojo build -I . -I benchmarks benchmarks/physics3d_cpu/bench_so101.mojo -o <bin>
    <bin> <so_arm101|so_arm101_f64|park_k0|park_k3|park_k6|park_k9> [warmup] [steps] [rounds]

⚠ EULER ON EVERY ROW. None of these XMLs carries `<option integrator>`, so
MuJoCo steps them with Euler; `So101ParkProbeConfig` inherits "rk4" for the
GPU probe, and matching that here would charge four stages against one. The
park scenes are the block-diagonal campaign's scenes
(`docs/BLOCK_DIAGONAL_MASS_MATRIX_PLAN.md` §1.1), so this is the CPU column
of that table: the same `k` slots, one lane, no launch overhead.

`so_arm101_f64` is the dtype control: the same model in the reference's own
precision, so a ratio can be split into "the algorithm" and "the float".
"""

from std.sys import argv

from mojo_rl.envs.robots import SoArm101Model, SoArm101ReachConfig
from mojo_rl.envs.robots.so101_park_config import So101ParkProbeConfig
from mojo_rl.envs.robots.so101_park_xml import (
    SoArm101ParkK0Model,
    SoArm101ParkK3Model,
    SoArm101ParkK6Model,
    SoArm101ParkK9Model,
)

from physics3d_cpu.harness import bench

comptime DT = DType.float32


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: bench_so101 <model> [warmup] [steps] [rounds]")
        return
    var name = String(args[1])
    var warmup = Int(atol(String(args[2]))) if len(args) > 2 else 2000
    var steps = Int(atol(String(args[3]))) if len(args) > 3 else 20000
    var rounds = Int(atol(String(args[4]))) if len(args) > 4 else 1
    if name == "so_arm101":
        bench[SoArm101Model, SoArm101ReachConfig, DT, True](name, warmup, steps, rounds)
    elif name == "so_arm101_f64":
        bench[SoArm101Model, SoArm101ReachConfig, DType.float64, True](
            name, warmup, steps, rounds
        )
    elif name == "park_k0":
        bench[SoArm101ParkK0Model, So101ParkProbeConfig[6, 6, 0], DT, True](
            name, warmup, steps, rounds
        )
    elif name == "park_k3":
        bench[SoArm101ParkK3Model, So101ParkProbeConfig[27, 24, 3], DT, True](
            name, warmup, steps, rounds
        )
    elif name == "park_k6":
        bench[SoArm101ParkK6Model, So101ParkProbeConfig[48, 42, 6], DT, True](
            name, warmup, steps, rounds
        )
    elif name == "park_k9":
        bench[SoArm101ParkK9Model, So101ParkProbeConfig[69, 60, 9], DT, True](
            name, warmup, steps, rounds
        )
    else:
        print("unknown model:", name)
