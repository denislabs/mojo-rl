"""Contact-rich models, CPU step vs MuJoCo — see `harness.mojo`.

    pixi run mojo build -I . -I benchmarks benchmarks/physics3d_cpu/bench_contact.mojo -o <bin>
    <bin> <sawyer_reach|dog_stand|humanoid_cmu|reassemble3|reassemble5> [warmup] [steps] [rounds]

All five XMLs step with Euler in MuJoCo (Sawyer and the reassemble scenes say
so; dog and humanoid_CMU say nothing and get the default). These are the rows
where the constraint solver, not collision, has a chance to be the gap.

The two `reassemble` scenes start from their TASK RESET (`TASK_POSE`), not
from qpos0 — see `harness.mojo` — and hand that pose to the twin through the
optional 5th argument. They are dm_control manipulation's brick piles — a
stack of 3 / 5 bricks in contact from step 1 (MuJoCo: 92 / 231 contacts on
average, max 138 / 361), ELLIPTIC cone, `noslip_iterations=5`. They are the
contact-dense end of the sweep, they take the elliptic Newton path, and they
have a history of being both slow and unstable here.
"""

from std.sys import argv

from mojo_rl.envs.metaworld import SawyerReachModel, SawyerReachConfig
from mojo_rl.envs.dm_control.dog import DMDogStandWalkModel, DMDogStandConfig
from mojo_rl.envs.dm_control.humanoid_cmu import (
    DMHumanoidCMUModel,
    DMHumanoidCMUConfig,
    WALK_SPEED,
)
from mojo_rl.envs.dm_control.manipulation_reassemble3_def import Reassemble3Model
from mojo_rl.envs.dm_control.manipulation_reassemble3_config import Reassemble3Config
from mojo_rl.envs.dm_control.manipulation_reassemble5_def import Reassemble5Model
from mojo_rl.envs.dm_control.manipulation_reassemble5_config import Reassemble5Config

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
    # Optional 5th arg: where the reassemble rows write the task-reset pose
    # for the MuJoCo twin (`--pose`). Ignored by the other rows.
    var pose_file = String(args[5]) if len(args) > 5 else String("")
    if name == "sawyer_reach":
        bench[SawyerReachModel, SawyerReachConfig, DT, True](name, warmup, steps, rounds)
    elif name == "dog_stand":
        bench[DMDogStandWalkModel, DMDogStandConfig, DT, True](name, warmup, steps, rounds)
    elif name == "humanoid_cmu":
        bench[DMHumanoidCMUModel, DMHumanoidCMUConfig[WALK_SPEED], DT, True](
            name, warmup, steps, rounds
        )
    elif name == "reassemble3":
        bench[Reassemble3Model, Reassemble3Config, DT, True, TASK_POSE=True](
            name, warmup, steps, rounds, pose_file
        )
    elif name == "reassemble3_f64":
        bench[
            Reassemble3Model, Reassemble3Config, DType.float64, True,
            TASK_POSE=True,
        ](name, warmup, steps, rounds, pose_file)
    elif name == "reassemble5":
        bench[Reassemble5Model, Reassemble5Config, DT, True, TASK_POSE=True](
            name, warmup, steps, rounds, pose_file
        )
    else:
        print("unknown model:", name)
