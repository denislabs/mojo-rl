from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel, HalfCheetahBodies, HalfCheetahJoints,
    HalfCheetahGeoms, HalfCheetahActuators,
)
from envs.hopper.hopper_def import (
    HopperModel, HopperBodies, HopperJoints,
    HopperGeoms, HopperActuators,
)

fn main():
    print("HalfCheetah: NQ=", HalfCheetahModel.NQ, "NV=", HalfCheetahModel.NV)
    print("Hopper: NQ=", HopperModel.NQ, "NV=", HopperModel.NV)
    # Test that *Like trait members work on Joints
    print("HCJoints.OBS_DIM=", HalfCheetahJoints.OBS_DIM, "ACTION_DIM=", HalfCheetahJoints.ACTION_DIM)
    print("HJoints.OBS_DIM=", HopperJoints.OBS_DIM, "ACTION_DIM=", HopperJoints.ACTION_DIM)
    # Test OBS_DIM/ACTION_DIM through ModelDef
    print("HCModel.OBS_DIM=", HalfCheetahModel.OBS_DIM, "ACTION_DIM=", HalfCheetahModel.ACTION_DIM)
    print("HModel.OBS_DIM=", HopperModel.OBS_DIM, "ACTION_DIM=", HopperModel.ACTION_DIM)
    print("All OK!")
