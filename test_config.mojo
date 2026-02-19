from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from envs.hopper.hopper_config import HopperConfig
from envs.half_cheetah.half_cheetah_def import HalfCheetahModel
from envs.hopper.hopper_def import HopperModel
from envs.phyics3d_env import Phyics3dEnv

fn main():
    print("HalfCheetahConfig: FRAME_SKIP=", HalfCheetahConfig.FRAME_SKIP, "INTEGRATOR_WS_EXTRA=", HalfCheetahConfig.INTEGRATOR_WS_EXTRA)
    print("HopperConfig: FRAME_SKIP=", HopperConfig.FRAME_SKIP, "INTEGRATOR_WS_EXTRA=", HopperConfig.INTEGRATOR_WS_EXTRA)

    # Test Phyics3dEnv type comptime constants
    comptime HalfCheetahEnv = Phyics3dEnv[HalfCheetahModel, HalfCheetahConfig]
    comptime HopperEnv = Phyics3dEnv[HopperModel, HopperConfig]

    print("HalfCheetahEnv STATE_SIZE=", HalfCheetahEnv.STATE_SIZE)
    print("  STEP_WS_SHARED=", HalfCheetahEnv.STEP_WS_SHARED)
    print("  STEP_WS_PER_ENV=", HalfCheetahEnv.STEP_WS_PER_ENV)
    print("HopperEnv STATE_SIZE=", HopperEnv.STATE_SIZE)
    print("  STEP_WS_SHARED=", HopperEnv.STEP_WS_SHARED)
    print("  STEP_WS_PER_ENV=", HopperEnv.STEP_WS_PER_ENV)

    # Test instantiation
    var cheetah = HalfCheetahEnv()
    print("HalfCheetahEnv created, current_step=", cheetah.get_current_step())

    var hopper = HopperEnv()
    print("HopperEnv created, current_step=", hopper.get_current_step())

    print("All Phyics3dEnv OK!")
