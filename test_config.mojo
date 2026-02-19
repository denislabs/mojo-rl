from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from envs.hopper.hopper_config import HopperConfig
from envs.mujoco_env import MuJoCoEnv

fn main():
    print("HalfCheetahConfig: NQ=", HalfCheetahConfig.NQ, "NV=", HalfCheetahConfig.NV)
    print("  OBS_DIM=", HalfCheetahConfig.OBS_DIM, "ACTION_DIM=", HalfCheetahConfig.ACTION_DIM)
    print("  FRAME_SKIP=", HalfCheetahConfig.FRAME_SKIP, "INTEGRATOR_WS_EXTRA=", HalfCheetahConfig.INTEGRATOR_WS_EXTRA)
    print("HopperConfig: NQ=", HopperConfig.NQ, "NV=", HopperConfig.NV)
    print("  OBS_DIM=", HopperConfig.OBS_DIM, "ACTION_DIM=", HopperConfig.ACTION_DIM)
    print("  FRAME_SKIP=", HopperConfig.FRAME_SKIP, "INTEGRATOR_WS_EXTRA=", HopperConfig.INTEGRATOR_WS_EXTRA)

    # Test MuJoCoEnv type comptime constants
    comptime HalfCheetahEnv = MuJoCoEnv[HalfCheetahConfig]
    comptime HopperEnv = MuJoCoEnv[HopperConfig]

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

    print("All MuJoCoEnv OK!")
