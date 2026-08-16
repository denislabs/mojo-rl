"""`dm_control` `manipulation/stack_3_bricks_random_order_features` task config.

`bricks.py::Stack` with three bricks, a fixed base and `randomize_order=True`,
`target_height=3`. Everything the task DOES — the relabeling, the
observation, the reward and the four-statement reset — is shared with
`stack_2_of_3_bricks_random_order_features` and lives in `manipulation_stack_random`;
this file is the model wiring and the two numbers that differ.

    observation = desired_order(3) + robot(42) + 3 x brick(13)   (84)
    reward      = mean over the two stacked pairs

⚠⚠ THE REFERENCE'S MODEL CHANGES EVERY EPISODE AND OURS DOES NOT. Read
`manipulation_stack_random`'s header before touching anything here: the task is
made correct by RELABELING (reference brick `r` is played by our physical brick
`sigma(r)`, with `sigma(order[0])` the one this XML welded down), and the
obvious alternative — keeping every brick free and freezing the base — is
measurably wrong, because a brick with no freejoint is welded to the WORLD and
so cannot contact the ground at all.

⚠ `target_height` DEFAULTS TO `num_bricks` — `_stack(target_height=None)` —
so all three bricks are in the order and the reward averages two pairs.

"""

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.manipulation_stack_random import (
    append_stack_random_obs,
    stack_random_reward,
    stack_random_set_grasp_and_order,
    stack_random_reset_full,
)

from .manipulation_stack3r_def import Stack3RandomModel


comptime TARGET_HEIGHT: Int = 3
comptime OBS_DIM: Int = 84


struct Stack3RandomConfig(Phyics3dEnvConfig):
    # === Physics === (identical to every other task in this family)
    comptime FRAME_SKIP: Int = 20
    comptime MAX_STEPS: Int = 250
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime RNE_POST: Bool = True  # required by `joints_torque`
    comptime NMESH_VERTS: Int = 60000
    comptime HAS_GPU_HOOKS: Bool = False
    comptime USES_MOCAP: Bool = False

    @staticmethod
    def get_timestep() -> Float64:
        return 0.002

    # === CPU: Observation ===
    @staticmethod
    def custom_extract_obs_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        try:
            append_stack_random_obs[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_bodies, m_joints, m_sites, TARGET_HEIGHT, obs)
        except:
            return False
        return True

    # === CPU: Reward ===
    @staticmethod
    def compute_reward_and_done_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        return (
            Scalar[DTYPE](
                stack_random_reward[
                    DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
                ](d, TARGET_HEIGHT)
            ),
            False,
        )

    # === CPU: per-episode STATE — the grasp and the order draw ============
    @staticmethod
    def custom_reset_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        try:
            stack_random_set_grasp_and_order[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_joints, TARGET_HEIGHT)
        except:
            pass

    # === CPU: the three bricks, the settle, then the arm ==================
    @staticmethod
    def custom_reset_full_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        NGEOM: Int,
        NEQ: Int,
        NTEN: Int,
        NSITE: Int,
        NEXCL: Int,
        NMESHV: Int,
        NPAIR: Int,
        MAX_CONTACTS: Int,
    ](
        mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    ) raises:
        stack_random_reset_full[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
            Stack3RandomModel.CONE_TYPE,
            Stack3RandomModel.MAX_CONDIM,
            Stack3RandomModel.NOSLIP_ITER,
        ](d, mf, Self.get_timestep())
