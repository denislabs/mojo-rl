"""`dm_control` `manipulation/stack_3_bricks_features` task config.

`bricks.py::Stack` with three bricks, a FIXED base and no order randomisation. Everything the task DOES — the observation, the
stud-to-hole reward and the four-statement reset — is shared with the other
fixed-order stack tasks and lives in `manipulation_stack_fixed`; this file is
the model wiring and the two numbers that differ.

    observation = robot(42) + 3 x brick(13)   (81)
    reward      = mean over the two stacked pairs

⚠ THE REWARD AVERAGES TWO PAIRS — (0,1) and (1,2) — so a tower with its bottom
pair clicked and its top pair scattered scores about 0.5. That is the shaping
`Stack` intends, and it is NOT the min or the sum.

"""

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.manipulation_stack_fixed import (
    append_stack_fixed_obs,
    stack_fixed_reward,
    stack_fixed_set_grasp,
    stack_fixed_reset_full,
    ROBOT_SITE_BASE,
    SITE_PINCH,
    stack_brick_body_of,
    stack_brick_frame_site_of,
    stack_brick_stud_0_of,
    stack_brick_hole_0_of,
    stack_free_slot_of,
    stack_qpos_adr_of,
    stack_dof_adr_of,
    CORNER_A,
    CORNER_B,
    CLOSE_COEF,
    PROP_BBOX_LOWER_X,
    PROP_BBOX_UPPER_X,
    TCP_BBOX_LOWER_Z,
    TCP_BBOX_UPPER_Z,
)

from .manipulation_stack_3_bricks_def import Stack3BricksModel


comptime N_BRICKS: Int = 3
comptime FIXED_BRICK: Int = 0
"""⚠ Brick 0, measured against the bake-time reference — `_add_or_remove_freejoints`
strips the freejoint from `desired_order[0]` and `desired_order` is `arange(3)`
here, so it is brick 0 every episode. Its pose is a MODEL field
(`body_pos`/`body_quat`) and it has NO qpos at all, which is why the other two
bricks live at qpos 9 and 16 rather than 9 and 16 counting from brick 0."""

comptime OBS_DIM: Int = 81


struct Stack3BricksConfig(Phyics3dEnvConfig):
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
            append_stack_fixed_obs[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_bodies, m_joints, m_sites, N_BRICKS, obs)
        except:
            return False
        return True

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
                stack_fixed_reward[
                    DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
                ](d, N_BRICKS)
            ),
            False,
        )

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
            stack_fixed_set_grasp[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_joints)
        except:
            pass

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
        mut mf: Model[
            DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR,
        ],
    ) raises:
        stack_fixed_reset_full[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
            Stack3BricksModel.CONE_TYPE,
            Stack3BricksModel.MAX_CONDIM,
            Stack3BricksModel.NOSLIP_ITER,
        ](d, mf, N_BRICKS, FIXED_BRICK, Self.get_timestep())
