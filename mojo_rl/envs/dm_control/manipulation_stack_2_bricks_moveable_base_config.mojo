"""`dm_control` `manipulation/stack_2_bricks_moveable_base_features` task config.

`bricks.py::Stack` with two bricks, NO fixed base and no order randomisation. Everything the task DOES — the observation, the
stud-to-hole reward and the four-statement reset — is shared with the other
fixed-order stack tasks and lives in `manipulation_stack_fixed`; this file is
the model wiring and the two numbers that differ.

    observation = robot(42) + 2 x brick(13)   (68)
    reward      = mean over the one stacked pair

⚠⚠ `nq` IS 23, NOT `stack_2_bricks`' 16, for the same two bricks and the same
185 geoms. The two tasks differ ONLY in whether the base brick has a freejoint,
and that one flag moves the whole coordinate layout. They are not the same
model with a flag.

⚠ AND THE ACCEPTED RESET SET IS GENUINELY WIDER. dm_control's TCP predicate
rejects a robot pose that penetrates an external body WITHOUT a freejoint; with
no fixed brick there is no such body among the props, so arm-versus-brick is
never a rejection reason here. That is a real difference in the reset
distribution, not a detail.

"""

from mojo_rl.physics3d.fields import Data, Model, Dims, DimsLike
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

from .manipulation_stack_2_bricks_moveable_base_def import Stack2MoveableModel


comptime N_BRICKS: Int = 2
comptime FIXED_BRICK: Int = -1
"""⚠⚠ -1 MEANS NO BRICK IS FIXED, and that is this task's entire content.
`moveable_base=True` gives `_add_or_remove_freejoints(fixed_indices=[])`, so
every brick keeps its freejoint: `nq` is 23 rather than `stack_2_bricks`' 16,
both bricks live in qpos (9 and 16), and NOTHING is written to `body_pos`.
Passing 0 here instead of -1 would write a model field that qpos then
overrides — silently, because brick 0 would still have coordinates too."""

comptime OBS_DIM: Int = 68


struct Stack2MoveableConfig(Phyics3dEnvConfig):
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
    def custom_extract_obs_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        try:
            append_stack_fixed_obs[DTYPE](d, m_bodies, m_joints, m_sites, N_BRICKS, obs)
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
        d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
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
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        try:
            stack_fixed_set_grasp[DTYPE, D](d, m_joints)
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
        mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
        mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    ) raises:
        stack_fixed_reset_full[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
            Stack2MoveableModel.CONE_TYPE,
            Stack2MoveableModel.MAX_CONDIM,
            Stack2MoveableModel.NOSLIP_ITER,
        ](d, mf, N_BRICKS, FIXED_BRICK, Self.get_timestep())
