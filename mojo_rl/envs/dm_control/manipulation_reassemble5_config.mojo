"""`dm_control` `manipulation/reassemble_5_bricks_random_order_features` config.

`bricks.py::Reassemble` with five bricks and BOTH randomisation flags on — the
thirteenth and last `_features` task in the manipulation suite. Everything the
task DOES lives in `manipulation_reassemble`; this file is the model wiring and
the three numbers that differ.

    observation = desired_order(5) + robot(42) + 5 x brick(13)      (112)
    reward      = mean over the four DESIRED pairs, close_coef = 0

⚠⚠⚠ THE REFERENCE CHANGES ITS MODEL EVERY EPISODE AND WE DO NOT. Read
`manipulation_reassemble`'s relabeling section, and `manipulation_stack_random`'s
header behind it, before touching anything here. `initialize_episode_mjcf`
draws `initial_order` and removes the freejoint from `initial_order[0]`, so
which BODY is welded permutes per episode — measured over 20 resets, all five
occur (3/4/6/4/3). The workaround is a RELABELING: reference brick `r` is
played by our physical brick `sigma[r]`, with `sigma[initial_order[0]]` the one
this XML welded down.

⚠⚠ TWO ORDERS, AND ONLY `initial_order` DECIDES THE MODEL. `desired_order`
shares its first entry (the welded brick cannot be restacked) and is otherwise
drawn independently. Both are stored in `Data.meta`, in REFERENCE labels; the
observation emits `desired_order` as drawn and the brick blocks through
`sigma`, and the reward pairs are `sigma[desired_order[i]]`.

⚠ THIS IS THE TASK THE `META_IDX_TASK_PARAM` BLOCK WAS WIDENED FOR. Ten of the
twelve slots hold the two orders; four would not have been enough for one.
"""

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.manipulation_reassemble import (
    append_reassemble_random_obs,
    reassemble_random_reward,
    reassemble_random_draw_orders,
    reassemble_random_reset_full,
)


comptime N_BRICKS: Int = 5
comptime FIXED_BRICK: Int = 2
"""Which PHYSICAL brick this bake left without a freejoint — `duplo2x4_4/`,
body 21.

⚠ NOT A TASK CONSTANT. It is whatever `initialize_episode_mjcf` drew when the
XML was baked; the gate asserts it against a freshly constructed, once-reset
reference env, which is what the generator saw. Everything else is written in
terms of it."""

comptime OBS_DIM: Int = 112


struct Reassemble5Config(Phyics3dEnvConfig):
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
            append_reassemble_random_obs[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_bodies, m_joints, m_sites, N_BRICKS, FIXED_BRICK, obs)
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
                reassemble_random_reward[
                    DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
                ](d, N_BRICKS, FIXED_BRICK)
            ),
            False,
        )

    # === CPU: per-episode STATE — the grasp and the two order draws ========
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
            reassemble_random_draw_orders[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_joints, N_BRICKS)
        except:
            pass

    # === CPU: the assembled stack, then the arm ===========================
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
        reassemble_random_reset_full[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
        ](d, mf, N_BRICKS, FIXED_BRICK, "reassemble_5")
