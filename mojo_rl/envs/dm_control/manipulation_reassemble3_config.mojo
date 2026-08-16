"""`dm_control` `manipulation/reassemble_3_bricks_fixed_order_features` config.

`bricks.py::Reassemble` with three bricks and BOTH randomisation flags off.
Everything the task DOES lives in `manipulation_reassemble`; this file is the
model wiring and the two orders.

    observation = robot(42) + 3 x brick(13)          (81, no `desired_order`)
    reward      = mean over the two DESIRED pairs, close_coef = 0

⚠⚠ THE TWO ORDERS ARE BOTH CONSTANTS AND THEY ARE NOT THE SAME CONSTANT.
`randomize_initial_order` is False so `initial_order` stays `arange(3)`, and
`initialize_episode_mjcf` then derives `desired_order` from it: entry 0 is
shared (that brick is welded down) and the rest are REVERSED. So the episode
starts in stack 0-1-2 and is rewarded for stack 0-2-1, and the reward at reset
is 0 by construction. Measured on the reference: reward 0.0 at a freshly built
initial stack and 1.0 at a built desired stack.

⚠ NO RELABELING AND NO `desired_order` OBSERVABLE. Both flags are off, so the
reference's model is stable across episodes and reference brick `r` is our
physical brick `r` — `sigma` is the identity. `reassemble_5` is the task that
needs `manipulation_stack_random`'s machinery; this one does not.
"""

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.manipulation_reassemble import (
    append_reassemble_obs,
    reassemble_reward,
    reassemble_set_grasp,
    reassemble_reset_full,
)


comptime N_BRICKS: Int = 3
comptime FIXED_BRICK: Int = 0
"""`initial_order[0]`, whose freejoint `_add_or_remove_freejoints` removes.
⚠ Asserted in the gate against MuJoCo's own tables, not assumed."""

comptime OBS_DIM: Int = 81


@always_inline
def initial_order() -> List[Int]:
    """`np.arange(num_bricks)` — never shuffled, `randomize_initial_order` is
    False."""
    var o = List[Int]()
    for i in range(N_BRICKS):
        o.append(i)
    return o^


@always_inline
def desired_order() -> List[Int]:
    """`desired_order[0] = initial_order[0]`, then `initial_order[-1:0:-1]`.

    For the identity initial order that is [0, 2, 1]. Written out rather than
    computed from a loop because the slice above is the kind of expression that
    is easy to transcribe off by one, and the gate pins it against the
    reference's own array.
    """
    var o = List[Int]()
    o.append(0)
    o.append(2)
    o.append(1)
    return o^


@always_inline
def identity_sigma() -> List[Int]:
    """No relabeling: the reference's model does not move under us."""
    var s = List[Int]()
    for i in range(N_BRICKS):
        s.append(i)
    return s^


struct Reassemble3Config(Phyics3dEnvConfig):
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
        d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        try:
            # ⚠ EMPTY `desired_obs` — `randomize_desired_order` is False, so
            # the task observable does not exist and the vector starts with the
            # robot block.
            append_reassemble_obs[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](
                d, m_bodies, m_joints, m_sites, List[Int](), identity_sigma(),
                obs,
            )
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
                reassemble_reward[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
                    d, desired_order()
                )
            ),
            False,
        )

    # === CPU: per-episode STATE — the grasp ===============================
    @staticmethod
    def custom_reset_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        try:
            reassemble_set_grasp[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_joints)
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
        mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
        mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    ) raises:
        reassemble_reset_full[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
        ](d, mf, initial_order(), N_BRICKS, FIXED_BRICK, "reassemble_3")
