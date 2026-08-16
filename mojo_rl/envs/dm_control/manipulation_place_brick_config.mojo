"""`dm_control` `manipulation/place_brick_features` task config.

`Place` with a second Duplo as the cradle. Everything the task DOES —
the 58-float observation, the three-term reward, the four-statement reset —
is shared with `place_cradle_features` and lives in
`manipulation_place_common`; this file is the model wiring and nothing else.

⚠ THE TWO `place_*` TASKS SHARE EVERY ELEMENT ID. That is a measurement, not a
convenience: the cradle entity attaches to the PEDESTAL, so it lands after
every element the task reads, and both gates assert the ids against MuJoCo
rather than inheriting them from here.

⚠ THE CRADLE IS A SECOND, FIXED DUPLO — `props.Duplo()` with no observable
options, attached to the pedestal. It brings 41 more geoms, 17 more sites and
five more sensors, NONE of which is an enabled observable, so the observation
is byte-for-byte `place_cradle`'s while the model is 60% larger. It also means
this model has TWO bricks whose stud radius `initialize_episode_mjcf` redraws;
see `manipulation_ref._load`.

"""

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.manipulation_place_common import (
    append_place_obs,
    place_reward,
    place_set_grasp,
    place_reset_full,
)

from .manipulation_place_brick_def import PlaceBrickModel


struct PlaceBrickConfig(Phyics3dEnvConfig):
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
        """Robot (42), brick (13), pedestal (3)."""
        try:
            append_place_obs[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
                d, m_bodies, m_joints, m_sites, obs
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
        """`Place.get_reward` — see `place_reward` for the three terms and why
        the first two are a SWITCH rather than a sum."""
        return (
            Scalar[DTYPE](
                place_reward[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](d)
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
            place_set_grasp[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
                d, m_joints
            )
        except:
            pass

    # === CPU: pedestal, then arm, then brick ==============================
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
        place_reset_full[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
            PlaceBrickModel.CONE_TYPE,
            PlaceBrickModel.MAX_CONDIM,
            PlaceBrickModel.NOSLIP_ITER,
        ](d, mf, Self.get_timestep())
