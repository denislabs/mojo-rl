"""The family's `Phyics3dEnvConfig` — the reward IS the goal. P3c.

    Phyics3dBatchedEnv[So101TabletopModel, So101TabletopConfig, N_ENVS]

One config per FAMILY, not per task. That is the fixed scene budget cashing in:
every task in the family shares this type, this model and this monomorphisation,
and what varies between lanes is DATA — the twelve-word tape in
`meta[env, META_IDX_TASK_PARAM_*]`.

## ⚠ WHAT THE HOST WRITES, AND WHEN

    once   : curriculum[0, 0..4]              the region table
    per ep : meta[env, TASK_PARAM_0.._11]     this lane's goal
    per ep : qpos / qvel                      placements + parked slots

`tasks/reset.reset_slots` and `tasks/tape.encode_goal` are those writes. None
of them is a kernel today, and none needs to be: a reset is a host operation in
the driver, and the tape survives `_reset_env_lane` because that only writes
`META_IDX_STEP_COUNT`.

## ⚠⚠ WHAT THIS DOES *NOT* DO YET, STATED RATHER THAN IMPLIED

* **No active mask in the observation.** `TASK_LAYER_PLAN.md` §3.4 asks for
  `(pose, active)` per slot so a parked slot's pose is not a constant that
  means "absent" by convention. It needs another PER-LANE channel and all
  twelve `TASK_PARAM` words are the tape. The obs is the model default here.
* **No per-step repark.** Gap D's fix — pinning a parked slot's pose every
  step — needs the same missing channel to know which slots are active. Parked
  slots therefore FALL, as they did for the P0 probe. That is harmless for the
  REWARD (a goal names only active slots, and a parked one starts 50 m away and
  10 m out) and it is NOT harmless for an observation that reports poses.

Both are the same missing thing: a per-lane active mask. The honest fix is one
more `Data` field or a widened `METADATA_SIZE`, and it belongs with whoever
adds the obs, not bolted on here.
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    META_IDX_PREV_X,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .gpu_eval import eval_tape_gpu
from .so101_tabletop_xml import So101TabletopModel
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig


struct So101TabletopConfig(Phyics3dEnvConfig):
    comptime FRAME_SKIP: Int = 2
    comptime HAS_GPU_HOOKS: Bool = True
    comptime MAX_STEPS: Int = 300
    """The family's `horizon=`. ⚠ RESTATED, NOT READ — a config is a comptime
    TYPE and the `.family` is a runtime file, so this cannot import it. Keep
    them in step by hand; a mismatch changes episode length, not correctness."""

    # ⚠⚠ SO-ARM101 SHIPS A MOCAP BODY (`target`). `Phyics3dBatchedEnv.__init__`
    # RAISES if a mocap-flagged body exists while this is False. Frozen at its
    # XML pose here on purpose: no goal in this family reads it, and a target
    # that moved per episode would make the contact set vary run to run.
    comptime USES_MOCAP: Bool = True

    # ⚠⚠ NONZERO OR THE ARM'S 30 COLLISION MESHES SILENTLY STOP COLLIDING.
    # 0 is not a size hint — both narrow phases gate their mesh branch on
    # `NMESH_VERTS > 0` and emit no contact otherwise. Set it to 1 and read the
    # error, which quotes the number it needs.
    comptime NMESH_VERTS: Int = 26198

    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        So101TabletopModel.NQ, So101TabletopModel.NV
    ]()

    # === CPU hooks — present for the trait; this config is GPU-only ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        pass

    @staticmethod
    def compute_reward_and_done_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # ⚠ THE CPU LEG EVALUATES THROUGH `tasks/eval.eval_goal`, which needs
        # the family and the bound goal — neither of which a static hook can
        # hold. The parity gate therefore drives the CPU side itself rather
        # than through this hook, and this returns zero so that a CPU env
        # wired to this config is obviously inert instead of subtly wrong.
        return (Scalar[DTYPE](0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return 0.002

    @staticmethod
    def get_reset_noise() -> Float64:
        # ⚠ ZERO. Every lane's variation comes from the SAMPLER, seeded by
        # (seed, lane); joint noise on top would add a second, uncontrolled
        # source and make a lane's state depend on two streams.
        return 0.0

    # === GPU: pre-step ===
    @always_inline
    @staticmethod
    def pre_step_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        # ⚠ NOT THE REPARK. See the module header: pinning a parked slot needs
        # a per-lane ACTIVE MASK, and all twelve TASK_PARAM words are the tape.
        # ⚠⚠ AND IT MUST NOT TOUCH `meta` — the tape lives there.
        pass

    # === GPU: reward + termination — constant zero, never done ===
    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        ACTION_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_F, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        site_xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            DTYPE,
            Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE),
            MutAnyOrigin,
        ],
        sites: LayoutTensor[
            DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        cfrc_ext: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        curriculum: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # ⚠⚠ THE WHOLE REWARD IS THE GOAL. §5.3: sparse, +1 on success,
        # matching LIBERO. Shaping is a per-FAMILY concern expressed in a
        # config, not in a task — a shaped reward is a research choice about
        # one experiment, and putting it in the task file would make two runs
        # incomparable while looking identical.
        var holds = eval_tape_gpu[DTYPE, BATCH_SIZE, NBODY_F, SITE_DIM](
            meta, curriculum, xpos, xquat, site_xpos, env
        )
        # ⚠ TERMINATES ON SUCCESS. A sparse task that keeps running after the
        # goal is met pays for steps that teach nothing and lets a policy
        # bank the reward repeatedly; the driver's truncation still ends the
        # unsolved ones at MAX_STEPS.
        var r = Scalar[DTYPE](1) if holds else Scalar[DTYPE](0)
        _ = qpos
        _ = qvel
        _ = xipos
        _ = xvel
        _ = bodies
        _ = contacts
        _ = sites
        _ = geoms
        _ = cfrc_ext
        _ = cvel
        _ = actions
        _ = xangvel
        _ = cacc
        _ = cfrc_int
        _ = subtree_com
        _ = site_xpos_acc
        _ = xquat_acc
        _ = act
        _ = step_count
        _ = frame_skip
        _ = timestep
        return (r, holds)
    # === GPU: qpos init — the scene's own pose is already right ===
    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NJOINT_F: Int,
        NV_F: Int,
        NBODY_M: Int,
        NGEOM_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DTYPE, Layout.row_major(NJOINT_F, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        mocap_pos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_M * 3), MutAnyOrigin
        ],
        mocap_quat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_M * 4), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_M, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        # ⚠ NOTHING. The episode's poses are written by the HOST before the
        # step loop — `tasks/reset.reset_slots` places the active slots where
        # the sampler put them and parks the rest, and `tasks/tape` writes the
        # goal into `meta`. Doing it here instead would mean reimplementing
        # the sampler as device code, which `sampler.mojo` is SHAPED for but
        # which is not needed until resets happen mid-run.
        #
        # ⚠⚠ AND THE TAPE MUST SURVIVE THIS. It does: `_reset_env_lane` writes
        # META_IDX_STEP_COUNT and leaves the rest (`gpu/constants.mojo:164`),
        # and this hook deliberately writes nothing to `meta` either. A hook
        # that zeroed `meta` here would blank every lane's goal at the first
        # reset and every reward would read 0 — a flat curve, not a crash.
        pass
