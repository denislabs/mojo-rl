"""`libero_goal` on the batched GPU env, driven by OSC_POSE — L6's wiring.

    var env = LiberoGoalOscEnv[N](ctx, seed)
    env.set_osc_refs(build_osc_refs(...), ctx)     # from the parsed scene
    env.reset_batch[N](ctx, seed)
    env.step_batch[N](ctx, seed)

## ⚠⚠ WHAT THIS CONFIG IS AND IS NOT

It is the CONTROL path — the family's scene, its timestep and frame skip, and
`HAS_OSC_CONTROLLER`, so a batch of Pandas is driven by the seven-word action
LIBERO's demonstrations and policies speak — PLUS the task layer's hooks, each
one a call into code every family shares, over this family's generated table:

    init_qpos_gpu            placement/table.reset_task_slots (jinit=, placements)
    pre_step_full_gpu        task_hooks.repark_inactive_slots
    custom_extract_obs_*     task_hooks.write_task_obs(_host)
    compute_reward_and_done  gpu_eval.eval_tape_gpu -> reward AND GOAL_HELD

What is per LANE is data in `meta`, written by the driver per episode: the tape
(`tape.encode_goal`), the mask (`active.active_mask`), the init words
(`active.init_region_words`), the joint words (`placement/check.
joint_init_words`); and, once, the region table in `curriculum`
(`gpu_eval.region_table_words`).

⚠ EXCEPT THE RESET: `jinit=` draws and the free-slot placement. `init_qpos_gpu`
calls the task layer's shared kernel (`tasks/placement/table.reset_task_slots`)
on this family's generated table, because both are DATA in `meta` and one
kernel serves every family — not a second copy of anything. Without words it is
a no-op.

That split is deliberate and it is the honest scope of "the batched device
wiring". A config that also carried a goal would be a second copy of the task
layer's hooks, and the thing being gated here — that a batched env can run
robosuite's controller — would be buried inside it.

⚠ WHAT IS STILL OWED IS THE NVIDIA RUN: the batch stepping, and a lane-by-lane
comparison against `examples/tasks/libero_eval.mojo`'s CPU loop.

## ⚠ FRAME_SKIP IS 25 AND IT IS NOT A TUNING KNOB

`libero_goal.family` says `control_freq=20` and the scene's timestep is 2 ms, so
one policy step is 1/20 / 0.002 = 25 substeps — robosuite's own arithmetic
(`MujocoEnv.__init__`: `self.control_timestep = 1. / control_freq`). The
controller's `set_goal` fires on substep 0 of those 25 and the torque law runs
on all of them; `dynamics/osc_control.mojo` says why that gate matters.

⚠ RESTATED FROM THE `.family`, NOT READ. A config is a comptime TYPE and the
`.family` is a runtime file — the same constraint `MAX_STEPS` lives under in
`So101TabletopConfig`. Keep them in step by hand.

## ⚠ `NORMALIZED_ACTIONS` IS FALSE, AND THAT IS THE OPPOSITE OF THE SO-101

There the action is [-1, 1] per joint mapped onto each actuator's `ctrlrange`,
because the policy commands joint POSITIONS. Here it commands end-effector
DELTAS in metres and radians, which `osc_set_goal_gpu` scales by
`output_max_pos` / `output_max_ori` itself (0.05 m, 0.5 rad — robosuite's
`osc_pose.json`). Normalising on top of that would scale the deltas twice.
"""

from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE, METADATA_SIZE, MODEL_BODY_SIZE, MODEL_CURRICULUM_SIZE,
    MODEL_GEOM_SIZE, MODEL_JOINT_SIZE, MODEL_SITE_SIZE, META_IDX_GOAL_HELD,
)
from mojo_rl.tasks.task_hooks import (
    repark_inactive_slots, write_task_obs, write_task_obs_host,
)
from mojo_rl.tasks.placement.table import reset_task_slots
from mojo_rl.tasks.placement.libero_goal import LiberoGoalPlacement
from mojo_rl.tasks.gpu_eval import eval_tape_gpu
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.tasks.libero_goal_dims import LIBERO_GOAL_DIMS
from mojo_rl.tasks.libero_goal_xml import (
    LiberoGoalModel, LIBERO_GOAL_MAX_CONTACTS,
)


comptime LIBERO_GOAL_FRAME_SKIP: Int = 25
"""`control_freq=20` against a 2 ms timestep. See the header."""

comptime LIBERO_GOAL_HORIZON: Int = 600
"""`horizon=` in the `.family`, and `cfg.eval.max_steps` in
`lifelong/metric.py`. Restated, not read."""


struct LiberoGoalOscConfig(Phyics3dEnvConfig):
    comptime FRAME_SKIP: Int = LIBERO_GOAL_FRAME_SKIP
    comptime MAX_STEPS: Int = LIBERO_GOAL_HORIZON

    # ⚠ EULER, AS robosuite RUNS THIS MODEL. `base.xml` sets no
    # `<option integrator>`, so MuJoCo steps it under the default; RK4 would be
    # four times the cost of a pipeline the demonstrations were never recorded
    # under. The same reasoning as `So101TabletopConfig`, which measured it.
    comptime INTEGRATOR: StaticString = "euler"

    comptime INTEGRATOR_WS_EXTRA: Int = 0
    """0 for Euler and RK4; only ImplicitFast wants workspace beyond them."""

    comptime HAS_GPU_HOOKS: Bool = True
    comptime HAS_OSC_CONTROLLER: Bool = True
    comptime NORMALIZED_ACTIONS: Bool = False

    # ⚠ THE SCENE IS MESH-HEAVY AND THE COLLIDERS ARE NOT. LIBERO's objects
    # collide as 859 BOXES with 14 meshes; the `.msh` visuals carry
    # `contype=0`. The mesh vertex budget still has to cover the colliding
    # ones, and `parse_model_runtime` doubles until it fits — this is the
    # comptime figure the batched path needs up front.
    comptime NMESH_VERTS: Int = 65536

    # `mj_step`'s own order: the FK products after a step describe the state
    # BEFORE the last substep. LIBERO's `_check_success` reads object poses in
    # sync with qpos, the way dm_control does, so the flag is on.
    comptime SYNC_FK_AFTER_STEP: Bool = True

    # === the task hooks — `tasks/task_hooks.mojo`, on this family's table ===
    @always_inline
    @staticmethod
    def pre_step_full_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        """Pin every INACTIVE free slot at its park pose — `task_hooks.
        repark_inactive_slots`. Every `libero_goal` task activates every slot
        today, so this is inert on this family; it is here because an inactive
        slot is what the observation's mask words describe, and a union family
        (`libero_object`, 7 of 11 active per task) needs exactly this."""
        repark_inactive_slots[LiberoGoalPlacement, DTYPE, BATCH_SIZE, NQ, NV](
            qpos, qvel, meta, env
        )

    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        OBS_DIM: Int,
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
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
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
    ) -> Bool:
        """`task_hooks.write_task_obs`: `qpos`, `qvel`, one active word per
        free slot, then the nine goal words — measured from `robot_grip_site`,
        with an `In`/`On` target at ITS OWN region's site.

        ⚠ `LIBERO_GOAL_OBS_DIM` IS THAT WIDTH, defined beside the model def,
        so the number the env allocates and the layout written here are one
        expression."""
        write_task_obs[
            LiberoGoalPlacement, DTYPE, BATCH_SIZE, NQ_F, NV_F, NBODY_F,
            SITE_DIM, OBS_DIM,
        ](qpos, qvel, xpos, site_xpos, meta, obs, env)
        _ = xquat
        _ = xvel
        _ = bodies
        _ = contacts
        _ = sites
        _ = geoms
        _ = xipos
        _ = xangvel
        _ = cvel
        _ = cacc
        _ = cfrc_int
        _ = subtree_com
        _ = site_xpos_acc
        _ = xquat_acc
        _ = act
        return True

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
        """The single-env twin — `task_hooks.write_task_obs_host`. A batched
        run writes the checkpoint a single-env eval loads, so the two must agree
        word for word; `tests/tasks/test_libero_task_hooks.mojo` demands it."""
        write_task_obs_host[LiberoGoalPlacement, DTYPE, D](d, obs)
        _ = m_bodies
        _ = m_joints
        _ = m_geoms
        _ = m_sites
        _ = act
        return True

    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        ACTION_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
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
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
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
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """LIBERO'S REWARD IS ITS GOAL: sparse, +1 when the tape holds.

        ⚠⚠ NO SHAPING, DELIBERATELY. `lifelong/metric.py` scores an episode by
        `_check_success()` alone and every published LIBERO number is that. A
        shaped reward is a research choice about one experiment; putting it in
        the config that also defines the benchmark would make two runs
        incomparable while looking identical. `So101TabletopConfig` shapes
        because its family is ours to design — this one is not.

        ⚠ THE WIDE OVERLOAD OF `eval_tape_gpu`, with this hook's own qpos,
        sites, bodies and contacts. Eight of `libero_goal`'s ten goals are
        CONTACT predicates (`On`, `In`); the narrow overload compiles those
        branches out and would read every one of them as False — a reward that
        is zero for a reason no log would show.

        ⚠ THE TAPE COMES FROM `meta`, WRITTEN PER EPISODE BY THE DRIVER. Until
        one writes it, `eval_tape_gpu` reads an all-zero tape and returns False,
        so the reward is 0 — the same contract the SO-101 family runs under, and
        the reason `tasks/gpu_eval.mojo` exists.
        """
        var holds = eval_tape_gpu[
            DTYPE, BATCH_SIZE, NBODY, SITE_DIM, NQ, NSITE_F, MC_F
        ](
            meta, curriculum, xpos, xquat, site_xpos, qpos, sites, bodies,
            contacts, env,
        )
        # ⚠⚠ THE SUCCESS BIT GOES TO `META_IDX_GOAL_HELD` TOO. Here the reward
        # IS that bit, so a driver could read either — but the drivers that
        # count success (`task_batched_gpu`, `task_eval_frozen`) read the
        # word, because on a SHAPED family the reward is not the bit. Writing
        # it on every family keeps one way to ask.
        meta[env, META_IDX_GOAL_HELD] = (
            Scalar[DTYPE](1) if holds else Scalar[DTYPE](0)
        )
        # ⚠ THE `Bool` IS AN ASK AND IS DISCARDED UNLESS THE ENV WAS BUILT WITH
        # `TERMINATE_ON_UNHEALTHY=True` (`phyics3d_batched_env.mojo`). A driver
        # reading success out of `_done` without that flag reads a constant
        # zero; read `_reward` instead, which needs no flag.
        _ = qvel
        _ = xipos
        _ = xvel
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
        return (Scalar[DTYPE](1) if holds else Scalar[DTYPE](0), holds)

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
        """ZERO REWARD, NEVER TERMINATED — and that is the honest state of it.

        ⚠⚠ THIS IS NOT A PLACEHOLDER THAT SOMEONE FORGOT. LIBERO's reward is
        its GOAL, which lives in the task layer (`tasks/predicates.mojo` bound
        per task, `tasks/tape.mojo` for the device, `tasks/gpu_eval.mojo` for
        the per-lane bit) and is selected by the `.task` file the episode is
        running — not by the config, which is one comptime type for all ten
        tasks. Wiring it here would be a second copy of the tape.

        A driver that trains on this config would see a flat zero curve, which
        is why the header says so in as many words. `examples/tasks/libero_eval.mojo`
        is the loop that evaluates the real goal, on the CPU, today.
        """
        _ = len(m_bodies) + len(m_joints) + len(m_geoms) + len(m_sites)
        _ = len(actions) + step_count + frame_skip
        _ = prev_x
        return (Scalar[DTYPE](0), False)

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
        """THE JOINT DRAWS AND FREE-SLOT PLACEMENT AT RESET — the first LIBERO
        device reset.

        ⚠ THE SAME KERNEL AS `so101_tabletop`, reading this family's GENERATED
        table (`placement/libero_goal.mojo`), and gated against the host
        sampler on every `libero_goal` task by `tests/tasks/
        test_device_placement.mojo`. It places what the lane's init words in
        `meta` say; a driver that writes none leaves every free slot where
        `_reset_env_lane` put it, which is what this config did before.

        ⚠ WORDS, NOT A TASK. The driver writes them per lane with
        `active.init_region_words` and `placement.check.joint_init_words`
        after `require_device_placement[LiberoGoalPlacement]` — the tape, the
        mask and the observation words are still owed (see the header).

        ⚠ `jinit=` FIRST, THEN PLACEMENTS (`reset_task_slots`): a prop drawn
        into a drawer region stands in the drawer the draw just opened."""
        reset_task_slots[LiberoGoalPlacement, DTYPE, BATCH_SIZE, NQ_F, NV_F](
            qpos, qvel, meta, env, seed
        )
        _ = joints
        _ = mocap_pos
        _ = mocap_quat
        _ = bodies
        _ = geoms

    @staticmethod
    def get_timestep() -> Float64:
        """The scene's own `<option timestep>`, read off the generated dims.

        ⚠ NOT RESTATED. `libero_goal_dims.mojo` is generated from the composed
        XML through `mujoco.MjModel` and CI-checked, so this is the one number
        in the config that does NOT need keeping in step by hand."""
        return LIBERO_GOAL_DIMS.TIMESTEP


comptime LiberoGoalOscEnv = Phyics3dBatchedEnv[
    LiberoGoalModel, LiberoGoalOscConfig, _, CRBA_TREEWALK=True
]
"""The batched env, parameterised on the lane count.

⚠ IT CANNOT RUN ON METAL. `nv = 37` and the per-thread stack the CRBA and
Newton kernels want exceeds what Apple's pipeline creation allows — the P0 park
probe died at nv = 24. It COMPILES here, which is what
`tests/tasks/test_libero_osc_env.mojo` checks; the stepping leg is owed on
NVIDIA."""
