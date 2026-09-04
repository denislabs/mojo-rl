"""The family's `Phyics3dEnvConfig` — the reward IS the goal. P3c.

    Phyics3dBatchedEnv[So101TabletopModel, So101TabletopConfig, N_ENVS]

One config per FAMILY, not per task. That is the fixed scene budget cashing in:
every task in the family shares this type, this model and this monomorphisation,
and what varies between lanes is DATA — the twelve-word tape in
`meta[env, META_IDX_TASK_PARAM_*]`.

## ⚠ WHAT THE HOST WRITES, AND WHEN

    once   : curriculum[0, 0..4]              the region table
    per ep : meta[env, TASK_PARAM_0.._11]     this lane's goal
    per ep : meta[env, TASK_ACTIVE]           this lane's active slots
    per ep : qpos / qvel                      placements + parked slots

`tasks/reset.reset_slots`, `tasks/tape.encode_goal` and
`tasks/active.active_mask` are those writes. None of them is a kernel today,
and none needs to be: a reset is a host operation in the driver, and all three
`meta` writes survive `_reset_env_lane` because that only writes
`META_IDX_STEP_COUNT`.

⚠⚠ THAT SAME PROPERTY IS WHY EVERY ONE OF THEM MUST BE REWRITTEN EVERY
EPISODE. `meta` is not zeroed between episodes, so a lane keeps the previous
episode's goal and the previous episode's mask unless the driver writes over
them. `encode_goal` handles its half by writing `OP_NONE` into the terms it
does not use; the mask is one word and is always written whole.

## THE OBSERVATION, AND THE ACTIVE MASK IN IT

§3.4 asked for `(pose, active)` per slot, and `META_IDX_TASK_ACTIVE` is that
channel — one word, widened out of `METADATA_SIZE`, because all twelve
`TASK_PARAM` words are the tape and `encode_goal` writes every one of them.
The two observation hooks below are its only consumers.

⚠ THE OBSERVATION IS NOT THE MODEL DEFAULT ANY MORE, and it is a word wider
than the mask alone accounts for. The default is `qpos[obs_qpos_skip:] +
qvel`, and `obs_qpos_skip` defaults to 1 — which on a FLOATING-BASE model
drops the root's redundant word and on a DESK ARM drops `shoulder_pan`. This
family has no floating base: the arm is bolted to the world and `qpos[0]` is a
hinge angle the policy needs. So the hook writes the FULL `qpos`.

    OBS_DIM = NQ + NV + N_FREE_SLOTS = 27 + 24 + 3 = 54

⚠ THERE ARE TWO OBSERVATION HOOKS AND THEY ARE PINNED TO EACH OTHER. `_gpu`
takes `LayoutTensor`s and `_cpu` takes a `List`; there is no type that is
both, so the loop is written twice. `tests/tasks/test_active_mask.mojo` runs
BOTH on one state and demands identical vectors — a permutation between them
is a policy that works on the GPU and is nonsense on the CPU, with no error
anywhere.

## ⚠⚠ WHAT THIS STILL DOES NOT DO

* **No per-step repark.** Gap D's fix — pinning a parked slot's pose every
  step — now has the mask it was missing, but `pre_step_gpu` is handed only
  `qpos` and `meta`: it has no way to learn WHERE a slot parks. That is a
  family constant, so its home is `curriculum` (shared, host-written once) and
  reaching it means widening the `pre_step_gpu` signature across the fourteen
  configs that override it. Parked slots therefore still FALL.

  ⚠ THAT IS NOW COSMETIC, WHICH IT WAS NOT BEFORE. The fall is invisible to
  the REWARD (a goal names only active slots), and it is invisible to the
  OBSERVATION (an inactive slot's pose words are zeroed below). What remains
  is the VIEWER, which draws props sinking through the sky, and the invariant
  itself. A parked slot never lands inside a horizon — 7.06 m of free fall
  against 43 m of headroom, `TASK_LAYER_IMPLEMENTATION.md` — so nothing
  downstream reads a wrong number today.
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
    META_IDX_TASK_ACTIVE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .gpu_eval import eval_tape_gpu
from .obs import (
    slot_active, write_free_slot_obs, write_free_slot_obs_host,
    FREE_JOINT_NV,
)
from .so101_tabletop_xml import (
    So101TabletopModel, SO101_TABLETOP_N_FREE_SLOTS,
)
from mojo_rl.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS
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

    # ⚠⚠ THE ACTION IS [-1, 1] PER JOINT, mapped affinely onto each
    # actuator's own `ctrlrange`. This defaulted to False — raw control values,
    # clamped — for every commit up to the first training run, and the six
    # ranges it would have been clamping against are
    #
    #     shoulder_pan  +-1.9199    wrist_flex  +-1.6581
    #     shoulder_lift +-1.7453    wrist_roll  -2.7438 .. 2.8412
    #     elbow_flex    +-1.6900    gripper     -0.1745 .. 1.7453
    #
    # `Phyics3dEnvConfig.NORMALIZED_ACTIONS` carries the measurement that
    # settled this on the SAME ROBOT: with one scalar `ACTION_SCALE = 2.0`
    # against that spread, the trained policy commanded an out-of-range pose
    # on 24% to 100% of control steps, `elbow_flex` sat at the tanh rail 49%
    # of the time, and the gripper — asymmetric against a symmetric +-2.0 —
    # was out of range on EVERY step. Past the clamp the gradient is zero, and
    # three successive reward shapes produced the same shaking arm before
    # anyone looked at the clamp.
    #
    # ⚠ SO `action_scale` MUST BE 1.0 in every script that builds an agent for
    # this family. A scale of 2.0 maps [-2, 2] onto the range and puts the
    # useful band back inside the rails — undoing the fix while still looking
    # configured.
    #
    # ⚠ AND "DO NOTHING" IS NO LONGER A ZERO ACTION. Zero maps to the CENTRE
    # of each ctrlrange, which for the gripper is 0.785 rad — half open. The
    # zero-action drivers (`examples/tasks/task_eval_frozen.mojo`,
    # `task_batched_gpu.mojo`) therefore command a pose rather than no torque;
    # they are gating determinism and per-lane goal routing, both of which
    # hold under any fixed action, but their printed numbers move.
    comptime NORMALIZED_ACTIONS: Bool = True

    # ⚠⚠ NONZERO OR THE ARM'S 30 COLLISION MESHES SILENTLY STOP COLLIDING.
    # 0 is not a size hint — both narrow phases gate their mesh branch on
    # `NMESH_VERTS > 0` and emit no contact otherwise.
    #
    # ⚠⚠ THE ARM'S OWN CONSTANT, NOT A NUMBER READ OFF AN ERROR. This said
    # 26198 for one commit — the figure `parse_model_runtime` quoted for this
    # exact scene — and the BATCHED path then demanded 26199. One vertex, two
    # code paths, same model: `dims_from_flat` and the batched env's
    # `ModelDims` do not agree to the last hull vertex.
    #
    # Chasing that one vertex is the wrong response. `so101_park_xml` already
    # records the right rule and I should have followed it: reuse the arm's
    # declared budget, which is correct-by-construction for this robot and
    # comfortably above what either path asks. The props are BOXES — a
    # primitive, not a mesh — so they add no hull vertices at all.
    #
    # ⚠ A drift here is LOUD: `fields_build` raises rather than truncating.
    comptime NMESH_VERTS: Int = SO_ARM101_NMESH_VERTS

    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        So101TabletopModel.NQ, So101TabletopModel.NV
    ]()

    # ── THE FREE-SLOT TABLE — the one thing this type restates ────────────
    #
    # A config is a comptime TYPE and the `.family` is a runtime file, so this
    # cannot read it — the same constraint `MAX_STEPS` above lives under. The
    # difference is that this restatement is CHECKED: `tests/tasks/
    # test_active_mask.mojo` loads the family, runs `free_slot_addresses`
    # against the composed scene, and asserts every number below. A drift is a
    # failing gate, not a silently permuted observation.
    #
    # Measured on `scenes/so101_tabletop.xml` through MuJoCo 3.10.0:
    #
    #   family slot   joint          qposadr   dofadr
    #   1  brick      brick_free       6         6
    #   2  cube_a     cube_a_free     13        12
    #   3  cube_b     cube_b_free     20        18
    #
    # Slot 0 is `table`, a STATIC fixture: no joint, no state, and therefore
    # nothing in the observation varies with it. It still owns bit 0 of the
    # mask — the mask is indexed by FAMILY slot, so there is no second
    # numbering to keep in step (`tasks/active.mojo`).
    #
    # ⚠ `qposadr` AND `dofadr` DIVERGE AFTER THE FIRST FREE JOINT, because a
    # free joint is 7 `qpos` against 6 `qvel`. Reusing one for the other is
    # right for slot 0 of the three and wrong for the rest — which is exactly
    # the shape that reads as "the last prop's velocity is somebody else's".
    comptime N_FREE_SLOTS: Int = SO101_TABLETOP_N_FREE_SLOTS
    comptime FREE_SLOT_IDX_0: Int = 1
    comptime FREE_SLOT_IDX_1: Int = 2
    comptime FREE_SLOT_IDX_2: Int = 3
    comptime FREE_QADR_0: Int = 6
    comptime FREE_QADR_1: Int = 13
    comptime FREE_QADR_2: Int = 20
    comptime FREE_DADR_0: Int = 6
    comptime FREE_DADR_1: Int = 12
    comptime FREE_DADR_2: Int = 18

    # ── THE PARK POSE, the second thing this type restates ────────────────
    #
    # `tasks/family.park_pos` is `(park_x + slot*PARK_SPACING, park_y,
    # park_z)`, read from the `.family`'s `park=` line. All family constants,
    # so a comptime type can hold them — and `test_active_mask` asserts each
    # against `park_pos(f, si)` on the loaded family, the same way it asserts
    # the address table.
    #
    # ⚠⚠ I SAID THIS NEEDED A NEW OPERAND AND IT DID NOT. The P3d note claimed
    # the repark was blocked because `pre_step_gpu` "has no way to learn WHERE
    # a slot parks" and that reaching it meant putting the pose in
    # `curriculum` and widening a signature across fourteen configs. The pose
    # is a FAMILY CONSTANT, exactly like `FREE_QADR_*` above, and restating it
    # here costs one gate assertion. Only `qvel` actually needed a wider hook.
    comptime PARK_X: Float64 = 10.0
    comptime PARK_Y: Float64 = 0.0
    comptime PARK_Z: Float64 = 50.0
    comptime PARK_SPACING: Float64 = 0.5

    comptime OBS_MASK_BASE: Int = (
        So101TabletopModel.NQ + So101TabletopModel.NV
    )
    """Where the `N_FREE_SLOTS` active words start in `obs`.

    ⚠ READ FROM THE MODEL DEF, NOT RESTATED. `So101TabletopModel.OBS_DIM` is
    `SO101_TABLETOP_OBS_DIM`, defined beside the model def as
    `NQ + NV + N_FREE_SLOTS` — so the number the ENV allocates and the number
    this hook lays out are the same expression, not two copies of a total that
    happen to match today."""

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
    def custom_extract_obs_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """The single-env twin of `custom_extract_obs_gpu`.

        ⚠⚠ THE ORDER IS THE CONTRACT, AND THE TWO HOOKS MUST AGREE WORD FOR
        WORD. A batched run writes a checkpoint a single-env eval loads; a
        permutation here is a policy that works on the GPU and is nonsense on
        the CPU, with no error anywhere. `test_active_mask` runs both on one
        state and demands identical vectors — it does not check either against
        a description.

        ⚠ THIS CONFIG IS OTHERWISE GPU-ONLY — `compute_reward_and_done_cpu`
        returns a constant zero, deliberately. The observation is the one hook
        that must work on both, because it is what a checkpoint is shaped by.
        Leaving it to the model default would NOT have been inert: the default
        writes `NQ - 1 + NV` words into a vector this family sizes at
        `NQ + NV + N_FREE_SLOTS`, which is a silently truncated observation,
        not a missing one.
        """
        # ⚠⚠ `d.dims.get_nq()`, NOT `D.NQ`. The comptime members are POISON
        # on the DYNAMIC provider — `DynDims.NQ` is `DIM_POISON`, a negative
        # sentinel — so `range(D.NQ)` copies NOTHING there and the hook
        # returns a three-word observation with no error until something
        # indexes past it. The runtime accessors are correct on BOTH
        # providers, which is why `fields/dims.mojo` has all three families.
        var nq = d.dims.get_nq()
        var nv = d.dims.get_nv()
        for i in range(nq):
            obs.append(d.qpos.data[i])
        for i in range(nv):
            obs.append(d.qvel.data[i])
        for _ in range(Self.N_FREE_SLOTS):
            obs.append(Scalar[DTYPE](0))

        var mask = d.meta.data[META_IDX_TASK_ACTIVE]
        comptime for j in range(Self.N_FREE_SLOTS):
            comptime si = (
                Self.FREE_SLOT_IDX_0 if j == 0
                else (Self.FREE_SLOT_IDX_1 if j == 1 else Self.FREE_SLOT_IDX_2)
            )
            comptime qa = (
                Self.FREE_QADR_0 if j == 0
                else (Self.FREE_QADR_1 if j == 1 else Self.FREE_QADR_2)
            )
            comptime da = (
                Self.FREE_DADR_0 if j == 0
                else (Self.FREE_DADR_1 if j == 1 else Self.FREE_DADR_2)
            )
            write_free_slot_obs_host[DTYPE](
                obs,
                slot_active[DTYPE](mask, si),
                qa,
                nq + da,
                nq + nv + j,
            )
        _ = m_bodies
        _ = m_joints
        _ = m_geoms
        _ = m_sites
        _ = act
        return True

    @staticmethod
    def get_timestep() -> Float64:
        return 0.002

    @staticmethod
    def get_reset_noise() -> Float64:
        # ⚠⚠ THIS WAS 0.0, AND 0.0 MADE EVERY LANE THE SAME PROBLEM. The
        # reasoning was sound as far as it went — "every lane's variation comes
        # from the SAMPLER, seeded by (seed, lane); joint noise on top would
        # add a second source" — and it is wrong for any task whose GOAL does
        # not depend on a placement. `so101_reach_brick` asks the gripper to
        # reach a FIXED region: the brick's sampled pose enters the
        # observation and nothing else, so with zero joint noise all N lanes
        # start in the identical arm pose, every episode, and "the task" is one
        # open-loop trajectory rather than a distribution.
        #
        # ⚠ THE DETERMINISM ARGUMENT SURVIVES. Both streams are seeded — the
        # sampler from `(seed, lane)` and this from `reset_batch`'s seed — so
        # two runs at one seed still agree bit for bit, which is what P4's
        # frozen-init-table gate actually asserts. What zero bought was not
        # reproducibility but the absence of a second source, and the cost of
        # that was a degenerate start distribution.
        #
        # ⚠ 0.05 rad is the value `SoArm101ReachConfig` uses on the SAME
        # robot, so the two reach tasks perturb their starts comparably.
        return 0.05

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
        # ⚠ THE REPARK IS IN `pre_step_full_gpu` BELOW, which is the same hook
        # plus `qvel`. Nothing here, and in particular NOTHING THAT TOUCHES
        # `meta` — the tape and the active mask live there.
        pass

    # === GPU: pre-step, with qvel — Gap D's repark ===
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
        """Pin every INACTIVE free slot at its park pose, every step.

        `TASK_LAYER_IMPLEMENTATION.md` Gap D. Gravity is a `Model` field
        shared by the batch, so a parked body FALLS — `reset.reset_slots`
        zeroing its velocity at reset stops the fall compounding across
        episodes and does not stop the fall.

        ⚠⚠ THE POSE **AND** THE VELOCITY, AND POSE-ONLY IS HALF A FIX. Writing
        `qpos` back each step pins where the body IS while the integrator
        keeps adding `g*dt` to where it is GOING: the position looks parked
        and `qvel` grows without bound — 11.8 m/s by the end of a 300-step
        horizon. It never becomes a NaN and it never moves the arm (a parked
        slot is its own kinematic tree), so nothing would have caught it; it
        is simply not what "parked" should mean. Zeroing both makes a parked
        slot's state CONSTANT, which is checkable.

        ⚠ AN ACTIVE SLOT IS NOT TOUCHED. This runs before physics on every
        step, so a stray write here would pin the props the task is about —
        and the reward would read a scene that never moves while the arm
        pushed at it. `test_active_mask` asserts the active slots' words are
        BIT-IDENTICAL across the call.

        ⚠ AND IT MUST NOT TOUCH `meta`. The tape and the active mask live
        there, and this hook also runs at the END of `_reset_env_lane` — a
        write here would land after `init_qpos_gpu` and before the first step.
        """
        var mask = rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_ACTIVE])
        comptime for j in range(Self.N_FREE_SLOTS):
            comptime si = (
                Self.FREE_SLOT_IDX_0 if j == 0
                else (Self.FREE_SLOT_IDX_1 if j == 1 else Self.FREE_SLOT_IDX_2)
            )
            comptime qa = (
                Self.FREE_QADR_0 if j == 0
                else (Self.FREE_QADR_1 if j == 1 else Self.FREE_QADR_2)
            )
            comptime da = (
                Self.FREE_DADR_0 if j == 0
                else (Self.FREE_DADR_1 if j == 1 else Self.FREE_DADR_2)
            )
            # ⚠ FOLDED AT COMPILE TIME. `si` is a comptime index, so the park
            # pose is three constants in the kernel and not an arithmetic
            # chain — and no `Float64` reaches the device, which Metal has no
            # instruction for.
            comptime px = Scalar[DTYPE](
                Self.PARK_X + Float64(si) * Self.PARK_SPACING
            )
            comptime py = Scalar[DTYPE](Self.PARK_Y)
            comptime pz = Scalar[DTYPE](Self.PARK_Z)
            if not slot_active[DTYPE](mask, si):
                # ⚠ THE QUATERNION IS IDENTITY AND W COMES FIRST IN `qpos`.
                # `reset.write_free_pose` writes the same seven words; the
                # trap it records — that `(0,0,0,0)` is a DEGENERATE rotation
                # forward kinematics turns into a NaN pose — applies here on
                # every step rather than once.
                qpos[env, qa + 0] = px
                qpos[env, qa + 1] = py
                qpos[env, qa + 2] = pz
                qpos[env, qa + 3] = Scalar[DTYPE](1)
                qpos[env, qa + 4] = Scalar[DTYPE](0)
                qpos[env, qa + 5] = Scalar[DTYPE](0)
                qpos[env, qa + 6] = Scalar[DTYPE](0)
                comptime for k in range(FREE_JOINT_NV):
                    qvel[env, da + k] = Scalar[DTYPE](0)

    # === GPU: the observation — full state, plus §3.4's active mask ===
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
        """`qpos` in full, `qvel`, then one active word per free slot.

        ⚠ THE FULL `qpos`, NOT `qpos[1:]`. The model default skips a leading
        word that is a floating base's redundant coordinate on a Gym model and
        is `shoulder_pan` here. See `so101_tabletop_xml.SO101_TABLETOP_OBS_DIM`.

        ⚠⚠ AN INACTIVE SLOT IS ZEROED **AND** FLAGGED. Either alone is a bug:
        zeroing alone reinstates the convention the mask exists to remove, and
        flagging alone leaves a +50 in the vector — a parked slot sits 50 m up
        and falls, because nothing reparks it — which dominates the first
        layer whatever the flag says. `tasks/obs.write_free_slot_obs` does
        both, in one place, so a future reader cannot do one of them.

        ⚠ THE MASK WORD IS READ, NEVER WRITTEN, HERE. The host writes it once
        per episode beside the tape; an observation hook that computed it
        would be deciding what the task is while reporting what the state is.
        """
        comptime for i in range(NQ_F):
            obs[env, i] = qpos[env, i]
        comptime for i in range(NV_F):
            obs[env, NQ_F + i] = qvel[env, i]

        var mask = rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_ACTIVE])
        comptime for j in range(Self.N_FREE_SLOTS):
            comptime si = (
                Self.FREE_SLOT_IDX_0 if j == 0
                else (Self.FREE_SLOT_IDX_1 if j == 1 else Self.FREE_SLOT_IDX_2)
            )
            comptime qa = (
                Self.FREE_QADR_0 if j == 0
                else (Self.FREE_QADR_1 if j == 1 else Self.FREE_QADR_2)
            )
            comptime da = (
                Self.FREE_DADR_0 if j == 0
                else (Self.FREE_DADR_1 if j == 1 else Self.FREE_DADR_2)
            )
            write_free_slot_obs[DTYPE, BATCH_SIZE, OBS_DIM](
                obs, env,
                slot_active[DTYPE](mask, si),
                qa,
                NQ_F + da,
                Self.OBS_MASK_BASE + j,
            )

        _ = xpos
        _ = xquat
        _ = xvel
        _ = bodies
        _ = site_xpos
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
        # ⚠ ASKS TO TERMINATE ON SUCCESS. A sparse task that keeps running
        # after the goal is met pays for steps that teach nothing and lets a
        # policy bank the reward repeatedly; the driver's truncation still
        # ends the unsolved ones at MAX_STEPS.
        #
        # ⚠⚠ **AND THE ASK IS IGNORED BY DEFAULT.** `Phyics3dBatchedEnv` takes
        # `TERMINATE_ON_UNHEALTHY` as a comptime parameter DEFAULTING TO
        # FALSE, and then does
        #
        #     comptime if not Self.TERMINATE_ON_UNHEALTHY:
        #         is_terminated = False        # phyics3d_batched_env.mojo:1161
        #
        # so this `Bool` is DISCARDED unless the env was instantiated with the
        # flag, and `_done` then carries only truncation. A driver that wants
        # success-termination must spell it:
        #
        #     Phyics3dBatchedEnv[So101TabletopModel, So101TabletopConfig,
        #                        N_ENVS, TERMINATE_ON_UNHEALTHY=True]
        #
        # ⚠ A DRIVER READING SUCCESS OUT OF `_done` WITHOUT IT READS ZERO —
        # not an error, a constant. `examples/tasks/task_eval_frozen.mojo` did
        # exactly that and reported 0/128 on a task that holds at reset; the
        # eval reads `_reward` instead, which is this hook's other return and
        # needs no flag.
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
