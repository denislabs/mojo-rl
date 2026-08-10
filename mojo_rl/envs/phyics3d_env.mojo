"""Generic physics3d environment on the PER-FIELD tensor path (migration P5).

`Phyics3dEnv[MODEL_DEF, CONFIG]` is the generic single-env CPU MuJoCo
environment: MODEL_DEF/CONFIG parameterization over the
`BoxContinuousActionEnv` surface (drop-in for the CPU training drivers), with
the PHYSICS running through `RK4Integrator` over `Data` /
`Model` — no state slab, no workspace slab, no offsets.

G2: the legacy `Model`/`Data` hooks-adapter bridge is GONE — every CPU hook
(`MODEL_DEF.reset_data` / `apply_actions` / `extract_obs` and the CONFIG
reward/termination/pre-step/custom hooks) is fields-native and reads/writes
`self.d` directly. Mocap targets live in `d.mocap_pos`/`d.mocap_quat`
(hook-written); `_sync_mocap_to_fields` presets the mocap body world pose
from them before FK/stepping.

Scope (full parity with the legacy CPU env):
- Contacts + joint limits: the integrator runs detection + the constraint
  solve (limits inside) after every stage.
- Equality/tendon constraints: solved by the SOLVER (newton) — Humanoid
  tendons, and SawyerReach's weld-equality mocap control (the mocap body pose
  is preset via `_sync_mocap_to_fields` and skipped in FK; the weld solve makes
  the hand track the target). Validated in `test_sawyer_fields_parity`.
- Fluid forces (Swimmer) applied inside the fields integrator step.
- CPU target (single-env driver ABI). The GPU-batched facade is
  `phyics3d_batched_env`.
"""

from std.collections import InlineArray
from std.memory import alloc
from std.random import random_float64
from max.gpu.host import DeviceContext

from mojo_rl.core.env_traits import BoxContinuousActionEnv, RenderableEnv
from mojo_rl.core.obs_state import ObsState
from mojo_rl.core.cont_action import ContAction

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.model.model_renderer import ModelRenderer
from mojo_rl.render.ui import UIRect, UIText
from mojo_rl.render.renderer3d import RendererHandoff
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_auto
from mojo_rl.physics3d.joint_types import JNT_FREE
from mojo_rl.physics3d.integrator.rk4 import RK4Integrator
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MOCAP,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    META_IDX_NUM_CONTACTS,
)
from mojo_rl.nn.core.tensor import TensorImpl

from .phyics3d_env_config import Phyics3dEnvConfig


struct Phyics3dEnv[
    MODEL_DEF: ModelDefLike,
    CONFIG: Phyics3dEnvConfig,
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = False,
    SOLVER: StaticString = "newton",
](BoxContinuousActionEnv, RenderableEnv):
    """Generic MuJoCo env, physics on the per-field tensor path. See module
    docstring for the bridge design and scope.

    SOLVER defaults to "newton" — the legacy env default physics
    (CONFIG.physics_substep = RK4 + Newton). This facade steps on CPU,
    where the GPU-only PARALLEL_GPU / CRBA_TREEWALK knobs never apply
    (CPU is always serial + dense, matching legacy production's CPU
    side)."""

    comptime dtype = Self.DTYPE
    comptime StateType = ObsState[Self.MODEL_DEF.OBS_DIM]
    comptime ActionType = ContAction[Self.MODEL_DEF.ACTION_DIM]
    comptime NAME: String = "Physics3dEnv"

    comptime OBS_DIM: Int = Self.MODEL_DEF.OBS_DIM
    comptime ACTION_DIM: Int = Self.MODEL_DEF.ACTION_DIM
    comptime NQ: Int = Self.MODEL_DEF.NQ
    comptime NV: Int = Self.MODEL_DEF.NV
    comptime NBODY: Int = Self.MODEL_DEF.NBODY
    comptime NJOINT: Int = Self.MODEL_DEF.NJOINT
    comptime MAX_CONTACTS: Int = Self.MODEL_DEF.MAX_CONTACTS
    # ⚠ FROM THE CONFIG, NOT HARDCODED. This was a literal `0` at every one of
    # the six sites below, which made mesh geoms non-colliding in EVERY
    # environment — both narrow phases gate their mesh branch on
    # `NMESH_VERTS > 0`. Default is still 0; only configs with collidable
    # meshes override it. See `Phyics3dEnvConfig.NMESH_VERTS`.
    comptime NMESH_VERTS: Int = Self.CONFIG.NMESH_VERTS
    comptime NGEOM: Int = Self.MODEL_DEF.NGEOM
    comptime NSITE: Int = Self.MODEL_DEF.NSITE

    # Fields path (the physics state; hooks read/write it directly)
    var mf: Model[
        Self.DTYPE,
        Self.NV,
        Self.NBODY,
        Self.NJOINT,
        Self.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY,
        Self.MODEL_DEF.MAX_TENDON,
        Self.NSITE,
        Self.MODEL_DEF.NEXCLUDE,
        Self.NMESH_VERTS,
    ]
    var d: Data[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NBODY,
        Self.MAX_CONTACTS,
        Self.NSITE,
        1,
    ]
    # Both integrators are held (host scratch only on the CPU path — cheap);
    # the step comptime-dispatches on CONFIG.INTEGRATOR. HalfCheetah/Pusher/
    # MetaWorld configure Euler+Newton; the other 9 envs use RK4+Newton.
    comptime IntegRK4 = RK4Integrator[
        Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
        Self.MAX_CONTACTS, Self.NGEOM, Self.MODEL_DEF.MAX_EQUALITY,
        Self.MODEL_DEF.MAX_TENDON, Self.NSITE, Self.MODEL_DEF.NEXCLUDE,
        Self.NMESH_VERTS,
        Self.MODEL_DEF.CONE_TYPE, 1, SOLVER = Self.SOLVER,
    ]
    # ⚠ `MAX_CONDIM` AND `NOSLIP_ITER` MUST BE FORWARDED FROM THE MODEL DEF.
    # Both default to a value that silently disables the feature (3 and 0), and
    # omitting them here does not fail — it just runs a different physics.
    #
    # That is not hypothetical: `MAX_CONDIM` was NOT forwarded until
    # 2026-08-03, so every env built through this class ran the pyramidal edge
    # builder at condim 3 no matter what its model declared. quadruped `fetch`
    # (condim-6 ball) and dog (42 condim-6 teeth) were both affected, and the
    # Phase 3 gate did not catch it because
    # `test_rolling_friction_vs_mujoco.mojo` constructs the integrator
    # DIRECTLY with `MAX_CONDIM=M.MAX_CONDIM` and never goes through
    # `Phyics3dEnv`. A gate that bypasses the production path proves the
    # production path works only by coincidence.
    comptime IntegEuler = EulerIntegrator[
        Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
        Self.MAX_CONTACTS, Self.NGEOM, Self.MODEL_DEF.MAX_EQUALITY,
        Self.MODEL_DEF.MAX_TENDON, Self.NSITE, Self.MODEL_DEF.NEXCLUDE,
        Self.NMESH_VERTS,
        Self.MODEL_DEF.CONE_TYPE, 1, SOLVER = Self.SOLVER,
        RNE_POST = Self.CONFIG.RNE_POST,
        MAX_CONDIM = Self.MODEL_DEF.MAX_CONDIM,
        NOSLIP_ITER = Self.MODEL_DEF.NOSLIP_ITER,
    ]
    var integ_rk4: Self.IntegRK4
    var integ_euler: Self.IntegEuler

    var max_steps: Int
    var current_step: Int
    var frame_skip: Int
    var _last_terminated: Bool
    var prev_x: Scalar[Self.DTYPE]
    # Actuator activation (MuJoCo `d->act`), one scalar per activation
    # variable. Lives on the ENV rather than in `Data` because no physics
    # kernel reads it — only `apply_actions` does, on the CPU, and putting it
    # in `Data` would force an NA parameter onto every FK / integrator /
    # solver signature that threads a `Data` through. Sized with a floor of 1:
    # NA == 0 for every model but quadruped, and a zero-length List is a
    # needless edge case for the indexing below.
    var act: List[Scalar[Self.DTYPE]]

    # Renderer (optional; RenderableEnv). Reads the fields FK products
    # (`self.d.xpos`/`xquat`), which the fields step refreshes every frame.
    var _renderer: Optional[
        Pointer[ModelRenderer[Self.MODEL_DEF], MutUntrackedOrigin]
    ]
    var _renderer_initialized: Bool
    # Owned device context: the model-record tensors upload to it once at build
    # (init_fields). Kept for the env's lifetime so the mf device buffers stay
    # valid (the CPU step path never reads them). The no-arg ctor creates one so
    # `Ant()` / `Hopper()` etc. work without the caller threading a ctx.
    var _ctx: DeviceContext

    def __init__(out self) raises:
        """Convenience ctor — owns a fresh `DeviceContext` (single-env CPU
        facade: one env, one context). Lets the `Ant()`/`Hopper()` API stay
        ctx-free when the alias points at this fields facade."""
        self = Self(DeviceContext())

    def __init__(
        out self,
        ctx: DeviceContext,
        max_steps: Int = Self.CONFIG.MAX_STEPS,
        frame_skip: Int = Self.CONFIG.FRAME_SKIP,
    ) raises:
        """`ctx` is used ONCE, for the model-record device upload at build
        (no device work on the CPU step path); it is stored to keep the mf
        device buffers valid for the env's lifetime."""
        comptime assert (
            (not Self.CONFIG.RNE_POST) or Self.CONFIG.INTEGRATOR == "euler"
        ), (
            "Phyics3dEnv: CONFIG.RNE_POST is wired into the Euler integrator"
            " only — an RK4 config would silently get zero cacc/cfrc_int, and"
            " with them zero accelerometer/force/torque readings"
        )
        self._ctx = ctx
        self.max_steps = max_steps
        self.current_step = 0
        self.frame_skip = frame_skip
        self.prev_x = Scalar[Self.DTYPE](0.0)
        self._last_terminated = False
        self.act = List[Scalar[Self.DTYPE]]()
        for _ in range(Self.MODEL_DEF.NA if Self.MODEL_DEF.NA > 0 else 1):
            self.act.append(Scalar[Self.DTYPE](0))
        self._renderer = None
        self._renderer_initialized = False

        # Build the model record tensors offset-free (P6 fields-native build):
        # no flat slab, no cross-family offset tables. `init_fields` writes
        # every record tensor via load_from_model and computes invweight0
        # fields-natively (G1).
        self.mf = type_of(self.mf)()
        Self.MODEL_DEF.init_fields[Self.DTYPE, Self.NMESH_VERTS](ctx, self.mf)

        self.d = type_of(self.d)()
        self.integ_rk4 = Self.IntegRK4()
        self.integ_euler = Self.IntegEuler()

        # Reference pose + fresh FK products so get_state()/renderer reads
        # are valid before the first reset().
        Self.MODEL_DEF.reset_data(self.d)
        self._sync_mocap_to_fields()
        self._fields_fk()
        Self.CONFIG.pre_step_cpu(self.d, self.prev_x)
        # Fluid forces (density/viscosity > 0) are applied inside the fields
        # integrator step (dynamics/fluid_forces.mojo); no guard needed.

    # ── kinematics / mocap helpers ─────────────────────────────────────────
    def _fields_fk(mut self):
        """Fields FK (CPU): refresh xpos/xquat/xipos from qpos so obs/reward
        hooks that read world poses see fresh values outside the integrator
        step (ctor / reset / set_state). Skips mocap bodies."""
        try:
            # CPU target: cannot actually raise (the `raises` exists for the
            # GPU branch's ctx handling).
            forward_kinematics[
                "cpu", Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
                Self.MAX_CONTACTS, Self.NGEOM, Self.MODEL_DEF.MAX_EQUALITY,
                Self.MODEL_DEF.MAX_TENDON, Self.NSITE,
                Self.MODEL_DEF.NEXCLUDE, Self.NMESH_VERTS, 1,
            ](self.d, self.mf, None)
        except e:
            print("Phyics3dEnv._fields_fk: FK error:", e)

    def _fields_vel(mut self):
        """Body world velocities (xvel/xangvel) from the current qvel.

        Companion to `_fields_fk`: the integrator computes these mid-step, so
        hooks that read them need a refresh once integration has finished.
        Only runs under CONFIG.SYNC_FK_AFTER_STEP."""
        try:
            compute_body_velocities[
                "cpu", Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
                Self.MAX_CONTACTS, Self.NGEOM, Self.MODEL_DEF.MAX_EQUALITY,
                Self.MODEL_DEF.MAX_TENDON, Self.NSITE,
                Self.MODEL_DEF.NEXCLUDE, Self.NMESH_VERTS, 1,
            ](self.d, self.mf, None)
        except e:
            print("Phyics3dEnv._fields_vel: velocity error:", e)

    def _sync_mocap_to_fields(mut self):
        """Mocap actuation: the CONFIG hooks write the mocap target into
        `d.mocap_pos`/`d.mocap_quat` (per step / on reset); preset the
        corresponding body world pose so the fields FK — which SKIPS mocap
        bodies — leaves the target in place and the weld/equality solve
        (SOLVER=newton) tracks it.

        Gated only on the per-body mocap flag, which is 0 for every non-mocap
        model, so this costs an NBODY float compare per call there. It used to
        carry an outer `MAX_EQUALITY > 0` comptime gate as well, on the
        assumption that mocap only ever means "weld-driven actuation"
        (SawyerReach). That silently disabled mocap for models that use a
        mocap body as a POSE CARRIER with no constraint attached — dm_control's
        reacher parks its randomized per-episode target on one — so the pose
        would never leave `mocap_pos` and the body would sit at its XML pos
        forever, with no error."""
        for b in range(Self.NBODY):
            if self.mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MOCAP] == 0:
                continue
            for k in range(3):
                var p = self.d.mocap_pos.data[b * 3 + k]
                self.d.xpos.data[b * 3 + k] = p
                self.d.xipos.data[b * 3 + k] = p
            for k in range(4):
                self.d.xquat.data[b * 4 + k] = self.d.mocap_quat.data[
                    b * 4 + k
                ]

    # ── state management ─────────────────────────────────────────────────
    def _reset_state(mut self):
        """Reset semantics (qpos0 + uniform noise + custom hook), all on the
        fields state."""
        # MuJoCo's mj_resetData zeroes `act` along with qpos/qvel.
        for _i in range(len(self.act)):
            self.act[_i] = Scalar[Self.DTYPE](0)
        Self.MODEL_DEF.reset_data(self.d)
        var noise_scale = Self.CONFIG.get_reset_noise()
        if noise_scale > 0.0:
            for i in range(Self.NQ):
                var noise = Scalar[Self.dtype](
                    (random_float64() * 2.0 - 1.0) * noise_scale
                )
                self.d.qpos.data[i] = self.d.qpos.data[i] + noise
            for i in range(Self.NV):
                var noise = Scalar[Self.dtype](
                    (random_float64() * 2.0 - 1.0) * noise_scale
                )
                self.d.qvel.data[i] = self.d.qvel.data[i] + noise
        # Per-episode MODEL randomization first, so the state hook below reads
        # whatever it wrote (point_mass `hard` randomizes the tendon mixing).
        Self.CONFIG.custom_reset_model_cpu(
            self.mf.bodies.data,
            self.mf.joints.data,
            self.mf.geoms.data,
            self.mf.sites.data,
            self.mf.tendons.data,
        )
        Self.CONFIG.custom_reset_cpu(
            self.d,
            self.mf.bodies.data,
            self.mf.joints.data,
            self.mf.geoms.data,
            self.mf.sites.data,
        )
        comptime if Self.CONFIG.RESET_FIND_HEIGHT:
            self._find_non_contacting_height()
        self._sync_mocap_to_fields()
        self._fields_fk()  # fresh obs before step 1
        self.current_step = 0
        self.prev_x = Scalar[Self.dtype](0)
        self._last_terminated = False
        Self.CONFIG.pre_step_cpu(self.d, self.prev_x)


    def _find_non_contacting_height(mut self):
        """Raise the free root in 1 cm steps until nothing is touching.

        `quadruped._find_non_contacting_height` (suite/quadruped.py:397): start
        embedded in the floor at z = 0 and step up by 1 cm until `data.ncon`
        is 0. The ORIENTATION is whatever `custom_reset_cpu` drew; this only
        moves the height, exactly as the reference does.

        Lives on the env rather than in the reset hook because it needs
        forward kinematics AND broadphase, and a config hook gets the record
        Lists but neither `Model` nor a way to run a pipeline stage. Gated by
        `CONFIG.RESET_FIND_HEIGHT` so no other model pays for it.

        Bounded at 10000 attempts like the reference — which RAISES there. We
        cannot raise from `_reset_state`, so this leaves the last height tried
        and lets the first step report the penetration; a model whose legs
        cannot clear the floor in 100 m is misbuilt in a way a reset failure
        would not explain anyway.
        """
        # The free root's z. Found from the joint records rather than assumed
        # at qpos[2]: a model may declare its free joint second.
        var zadr = -1
        for j in range(Self.NJOINT):
            var jt = Int(self.mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
            if jt == JNT_FREE:
                zadr = Int(
                    self.mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR]
                ) + 2
                break
        if zadr < 0:
            return

        for attempt in range(10000):
            self.d.qpos.data[zadr] = Scalar[Self.dtype](0.01 * Float64(attempt))
            self._fields_fk()
            try:
                detect_contacts_auto[
                    "cpu", Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                    Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM,
                    Self.MODEL_DEF.MAX_EQUALITY, Self.MODEL_DEF.MAX_TENDON,
                    Self.NSITE, Self.MODEL_DEF.NEXCLUDE, Self.NMESH_VERTS, 1,
                ](self.d, self.mf, None)
            except:
                return
            if Int(self.d.meta.data[META_IDX_NUM_CONTACTS]) == 0:
                return

    def set_state(mut self, qpos: List[Float64], qvel: List[Float64]):
        """Deterministic state injection (tests / eval)."""
        for i in range(min(Self.NQ, len(qpos))):
            self.d.qpos.data[i] = Scalar[Self.dtype](qpos[i])
        for i in range(min(Self.NV, len(qvel))):
            self.d.qvel.data[i] = Scalar[Self.dtype](qvel[i])
        self._sync_mocap_to_fields()
        self._fields_fk()
        # Velocities too, not just poses: injecting a qvel and leaving
        # xvel/xangvel at whatever the last step left is exactly the kind of
        # half-updated state this method exists to avoid. Every velocimeter /
        # gyro / subtreelinvel read goes through those two.
        self._fields_vel()

    def reward_at(
        mut self,
        qpos: List[Float64],
        qvel: List[Float64],
        action: List[Float64],
        prev_x: Scalar[Self.DTYPE] = Scalar[Self.DTYPE](0),
        step_count: Int = 1,
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Reward of an ARBITRARY state, without having reached it by stepping.

        This is the relabelling primitive: `set_state` puts the engine in
        `(qpos, qvel)` — refreshing FK *and* body velocities — and the config's
        reward hook is then called on it. It is what lets a dataset of
        `(qpos, qvel, action)` be scored under a reward invented long after
        collection, which is the whole reason the BFM dataset stores generalized
        coordinates rather than observations (`docs/BFM_ZERO_SHOT_RL.md` §6).

        ⚠ **Destructive.** The env's state IS the injected one afterwards —
        `reward_at` is not a peek. Relabelling a dataset therefore wants a
        dedicated env instance, not the one driving a rollout.

        ⚠ **Purity is a per-task property, not a guarantee.** The hook also
        receives `prev_x` and `step_count`, so a task that computes velocity as
        a finite difference of positions is NOT a function of `(qpos, qvel,
        action)` and cannot be relabelled from this dataset. The defaults here
        (`prev_x = 0`, `step_count = 1`) are what a stateless task ignores; for
        one that reads them, they are wrong, silently.

        Do not assume — measure. `tests/dm_control/test_reward_relabel.mojo`
        replays a rollout and diffs the relabelled reward against the online
        one; a task that passes is Markovian in the only sense that matters
        here. dm_control's suite is clean because its rewards read `Data` (FK
        products of qpos, and `xvel`, itself FK of qvel), but the Gym-derived
        configs in this same package are NOT: HalfCheetah's forward reward is
        literally `(x - prev_x) / dt`.
        """
        self.set_state(qpos, qvel)
        return Self.CONFIG.compute_reward_and_done_cpu(
            self.d,
            self.mf.bodies.data,
            self.mf.joints.data,
            self.mf.geoms.data,
            self.mf.sites.data,
            prev_x,
            action,
            step_count,
            self.frame_skip,
        )

    def obs_at(
        mut self, qpos: List[Float64], qvel: List[Float64]
    ) -> Self.StateType:
        """Observation of an arbitrary state. Same contract as `reward_at`.

        The dataset stores `qpos`/`qvel` and recovers the observation through
        this rather than storing it: for walker that is 18 floats against 24,
        and unlike the observation, generalized coordinates are sufficient for
        rewards the collection run had never heard of.
        """
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self) -> ObsState[Self.MODEL_DEF.OBS_DIM]:
        var obs_list = List[Scalar[Self.DTYPE]](capacity=Self.OBS_DIM)
        var custom = Self.CONFIG.custom_extract_obs_cpu(
            self.d,
            self.mf.bodies.data,
            self.mf.joints.data,
            self.mf.geoms.data,
            self.mf.sites.data,
            self.act,
            obs_list,
        )
        if not custom:
            Self.MODEL_DEF.extract_obs(self.d, obs_list)
        var obs = ObsState[Self.MODEL_DEF.OBS_DIM]()
        for i in range(Self.OBS_DIM):
            obs.data[i] = Float64(obs_list[i])
        return obs^

    # ── Env interface ─────────────────────────────────────────────────────
    def reset(mut self) -> Self.StateType:
        self._reset_state()
        return self._get_obs()

    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        Self.CONFIG.pre_step_cpu(self.d, self.prev_x)

        # Actions via the comptime actuator logic (per-motor ctrlrange clamp
        # + gear), written straight into the fields qfrc.
        var clamped_action = action.copy()
        var action_list = clamped_action.to_list()
        # The CONFIG hook keeps ONCE-PER-CONTROL-STEP semantics: it is action
        # semantics, not a force law, and SawyerReach's applies a mocap DELTA
        # that would compound `frame_skip` times inside the loop below.
        var custom_applied = Self.CONFIG.custom_apply_actions_cpu(
            self.d,
            self.mf.bodies.data,
            self.mf.joints.data,
            self.mf.geoms.data,
            self.mf.sites.data,
            self.mf.tendons.data,
            action_list,
        )
        # Mocap-controlled models (SawyerReach): push the updated mocap target
        # into the fields body poses before the step so the weld solve tracks it.
        self._sync_mocap_to_fields()

        # Physics: fields integrator (RK4 or Euler per CONFIG.INTEGRATOR) with
        # per-substep contact/limit solving.
        for _ in range(self.frame_skip):
            # Actuator + tendon-spring forces are recomputed EVERY SUBSTEP,
            # as MuJoCo recomputes qfrc_actuator inside every mj_step. For a
            # `<motor>` this rewrites the same constant and is bit-identical
            # to hoisting it; for a `<position>` servo, whose force reads
            # qpos, hoisting it would freeze the spring at its start-of-step
            # value for the whole control step.
            if not custom_applied:
                Self.MODEL_DEF.apply_actions(
                    self.d, action_list, self.act
                )
            try:
                # CPU target: cannot actually raise (the `raises` on the
                # dispatchers exists for the GPU branch's ctx handling).
                comptime if Self.CONFIG.INTEGRATOR == "euler":
                    self.integ_euler.step["cpu"](self.d, self.mf)
                else:
                    self.integ_rk4.step["cpu"](self.d, self.mf)
            except e:
                print("Phyics3dEnv.step: physics error:", e)

        # Put the FK products in sync with the integrated qpos before the
        # reward/obs hooks read them (dm_control convention). Off for the
        # Gym-derived envs, which are calibrated against raw `mj_step`, where
        # xpos/xquat/xipos still describe the pre-integration state. See
        # Phyics3dEnvConfig.SYNC_FK_AFTER_STEP.
        comptime if Self.CONFIG.SYNC_FK_AFTER_STEP:
            self._fields_fk()
            # …and the body velocities with the integrated qvel. `xvel` is
            # written inside the integrator, so without this it describes the
            # state before the last substep — the same staleness the FK sync
            # exists to fix, one field over. Mass-weighted sensors
            # (subtreelinvel) read `xvel` directly.
            self._fields_vel()

        self.current_step += 1

        var result = Self.CONFIG.compute_reward_and_done_cpu(
            self.d,
            self.mf.bodies.data,
            self.mf.joints.data,
            self.mf.geoms.data,
            self.mf.sites.data,
            self.prev_x,
            clamped_action.to_list(),
            self.current_step,
            self.frame_skip,
        )
        var reward = result[0]
        var terminated = result[1]
        comptime if not Self.TERMINATE_ON_UNHEALTHY:
            terminated = False
        var truncated = self.current_step >= self.max_steps
        var done = terminated or truncated
        self._last_terminated = terminated
        return (self._get_obs(), Scalar[Self.dtype](reward), done)

    def was_terminated(self) -> Bool:
        return self._last_terminated

    def get_state(self) -> Self.StateType:
        return self._get_obs()

    def close(mut self):
        pass

    # ── Render accessors (read the fields FK products) ────────────────────
    def get_xpos(self, idx: Int) -> Scalar[Self.DTYPE]:
        return self.d.xpos.data[idx]

    def get_xquat(self, idx: Int) -> Scalar[Self.DTYPE]:
        return self.d.xquat.data[idx]

    def get_x_velocity(self) -> Scalar[Self.DTYPE]:
        return self.d.qvel.data[0]

    # ── RenderableEnv (mirrors the legacy env; renders the fields poses) ──
    def init_renderer(mut self) raises -> Bool:
        return self._init_renderer(show_velocity=True, adopt=None)

    def init_renderer(mut self, show_velocity: Bool) raises -> Bool:
        return self._init_renderer(show_velocity=show_velocity, adopt=None)

    def init_renderer(
        mut self, show_velocity: Bool, adopt: Optional[RendererHandoff]
    ) raises -> Bool:
        """Open a window, or take over one another env's renderer detached.

        The adopt path is for tools that swap MODELS behind one window — each
        model is a different `Phyics3dEnv[...]` type, so the env cannot swap
        itself; the window has to outlive it. `detach_renderer` produces the
        handoff. See `RendererHandoff`.
        """
        return self._init_renderer(show_velocity=show_velocity, adopt=adopt)

    comptime RENDER_WIDTH: Int = 1280
    comptime RENDER_HEIGHT: Int = 720
    """Window size the env asks for. Exposed because screen-space UI has to
    lay itself out against it, and a hardcoded 1280 in a viewer would silently
    drift the day this changes."""

    def _init_renderer(
        mut self, show_velocity: Bool, adopt: Optional[RendererHandoff]
    ) raises -> Bool:
        if self._renderer_initialized:
            return True

        self._renderer = alloc[ModelRenderer[Self.MODEL_DEF]](1)

        var renderer = ModelRenderer[Self.MODEL_DEF](
            width=Self.RENDER_WIDTH,
            height=Self.RENDER_HEIGHT,
            visual_radius_scale=1.0,
            axes_offset=1.5,
            vel_arrow_height=0.15,
            vel_arrow_scale=0.1,
            show_velocity=show_velocity,
        )
        renderer.init(adopt)

        self._renderer.value().unsafe_write(renderer^)
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        if not self._renderer.value()[].is_open():
            return

        var xpos = InlineArray[Scalar[Self.DTYPE], Self.MODEL_DEF.NBODY * 3](
            uninitialized=True
        )
        var xquat = InlineArray[Scalar[Self.DTYPE], Self.MODEL_DEF.NBODY * 4](
            uninitialized=True
        )
        for i in range(Self.MODEL_DEF.NBODY * 3):
            xpos[i] = self.get_xpos(i)
        for i in range(Self.MODEL_DEF.NBODY * 4):
            xquat[i] = self.get_xquat(i)
        self._renderer.value()[].render_from_body_state(
            xpos,
            xquat,
            Self.MODEL_DEF.NBODY,
            vel_x=Float64(self.get_x_velocity()),
        )

    def close_renderer(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].close()
        self._renderer.value().unsafe_free()
        self._renderer_initialized = False

    def detach_renderer(mut self) raises -> Optional[RendererHandoff]:
        """Close this env's renderer but KEEP its window and device alive.

        Returns the handoff to give to the next env's
        `init_renderer(show_velocity, adopt)`, or `None` if there was no
        renderer to detach.

        ⚠ THE HANDOFF IS NOW THE CALLER'S TO END. Once it exists nothing else
        owns the window: adopt it, or call `Renderer3D.close_handoff` on it.
        Dropping it leaks the window, the device and every pipeline on it.
        """
        if not self._renderer_initialized:
            return None
        var h = self._renderer.value()[].detach()
        self._renderer.value().unsafe_free()
        self._renderer_initialized = False
        return h^

    def is_renderer_open(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].is_open()

    def check_renderer_quit(mut self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].check_quit()

    def renderer_take_key(mut self) -> Int:
        """A keycode the renderer's own bindings did not claim, 0 if none.

        Clears on read, so each press is delivered once. Lets a tool bind keys
        without the env or the renderer knowing what they mean.
        """
        if not self._renderer_initialized:
            return 0
        return self._renderer.value()[].take_key()

    def set_hud_extra(mut self, lines: List[String]) -> None:
        """Application-owned HUD lines, drawn under the engine's controls."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].set_hud_extra(lines)

    def renderer_take_click(mut self) -> Bool:
        """True once per mouse press; clears on read."""
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].take_click()

    def renderer_mouse_x(self) -> Float32:
        if not self._renderer_initialized:
            return 0
        return self._renderer.value()[].mouse_x()

    def renderer_mouse_y(self) -> Float32:
        if not self._renderer_initialized:
            return 0
        return self._renderer.value()[].mouse_y()

    def renderer_n_cameras(self) -> Int:
        if not self._renderer_initialized:
            return 0
        return self._renderer.value()[].n_cameras()

    def renderer_current_camera(self) -> Int:
        if not self._renderer_initialized:
            return 0
        return self._renderer.value()[].current_camera()

    def renderer_request_camera(mut self, index: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].request_camera(index)

    def renderer_request_screenshot(mut self) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].request_screenshot()

    def renderer_is_recording(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].is_recording()

    def renderer_recording_frames(self) -> Int:
        if not self._renderer_initialized:
            return 0
        return self._renderer.value()[].recording_frames()

    def renderer_toggle_recording(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].toggle_recording()

    def renderer_paused(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].paused()

    def renderer_toggle_pause(mut self) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].toggle_pause()

    def renderer_set_text_input_mode(mut self, on: Bool) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].set_text_input_mode(on)

    def set_ui_sidebar_width(mut self, w: Int) -> None:
        """Reserve `w` px on the left of the window for screen-space UI."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].set_ui_sidebar_width(w)

    def imgui_init(mut self) raises -> Bool:
        """Attach a Dear ImGui overlay to the renderer window.

        Returns False when the shim is not built (`pixi run build-imgui`) or
        the device refuses it — callers should degrade, not abort.
        """
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].imgui_init()

    def imgui_new_frame(mut self) raises -> None:
        """Open an ImGui frame. Widgets go between this and `render_frame`."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].imgui_new_frame()

    def imgui_active(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].imgui_active()

    def renderer_request_free_camera(mut self) -> None:
        """Detach from model cameras — dm_control's free camera (`-1`).

        Model cameras in `trackcom`/`targetbody` mode are re-aimed every frame,
        so mouse orbit fights them; only this one is actually free.
        """
        if not self._renderer_initialized:
            return
        self._renderer.value()[].request_free_camera()

    def renderer_set_capture_scene_only(mut self, on: Bool) -> None:
        """Whether screenshots and recordings exclude the reserved UI strip.

        On by default: a capture of the environment is what these are for, and
        the sidebar is a control surface, not part of the scene.
        """
        if not self._renderer_initialized:
            return
        self._renderer.value()[].set_capture_scene_only(on)

    def renderer_set_show_hud(mut self, on: Bool) -> None:
        """Show or hide the built-in text HUD (keybinds, camera, step).

        Turn it OFF alongside an ImGui sidebar: both report the same facts, and
        the HUD is drawn over the scene rather than beside it.
        """
        if not self._renderer_initialized:
            return
        self._renderer.value()[].set_show_hud(on)

    def set_ui(
        mut self, rects: List[UIRect], texts: List[UIText]
    ) -> None:
        """Hand a widget command list to the renderer for the next frame."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].set_ui(rects, texts)

    def renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].delay(ms)

    def renderer_is_paused(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].renderer.is_paused

    def renderer_step_once(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].renderer.step_once

    def start_recording(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer.start_recording(filename, fps, skip)

    def stop_recording(mut self) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer.stop_recording()

    # ── ContinuousStateEnv ────────────────────────────────────────────────
    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=Self.OBS_DIM)
        var custom = Self.CONFIG.custom_extract_obs_cpu(
            self.d,
            self.mf.bodies.data,
            self.mf.joints.data,
            self.mf.geoms.data,
            self.mf.sites.data,
            self.act,
            obs,
        )
        if not custom:
            Self.MODEL_DEF.extract_obs(self.d, obs)
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        self._reset_state()
        return self.get_obs_list()

    def obs_dim(self) -> Int:
        return Self.OBS_DIM

    # ── ContinuousActionEnv ───────────────────────────────────────────────
    def action_dim(self) -> Int:
        return Self.ACTION_DIM

    def action_low(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](Self.MODEL_DEF.CTRL_MIN)

    def action_high(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](Self.MODEL_DEF.CTRL_MAX)

    # ── BoxContinuousActionEnv ────────────────────────────────────────────
    def step_continuous[
        DTYPE2: DType
    ](mut self, action: Scalar[DTYPE2]) -> Tuple[
        List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool
    ]:
        var actions = List[Scalar[DTYPE2]]()
        for _ in range(Self.ACTION_DIM):
            actions.append(Scalar[DTYPE2](action))
        return self.step_continuous_vec[DTYPE2](actions)

    def step_continuous_vec[
        DTYPE2: DType
    ](
        mut self, action: List[Scalar[DTYPE2]], verbose: Bool = False
    ) -> Tuple[List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool]:
        var act = ContAction[Self.MODEL_DEF.ACTION_DIM]()
        for i in range(min(Self.ACTION_DIM, len(action))):
            act.data[i] = Float64(action[i])
        var result = self.step(act, verbose=verbose)
        var obs_list = self.get_obs_list()
        var obs = List[Scalar[DTYPE2]](capacity=Self.OBS_DIM)
        for i in range(Self.OBS_DIM):
            obs.append(Scalar[DTYPE2](obs_list[i]))
        return (obs^, Scalar[DTYPE2](result[1]), result[2])
