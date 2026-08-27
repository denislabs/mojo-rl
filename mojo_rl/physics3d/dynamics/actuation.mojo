"""`apply_actions_fields` — generalized forces from the spec records.

⚠ EXTRACTED FROM `ModelDefFromXML.apply_actions` (3d), NOT COPIED. That body
was a `@staticmethod` reading `Self.NV`/`Self.NQ`/`Self.nact`/`Self.NTEN_F`/
`Self.TIMESTEP`, so DRIVING a model was reachable only through a comptime
model def — however runtime its records were. `ModelDefFromXML.apply_actions`
now delegates here, so the CPU force path has ONE body and the two cannot
drift. Same pattern as `fields_build.apply_auto_spring_damper`, and both
replaced a duplicate rather than adding one.

The GPU twin (`apply_actions_kernel_gpu`) stays where it is and stays
comptime — decision 3.
"""

from mojo_rl.physics3d.fields import Data, DimsLike, SpecFields
# ⚠ The actuator KIND enum lives in the parser, beside the `ActuatorData`
# that carries it — not in `gpu/constants`, which holds the record LAYOUTS.
from mojo_rl.physics3d.parser.flat_model import (
    ACT_KIND_POSITION,
    ACT_KIND_VELOCITY,
    ACT_KIND_PID,
)
from mojo_rl.physics3d.gpu.constants import (
    ACTTEN_IDX_SPRING_HI,
    ACTTEN_IDX_SPRING_LO,
    ACTTEN_IDX_STIFFNESS,
    ACTTEN_IDX_TRN_COEF_0,
    ACTTEN_IDX_TRN_DADR_0,
    ACTTEN_IDX_TRN_N,
    ACTTEN_IDX_TRN_QADR_0,
    ACT_IDX_ACT_ADR,
    ACT_IDX_PID_KI,
    ACT_IDX_PID_KD,
    ACT_IDX_PID_IMAX,
    ACT_IDX_PID_SLEW,
    META_IDX_SIM_TIME,
    ACT_IDX_CTRL_LIMITED,
    ACT_IDX_CTRL_MAX,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_DYN_TAU,
    ACT_IDX_FORCE_LIMITED,
    ACT_IDX_FORCE_MAX,
    ACT_IDX_FORCE_MIN,
    JLIM_SIZE,
    META_IDX_ACTDAMP_LIVE,
    JLIM_IDX_DOF_ADR,
    JLIM_IDX_ACTFRC_LIMITED,
    JLIM_IDX_ACTFRC_MIN,
    JLIM_IDX_ACTFRC_MAX,
    ACT_IDX_GEAR,
    ACT_IDX_KIND,
    ACT_IDX_KP,
    ACT_IDX_KV,
    ACT_IDX_BIAS0,
    ACT_IDX_BIAS1,
    ACT_IDX_TRN_COEF_0,
    ACT_IDX_TRN_DADR_0,
    ACT_IDX_TRN_N,
    ACT_IDX_TRN_QADR_0,
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
)


@always_inline
def _floor1(n: Int) -> Int:
    """`SpecFields`' `*_F` floor, on the live dimension — the record tensors
    are sized `max(n, 1)`, so the loop bound must match."""
    return n if n > 0 else 1


@always_inline
def actuator_scalar_force(
    gain: Float64,
    u: Float64,
    has_bias: Bool,
    b0: Float64,
    b1: Float64,
    kv: Float64,
    length: Float64,
    vel: Float64,
) -> Float64:
    """`mj_fwdActuation`'s scalar `force[i]`, before `forcerange`.

    ONE COPY OF THE LAW, FIVE CALLERS. `apply_actions_fields` (CPU),
    `apply_actions_kernel_gpu`, and the three loops in `pose_transmission`
    (spatial-tendon actuator, site actuator, spatial-tendon spring) each used
    to spell it out. ⚠ The `<default>`-class rule that drifted between two
    inline copies (`ctrllimited`, 322 unclamped actuators) is the same shape
    of accident; a law with five copies is five chances to fix four of them.

    MuJoCo, `engine_forward.c:508-628`:

        gain  = gainprm[0]                                (gaintype FIXED)
        force = gain * u                                  (u = ctrl or act)
        bias  = biasprm[0] + biasprm[1]*length + biasprm[2]*velocity
        force += bias                                     (biastype AFFINE)

    ⚠ `kv` IS `-biasprm[2]`, which is how this record has always stored it,
    so the third term is written `(-kv) * vel`. Negation is exact, so this is
    `biasprm[2] * velocity` bit for bit.

    ⚠ `has_bias` IS `kind != ACT_KIND_MOTOR`, i.e. MuJoCo's `biastype ==
    mjBIAS_NONE`. It is passed rather than derived so the callers that
    already branch on `kind` do not load it twice, and so a caller that has
    no `length` to give (a site transmission, where MuJoCo defines it as 0)
    can still say honestly that it has a bias.

    ⚠ THE GROUPING IS THE REFERENCE'S, NOT AN EQUIVALENT ONE. The old
    position law was `kp*(u - length) - kv*vel`, which is the same number in
    exact arithmetic and a different one in float64. Every model whose
    actuators already matched to 0.000e+00 does so through this expression
    now; `gainprm[0] == -biasprm[1]` makes the two agree to the last bit only
    when nothing rounds.
    """
    var force = gain * u
    if has_bias:
        force = force + (b0 + b1 * length + (-kv) * vel)
    return force


def apply_actions_fields[
    DTYPE: DType, D: DimsLike, D2: DimsLike, NORMALIZED: Bool = False
](
    sf: SpecFields[DTYPE, D],
    mut d: Data[DTYPE, D2, 1],
    actions: List[Float64],
    mut act: List[Scalar[DTYPE]],
    timestep: Float64,
):
    """Generalized forces from the model spec: actuators + tendon springs.

    MuJoCo recomputes `qfrc_actuator` inside every `mj_step`, and for a
    `<motor>` that is redundant — its force is `gear * ctrl`, constant
    across a control step. A `<position>` servo is not: its force reads
    `qpos`, which moves every substep. So `Phyics3dEnv.step` calls this
    ONCE PER SUBSTEP rather than once per control step. For a motor-only
    model that is bit-identical (the same constant is written each time);
    for a servo it is the difference between a spring and a constant push.

    Both actuator kinds go through the same
    `force -> moment^T force` shape, over the transmission triples the
    comptime parser resolved (`motor_trn_*`):

        MOTOR     force = ctrl
        POSITION  force = kp*(ctrl - length) - kv*velocity
        length    = gear * sum_k coef_k qpos[qadr_k]
        velocity  = gear * sum_k coef_k qvel[dadr_k]
        qfrc[dadr_k] += gear * coef_k * force

    A joint transmission is one triple with coef 1, so the motor path
    reduces to the previous `qfrc[dof] = gear * ctrl` exactly.

    Accumulates rather than assigns, because a tendon transmission and a
    tendon spring can land on the same DOF (fish's `fins_flap` actuator
    and `fins_sym` spring share both fin roll joints) — hence the zeroing
    pass first. `d.qfrc` has exactly two other writers: `reset_data`,
    which zeroes it, and a CONFIG's `custom_apply_actions_cpu`, which
    returns True and suppresses this method entirely.
    """
    # ⚠ READ FROM `sf.dims`, NOT FROM A MODEL DEF (3d). This body used to be
    # `ModelDefFromXML.apply_actions`, a @staticmethod reading `Self.NV` /
    # `Self.NQ` / `Self.nact` / `Self.NTEN_F` / `Self.TIMESTEP` — which made
    # DRIVING a model reachable only through a comptime model def, however
    # runtime its records were. `timestep` is an argument for the same
    # reason: it is the one value here that is not a dimension.
    var nq = sf.dims.get_nq()
    var nv = sf.dims.get_nv()
    var n_act = sf.dims.get_nact()
    var n_ten = _floor1(sf.dims.get_nten())
    # ⚠ THE VALUES NOW COME FROM `sf`, NOT FROM `_acd`. This used to
    # materialize twenty-three comptime `InlineArray`s per call (Mojo 1.0
    # cannot index one at runtime), which is also why they were hoisted.
    # `SpecFields` is `List`-backed, so a read is a load and there is
    # nothing to hoist — but `o`/`to` below are the record base offsets and
    # every column is `base + IDX`, exactly as the record kernels address
    # `Model.bodies` / `Model.joints`.
    #
    # ⚠⚠ THE ARITHMETIC IS STILL `Float64`, DELIBERATELY. `DTYPE` is
    # float64 on the CPU env, so every read is exact — but a caller that
    # instantiated this at float32 would otherwise silently drop the gains
    # to float32 and move every force. Widening at the load keeps the
    # chain identical to the `_acd` version term for term.
    # ⚠⚠ `act` IS SIZED BY THE CALLER AND `<plugin plugin="mujoco.pid">` NEEDS
    # MORE THAN ONE SLOT PER ACTUATOR. Every caller in this tree allocates
    # `List(length=nact)`, which was exactly right while the only activation
    # was `mjDYN_FILTER`'s single lag variable. A PID actuator owns TWO (the
    # error integral and the previous control), so shadow_dexee's twelve want
    # 24 and the loop below would have silently skipped every one of them on
    # the `adr < len(act)` guard — a zero-gain PID, which is the same failure
    # mode as the unresolved `instance=` the parser now refuses.
    #
    # ⚠ GROWN HERE RATHER THAN PLUMBED THROUGH `Dims`. `na` is not a
    # dimension of any tensor the kernels bind; it sizes ONE host list, and
    # this is the only function that knows what each actuator's kind spends.
    # Growth is zero-filled, which is what `mj_resetData` leaves in `d->act`.
    var need = 0
    for i in range(n_act):
        var oo = i * MODEL_ACTUATOR_SIZE
        var aa = Int(sf.actuators.data[oo + ACT_IDX_ACT_ADR])
        if aa < 0:
            continue
        var slots = 1
        if Int(sf.actuators.data[oo + ACT_IDX_KIND]) == ACT_KIND_PID:
            slots = 0
            if Float64(sf.actuators.data[oo + ACT_IDX_PID_KI]) != 0.0:
                slots += 1
            if Float64(sf.actuators.data[oo + ACT_IDX_PID_SLEW]) >= 0.0:
                slots += 1
        if aa + slots > need:
            need = aa + slots
    while len(act) < need:
        act.append(Scalar[DTYPE](0))

    for i in range(nv):
        d.qfrc.data[i] = Scalar[DTYPE](0)
    # ── THIS STEP's actuator damping diagonal ────────────────────────────
    #
    # ⚠⚠ IT IS NOT `Model.dof_actdamp`. MuJoCo's `mjd_actuator_vel` opens by
    # SKIPPING any actuator whose force is clamped by its `forcerange` — a
    # saturated actuator's force is pinned at the bound and no longer depends
    # on velocity, so it contributes NOTHING to `qDeriv`. Whether it is
    # saturated changes every step, so the model-time value (baked from `kv`)
    # is right only while nothing is clamped.
    #
    # ⚠ MEASURED ON rby1, whose 24 position servos are `forcerange="-270 270"`
    # and saturate at `qpos0`: MuJoCo's own `qDeriv` diagonal reads -5 on
    # those dofs (joint damping alone) and -4005 on the two `<velocity>` wheel
    # dofs, which have no `forcerange` and are not clamped. We used -405
    # everywhere and over-damped every saturated dof.
    for i in range(nv):
        d.dof_actdamp.data[i] = Scalar[DTYPE](0)
    # ⚠ THE OFF-DIAGONAL'S LIVE GATE, cleared with the diagonal. 1.0 means
    # "this actuator contributed to `qDeriv` this step"; `Model.actdamp_trn`
    # holds the constants it is multiplied by. Zeroed here and raised below
    # under exactly the test the diagonal uses, so the two halves of
    # `mjd_actuator_vel` cannot disagree about a saturated actuator.
    for i in range(n_act):
        d.actdamp_act.data[i] = Scalar[DTYPE](0)
    d.meta.data[META_IDX_ACTDAMP_LIVE] = Scalar[DTYPE](1)

    for i in range(n_act):
        if i >= len(actions):
            break
        var o = i * MODEL_ACTUATOR_SIZE
        var n = Int(sf.actuators.data[o + ACT_IDX_TRN_N])
        if n == 0:
            continue
        # Clamp to per-actuator ctrlrange (per-element overrides default),
        # but ONLY when the actuator is `ctrllimited`.
        #
        # ⚠⚠ THE GUARD IS THE POINT. This clamp used to be unconditional,
        # against a `ctrlrange` that falls back to (-1, 1) when the model
        # supplies none — so an actuator MuJoCo leaves unclamped had its
        # command silently squeezed into +-1. See
        # `ComptimeActData.motor_ctrl_limited` for the measured semantics
        # and for why no dm_control or Gymnasium model here could reveal it
        # (0 of 254 actuators unlimited) while ToddlerBot is 30 of 30.
        var ctrl = actions[i]
        if sf.actuators.data[o + ACT_IDX_CTRL_LIMITED] != 0:
            var c_max = Float64(sf.actuators.data[o + ACT_IDX_CTRL_MAX])
            var c_min = Float64(sf.actuators.data[o + ACT_IDX_CTRL_MIN])
            # ⚠⚠ NORMALIZED ACTION SPACE — see `Phyics3dEnvConfig.
            # NORMALIZED_ACTIONS`. The action arrives in [-1, 1] and is mapped
            # AFFINELY onto this actuator's own `ctrlrange`, so a policy's
            # `tanh` rails land exactly ON the joint limits instead of outside
            # them. The clamp below then only catches inputs that were already
            # out of [-1, 1], and the dead band where many actions map to one
            # clamped pose — the band whose gradient is zero — disappears.
            #
            # ⚠ ONLY FOR A `ctrllimited` ACTUATOR, because an unlimited one has
            # no range to map onto; its action passes through raw. A model
            # mixing the two gets a mixed action space, which is why the flag
            # is per-CONFIG and not per-actuator.
            comptime if NORMALIZED:
                ctrl = c_min + (ctrl + 1.0) * 0.5 * (c_max - c_min)
            if ctrl > c_max:
                ctrl = c_max
            elif ctrl < c_min:
                ctrl = c_min

        var gear = Float64(sf.actuators.data[o + ACT_IDX_GEAR])

        # ACTIVATION (MuJoCo `d->act`). `force = gain .* [ctrl/act]`
        # (mj_fwdActuation): an actuator with a `dyntype` feeds its
        # activation to the gain where a plain one feeds `ctrl`. The
        # activation itself is a first-order lag of `ctrl`.
        #
        # `u` is what the gain multiplies. `act` is integrated AFTER the
        # force is computed, matching MuJoCo's order — `mj_fwdActuation`
        # reads the current `act`, and `mj_advance` advances it at the end
        # of the same step (`actearly` is off here). This function runs
        # ONCE PER SUBSTEP, which is the same cadence, so the two agree
        # step for step.
        var adr = Int(sf.actuators.data[o + ACT_IDX_ACT_ADR])
        # ⚠⚠ READ BEFORE THE ACTIVATION, BECAUSE PID'S SLOTS ARE NOT A `u`.
        # `mjDYN_FILTER`'s single activation IS the value the gain multiplies;
        # a `mujoco.pid` actuator's first slot is its ERROR INTEGRAL, and
        # feeding that to `force = kp * u` would multiply the wrong quantity
        # by the wrong gain. `kind` used to be read forty lines further down.
        var kind = Int(sf.actuators.data[o + ACT_IDX_KIND])
        var u = ctrl
        if kind != ACT_KIND_PID and adr >= 0 and adr < len(act):
            u = Float64(act[adr])

        # `motor_kp` is MuJoCo's `gainprm[0]`, whose default is 1 — so a
        # plain `<motor>`, which never writes it, is `force = ctrl`. A
        # bias-free `<general>` lands here too and its gain is real: dog's
        # actuators are `force = 0.02 * act`.
        var kp = Float64(sf.actuators.data[o + ACT_IDX_KP])
        var force = kp * u
        comptime _POS = ACT_KIND_POSITION
        comptime _VEL = ACT_KIND_VELOCITY
        # ⚠ `kind != MOTOR` IS `biastype != mjBIAS_NONE`. The parser sets
        # MOTOR exactly when `<general>` has no bias block and for every
        # `<motor>` element, so this branch is the bias, not a heuristic.
        if kind == _POS or kind == _VEL:
            var length = Float64(0)
            var vel = Float64(0)
            for k in range(n):
                var qadr = Int(
                    sf.actuators.data[o + ACT_IDX_TRN_QADR_0 + k]
                )
                var dadr = Int(
                    sf.actuators.data[o + ACT_IDX_TRN_DADR_0 + k]
                )
                var coef = Float64(
                    sf.actuators.data[o + ACT_IDX_TRN_COEF_0 + k]
                )
                if qadr >= 0 and qadr < nq:
                    length += coef * Float64(d.qpos.data[qadr])
                if dadr >= 0 and dadr < nv:
                    vel += coef * Float64(d.qvel.data[dadr])
            length *= gear
            vel *= gear
            # MuJoCo writes the same gaintype/biastype for both servo laws;
            # the ONLY difference is `biasprm[1]`, which is `-gainprm[0]`
            # for `<position>` and 0 for `<velocity>`. So the two share this
            # whole transmission walk and differ in one term:
            #     POSITION  force = kp*(u - length) - kv*vel
            #     VELOCITY  force = kp*u            - kv*vel
            # ⚠ VELOCITY must NOT subtract `length`. Doing so would add a
            # position feedback MuJoCo does not have, and on Jaco (kv=500)
            # a 0.1 rad offset would inject 50 N·m of phantom torque.
            # `u`, not `ctrl` — for a dyntype actuator the servo setpoint
            # is the ACTIVATION, which lags the control. They coincide
            # only when the actuator has no activation (then u == ctrl).
            force = actuator_scalar_force(
                kp,
                u,
                True,
                Float64(sf.actuators.data[o + ACT_IDX_BIAS0]),
                Float64(sf.actuators.data[o + ACT_IDX_BIAS1]),
                Float64(sf.actuators.data[o + ACT_IDX_KV]),
                length,
                vel,
            )

        # ── `<plugin plugin="mujoco.pid">` ───────────────────────────────
        #
        # `plugin/actuator/pid.cc` (MuJoCo 3.10.0), `Pid::Compute`, for
        # `dyntype="none"` — the only dyntype the parser admits:
        #
        #     ctrl  = clip(d->ctrl[i], ctrlrange)                [ctrllimited]
        #     ctrl  = clip(ctrl, prev +- slewmax*dt)          [slew AND t > 0]
        #     err   = ctrl - actuator_length[i]
        #     integ = clip(act[adr] + err*dt, +- imax)     [ki != 0; imax/ki]
        #     force = kp*err + kd*(0 - actuator_velocity[i]) + ki*integ
        #
        # ⚠ NOT `actuator_scalar_force`. That function is `gain*u + affine
        # bias`, which is every OTHER law in this engine; a plugin's
        # `gaintype`/`biastype` stay at their defaults and its gains are not
        # in `gainprm`/`biasprm` at all. Routing PID through it would compute
        # `1 * ctrl`.
        #
        # ⚠ THE ERROR DERIVATIVE IS `-velocity`, NOT `d(err)/dt`. `ctrl_dot`
        # is 0 for `dyntype="none"` (there is no activation behind the
        # setpoint), so the whole of it is the transmission velocity, and the
        # sign matters: `kd` DAMPS.
        #
        # ⚠⚠ AND IT CONTRIBUTES NOTHING TO `qDeriv`, DELIBERATELY.
        # `mjd_actuator_vel` branches on `biastype`/`gaintype` only
        # (engine_derivative.c:1245), so MuJoCo does not differentiate a
        # plugin's force either — `kd` is invisible to the implicit
        # integrator on BOTH sides. The `dof_actdamp` block below is guarded
        # on `_POS`/`_VEL` and so already does the right thing; this note is
        # here because "we have a velocity gain, so we must damp implicitly"
        # is the obvious wrong move.
        var pid_integral = Float64(0)
        var pid_ctrl = Float64(0)
        if kind == ACT_KIND_PID:
            var length_p = Float64(0)
            var vel_p = Float64(0)
            for k in range(n):
                var qadr_p = Int(
                    sf.actuators.data[o + ACT_IDX_TRN_QADR_0 + k]
                )
                var dadr_p = Int(
                    sf.actuators.data[o + ACT_IDX_TRN_DADR_0 + k]
                )
                var coef_p = Float64(
                    sf.actuators.data[o + ACT_IDX_TRN_COEF_0 + k]
                )
                if qadr_p >= 0 and qadr_p < nq:
                    length_p += coef_p * Float64(d.qpos.data[qadr_p])
                if dadr_p >= 0 and dadr_p < nv:
                    vel_p += coef_p * Float64(d.qvel.data[dadr_p])
            length_p *= gear
            vel_p *= gear

            var ki = Float64(sf.actuators.data[o + ACT_IDX_PID_KI])
            var kd = Float64(sf.actuators.data[o + ACT_IDX_PID_KD])
            var imax = Float64(sf.actuators.data[o + ACT_IDX_PID_IMAX])
            var slew = Float64(sf.actuators.data[o + ACT_IDX_PID_SLEW])

            # The two activation slots, in `Pid::GetState`'s order: the
            # integral first (present iff ki != 0), then the previous
            # control (present iff slewmax was configured).
            var i_slot = adr if ki != 0.0 else -1
            var s_slot = -1
            if slew >= 0.0:
                s_slot = adr + (1 if ki != 0.0 else 0)

            pid_ctrl = ctrl
            # ⚠⚠ `previous_ctrl_exists = d->time > 0`. On the FIRST step the
            # slot holds a zero that is not a control, and clamping to
            # `0 +- slewmax*dt` would cap every actuator at a few
            # milliradians of command. This is the whole reason
            # `META_IDX_SIM_TIME` exists.
            if (
                s_slot >= 0
                and s_slot < len(act)
                and Float64(d.meta.data[META_IDX_SIM_TIME]) > 0.0
            ):
                var prev = Float64(act[s_slot])
                var lo_s = prev - slew * timestep
                var hi_s = prev + slew * timestep
                if pid_ctrl > hi_s:
                    pid_ctrl = hi_s
                elif pid_ctrl < lo_s:
                    pid_ctrl = lo_s

            var err = pid_ctrl - length_p
            var err_dot = -vel_p
            if i_slot >= 0 and i_slot < len(act):
                pid_integral = Float64(act[i_slot]) + err * timestep
                # ⚠ `imax` IS ALREADY DIVIDED BY `ki` — see `ACT_IDX_PID_IMAX`.
                if imax >= 0.0:
                    if pid_integral > imax:
                        pid_integral = imax
                    elif pid_integral < -imax:
                        pid_integral = -imax
            # ⚠⚠ AN EXPLICIT `fma` ON THE **FIRST** PRODUCT, AND THAT IS THE
            # DIFFERENCE BETWEEN 0.0 AND 1 ulp. `pid.cc` writes
            #
            #     p_gain*error + d_gain*error_dot + i_gain*integral
            #
            # and the shipped build contracts `p_gain*error` into the add,
            # evaluating `fma(kp, err, kd*err_dot) + ki*integral`. Measured by
            # enumerating every association and every contraction of that
            # expression against MuJoCo 3.10.0's own `actuator_force` at four
            # states: of 24 candidates, ONLY the two that fuse the p-term hit
            # it exactly, and the plain left-to-right sum is 3.469e-18 low.
            #
            # ⚠ FUSING THE d-TERM INSTEAD DOES NOT WORK — that was the first
            # guess here and it measured no change at all. The two products
            # are not interchangeable.
            #
            # ⚠ IT IS INVISIBLE UNTIL THE JOINT MOVES. At step 0 `err_dot` is
            # 0, so the fused and unfused forms agree bit for bit; a gate that
            # only compared the first step would have called the plain sum
            # exact.
            var d_term = kd * err_dot
            force = kp.fma(err, d_term) + ki * pid_integral

        # `forcerange` (mj_fwdActuation). ⚠ THE CLAMP IS HERE — on the
        # SCALAR force, BEFORE the moment loop below multiplies by
        # `gear * coef`. Measured on 3.10.0: `<motor gear="3"
        # forcerange="-1 1">` at ctrl 5 gives actuator_force 1, moment 3,
        # qfrc 3. Clamping the accumulated `qfrc` instead would cap this
        # actuator at 1 N·m where MuJoCo delivers 3.
        var saturated = False
        if sf.actuators.data[o + ACT_IDX_FORCE_LIMITED] != 0:
            var f_hi = Float64(sf.actuators.data[o + ACT_IDX_FORCE_MAX])
            var f_lo = Float64(sf.actuators.data[o + ACT_IDX_FORCE_MIN])
            if force > f_hi:
                force = f_hi
            elif force < f_lo:
                force = f_lo
            # ⚠ THE TEST IS ON THE CLAMPED FORCE AND IT IS `<=` / `>=`, not a
            # strict compare — MuJoCo's is `force <= range[0] || force >=
            # range[1]`, so an actuator sitting EXACTLY on its bound also
            # loses its derivative. Testing "did the clamp change the value"
            # instead would disagree on that boundary.
            saturated = force <= f_lo or force >= f_hi

        # `-d force / d qvel` for this actuator, onto each transmission dof.
        # POSITION and VELOCITY are the two laws with a `kv` term; a MOTOR's
        # force does not depend on velocity at all.
        if (kind == _POS or kind == _VEL) and not saturated:
            var kv_d = Float64(sf.actuators.data[o + ACT_IDX_KV])
            if kv_d != 0.0:
                for k in range(n):
                    var dadr_d = Int(
                        sf.actuators.data[o + ACT_IDX_TRN_DADR_0 + k]
                    )
                    if dadr_d < 0 or dadr_d >= nv:
                        continue
                    var gc = gear * Float64(
                        sf.actuators.data[o + ACT_IDX_TRN_COEF_0 + k]
                    )
                    d.dof_actdamp.data[dadr_d] += Scalar[DTYPE](
                        kv_d * gc * gc
                    )
                # The off-diagonal half — see `Model.actdamp_trn`. Set
                # INSIDE the `kv_d != 0` guard so an actuator with no
                # velocity feedback never lights up.
                d.actdamp_act.data[i] = Scalar[DTYPE](1)

        for k in range(n):
            var dadr = Int(sf.actuators.data[o + ACT_IDX_TRN_DADR_0 + k])
            if dadr < 0 or dadr >= nv:
                continue
            d.qfrc.data[dadr] += Scalar[DTYPE](
                gear
                * Float64(sf.actuators.data[o + ACT_IDX_TRN_COEF_0 + k])
                * force
            )

        # mjDYN_FILTER, integrated by Euler exactly as `nextActivation`
        # does for a non-`filterexact` dyntype (engine_forward.c:341):
        #     act_dot = (ctrl - act) / tau ;  act += act_dot * timestep
        # `ctrl` here is already ctrlrange-clamped, matching MuJoCo, which
        # clamps `d->ctrl` before computing act_dot.
        if kind != ACT_KIND_PID and adr >= 0 and adr < len(act):
            var tau = Float64(sf.actuators.data[o + ACT_IDX_DYN_TAU])
            if tau < 1e-10:
                tau = 1e-10  # mjMINVAL guard, as MuJoCo applies
            act[adr] = Scalar[DTYPE](
                u + (ctrl - u) / tau * timestep
            )

        # The PID plugin's activations. `Pid::ActDot` writes
        # `act_dot = (target - act) / dt` for both slots and MuJoCo then
        # integrates `act += act_dot * dt`, so the slot simply BECOMES the
        # target — exactly, not to within a timestep. Writing the target is
        # the same number without the round trip through a division.
        #
        # ⚠ AFTER THE FORCE AND AFTER THE `forcerange` CLAMP, matching
        # `mj_fwdActuation` -> `mj_advance`: the integral that was clamped
        # into the force above is the one stored, and a saturated actuator
        # still integrates its error (which is what makes PID wind-up real
        # and `imax` necessary).
        if kind == ACT_KIND_PID and adr >= 0:
            var ki_w = Float64(sf.actuators.data[o + ACT_IDX_PID_KI])
            var slew_w = Float64(sf.actuators.data[o + ACT_IDX_PID_SLEW])
            var w = adr
            if ki_w != 0.0:
                if w < len(act):
                    act[w] = Scalar[DTYPE](pid_integral)
                w += 1
            if slew_w >= 0.0 and w < len(act):
                act[w] = Scalar[DTYPE](pid_ctrl)

    # ── `jnt_actfrcrange` — MuJoCo's SECOND force clamp ──────────────────
    #
    #     clampVec(d->qfrc_actuator, m->jnt_actfrcrange, m->jnt_actfrclimited,
    #              m->njnt, m->jnt_dofadr);          // engine_forward.c:477
    #
    # ⚠⚠ NOT THE SAME LIMIT AS `forcerange` ABOVE, AND HAVING ONE IS NOT
    # HAVING THE OTHER. `forcerange` is per-ACTUATOR and clamps that
    # actuator's SCALAR force before the moment; this is per-JOINT and clamps
    # the ACCUMULATED `qfrc_actuator` at the joint's dof address, after every
    # actuator has contributed. On unitree_g1 `actuator_forcelimited` is FALSE
    # on all 29 actuators while `jnt_actfrclimited` is TRUE on 29 of 30
    # joints, so this is the only force limit that model has — and it was the
    # one we did not implement. 481 of the tree's 2519 joints declare it,
    # across 20 robots.
    #
    # ⚠ BEFORE THE SPRINGS, NOT AFTER. A fixed-tendon spring is
    # `qfrc_passive` and is NOT subject to this limit; clamping after the
    # loop below would clamp a sum MuJoCo never clamps. `d.qfrc` is our
    # single accumulator for both, so the ORDER is what keeps them separable.
    #
    # ⚠ THE DOF ADDRESS, NOT THE QPOS ONE — `jnt_dofadr` is the index in
    # MuJoCo's call, and the two differ on every model with a free or ball
    # joint (g1's are 7 and 6).
    var n_jnt = sf.dims.get_njoint()
    for j in range(n_jnt):
        var jo = j * JLIM_SIZE
        if sf.joint_limits.data[jo + JLIM_IDX_ACTFRC_LIMITED] == 0:
            continue
        var jdof = Int(sf.joint_limits.data[jo + JLIM_IDX_DOF_ADR])
        if jdof < 0 or jdof >= nv:
            continue
        var a_hi = Float64(sf.joint_limits.data[jo + JLIM_IDX_ACTFRC_MAX])
        var a_lo = Float64(sf.joint_limits.data[jo + JLIM_IDX_ACTFRC_MIN])
        var cur = Float64(d.qfrc.data[jdof])
        if cur > a_hi:
            d.qfrc.data[jdof] = Scalar[DTYPE](a_hi)
        elif cur < a_lo:
            d.qfrc.data[jdof] = Scalar[DTYPE](a_lo)

    # Fixed-tendon springs (`engine_passive.c`, tendon-level spring):
    # a DEADBAND on `tendon_lengthspring`, zero inside the band.
    # ⚠ THE BOUND IS `_NTEN` (the RECORD CAPACITY) WHERE IT USED TO BE
    # `_acd.ntendon` (the real count). Padding rows are zero-filled by
    # `TensorImpl.alloc` and `build_spec_fields` never touches them, so
    # `stiffness == 0` skips them on the first test — the same test that
    # already skipped every real tendon without a spring. Iterating the
    # capacity is what lets the count stop being a comptime quantity.
    for t in range(n_ten):
        var to = t * MODEL_ACT_TENDON_SIZE
        var k_spring = Float64(sf.act_tendons.data[to + ACTTEN_IDX_STIFFNESS])
        if k_spring == 0.0:
            continue
        var n = Int(sf.act_tendons.data[to + ACTTEN_IDX_TRN_N])
        if n == 0:
            continue
        var length = Float64(0)
        for k in range(n):
            var qadr = Int(
                sf.act_tendons.data[to + ACTTEN_IDX_TRN_QADR_0 + k]
            )
            if qadr >= 0 and qadr < nq:
                length += (
                    Float64(
                        sf.act_tendons.data[to + ACTTEN_IDX_TRN_COEF_0 + k]
                    )
                    * Float64(d.qpos.data[qadr])
                )
        var lo = Float64(sf.act_tendons.data[to + ACTTEN_IDX_SPRING_LO])
        var hi = Float64(sf.act_tendons.data[to + ACTTEN_IDX_SPRING_HI])
        var frc = Float64(0)
        if length > hi:
            frc = k_spring * (hi - length)
        elif length < lo:
            frc = k_spring * (lo - length)
        if frc == 0.0:
            continue
        for k in range(n):
            var dadr = Int(
                sf.act_tendons.data[to + ACTTEN_IDX_TRN_DADR_0 + k]
            )
            if dadr < 0 or dadr >= nv:
                continue
            d.qfrc.data[dadr] += Scalar[DTYPE](
                Float64(
                    sf.act_tendons.data[to + ACTTEN_IDX_TRN_COEF_0 + k]
                )
                * frc
            )

    # ── `d->time`, advanced where MuJoCo advances it ─────────────────────
    #
    # ⚠⚠ NOTHING ELSE IN THIS ENGINE HAS EVER NEEDED IT, which is why there
    # was no clock at all. `mujoco.pid`'s slew limiter does: its
    # `previous_ctrl_exists` is literally `d->time > 0`, and a zero in the
    # previous-control slot is a legal control, so no value in `act` can
    # answer the question. See `META_IDX_SIM_TIME`.
    #
    # ⚠ LAST, NOT FIRST. `mj_advance` runs at the END of `mj_step`, after
    # `mj_fwdActuation` has read the time — so a step-0 call must see 0. This
    # function's cadence is one call per control substep, the same as one
    # `mj_step`.
    #
    # ⚠ THIS IS NOT A GENERAL-PURPOSE CLOCK YET. A caller that steps the
    # integrator without calling this function does not advance it, and a
    # caller that calls this twice for one step advances it twice. Both are
    # true of `META_IDX_ACTDAMP_LIVE` beside it for the same reason: these
    # slots describe what THIS function did.
    d.meta.data[META_IDX_SIM_TIME] += Scalar[DTYPE](timestep)


# =========================================================================
# Model build (spec-direct; G4)
# =========================================================================

# =========================================================================
# GPU: _compute_invweight0_gpu (duplicated from ModelDef, dims from params)
# =========================================================================
