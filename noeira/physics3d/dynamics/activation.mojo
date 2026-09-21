"""`mj_nextActivation` — the actuator activation ODE and its integration.

MuJoCo advances an actuator's activation state in two places that must agree:
`mj_fwdActuation` computes `act_dot` from the dyntype (engine_forward.c:434),
and `mj_nextActivation` (engine_support.c) integrates it and clamps it to
`actrange`. The three dyntypes this engine models are

    mjDYN_INTEGRATOR    act_dot = ctrl
    mjDYN_FILTER        act_dot = (ctrl - act) / max(mjMINVAL, tau)
    mjDYN_FILTEREXACT   the same act_dot, integrated EXACTLY

and only the last two share an `act_dot`. The integration then differs again:

    filterexact         act += act_dot * tau * (1 - exp(-h/tau))
    everything else     act += act_dot * h

followed, for every dyntype, by `clip(act, actrange)` when `actlimited`.

⚠⚠ THIS IS ONE FUNCTION BECAUSE THE RULE HAD FOUR CALL SITES. `filter` was
integrated inline in `dynamics/actuation.mojo`, in the GPU kernel
(`model_def_from_xml.mojo`), and TWICE in `dynamics/pose_transmission.mojo`
(the spatial-tendon loop and the site loop). Four copies of a rule that was
about to grow two branches and a clamp is the shape this tree has been burned
by more than any other — see the `<geom gap>` and `ctrllimited` cases. Every
one of those sites calls this now.

⚠ `filterexact` IS NOT A REFINEMENT OF `filter`, IT IS A DIFFERENT ANSWER.
Both integrate `(ctrl - act)/tau`, but Euler overshoots when `h` is not small
against `tau`: at `h == tau` Euler reaches the setpoint in ONE step while the
exact form reaches `1 - 1/e = 63%` of it. MuJoCo's oracle for
`dynprm=0.1`, `ctrl=1`, `h=0.002`: `act = 0.0198`, where Euler gives 0.02.
That 1% is the whole of AUD-02's error on an aloha filtered actuator, and it
compounds every step.

⚠ `<position timeconst="t">` IS `filterexact`, NOT A THIRD THING (AUD-21).
`mjs_setToPosition` (user_api.cc:1291-1294) writes `dynprm[0] = t` and
`dyntype = t == 0 ? mjDYN_NONE : mjDYN_FILTEREXACT`. `<intvelocity timeconst>`
routes through the same function. So the two audit ids are one feature under
two spellings and the parser resolves both to this.
"""

from std.math import exp

from ..gpu.constants import (
    ACT_DYN_FILTER,
    ACT_DYN_FILTEREXACT,
    ACT_DYN_INTEGRATOR,
)

# `mjMINVAL`. Guards the `tau` division exactly where MuJoCo guards it
# (`mju_max(mjMINVAL, dynprm[0])`), and no wider: a small POSITIVE tau is a
# fast filter, not a degenerate one.
comptime ACT_MINVAL: Float64 = 1e-15


@always_inline
def _exp_local[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """`exp`, with the floating-point evidence supplied LOCALLY.

    ⚠⚠ NOT `exp(x)` DIRECTLY, AND NOT A `where DTYPE.is_floating_point()` ON
    THE CALLER. The stdlib's `exp` carries that constraint; adding it to
    `next_activation` would propagate up through `apply_actions_fields` to the
    env-config trait's `custom_apply_actions_cpu`, whose signature every
    environment in this tree implements — and the compiler then rejects the
    WHOLE conformance ("method ... has constraints that cannot be proven or
    disproven"). `sensors/touch.mojo:352` documents the same wall and takes
    the same way round it: name the two concrete float types.

    ⚠ THE FALLBACK RETURNS 1, NOT `x`. This engine instantiates float32 and
    float64 only, so the branch is unreachable; `exp(0) == 1` is the identity
    for the `1 - exp(...)` it feeds, so an unreachable branch that somehow ran
    would freeze the activation rather than corrupt it.
    """
    comptime if DTYPE == DType.float64:
        return rebind[Scalar[DTYPE]](exp(rebind[Scalar[DType.float64]](x)))
    elif DTYPE == DType.float32:
        return rebind[Scalar[DTYPE]](exp(rebind[Scalar[DType.float32]](x)))
    else:
        return Scalar[DTYPE](1)


@always_inline
def next_activation[
    DTYPE: DType
](
    dyn_type: Int,
    act: Scalar[DTYPE],
    ctrl: Scalar[DTYPE],
    tau_in: Scalar[DTYPE],
    timestep: Scalar[DTYPE],
    act_limited: Bool,
    act_min: Scalar[DTYPE],
    act_max: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """The activation after one step. `ctrl` must ALREADY be ctrlrange-clamped.

    MuJoCo clamps `d->ctrl` before computing `act_dot` (`mj_fwdActuation`
    reads the clamped vector), so a caller that passes the raw control
    integrates a setpoint the force law never saw.

    ⚠ CALL THIS AFTER THE FORCE, NOT BEFORE. `mj_fwdActuation` reads the
    CURRENT activation to build the force and `mj_advance` advances it at the
    end of the same step; advancing first applies a force from an activation
    that is one step ahead of the one the reference used.
    """
    comptime MINVAL = Scalar[DTYPE](ACT_MINVAL)

    var tau = tau_in
    if tau < MINVAL:
        tau = MINVAL

    var act_dot: Scalar[DTYPE]
    if dyn_type == ACT_DYN_INTEGRATOR:
        act_dot = ctrl
    else:
        # FILTER and FILTEREXACT share this; every other dyntype reaching
        # here has been refused at load.
        act_dot = (ctrl - act) / tau

    var out: Scalar[DTYPE]
    if dyn_type == ACT_DYN_FILTEREXACT:
        # `act(h) = act(0) + act_dot(0) * tau * (1 - exp(-h/tau))`
        out = act + act_dot * tau * (
            Scalar[DTYPE](1) - _exp_local[DTYPE](-timestep / tau)
        )
    else:
        out = act + act_dot * timestep

    if act_limited:
        if out < act_min:
            out = act_min
        elif out > act_max:
            out = act_max
    return out


@always_inline
def act_is_stateful(dyn_type: Int) -> Bool:
    """Does this dyntype carry an activation the force law reads?

    `mjDYN_NONE` does not; the three modelled kinds do. Kept beside the
    integrator so "which dyntypes exist" has one answer.
    """
    return (
        dyn_type == ACT_DYN_INTEGRATOR
        or dyn_type == ACT_DYN_FILTER
        or dyn_type == ACT_DYN_FILTEREXACT
    )
