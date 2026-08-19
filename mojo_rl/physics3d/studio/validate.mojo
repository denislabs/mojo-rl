"""What is wrong with this model — as MARKERS, not as an abort. V2.0.

## ⚠⚠ WHY THIS IS A NEW REQUIREMENT AND NOT A NICE-TO-HAVE

Every invariant this tree enforces is enforced **at parse time**, by raising.
That is right for a loader and wrong for an editor: V1 could only change
numbers, so no edit could reach an invalid model, but V2 deletes a geom, and

    <body name="thigh"><joint type="hinge"/><geom .../></body>
                                            ^^^^^^^^ delete this

is a model MuJoCo REFUSES — "mass and inertia of moving bodies must be larger
than mjMINVAL". Two clicks, and the tool's only vocabulary for it would be an
exception. The user cannot get back: the state they need to pass through to
fix it is the state that will not load.

⇒ **The answer is a list, not a raise.** An invalid intermediate state must be
workable — red in the panel, unsteppable if need be, but still on screen and
still editable. `docs/PHYSICS3D_STUDIO_PLAN.md` §4.1.

## ⚠⚠⚠ THE RULE THAT DEFINES `SEV_ERROR`, AND IT IS NOT "LOOKS WRONG"

    ERROR  ==  this model WILL NOT LOAD — MuJoCo refuses it, or (one case,
               tagged) it uses a feature this engine does not implement and
               refuses rather than building a fraction of silently
    WARN   ==  it loads, and it is probably not what was meant

Nothing else. Every ERROR below was established by handing the case to the
3.10.0 runtime and recording the verdict, and the gate re-checks that
correspondence in both directions. The engine-only exemption is ONE named
fixture (`<replicate>`); keeping it a list rather than a principle is what
stops it becoming the place wrong checks go to hide.

The correspondence matters because the tempting checks
are the wrong ones — measured, MuJoCo **accepts** all of these:

    <pair geom1="g" geom2="g"/>          a geom paired with itself
    <exclude body1="b" body2="b"/>       a body excluded against itself
    contype="0" conaffinity="0"          a geom that can never collide
    gear="0"                             an actuator that applies no force
    damping="-1"  armature="-1"          negative dissipation
    solimp="0.9 2.0 0.001"               an impedance above 1
    limited="false" range="1 -1"         an inverted range that is not used
    <inertial mass="0"/>                 on a body with no joints

A validator that called those errors would light up real, working models and
be turned off within a day. Several are worth a WARNING; none is an error.
And the converse trap is worse — these look harmless and MuJoCo refuses them:

    range="1 1"       on a LIMITED joint   (strictly smaller, not <=)
    size="1 1 0"      on a plane           (only the THIRD number must be > 0)
    <joint type="hinge" axis="0 0 0"/>     (a ball joint's axis may be zero)
    a repeated name WITHIN one element kind (a body and a geom may share one)

⚠ THE ORDERING TRAP THIS FILE EXISTS TO AVOID. Being stricter than the
reference is not "safe". A marker on a model that loads is a false alarm, and
false alarms are how a diagnostics panel becomes invisible.

## Two entry points, because they answer questions at different times

`validate_document(xml)` runs on TEXT, before anything is parsed, and is the
only one that can speak about a model that does not load at all — a dangling
`joint=` makes `full_parser` raise, so by the time a `FlatModelDef` exists the
question is already settled. `validate_model(fmd, m)` runs after the build and
covers everything that needs numbers.

⚠ THE MASS CHECK NEEDS THE BUILT `Model`, NOT THE RECORD. `fmd.bodies[i].mass`
is what the FILE said; a body whose mass comes from its geoms (the common
case) carries whatever the parser defaulted to, and the derivation
(`_inertia_from_geoms_staging`, `<compiler settotalmass>`, `boundmass`) runs in
`fields_build`. Validating the record would report the mass of a body nobody
wrote a mass for.
"""

from ..parser.flat_model import FlatModelDef
from ..parser.expander import dangling_references, generator_elements
from ..fields import Model, DynDims
from ..gpu.constants import (
    MODEL_BODY_SIZE, BODY_IDX_MASS, BODY_IDX_IXX, BODY_IDX_IYY, BODY_IDX_IZZ,
)
from .writer import unwritable


comptime DT = DType.float64

comptime SEV_INFO: Int = 0
comptime SEV_WARN: Int = 1
comptime SEV_ERROR: Int = 2

# MuJoCo's `mjMINVAL` (`mjmodel.h`). The mass/inertia check compares against
# this exact constant, so ours has to be the same number and not a rounder one.
comptime MJMINVAL: Float64 = 1e-15


def severity_name(s: Int) -> String:
    if s >= SEV_ERROR:
        return String("ERROR")
    if s == SEV_WARN:
        return String("warning")
    return String("info")


struct Diagnostic(Copyable, Movable):
    """One finding, addressed to a human and locatable in the model.

    ⚠ `code` IS A STABLE SLUG AND `message` IS PROSE. The gate asserts on the
    code — a test that matched on the message would break every time the
    wording improved, which is how diagnostics stop being reworded.
    """

    var severity: Int
    var code: String
    var subject: String
    """What it is about, named the way the user sees it: `body 'thigh'`."""
    var message: String

    def __init__(
        out self, severity: Int, code: String, subject: String, message: String
    ):
        self.severity = severity
        self.code = code
        self.subject = subject
        self.message = message


def format_diagnostic(d: Diagnostic) -> String:
    return severity_name(d.severity) + ": " + d.subject + " — " + d.message


def worst_severity(ds: List[Diagnostic]) -> Int:
    """The highest severity present, or `SEV_INFO` for an empty list.

    ⚠ THE LOAD PATH BRANCHES ON THIS, not on `len(ds)`. A model with three
    warnings is a model that loads.
    """
    var w = SEV_INFO
    for d in ds:
        if d.severity > w:
            w = d.severity
    return w


def count_at(ds: List[Diagnostic], severity: Int) -> Int:
    var n = 0
    for d in ds:
        if d.severity == severity:
            n += 1
    return n


# =============================================================================
# Text level — before anything is parsed
# =============================================================================


def validate_document(xml: String) -> List[Diagnostic]:
    """What is wrong with this MJCF text, without loading it.

    ⚠⚠ THIS FUNCTION MUST NOT PROPAGATE. It is the thing standing between an
    invalid edit and an abort, so every call it makes is wrapped: a checker
    that itself raises becomes a diagnostic saying so. `def` in Mojo raises by
    default, so "cannot fail" has to be built, not declared.

    ⚠ THE DANGLING-REFERENCE SCAN IS `expander.dangling_references`, THE SAME
    FUNCTION `check_references` RAISES FROM. Re-implementing it here is the
    two-parsers-one-wrong-default failure this tree keeps meeting: the copy
    would agree with the original until one of them learned about a new
    reference attribute, and then a panel would say a model was clean while
    the loader refused it.
    """
    var out = List[Diagnostic]()

    try:
        var gens = generator_elements(xml)
        for g in gens:
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("generator-unsupported"),
                    g,
                    String(
                        "generates bodies from a description; this engine"
                        " does not implement it, and the loader refuses the"
                        " model rather than building the few elements that"
                        " were written out literally."
                    ),
                )
            )
    except e:
        out.append(_checker_failed(String("generator-unsupported"), String(e)))

    try:
        var bad = dangling_references(xml)
        for b in bad:
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("dangling-ref"),
                    b,
                    String(
                        "names nothing this document declares. MuJoCo refuses"
                        " the model; deleting the element that DECLARED the"
                        " name is the usual cause."
                    ),
                )
            )
    except e:
        out.append(_checker_failed(String("dangling-ref"), String(e)))

    return out^


def _checker_failed(code: String, msg: String) -> Diagnostic:
    """A checker that raised is itself a finding — never a lost check.

    ⚠ SEV_ERROR, NOT A SILENT SKIP. "the validator could not answer" and "the
    validator found nothing" must not look the same in the panel; the second
    is the one that lets a broken model through.
    """
    return Diagnostic(
        SEV_ERROR,
        String("checker-failed"),
        String("validator (") + code + ")",
        String("this check could not run: ") + msg,
    )


# =============================================================================
# Model level — after the parse and the build
# =============================================================================


def validate_model(
    fmd: FlatModelDef, m: Model[DT, DynDims]
) -> List[Diagnostic]:
    """Everything that needs the numbers, on a model that already loaded."""
    var out = List[Diagnostic]()
    var nb = len(fmd.bodies)

    _check_body_mass(fmd, m, out)
    _check_geom_sizes(fmd, out)
    _check_joints(fmd, out)
    _check_actuators(fmd, out)
    _check_placement(fmd, out)
    _check_duplicate_names(fmd, out)
    _check_soft(fmd, out)

    try:
        var missing = unwritable(fmd)
        if missing.byte_length() > 0:
            out.append(
                Diagnostic(
                    SEV_INFO,
                    String("not-exportable"),
                    String("model"),
                    String(
                        "flattened export would drop:"
                    ) + missing + ". The scene document still round-trips.",
                )
            )
    except e:
        out.append(_checker_failed(String("not-exportable"), String(e)))

    _ = nb
    return out^


def _body_name(fmd: FlatModelDef, b: Int) -> String:
    """`body 'thigh'`, or `body 4` when the source named nothing.

    ⚠ NOT A SYNTHESISED NAME. `body_names[b]` is "" for an unnamed body and
    inventing `body4` here would let a user search the file for a string that
    is not in it.
    """
    if b >= 0 and b < len(fmd.body_names):
        var n = fmd.body_names[b]
        if n.byte_length() > 0:
            return String("body '") + n + "'"
    return String("body ") + String(b)


def _named(names: List[String], i: Int, kind: String) -> String:
    if i >= 0 and i < len(names):
        var n = names[i]
        if n.byte_length() > 0:
            return kind + " '" + n + "'"
    return kind + " " + String(i)


def _body_has_joint(fmd: FlatModelDef, b: Int) -> Bool:
    for j in fmd.joints:
        if j.body_id == b:
            return True
    return False


def _body_is_static(fmd: FlatModelDef, b: Int) -> Bool:
    """No joint anywhere between `b` and the world.

    MuJoCo's "plane only allowed in static bodies" reads the whole chain, not
    the body — verified: a plane in a jointless child of a hinged parent is
    refused.
    """
    var cur = b
    var guard = 0
    while cur > 0 and guard < 1024:
        if _body_has_joint(fmd, cur):
            return False
        cur = fmd.bodies[cur - 1].parent
        guard += 1
    return True


def _mass_ok(m: Model[DT, DynDims], b: Int) -> Bool:
    var o = b * MODEL_BODY_SIZE
    return (
        Float64(m.bodies.data[o + BODY_IDX_MASS]) >= MJMINVAL
        and Float64(m.bodies.data[o + BODY_IDX_IXX]) >= MJMINVAL
        and Float64(m.bodies.data[o + BODY_IDX_IYY]) >= MJMINVAL
        and Float64(m.bodies.data[o + BODY_IDX_IZZ]) >= MJMINVAL
    )


def _mass_ok_with_static_children(
    fmd: FlatModelDef, m: Model[DT, DynDims], b: Int, depth: Int
) -> Bool:
    """MuJoCo's `mjCModel::CheckBodyMassInertia`, transcribed.

    ⚠ THE RECURSION IS THE RULE, not a refinement of it. A body with a DOF and
    no mass of its own is LEGAL when a jointless child carries the mass — the
    attachment-frame pattern half of Menagerie is built from. Checking the
    body alone would flag `agility_cassie` and `jaco` on load, which is
    exactly the false alarm that gets a panel switched off. The walk stops at
    any child with a joint: that child is its own moving body and answers for
    itself.
    """
    if _mass_ok(m, b):
        return True
    if depth > 64:
        return False
    for c in range(1, len(fmd.bodies) + 1):
        if fmd.bodies[c - 1].parent != b:
            continue
        if _body_has_joint(fmd, c):
            continue
        if _mass_ok_with_static_children(fmd, m, c, depth + 1):
            return True
    return False


def _check_body_mass(
    fmd: FlatModelDef, m: Model[DT, DynDims], mut out: List[Diagnostic]
):
    """The one an edit reaches first: delete the geom that carried the mass."""
    for b in range(1, len(fmd.bodies) + 1):
        if not _body_has_joint(fmd, b):
            continue
        if _mass_ok_with_static_children(fmd, m, b, 0):
            continue
        out.append(
            Diagnostic(
                SEV_ERROR,
                String("zero-mass-moving-body"),
                _body_name(fmd, b),
                String(
                    "has a joint but no mass or inertia, and no jointless"
                    " child supplies one. MuJoCo refuses the model ('mass and"
                    " inertia of moving bodies must be larger than mjMINVAL')."
                    " Give it a geom, or an explicit <inertial>."
                ),
            )
        )


def _check_geom_sizes(fmd: FlatModelDef, mut out: List[Diagnostic]):
    """Per TYPE, and the plane is the one that is not like the others."""
    for gi in range(len(fmd.geoms)):
        ref g = fmd.geoms[gi]
        var t = g.geom_type
        var bad = String("")
        if t == 0:  # plane
            # ⚠ ONLY size[2]. `size="0 0 1"` is an INFINITE plane and MuJoCo
            # accepts it; flagging the first two would mark the floor of every
            # dm_control model.
            if not (g.half_z > 0.0):
                bad = String("size[2] (the render grid spacing)")
        elif t == 1:  # sphere
            if not (g.radius > 0.0):
                bad = String("size[0] (radius)")
        elif t == 2 or t == 4:  # capsule, cylinder
            if not (g.radius > 0.0):
                bad = String("size[0] (radius)")
            elif not (g.half_length > 0.0):
                bad = String("size[1] (half-length)")
        elif t == 3 or t == 6:  # box, ellipsoid
            if not (g.half_x > 0.0):
                bad = String("size[0]")
            elif not (g.half_y > 0.0):
                bad = String("size[1]")
            elif not (g.half_z > 0.0):
                bad = String("size[2]")
        # ⚠ A MESH GEOM TAKES ITS SHAPE FROM THE ASSET and carries no size at
        # all; checking one here would flag every mesh in Menagerie.
        if bad.byte_length() > 0:
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("nonpositive-geom-size"),
                    _named(fmd.geom_names, gi, String("geom")),
                    bad + " must be positive; MuJoCo refuses the model.",
                )
            )

        if g.condim != 1 and g.condim != 3 and g.condim != 4 and g.condim != 6:
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("invalid-condim"),
                    _named(fmd.geom_names, gi, String("geom")),
                    String("condim=") + String(g.condim)
                    + "; MuJoCo allows 1, 3, 4 or 6.",
                )
            )


def _check_joints(fmd: FlatModelDef, mut out: List[Diagnostic]):
    for ji in range(len(fmd.joints)):
        ref j = fmd.joints[ji]
        var subj = _named(fmd.joint_names, ji, String("joint"))

        # ⚠ LIMITED ONLY. `limited="false" range="1 -1"` loads — the range is
        # not read — and a marker on it would fire on models that work.
        if j.is_limited and not (j.range_min < j.range_max):
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("inverted-joint-range"),
                    subj,
                    String("range=") + String(j.range_min) + " "
                    + String(j.range_max)
                    + "; MuJoCo requires range[0] STRICTLY smaller than"
                    + " range[1] on a limited joint, so an equal pair is"
                    + " refused too.",
                )
            )

        # ⚠ A BALL JOINT'S AXIS IS UNUSED and may be zero — measured. Only
        # hinge (3) and slide (2) read it.
        #
        # ⚠⚠ THE TEST IS `not (n2 > eps)`, NOT `n2 < eps`, AND THAT IS THE
        # WHOLE CHECK. `_fill_joint` normalises every axis by dividing by its
        # length, so `axis="0 0 0"` arrives here as 0/0 = **NaN**, not as
        # zero — and every comparison against NaN is false, so `n2 < eps`
        # reports nothing. This gate caught exactly that: MuJoCo refuses the
        # model, we built one whose joint axis is NaN, and the validator said
        # it was clean. Negating the positive test is what makes NaN fail.
        if j.jnt_type == 2 or j.jnt_type == 3:
            var n2 = (
                j.axis_x * j.axis_x + j.axis_y * j.axis_y + j.axis_z * j.axis_z
            )
            if not (n2 > 1e-24):
                out.append(
                    Diagnostic(
                        SEV_ERROR,
                        String("zero-joint-axis"),
                        subj,
                        String(
                            "axis is zero-length or NaN; MuJoCo refuses the"
                            " model ('axis too small in joint'). This parser"
                            " normalises by dividing by the length, so a"
                            " zero axis reaches the model as NaN rather than"
                            " as an error."
                        ),
                    )
                )

        if j.damping < 0.0 or j.armature < 0.0:
            out.append(
                Diagnostic(
                    SEV_WARN,
                    String("negative-dissipation"),
                    subj,
                    String(
                        "damping or armature is negative. MuJoCo accepts it —"
                        " this is a warning, not an error — but a negative"
                        " value ADDS energy and the integrator will diverge."
                    ),
                )
            )

    # ── per-body joint composition ────────────────────────────────────────
    for b in range(1, len(fmd.bodies) + 1):
        var nv = 0
        var has_free = False
        var rot_after_ball = False
        var seen_ball_or_free = False
        for j in fmd.joints:
            if j.body_id != b:
                continue
            nv += j.nv
            if j.jnt_type == 0:
                has_free = True
            if seen_ball_or_free and (j.jnt_type == 0 or j.jnt_type == 1
                                      or j.jnt_type == 3):
                rot_after_ball = True
            if j.jnt_type == 0 or j.jnt_type == 1:
                seen_ball_or_free = True

        if has_free and fmd.bodies[b - 1].parent != 0:
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("free-joint-nested"),
                    _body_name(fmd, b),
                    String(
                        "has a free joint but is not a child of the world."
                        " MuJoCo refuses it ('free joint can only be used on"
                        " top level'); re-parent the body to the world, or"
                        " make the joint a slide/hinge chain."
                    ),
                )
            )
        if nv > 6:
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("body-too-many-dofs"),
                    _body_name(fmd, b),
                    String("carries ") + String(nv)
                    + " dofs; MuJoCo allows at most 6 per body.",
                )
            )
        elif rot_after_ball:
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("rotation-after-ball"),
                    _body_name(fmd, b),
                    String(
                        "has a rotational joint after a ball or free joint."
                        " MuJoCo refuses it ('ball followed by rotation');"
                        " the ball must come last."
                    ),
                )
            )


def _check_actuators(fmd: FlatModelDef, mut out: List[Diagnostic]):
    for ai in range(len(fmd.actuators)):
        ref a = fmd.actuators[ai]
        var subj = _named(fmd.actuator_names, ai, String("actuator"))

        if a.joint_id < 0 and a.tendon_id < 0:
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("actuator-no-transmission"),
                    subj,
                    String(
                        "drives neither a joint nor a tendon. MuJoCo refuses"
                        " it ('missing transmission target'); this engine"
                        " would give it zero force, which is worse."
                    ),
                )
            )

        # ⚠ MuJoCo REFUSES an equal pair here, `ctrlrange="0 0"` included —
        # that spelling is the "undefined" marker some models use and it does
        # NOT mean unlimited.
        if a.is_ctrl_limited and not (a.ctrl_min < a.ctrl_max):
            out.append(
                Diagnostic(
                    SEV_ERROR,
                    String("invalid-ctrlrange"),
                    subj,
                    String("ctrlrange=") + String(a.ctrl_min) + " "
                    + String(a.ctrl_max)
                    + "; MuJoCo requires the first STRICTLY smaller than the"
                    + " second when ctrllimited is set.",
                )
            )

        if a.gear == 0.0:
            out.append(
                Diagnostic(
                    SEV_WARN,
                    String("zero-gear"),
                    subj,
                    String(
                        "has gear=0 and therefore applies no force. MuJoCo"
                        " loads it; nothing downstream will say so."
                    ),
                )
            )


def _check_placement(fmd: FlatModelDef, mut out: List[Diagnostic]):
    for gi in range(len(fmd.geoms)):
        ref g = fmd.geoms[gi]
        if g.geom_type != 0:
            continue
        if g.body_id == 0:
            continue
        if _body_is_static(fmd, g.body_id):
            continue
        out.append(
            Diagnostic(
                SEV_ERROR,
                String("plane-in-moving-body"),
                _named(fmd.geom_names, gi, String("geom")),
                String("is a plane inside ") + _body_name(fmd, g.body_id)
                + ", which is not static. MuJoCo refuses it ('plane only"
                + " allowed in static bodies').",
            )
        )

    for b in range(1, len(fmd.bodies) + 1):
        if not fmd.bodies[b - 1].is_mocap:
            continue
        if fmd.bodies[b - 1].parent == 0 and not _body_has_joint(fmd, b):
            continue
        out.append(
            Diagnostic(
                SEV_ERROR,
                String("mocap-not-world-child"),
                _body_name(fmd, b),
                String(
                    "is a mocap body, so MuJoCo requires it to be a JOINTLESS"
                    " direct child of the world."
                ),
            )
        )


def _check_duplicate_names(fmd: FlatModelDef, mut out: List[Diagnostic]):
    """Repeated names, ONE KIND AT A TIME.

    ⚠⚠ PER KIND, AND THAT IS NOT A DETAIL. MuJoCo scopes names by element
    type: `<body name="x">` and `<geom name="x">` in the same file is legal —
    measured. Checking one global set would flag working models, and the
    common editor accident (duplicate an instance without re-prefixing it) is
    a collision WITHIN a kind anyway.

    ⚠ AN UNNAMED ELEMENT IS "" AND IS NOT A DUPLICATE. Most geoms in this tree
    have no name at all.
    """
    _dupes_in(fmd.body_names, String("body"), out)
    _dupes_in(fmd.joint_names, String("joint"), out)
    _dupes_in(fmd.geom_names, String("geom"), out)
    _dupes_in(fmd.site_names, String("site"), out)
    _dupes_in(fmd.actuator_names, String("actuator"), out)


def _dupes_in(names: List[String], kind: String, mut out: List[Diagnostic]):
    for i in range(len(names)):
        if names[i].byte_length() == 0:
            continue
        var first = True
        for k in range(i):
            if names[k] == names[i]:
                first = False
        if first:
            continue
        out.append(
            Diagnostic(
                SEV_ERROR,
                String("duplicate-name"),
                kind + " '" + names[i] + "'",
                String(
                    "is used by more than one "
                ) + kind + "; MuJoCo refuses a repeated name within one"
                + " element kind (across kinds it is fine).",
            )
        )


def _check_soft(fmd: FlatModelDef, mut out: List[Diagnostic]):
    """Things MuJoCo accepts and a human probably did not mean.

    ⚠ EVERY ONE OF THESE WAS CHECKED AGAINST THE RUNTIME AND LOADS. They are
    warnings for that reason and no other.
    """
    for pi in range(len(fmd.pairs)):
        ref p = fmd.pairs[pi]
        if p.geom1 == p.geom2:
            out.append(
                Diagnostic(
                    SEV_WARN,
                    String("self-pair"),
                    String("<contact><pair> ") + String(pi),
                    String("pairs ")
                    + _named(fmd.geom_names, p.geom1, String("geom"))
                    + " with itself. It loads and can never produce a"
                    + " contact.",
                )
            )
    for xi in range(len(fmd.excludes)):
        ref x = fmd.excludes[xi]
        if x.body1 == x.body2:
            out.append(
                Diagnostic(
                    SEV_WARN,
                    String("self-exclude"),
                    String("<contact><exclude> ") + String(xi),
                    String("excludes ") + _body_name(fmd, x.body1)
                    + " against itself, which was already excluded.",
                )
            )
