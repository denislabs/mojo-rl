"""Compile-both-sides mjModel diff — the "layer 1" gate of the dm_control port.

The pattern, from `docs/DM_CONTROL_PORT.md`'s standing invariants:

    compile OUR ported XML with MuJoCo, compile the REFERENCE's XML with
    MuJoCo, and diff every table.

Both sides are MuJoCo, so our parser and our engine are not in the loop at all:
a mismatch isolates the XML TEXT. Layer 2 (our `fields.Model` against MuJoCo
compiled from that same string) is only a valid comparison BECAUSE layer 1
proves the string is the reference model.

⚠ THE TABLE LIST IS EXHAUSTIVE ON PURPOSE. quadruped ran for two days on
sensor, observation, reward and dynamics gates before a constant-by-constant
diff was written, and it found bug 44 on its first run — `<freejoint>`
inheriting `solimplimit` from an enclosing default class, one wrong cell in
`jnt_solimp`. Picking a subset of tables is exactly how that hid.

Extracted from `quadruped_ref.py` on 2026-08-03 when humanoid_CMU became the
second user; `docs/DM_CONTROL_PORT_PHASE2.md` has ~35 more ports queued behind
it, every one of which owes this gate.
"""

# Every mjModel table that describes the MODEL rather than a simulation state.
_TABLES = [
    "body_parentid", "body_rootid", "body_weldid", "body_jntnum", "body_jntadr",
    "body_dofnum", "body_dofadr", "body_geomnum", "body_geomadr",
    "body_pos", "body_quat", "body_ipos", "body_iquat",
    "body_mass", "body_inertia", "body_invweight0",
    "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_axis",
    "jnt_pos", "jnt_range", "jnt_limited", "jnt_solimp", "jnt_solref",
    "jnt_stiffness", "jnt_margin",
    "dof_bodyid", "dof_jntid", "dof_parentid", "dof_armature", "dof_damping",
    "dof_invweight0", "dof_M0", "dof_frictionloss",
    "qpos0", "qpos_spring",
    "geom_type", "geom_bodyid", "geom_contype", "geom_conaffinity",
    "geom_condim", "geom_group", "geom_pos", "geom_quat", "geom_size",
    "geom_friction", "geom_solimp", "geom_solref", "geom_margin", "geom_gap",
    "geom_rbound",
    "site_type", "site_bodyid", "site_pos", "site_quat", "site_size",
    "actuator_trntype", "actuator_trnid", "actuator_dyntype",
    "actuator_gaintype", "actuator_biastype", "actuator_dynprm",
    "actuator_gainprm", "actuator_biasprm", "actuator_gear",
    "actuator_ctrllimited", "actuator_ctrlrange", "actuator_forcerange",
    "actuator_actadr", "actuator_actnum",
    "tendon_adr", "tendon_num", "tendon_limited", "tendon_range",
    "tendon_stiffness", "tendon_damping", "tendon_invweight0",
    "tendon_length0", "tendon_lengthspring",
    "tendon_solimp_lim", "tendon_solref_lim",
    "wrap_type", "wrap_objid", "wrap_prm",
    "eq_type", "eq_obj1id", "eq_obj2id", "eq_active0", "eq_solimp",
    "eq_solref", "eq_data",
    "sensor_type", "sensor_objid", "sensor_adr", "sensor_dim",
    "exclude_signature",
]

_COUNTS = ["nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nsite",
           "ntendon", "neq", "nsensor", "nmocap", "nexclude"]

_OPTS = ["timestep", "cone", "jacobian", "solver", "iterations", "integrator",
         "impratio", "tolerance"]

_ORDERED = ["mjOBJ_BODY", "mjOBJ_JOINT", "mjOBJ_GEOM", "mjOBJ_SITE",
            "mjOBJ_TENDON", "mjOBJ_ACTUATOR", "mjOBJ_SENSOR"]

_ORDER_COUNTS = {"mjOBJ_BODY": "nbody", "mjOBJ_JOINT": "njnt",
                 "mjOBJ_GEOM": "ngeom", "mjOBJ_SITE": "nsite",
                 "mjOBJ_TENDON": "ntendon", "mjOBJ_ACTUATOR": "nu",
                 "mjOBJ_SENSOR": "nsensor"}


def n_tables():
    """How many mjModel TABLES `diff_models` sweeps.

    Asserted on the Mojo side so that deleting entries from `_TABLES` to make a
    failure go away shows up as a failure of its own.
    """
    return len(_TABLES)


def n_checks():
    """Total comparisons: tables + counts + `<option>` fields + gravity."""
    return len(_TABLES) + len(_COUNTS) + len(_OPTS) + 1


def diff_models(ref, got, skip_tables=()):
    """Diff two compiled `mjModel`s. Returns human-readable mismatch strings.

    Empty means the two XMLs compile to the same model EXACTLY — the tolerance
    is 0.0, not an epsilon, because both sides ran the same compiler on the same
    numbers.

    Element ORDER is compared too, by name. A table comparison indexed by id
    says nothing if the ids name different things, and MuJoCo's geom ordering
    (by body id) is known to differ from our parser's (XML text order), so this
    is not a theoretical concern.

    `skip_tables` exists for a DELIBERATE, DOCUMENTED deviation only — pass the
    table names and say in the caller why. It is not a way to make a red gate
    green.
    """
    import numpy as np
    import mujoco

    bad = []

    for n in _COUNTS:
        a, b = getattr(ref, n), getattr(got, n)
        if a != b:
            bad.append(f"{n}: ref {a} != ours {b}")
    for n in _OPTS:
        a, b = getattr(ref.opt, n), getattr(got.opt, n)
        if a != b:
            bad.append(f"opt.{n}: ref {a} != ours {b}")
    if not np.array_equal(ref.opt.gravity, got.opt.gravity):
        bad.append(
            f"opt.gravity: ref {ref.opt.gravity} != ours {got.opt.gravity}")

    for n in _TABLES:
        if n in skip_tables:
            continue
        if not hasattr(ref, n) or not hasattr(got, n):
            continue  # table absent in this MuJoCo version
        a = np.asarray(getattr(ref, n), dtype=np.float64)
        b = np.asarray(getattr(got, n), dtype=np.float64)
        if a.shape != b.shape:
            bad.append(f"{n}: shape {a.shape} != {b.shape}")
            continue
        if a.size == 0:
            continue
        d = np.abs(a - b)
        if d.max() > 0.0:
            i = np.unravel_index(int(np.argmax(d)), d.shape)
            bad.append(f"{n}{list(i)}: ref {a[i]!r} != ours {b[i]!r}")

    for name in _ORDERED:
        objtype = getattr(mujoco.mjtObj, name)
        count = _ORDER_COUNTS[name]
        for i in range(min(getattr(ref, count), getattr(got, count))):
            x = mujoco.mj_id2name(ref, objtype, i)
            y = mujoco.mj_id2name(got, objtype, i)
            if x != y:
                bad.append(f"{name} order at {i}: ref {x!r} != ours {y!r}")

    return bad
