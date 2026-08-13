"""Reference model builder for the 13 dm_control `manipulation` *_features tasks.

## Why this one is different from every other `*_ref.py` here

The suite refs (`quadruped_ref.py`, `stacker_ref.py`, ...) all work the same
way: dm_control is NOT importable in this environment, so `make_model()` is
COPIED out of the suite module with two mechanical substitutions (lxml.etree ->
stdlib etree, and the asset loader inlined). That works because a suite model is
a static XML file plus a little Python surgery.

**Manipulation cannot be copied that way.** Its models come out of `composer`:
`arena.attach(arm)`, `arm.attach(hand)`, `arena.add_free_entity(prop)` — a full
PyMJCF attach/namespace/merge engine (that is where every `jaco_arm/` prefix in
the baked XML comes from). Re-implementing PyMJCF to produce a reference would
mean the reference and the thing under test share an author, which is not a
reference at all.

So this module **runs the real dm_control** instead, out of the checked-out
source tree at `references/dm_control-main/`, and bakes the assembled model to
a flattened MJCF + STL assets. `bake()` is §5.2's "composer bake pipeline".

## Bootstrap — what `_bootstrap()` does and why it is needed

Three things stand between `references/dm_control-main/` and an import:

1. **Missing Python deps** — `lxml` (PyMJCF), `dm_env`/`dm_tree` (composer),
   `pyparsing`, `protobuf`, `scipy`. Installed on demand into
   `references/.dmc_deps/` (gitignored with the rest of `references/`).
   `labmaze` is deliberately NOT installed: it has no cp313 wheel and builds
   with bazel, and nothing outside the Phase 11 maze tasks imports it.

2. **`mjbindings` is generated code.** `dm_control/mujoco/wrapper/mjbindings/`
   ships only `__init__.py` + `functions.py` in the source tree; `constants.py`,
   `enums.py` and `sizes.py` are produced at install time by
   `dm_control/autowrap/autowrap.py` parsing the MuJoCo headers. Without them
   the import dies on a confusing circular-import error. `_bootstrap()` runs
   autowrap once and caches the result in the (gitignored) reference tree.

3. ⚠ **Which MuJoCo it is generated against.** dm_control's `setup.py` requires
   `mujoco >= 3.11.0`, so `pip install dm_control` would drag the runtime up
   from 3.10.0 — silently re-baselining every existing dm_control gate in this
   repo, all of which were measured against 3.10.0. Running autowrap against
   `mujoco.HEADERS_DIR` instead pins the reference to **the exact MuJoCo the
   rest of the port is gated against**. Do not replace this with a pip install
   of dm_control. See `feedback_reference_tree_version_drift`.

## Usage

    import manipulation_ref
    manipulation_ref.bake('reach_site_features', out_dir)   # -> model.xml + STLs
    m = manipulation_ref.model('reach_site_features')       # -> mujoco.MjModel
    manipulation_ref.compare_xml_to_reference(our_xml, 'reach_site_features')

`ALL_FEATURES` lists the 13 task names in registry order.
"""

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_DMC_TREE = os.path.join(_REPO, 'references', 'dm_control-main')
_DEPS_DIR = os.path.join(_REPO, 'references', '.dmc_deps')
_MJBINDINGS = os.path.join(
    _DMC_TREE, 'dm_control', 'mujoco', 'wrapper', 'mjbindings')

# The 13 `_features` tasks, in `manipulation.ALL` order. The 12 `_vision`
# variants are the SAME tasks with a camera observation and are Phase 12.
ALL_FEATURES = (
    'stack_2_bricks_features',
    'stack_2_bricks_moveable_base_features',
    'stack_3_bricks_features',
    'stack_3_bricks_random_order_features',
    'stack_2_of_3_bricks_random_order_features',
    'reassemble_3_bricks_fixed_order_features',
    'reassemble_5_bricks_random_order_features',
    'lift_brick_features',
    'lift_large_box_features',
    'place_brick_features',
    'place_cradle_features',
    'reach_duplo_features',
    'reach_site_features',
)

# Deps dm_control needs that this environment does not ship. Pinned loosely:
# these only build the reference model, they are not in any measured path.
_PIP_DEPS = ('lxml', 'dm-env', 'dm-tree', 'pyparsing', 'protobuf', 'scipy')

# `setup.py::HEADER_FILENAMES` — listed explicitly there (and here) because
# HEADERS_DIR also holds unrelated headers.
_HEADER_FILENAMES = (
    'mjdata.h', 'mjmodel.h', 'mjrender.h', 'mjtype.h',
    'mjui.h', 'mjvisualize.h', 'mjxmacro.h', 'mujoco.h',
)

_bootstrapped = False


def _install_deps():
    """Install the missing deps into `_DEPS_DIR` with `uv`.

    ⚠ `--no-deps` is deliberate. Without it the resolver pulls its own numpy
    into `_DEPS_DIR`, which then SHADOWS the environment's numpy — and mujoco's
    compiled bindings are built against that one, so the shadow shows up as an
    ABI error somewhere far away rather than as an install problem.
    """
    missing = []
    for mod, pkg in (('lxml.etree', 'lxml'), ('dm_env', 'dm-env'),
                     ('tree', 'dm-tree'), ('pyparsing', 'pyparsing'),
                     ('google.protobuf', 'protobuf'), ('scipy', 'scipy')):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    cmd = ['uv', 'pip', 'install', '--quiet',
           '--python', sys.executable, '--target', _DEPS_DIR, '--no-deps']
    cmd.extend(_PIP_DEPS)
    try:
        subprocess.check_call(cmd)
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            'manipulation_ref needs {} for dm_control\'s composer, and the '
            'automatic install failed ({}). Install them by hand with:\n    {}'
            .format(', '.join(missing), e, ' '.join(cmd)))


def _generate_mjbindings():
    """Run `autowrap.py` against the RUNTIME MuJoCo's headers.

    Writes `constants.py` / `enums.py` / `sizes.py` into the reference tree,
    which is gitignored in its entirety, so this leaves no tracked footprint.
    """
    if os.path.exists(os.path.join(_MJBINDINGS, 'constants.py')):
        return
    import mujoco
    headers = []
    for fn in _HEADER_FILENAMES:
        p = os.path.join(mujoco.HEADERS_DIR, fn)
        if not os.path.exists(p):
            raise RuntimeError(
                'MuJoCo header {!r} not found — cannot generate dm_control\'s '
                'mjbindings.'.format(p))
        headers.append(p)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [_DEPS_DIR, _DMC_TREE] + ([env['PYTHONPATH']] if 'PYTHONPATH' in env else []))
    subprocess.check_call(
        [sys.executable,
         os.path.join(_DMC_TREE, 'dm_control', 'autowrap', 'autowrap.py'),
         '--header_paths={}'.format(' '.join(headers)),
         '--output_dir={}'.format(_MJBINDINGS)],
        env=env)


def _bootstrap():
    """Make `from dm_control import manipulation` work. Idempotent."""
    global _bootstrapped
    if _bootstrapped:
        return
    for p in (_DEPS_DIR, _DMC_TREE):
        if p not in sys.path:
            sys.path.insert(0, p)
    _install_deps()
    # ⚠ `_DEPS_DIR` was put on `sys.path` before it existed (or while empty),
    # and the path finder caches a directory's contents keyed on its mtime.
    # Without this the freshly-installed `lxml` is invisible for the rest of
    # the process and the import fails as if nothing had been installed.
    import importlib
    importlib.invalidate_caches()
    _generate_mjbindings()
    _bootstrapped = True


_env_cache = {}


def _load(task_name, seed=0):
    """`manipulation.load(task_name)` — a composer `Environment`.

    `seed` is fixed by default: several tasks randomize prop placement in
    `initialize_episode`, and a reference that moves between runs is not one.

    ⚠ The result is CACHED because it must outlive the caller. `Physics` holds
    its `MjModel` through a weakref-backed binding, so an `Environment` that
    goes out of scope takes the model with it and the next attribute access
    raises `ReferenceError` — from the line that reads the model, not from the
    line that dropped the env.

    ⚠⚠ A FRESHLY CONSTRUCTED ENVIRONMENT DOES NOT HOLD THE TASK'S MODEL.
    `composer.Environment.__init__` compiles the MJCF as authored;
    `initialize_episode_mjcf` — which is allowed to EDIT the MJCF and is
    followed by a recompile — runs from `reset()`. So the model an episode
    actually uses only exists after one reset, and the two differ for every
    task with a Duplo:

        `props.Duplo.initialize_episode_mjcf` draws the stud radius, and the
        `stud` default class goes 0.0047 -> 0.004647 (`variation=0.0` makes the
        draw deterministic, so it is always exactly that). Every stud cylinder
        in the brick shrinks by 1.1%.

    That is a MODEL constant read by a comptime port, so baking the
    pre-reset tree would ship 16 geoms sized to a value no episode ever runs
    with. One reset here, once per cached env, and every consumer —
    `bake`, `model`, `counts`, the `*_state` helpers — sees the same model.

    ⚠ This is deliberately UNCONDITIONAL rather than duplo-only. It was
    verified to be a no-op where nothing edits the MJCF: `reach_site_features`
    and `lift_large_box_features` export byte-identical XML (15695 and 18078
    bytes) with and without it, so the two committed XML modules and every
    gate measured against them are unaffected.
    """
    _bootstrap()
    if task_name not in ALL_FEATURES:
        raise ValueError(
            '{!r} is not one of the 13 _features tasks: {}'
            .format(task_name, ', '.join(ALL_FEATURES)))
    key = (task_name, seed)
    if key not in _env_cache:
        from dm_control import manipulation
        env = manipulation.load(task_name, seed=seed)
        env.reset()
        _env_cache[key] = env
    return _env_cache[key]


def bake(task_name, out_dir, seed=0):
    """Assemble `task_name` with composer and write a flattened MJCF + assets.

    Returns the path to the written `model.xml`. Assets (STL meshes) land
    beside it under content-hashed filenames, which is PyMJCF's own naming —
    the same mesh referenced from two entities is written once.
    """
    _bootstrap()
    from dm_control import mjcf
    env = _load(task_name, seed=seed)
    root = env.task.root_entity.mjcf_model
    os.makedirs(out_dir, exist_ok=True)
    mjcf.export_with_assets(root, out_dir, out_file_name='model.xml')
    return os.path.join(out_dir, 'model.xml')


def xml_string(task_name, seed=0):
    """The flattened MJCF as a string, without touching the filesystem.

    ⚠ Mesh `file=` attributes in this string point at assets that only exist
    once `bake()` has written them, so this is for TEXT inspection. Use
    `model()` or `bake()` when MuJoCo has to compile it.
    """
    _bootstrap()
    env = _load(task_name, seed=seed)
    return env.task.root_entity.mjcf_model.to_xml_string()


def model(task_name, seed=0):
    """The reference `mujoco.MjModel`, compiled by dm_control itself."""
    return _load(task_name, seed=seed).physics.model.ptr


def physics(task_name, seed=0):
    """The reference `Physics` after `reset()` — model AND a settled state.

    Reset runs the task's `initialize_episode`, which for most of these tasks
    means the IK-based TCP initializer and `PropPlacer(settle_physics=True)`.
    That is state, not model, so keep it out of model-constant gates.
    """
    env = _load(task_name, seed=seed)
    env.reset()
    return env.physics


def compare_xml_to_reference(xml_string_, task_name, seed=0):
    """Layer-1 gate: compile OUR XML with MuJoCo, diff against the reference.

    Both sides are MuJoCo, so a mismatch isolates the XML text rather than our
    parser or engine. Returns `mjmodel_diff.diff_models`' report.
    """
    import mujoco
    sys.path.insert(0, _HERE)
    from mjmodel_diff import diff_models
    ours = mujoco.MjModel.from_xml_string(xml_string_)
    return diff_models(ours, model(task_name, seed=seed))


def n_tables_compared():
    sys.path.insert(0, _HERE)
    from mjmodel_diff import n_tables
    return n_tables()


def counts(task_name, seed=0):
    """`{name: value}` for the model counts, for a cheap first-look gate."""
    m = model(task_name, seed=seed)
    return {k: int(getattr(m, k)) for k in (
        'nq', 'nv', 'nu', 'na', 'nbody', 'njnt', 'ngeom', 'nsite', 'nmesh',
        'ntendon', 'neq', 'nsensor', 'nmocap', 'nexclude')}


if __name__ == '__main__':
    # `pixi run python tests/dm_control/manipulation_ref.py [task ...]`
    import warnings
    warnings.filterwarnings('ignore')
    names = sys.argv[1:] or list(ALL_FEATURES)
    for n in names:
        print('{:46s} {}'.format(n, counts(n)))


# -- inverse kinematics -----------------------------------------------------
#
# The reset path's TCP initializer runs dm_control's OWN damped-least-squares
# site IK. These helpers expose it so a Mojo gate can compare against the real
# `qpos_from_site_pose` rather than a transcription of it.

# `entities/manipulators/base.py::DOWN_QUATERNION`, MuJoCo order (w, x, y, z).
DOWN_QUATERNION = (0.0, 0.70710678118, 0.70710678118, 0.0)

# The site the TCP initializer drives, and the joints it is allowed to move.
# ⚠ `set_site_to_xpos` passes `joint_names=arm_joint_names`, restricting the
# solve to the ARM — the hand's finger joints are held. That restriction is
# what keeps the normal matrix full rank; see `physics3d/dynamics/ik_site.mojo`.
TCP_SITE = 'jaco_arm/jaco_hand/pinchsite'


def arm_joint_names(task_name='reach_site_features', seed=0):
    """Names of the arm joints, in model order, excluding the hand's."""
    env = _load(task_name, seed=seed)
    arm = env.task._arm  # pylint: disable=protected-access
    return [j.full_identifier for j in arm.joints]


def ik_reference(task_name, q0, target_pos, target_quat=None,
                 rot_weight=2.0, max_steps=100, seed=0):
    """Run dm_control's `qpos_from_site_pose` from `q0`.

    Arguments mirror `set_site_to_xpos`'s call, NOT the IK function's own
    defaults — in particular `rot_weight=2` and the arm-only joint set.

    Returns `(qpos, err_norm, steps, success, site_xpos)`; `qpos` is the FULL
    nq vector. ⚠ `success` is not implied by `qpos` looking reasonable: the
    progress guard breaks out with `success=False` while leaving `qpos` at the
    last accepted step.
    """
    _bootstrap()
    import numpy as np
    from dm_control.utils import inverse_kinematics
    env = _load(task_name, seed=seed)
    physics_ = env.physics
    with physics_.reset_context():
        physics_.data.qpos[:] = np.asarray(q0, dtype=float)
    if target_quat is None:
        target_quat = DOWN_QUATERNION
    result = inverse_kinematics.qpos_from_site_pose(
        physics=physics_,
        site_name=TCP_SITE,
        target_pos=np.asarray(target_pos, dtype=float),
        target_quat=np.asarray(target_quat, dtype=float),
        joint_names=arm_joint_names(task_name, seed=seed),
        rot_weight=rot_weight,
        max_steps=max_steps,
        inplace=True)
    physics_.forward()
    sid = physics_.model.name2id(TCP_SITE, 'site')
    return (list(result.qpos), float(result.err_norm), int(result.steps),
            bool(result.success), list(physics_.data.site_xpos[sid]))


def arm_joint_bounds(task_name='reach_site_features', seed=0):
    """`_get_joint_pos_sampling_bounds` — (lower, upper) per arm joint.

    ⚠ Unlimited HINGES get `[0, 2*pi]`, not their (absent) range; non-hinge
    joints without limits are a hard error in the reference.
    """
    env = _load(task_name, seed=seed)
    arm = env.task._arm  # pylint: disable=protected-access
    # pylint: disable=protected-access
    lower, upper = arm._get_joint_pos_sampling_bounds(env.physics)
    return list(lower), list(upper)


def arm_qpos_adr(task_name='reach_site_features', seed=0):
    """`jnt_qposadr` for each arm joint, in the same order as the bounds."""
    import mujoco
    env = _load(task_name, seed=seed)
    mm = env.physics.model.ptr
    out = []
    for n in arm_joint_names(task_name, seed=seed):
        out.append(int(mm.jnt_qposadr[
            mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, n)]))
    return out


def retry_pose_draws(n_draws, rng_seed, task_name='reach_site_features',
                     seed=0):
    """The exact poses `randomize_arm_joints` would draw, flattened.

    `set_site_to_xpos` re-randomises with `random_state.uniform(lower, upper)`
    once per FAILED attempt. Reproducing that stream here — same
    `RandomState`, same seed, same call order — lets a Mojo port be driven
    down an identical trajectory instead of merely a statistically similar
    one.
    """
    import numpy as np
    lower, upper = arm_joint_bounds(task_name, seed=seed)
    rs = np.random.RandomState(rng_seed)
    out = []
    for _ in range(n_draws):
        out.extend(list(rs.uniform(lower, upper)))
    return out


def set_site_to_xpos_reference(task_name, q0, target_pos, rng_seed,
                               target_quat=None, max_ik_attempts=10, seed=0):
    """dm_control's own `set_site_to_xpos`, from `q0`.

    Returns `(success, qpos)`. The `RandomState` is seeded with `rng_seed`, so
    `retry_pose_draws(k, rng_seed)` yields the poses it will use.
    """
    _bootstrap()
    import numpy as np
    env = _load(task_name, seed=seed)
    physics_ = env.physics
    arm = env.task._arm  # pylint: disable=protected-access
    with physics_.reset_context():
        physics_.data.qpos[:] = np.asarray(q0, dtype=float)
    if target_quat is None:
        target_quat = DOWN_QUATERNION
    ok = arm.set_site_to_xpos(
        physics=physics_,
        random_state=np.random.RandomState(rng_seed),
        site=TCP_SITE,
        target_pos=np.asarray(target_pos, dtype=float),
        target_quat=np.asarray(target_quat, dtype=float),
        max_ik_attempts=max_ik_attempts)
    return bool(ok), list(physics_.data.qpos)


def hand_joint_info(task_name='reach_site_features', seed=0):
    """(names, qpos addresses, lower, upper) for the HAND's finger joints."""
    import mujoco
    env = _load(task_name, seed=seed)
    hand = env.task._hand  # pylint: disable=protected-access
    mm = env.physics.model.ptr
    names, adr, lo, hi = [], [], [], []
    for j in hand.joints:
        n = j.full_identifier
        jid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, n)
        names.append(n)
        adr.append(int(mm.jnt_qposadr[jid]))
        lo.append(float(mm.jnt_range[jid][0]))
        hi.append(float(mm.jnt_range[jid][1]))
    return names, adr, lo, hi


def set_grasp_reference(close_factor, task_name='reach_site_features',
                        seed=0):
    """`hand.set_grasp(physics, close_factors=close_factor)` — returns qpos.

    ⚠ `reach` passes a SCALAR, which the hand broadcasts to every finger.
    """
    _bootstrap()
    env = _load(task_name, seed=seed)
    physics_ = env.physics
    env.task._hand.set_grasp(  # pylint: disable=protected-access
        physics_, close_factors=close_factor)
    return list(physics_.data.qpos)


def reach_workspace(task_name='reach_site_features', seed=0):
    """(tcp_bbox, target_bbox) as ((lo3), (hi3)) pairs, from reach.py.

    ⚠ For `reach_site_features` these two boxes are IDENTICAL, so a gate on
    this task cannot detect them being swapped. `reach_duplo` differs.
    """
    _bootstrap()
    from dm_control.manipulation import reach
    ws = reach._SITE_WORKSPACE  # pylint: disable=protected-access
    return ((tuple(ws.tcp_bbox.lower), tuple(ws.tcp_bbox.upper)),
            (tuple(ws.target_bbox.lower), tuple(ws.target_bbox.upper)))


def target_placer_reference(rng_seed, task_name='reach_site_features',
                            seed=0):
    """What `reach`'s `_target_placer` yields, plus the raw uniforms.

    Returns `(pos, u)` where `pos` is the sampled target position and `u` are
    the three [0, 1) draws that reproduce it as
    `lower + (upper - lower) * u`. A Mojo port fed `u` must land on `pos`.
    """
    _bootstrap()
    import numpy as np
    from dm_control.composer.variation import distributions
    from dm_control.manipulation import reach
    ws = reach._SITE_WORKSPACE  # pylint: disable=protected-access
    dist = distributions.Uniform(*ws.target_bbox)
    pos = dist(random_state=np.random.RandomState(rng_seed))
    u = np.random.RandomState(rng_seed).random_sample(3)
    return list(np.atleast_1d(pos)), list(u)


# -- collision rejection ----------------------------------------------------
#
# `ToolCenterPointInitializer._has_relevant_collisions` classifies a GEOM by
# which entity model owns it. A baked MJCF is flat and our parser keeps no
# body names, so the Mojo port takes the classification as input; these
# helpers derive it here, with dm_control's own objects, so a gate tests the
# RULE rather than a second guess at the labelling.

# Must match `envs/dm_control/manipulation_reset.mojo`.
BODY_ARM, BODY_HAND, BODY_FREE, BODY_FIXED = 0, 1, 2, 3


def body_classes_reference(task_name='reach_site_features', seed=0):
    """Per-BODY class array, indexed by MuJoCo body id.

    The reference works per geom (`geom.root is arm_model`); this maps that
    onto bodies, which is what our contact records carry. ⚠ That is only
    equivalent if no body owns geoms from two different entity roots — asserted
    below rather than assumed, because the whole predicate silently changes
    meaning if it is ever false.

    ⚠ GEOMLESS BODIES KEEP THE `BODY_FIXED` DEFAULT, and two of Jaco's do:
    `jaco_arm/` (body 1) and `jaco_arm/jaco_hand/` (body 9) are the entity
    ATTACHMENT FRAMES and own no geoms at all. Labelling them "external" is
    wrong in spirit and harmless in fact — a body with no geoms cannot appear
    in a contact, so the entry is never read. `test_tcp_initializer_vs_dm_control`
    asserts exactly that rather than leaving it to luck.
    """
    _bootstrap()
    from dm_control import mjcf
    env = _load(task_name, seed=seed)
    physics = env.physics
    m = physics.model.ptr
    task = env.task
    arm_model = task._arm.mjcf_model          # pylint: disable=protected-access
    hand_model = task._hand.mjcf_model        # pylint: disable=protected-access
    mjcf_root = arm_model.root_model

    free_body_geoms = set()
    for body in mjcf_root.worldbody.get_children('body'):
        if mjcf.get_freejoint(body):
            free_body_geoms.update(body.find_all('geom'))

    def geom_class(g):
        if g.root is arm_model:
            return BODY_ARM
        if g.root is hand_model:
            return BODY_HAND
        if g in free_body_geoms:
            return BODY_FREE
        return BODY_FIXED

    all_geoms = mjcf_root.find_all('geom')
    # World starts FIXED: it owns the arena plane and has no freejoint, which
    # is exactly what makes arm-versus-ground a relevant collision.
    classes = [BODY_FIXED] * m.nbody
    seen = {}
    for gid, g in enumerate(all_geoms):
        b = int(m.geom_bodyid[gid])
        c = geom_class(g)
        if b in seen and seen[b] != c:
            raise AssertionError(
                'body %d owns geoms of two entity classes (%d and %d) — the '
                'body-level port of _has_relevant_collisions is invalid'
                % (b, seen[b], c))
        seen[b] = c
        classes[b] = c
    return classes


def has_relevant_collisions_at(qpos, task_name='reach_site_features', seed=0):
    """dm_control's predicate at a given `qpos`. Returns `(verdict, ncon)`."""
    _bootstrap()
    import numpy as np
    import mujoco
    from dm_control import mjcf
    env = _load(task_name, seed=seed)
    physics = env.physics
    m, d = physics.model.ptr, physics.data.ptr
    task = env.task
    arm_model = task._arm.mjcf_model          # pylint: disable=protected-access
    hand_model = task._hand.mjcf_model        # pylint: disable=protected-access
    mjcf_root = arm_model.root_model
    all_geoms = mjcf_root.find_all('geom')

    free_body_geoms = set()
    for body in mjcf_root.worldbody.get_children('body'):
        if mjcf.get_freejoint(body):
            free_body_geoms.update(body.find_all('geom'))

    def is_robot(g):
        return g.root is arm_model or g.root is hand_model

    def is_external_fixed(g):
        return not (is_robot(g) or g in free_body_geoms)

    d.qpos[:] = np.asarray(qpos, dtype=float)
    mujoco.mj_forward(m, d)

    for k in range(d.ncon):
        con = d.contact[k]
        g1, g2 = all_geoms[con.geom1], all_geoms[con.geom2]
        if con.dist > 0:
            continue
        if ((g1.root is arm_model and g2.root is arm_model) or
                (g1.root is arm_model and g2.root is hand_model) or
                (g1.root is hand_model and g2.root is arm_model) or
                (is_robot(g1) and is_external_fixed(g2)) or
                (is_external_fixed(g1) and is_robot(g2))):
            return True, int(d.ncon)
    return False, int(d.ncon)


def bodies_without_geoms(task_name='reach_site_features', seed=0):
    """Body ids owning no geoms — the entries of `body_classes_reference`
    that keep the default label because nothing ever reads them."""
    _bootstrap()
    import numpy as np
    m = model(task_name, seed=seed)
    return [b for b in range(m.nbody) if not (m.geom_bodyid == b).any()]


def action_spec_reference(task_name='reach_site_features', seed=0):
    """dm_control's `action_spec()` bounds for a MANIPULATION task.

    Returns `(minimum, maximum)` as plain lists, one entry per actuator. This
    is the spec a policy is handed — per-actuator, not a scalar pair — and is
    what `Phyics3dEnv.action_low_at/action_high_at` must reproduce.
    """
    _bootstrap()
    import numpy as np
    env = _load(task_name, seed=seed)
    spec = env.action_spec()
    return (list(np.atleast_1d(spec.minimum).astype(float)),
            list(np.atleast_1d(spec.maximum).astype(float)))


def suite_action_spec_reference(domain, task):
    """The same, for a `dm_control.suite` task (quadruped, walker, ...).

    Separate from `action_spec_reference` because `manipulation` and `suite`
    are different registries with different loaders; `_load` only knows the
    manipulation one.
    """
    _bootstrap()
    import numpy as np
    from dm_control import suite
    env = suite.load(domain, task)
    spec = env.action_spec()
    return (list(np.atleast_1d(spec.minimum).astype(float)),
            list(np.atleast_1d(spec.maximum).astype(float)))


# -- the `reach_site_features` task layer -----------------------------------
#
# `Reach.get_reward` and the eight `_features` observables, evaluated at an
# INJECTED state so a Mojo gate can drive both engines from the same numbers.
# Everything here calls dm_control's own objects: the observables are the real
# `Observable` instances (so the FTT corruptor runs), and the reward is
# `task.get_reward`. Nothing is re-implemented.

# `observation_spec()` order, which is what a policy sees when the dict is
# flattened. NOT declaration order — the task's own observable comes first.
REACH_OBS_ORDER = (
    'target_position',
    'jaco_arm/joints_pos',
    'jaco_arm/joints_torque',
    'jaco_arm/joints_vel',
    'jaco_arm/jaco_hand/joints_pos',
    'jaco_arm/jaco_hand/joints_vel',
    'jaco_arm/jaco_hand/pinch_site_pos',
    'jaco_arm/jaco_hand/pinch_site_rmat',
)


def _reach_observables(task):
    """`{spec name: Observable}` for the enabled `_features` observables.

    ⚠ `entity.observables.as_dict()` keys are ALREADY the fully-qualified
    names composer puts in `observation_spec()` (`jaco_arm/joints_pos`, not
    `joints_pos`), because the entities are ATTACHED. Prefixing them again
    yields `jaco_arm/jaco_arm/joints_pos` and a KeyError far from here.
    """
    out = {'target_position': task.task_observables['target_position']}
    for entity in (task._arm, task._hand):
        for k, v in entity.observables.as_dict().items():
            if v.enabled:
                out[k] = v
    return out


def reach_state(qpos, qvel, ctrl=None, target_pos=None,
                task_name='reach_site_features', seed=0,
                zero_frictionloss=False):
    """Evaluate the task at an injected state. Returns a plain dict.

    ⚠ `target_pos` writes the SITE's model `pos`, which is where dm_control
    keeps it (`physics.bind(self._target).pos = ...`) — the target is a model
    constant that `initialize_episode` rewrites, not a body pose. Passing None
    leaves whatever the last reset drew, which is not reproducible across
    dm_control versions; a gate should always pass one.

    ⚠ `mj_forward` is what fills `sensordata`, so the returned
    `jaco_arm/joints_torque` is the ACCELERATION STAGE AT THIS STATE — not at
    the state some previous step left. A Mojo side comparing it must produce
    `cfrc_int` at the same state (one integrator substep FROM here), not after
    a full control step.
    """
    _bootstrap()
    import numpy as np
    env = _load(task_name, seed=seed)
    task, p = env.task, env.physics
    if target_pos is not None:
        p.bind(task._target).pos = np.asarray(target_pos, dtype=float)
    p.data.qpos[:] = np.asarray(qpos, dtype=float)
    p.data.qvel[:] = np.asarray(qvel, dtype=float)
    p.data.ctrl[:] = (0.0 if ctrl is None
                      else np.asarray(ctrl, dtype=float))
    # ⚠ `zero_frictionloss` REMOVES ALL NINE CONSTRAINT ROWS. Every dof of this
    # model carries `frictionloss` (2 / 1 / 0.1 by joint class), so `nefc` is 9
    # at every contact-free pose and the acceleration comes out of an iterative
    # solve. Zeroing it leaves an unconstrained forward dynamics, which is how
    # a gate can ask whether a residual belongs to the solver or to what reads
    # its output. Restored before returning — a leaked model edit would make
    # every later call in the process measure a different model.
    saved = None
    if zero_frictionloss:
        saved = np.array(p.model.ptr.dof_frictionloss)
        p.model.ptr.dof_frictionloss[:] = 0.0
    p.forward()
    rs = np.random.RandomState(0)
    obs = _reach_observables(task)
    # ⚠ EVERYTHING BELOW `flat` IS DIAGNOSTIC, and it is here because it paid
    # for itself: a 3.3e-07 disagreement in `joints_torque` was localised by
    # walking `cfrc_int` -> `cacc` -> `subtree_com` -> `site_xpos` -> `xquat`
    # -> raw `sensordata`, finding every one of them exact to 1e-15, and so
    # arriving at the arithmetic (`std.math.log1p`) rather than the physics.
    # Keep them; the next residual will want the same ladder.
    #
    # ⚠⚠ `qacc` IS A TRAP TO COMPARE NAIVELY. This is `mj_forward`'s, i.e. the
    # acceleration BEFORE integration. A Mojo side reading `d.qacc` after an
    # Euler substep holds a different quantity — `mj_Euler` treats
    # `dof_damping` implicitly — and the two differ by ~1.5% on this model with
    # nothing wrong. See `feedback_mj_forward_qacc_is_not_what_mj_step_
    # integrates`.
    out = {'reward': float(task.get_reward(p)),
           'ncon': int(p.data.ncon),
           'nefc': int(p.data.nefc),
           'qacc': [float(x) for x in p.data.qacc],
           'qfrc_constraint': [float(x) for x in p.data.qfrc_constraint],
           'cfrc_int': [float(x) for x in np.asarray(p.data.cfrc_int).ravel()],
           'cacc': [float(x) for x in np.asarray(p.data.cacc).ravel()],
           'subtree_com': [float(x) for x in
                           np.asarray(p.data.subtree_com).ravel()],
           'site_xpos': [float(x) for x in
                         np.asarray(p.data.site_xpos).ravel()],
           'sensordata': [float(x) for x in np.asarray(p.data.sensordata)],
           # ⚠ MuJoCo's xquat is (w, x, y, z); ours is (x, y, z, w).
           'xquat': [float(x) for x in np.asarray(p.data.xquat)[:, [1, 2, 3, 0]].ravel()],
           'flat': []}
    for name in REACH_OBS_ORDER:
        v = np.asarray(obs[name](p, rs), dtype=float).ravel()
        out[name] = list(v)
        out['flat'].extend(float(x) for x in v)
    if saved is not None:
        p.model.ptr.dof_frictionloss[:] = saved
    return out


def reach_indices(task_name='reach_site_features', seed=0):
    """The element ids `manipulation_reach_config` hardcodes, from MuJoCo.

    Returned so the gate can assert them rather than trust a comment: a model
    rebake that renumbers a site would otherwise leave the config reading the
    wrong element with every shape still correct.
    """
    _bootstrap()
    import mujoco
    import numpy as np
    m = model(task_name, seed=seed)
    task = _load(task_name, seed=seed).task
    p = _load(task_name, seed=seed).physics
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    return {
        'site_target': int(p.bind(task._target).element_id),
        'site_pinch': int(p.bind(task._hand.tool_center_point).element_id),
        'body_pinch': int(m.site_bodyid[
            p.bind(task._hand.tool_center_point).element_id]),
        # `<torque site=...>` sensors, in arm joint order, and the body each
        # site hangs off.
        'torque_sites': [int(m.sensor_objid[i]) for i in range(m.nsensor)],
        'torque_bodies': [int(m.site_bodyid[m.sensor_objid[i]])
                          for i in range(m.nsensor)],
        'sensor_types': [int(m.sensor_type[i]) for i in range(m.nsensor)],
        'arm_axes': [list(map(float, m.jnt_axis[j])) for j in range(6)],
        'hand_range': [list(map(float, m.jnt_range[j])) for j in range(6, 9)],
        'nq': int(m.nq), 'nv': int(m.nv), 'nsite': int(m.nsite),
    }


# -- the `lift_large_box_features` task layer --------------------------------

LIFT_OBS_ORDER = (
    'jaco_arm/joints_pos',
    'jaco_arm/joints_torque',
    'jaco_arm/joints_vel',
    'jaco_arm/jaco_hand/joints_pos',
    'jaco_arm/jaco_hand/joints_vel',
    'jaco_arm/jaco_hand/pinch_site_pos',
    'jaco_arm/jaco_hand/pinch_site_rmat',
    'unnamed_model/angular_velocity',
    'unnamed_model/linear_velocity',
    'unnamed_model/orientation',
    'unnamed_model/position',
)


def _lift_observables(task):
    """`{spec name: Observable}` — the arm, the hand and the prop.

    ⚠ `as_dict()` keys are already fully qualified (the entities are
    ATTACHED); prefixing them again yields `jaco_arm/jaco_arm/...`.
    """
    out = {}
    for entity in (task._arm, task._hand, task._prop):
        for k, v in entity.observables.as_dict().items():
            if v.enabled:
                out[k] = v
    return out


def lift_state(qpos, qvel, ctrl=None, target_height=None,
               task_name='lift_large_box_features', seed=0):
    """Evaluate `Lift` at an injected state. Returns a plain dict.

    ⚠ `target_height` IS EPISODE STATE, not a model constant: `Lift` computes
    it in `initialize_episode` from where the prop settled. A gate must pass
    the same value it gave the Mojo side, or the two rewards are answering
    different questions.

    ⚠ `mj_forward` fills `sensordata`, so `jaco_arm/joints_torque` here is the
    ACCELERATION STAGE AT THIS STATE — the Mojo side must produce `cfrc_int`
    at the same state (one substep FROM here), not after a control step.
    """
    _bootstrap()
    import numpy as np
    env = _load(task_name, seed=seed)
    task, p = env.task, env.physics
    p.data.qpos[:] = np.asarray(qpos, dtype=float)
    p.data.qvel[:] = np.asarray(qvel, dtype=float)
    p.data.ctrl[:] = 0.0 if ctrl is None else np.asarray(ctrl, dtype=float)
    p.forward()
    # ⚠ `_target_height` DOES NOT EXIST until `initialize_episode` has run —
    # it is episode state, not a task attribute. A caller that only wants the
    # observation must still give `get_reward` something to read, so an absent
    # one defaults to 0. A gate that cares about the reward passes its own.
    if target_height is not None:
        task._target_height = float(target_height)
    elif not hasattr(task, '_target_height'):
        task._target_height = 0.0
    rs = np.random.RandomState(0)
    obs = _lift_observables(task)
    out = {'reward': float(task.get_reward(p)),
           'ncon': int(p.data.ncon),
           'lowest_vertex_z': float(task._get_height_of_lowest_vertex(p)),
           'target_height': float(task._target_height),
           'flat': []}
    for name in LIFT_OBS_ORDER:
        v = np.asarray(obs[name](p, rs), dtype=float).ravel()
        out[name] = list(v)
        out['flat'].extend(float(x) for x in v)
    return out


def lift_indices(task_name='lift_large_box_features', seed=0):
    """The element ids `manipulation_lift_box_config` hardcodes."""
    _bootstrap()
    import mujoco
    m = model(task_name, seed=seed)
    env = _load(task_name, seed=seed)
    task, p = env.task, env.physics
    prop_geoms = task._prop.mjcf_model.find_all('geom')
    return {
        'prop_geom': int(p.bind(prop_geoms[0]).element_id),
        'prop_body': int(m.geom_bodyid[p.bind(prop_geoms[0]).element_id]),
        'vertex_sites': [int(p.bind(v).element_id) for v in task._prop.vertices],
        'target_height_site': int(p.bind(task._target_height_site).element_id),
        # The prop's free joint.
        'prop_qposadr': int(m.jnt_qposadr[m.njnt - 1]),
        'prop_dofadr': int(m.jnt_dofadr[m.njnt - 1]),
        'prop_jnt_type': int(m.jnt_type[m.njnt - 1]),
        'nbody': int(m.nbody), 'nq': int(m.nq), 'nv': int(m.nv),
    }


def lift_reset_qpos(n, task_name='lift_large_box_features', seed=0):
    """`n` real `initialize_episode` draws — qpos after dm_control's own reset.

    Used to check that OUR reset produces poses from the same region, and to
    report what the reference's settle actually does to the prop.
    """
    _bootstrap()
    import numpy as np
    env = _load(task_name, seed=seed)
    out = []
    for _ in range(n):
        env.reset()
        p = env.physics
        out.append({
            'qpos': [float(x) for x in p.data.qpos],
            'lowest_vertex_z': float(
                env.task._get_height_of_lowest_vertex(p)),
            'target_height': float(env.task._target_height),
            'ncon': int(p.data.ncon),
        })
    return out


# -- the `reach_duplo_features` task layer -----------------------------------
#
# `Reach` again, but with `prop=Duplo` instead of `use_site=True`, which
# changes three things and nothing else:
#
#   * the `target_position` task observable DISAPPEARS (the prop IS the
#     target), and the free-prop block appears — 55 numbers, not 45;
#   * `get_reward` measures to `physics.bind(self._target).xpos`, and
#     `self._target` is the ATTACHMENT FRAME `add_free_entity` returned, i.e.
#     the prop's body — not the invisible `target_site` bolted to it;
#   * `initialize_episode` runs `_prop_placer` in place of `_target_placer`,
#     which is a rejection loop plus a settle rather than one assignment.

REACH_DUPLO_OBS_ORDER = (
    'jaco_arm/joints_pos',
    'jaco_arm/joints_torque',
    'jaco_arm/joints_vel',
    'jaco_arm/jaco_hand/joints_pos',
    'jaco_arm/jaco_hand/joints_vel',
    'jaco_arm/jaco_hand/pinch_site_pos',
    'jaco_arm/jaco_hand/pinch_site_rmat',
    'duplo2x4/angular_velocity',
    'duplo2x4/linear_velocity',
    'duplo2x4/orientation',
    'duplo2x4/position',
)


def reach_duplo_state(qpos, qvel, ctrl=None,
                      task_name='reach_duplo_features', seed=0):
    """Evaluate `Reach` with a Duplo at an injected state. Returns a dict.

    ⚠ The reward's target is `p.bind(task._target).xpos` and `task._target` is
    a `_AttachmentFrame`, so this binds a BODY. The `target_site` the task also
    creates is invisible, sits at the frame origin and is read by nothing —
    checking against it would agree by accident and stop agreeing for the
    `place_*` tasks, where the two are not the same element.

    ⚠ `mj_forward` fills `sensordata`, so `jaco_arm/joints_torque` is the
    ACCELERATION STAGE AT THIS STATE; the Mojo side must produce `cfrc_int` at
    the same state, not after a control step. Same trap as `reach_state`.
    """
    _bootstrap()
    import numpy as np
    env = _load(task_name, seed=seed)
    task, p = env.task, env.physics
    p.data.qpos[:] = np.asarray(qpos, dtype=float)
    p.data.qvel[:] = np.asarray(qvel, dtype=float)
    p.data.ctrl[:] = 0.0 if ctrl is None else np.asarray(ctrl, dtype=float)
    p.forward()
    rs = np.random.RandomState(0)
    obs = {}
    for entity in (task._arm, task._hand, task._prop):
        for k, v in entity.observables.as_dict().items():
            if v.enabled:
                obs[k] = v
    hand_pos = p.bind(task._hand.tool_center_point).xpos
    target_pos = p.bind(task._target).xpos
    out = {'reward': float(task.get_reward(p)),
           'ncon': int(p.data.ncon),
           'nefc': int(p.data.nefc),
           'tcp_xpos': [float(x) for x in hand_pos],
           'target_xpos': [float(x) for x in target_pos],
           'distance': float(np.linalg.norm(hand_pos - target_pos)),
           'flat': []}
    for name in REACH_DUPLO_OBS_ORDER:
        v = np.asarray(obs[name](p, rs), dtype=float).ravel()
        out[name] = list(v)
        out['flat'].extend(float(x) for x in v)
    return out


def reach_duplo_indices(task_name='reach_duplo_features', seed=0):
    """The element ids `manipulation_reach_duplo_config` hardcodes.

    ⚠ `frame_site` is the element the prop's `framepos`/`framequat`/
    `framelinvel`/`frameangvel` sensors NAME, read out of `sensor_objid`
    rather than looked up by name — the whole point is to catch a rebake that
    moves it. For the Duplo it is `bounding_box`, a SITE 11.9 mm above the
    body origin; for `props.Primitive` it is a GEOM. Same observable, different
    element type, and reading the wrong one is a small plausible offset.
    """
    _bootstrap()
    import mujoco
    m = model(task_name, seed=seed)
    env = _load(task_name, seed=seed)
    task, p = env.task, env.physics
    prop_geoms = task._prop.mjcf_model.find_all('geom')
    sensor_of = {}
    for i in range(m.nsensor):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_of[n] = i
    fp = sensor_of['duplo2x4/position']
    # Which duplo geoms can actually collide with the arm or the ground, by
    # MuJoCo's own contype/conaffinity rule against the compiler default (1, 1).
    collidable = [int(g) for g in range(m.ngeom)
                  if m.geom_bodyid[g] == m.geom_bodyid[
                      p.bind(prop_geoms[0]).element_id]
                  and ((int(m.geom_contype[g]) & 1)
                       or (int(m.geom_conaffinity[g]) & 1))]
    return {
        'prop_body': int(p.bind(task._target).element_id),
        # ⚠ The TCP and the six torque sites, so a gate can assert the ROBOT
        # SITE BASE rather than inherit another task's. `Reach` with a prop
        # puts its target site on the brick, so this model has one fewer
        # worldbody site than `reach_site_features` or `lift_large_box` and
        # every robot site shifts down by one.
        'pinch_site': int(p.bind(task._hand.tool_center_point).element_id),
        'torque_sites': [int(m.sensor_objid[i]) for i in range(m.nsensor)
                         if int(m.sensor_type[i]) == 5],
        'frame_site': int(m.sensor_objid[fp]),
        'frame_objtype': int(m.sensor_objtype[fp]),
        'n_prop_geoms': len(prop_geoms),
        'n_collidable_prop_geoms': len(collidable),
        'prop_qposadr': int(m.jnt_qposadr[m.njnt - 1]),
        'prop_dofadr': int(m.jnt_dofadr[m.njnt - 1]),
        'prop_jnt_type': int(m.jnt_type[m.njnt - 1]),
        'stud_radius': float(max(
            m.geom_size[g][0] for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '')
            .startswith('duplo2x4/stud_'))),
        'nbody': int(m.nbody), 'nq': int(m.nq), 'nv': int(m.nv),
        'ngeom': int(m.ngeom), 'nsite': int(m.nsite),
    }


def reach_duplo_reset_qpos(n, task_name='reach_duplo_features', seed=0):
    """`n` real `initialize_episode` draws — qpos after dm_control's own reset.

    Reports where the brick came to rest and how far the TCP ended up from it,
    which is what OUR reset has to land in the same region of.
    """
    _bootstrap()
    import numpy as np
    env = _load(task_name, seed=seed)
    out = []
    for _ in range(n):
        env.reset()
        p, task = env.physics, env.task
        hand_pos = p.bind(task._hand.tool_center_point).xpos
        target_pos = p.bind(task._target).xpos
        out.append({
            'qpos': [float(x) for x in p.data.qpos],
            'tcp_xpos': [float(x) for x in hand_pos],
            'prop_xpos': [float(x) for x in target_pos],
            'distance': float(np.linalg.norm(hand_pos - target_pos)),
            'reward': float(task.get_reward(p)),
            'ncon': int(p.data.ncon),
        })
    return out


def prop_pose_accepted(qpos, task_name='reach_duplo_features', seed=0):
    """dm_control's OWN `PropPlacer` rejection predicate at an injected state.

    `_has_collisions_with_prop` — any contact with `dist <= 0` touching a geom
    of the prop. Returns `(accepted, ncon)`; `accepted` is the negation, i.e.
    what the placer's loop breaks on.

    ⚠ NOT the same predicate as `tcp_pose_accepted`. The TCP initializer
    ignores robot-versus-FREE-body contacts entirely; this one is only about
    the prop, and a prop resting on the table IS a rejection — which is why
    `reach.py` places it 1 mm up (`_PROP_Z_OFFSET`) and settles afterwards.
    """
    _bootstrap()
    import numpy as np
    env = _load(task_name, seed=seed)
    task, p = env.task, env.physics
    p.data.qpos[:] = np.asarray(qpos, dtype=float)
    p.forward()
    bad = task._prop_placer._has_collisions_with_prop(p, task._prop)
    return (not bool(bad), int(p.data.ncon))
