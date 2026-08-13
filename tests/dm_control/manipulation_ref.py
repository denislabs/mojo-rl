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
    """
    _bootstrap()
    if task_name not in ALL_FEATURES:
        raise ValueError(
            '{!r} is not one of the 13 _features tasks: {}'
            .format(task_name, ', '.join(ALL_FEATURES)))
    key = (task_name, seed)
    if key not in _env_cache:
        from dm_control import manipulation
        _env_cache[key] = manipulation.load(task_name, seed=seed)
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
