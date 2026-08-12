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
