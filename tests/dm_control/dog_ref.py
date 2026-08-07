"""Reference dog model builder for `test_dog_vs_dm_control.mojo`.

Two jobs, and the second one is the unusual part.

1. `make_model_xml` is `dog.py::make_model` COPIED, with the same mechanical
   substitutions `quadruped_ref.py`/`manipulator_ref.py`/`swimmer_ref.py`
   needed (`lxml.etree` -> stdlib `ElementTree`, plus local stand-ins for
   `xml_tools.find_element` and lxml's `getparent()`). `from dm_control.suite
   import dog` is not importable here: `dm_env` and `lxml` are both absent.
   Keeping it a copy rather than an import is deliberate — if the reference
   generator changes, this diverges visibly instead of silently agreeing with
   our port because both were written by the same hand.

2. `bake_xml` performs THE DEVIATION (`docs/DM_CONTROL_PORT_PHASE2.md` §4.2):
   it replaces dog's 162 mesh "bone" geoms with the explicit `<inertial>` they
   exist to produce, and deletes them and the `<asset>` mesh block.

WHY THE BAKE IS SOUND, in the two parts that have to hold separately:

  * The mesh geoms never collide. `<default class="bone">` sets
    `contype="0" conaffinity="0"`, and this is CHECKED, not assumed —
    `check_bake` fails if any mesh geom has either flag set. Measured on
    MuJoCo 3.10.0: 162 mesh geoms, 0 colliding. All 120 colliding geoms are
    plane/sphere/capsule/ellipsoid/box `collision_primitive`s.
  * Their only other contribution is INERTIA. dog declares just 3 explicit
    `<inertial>` elements, so MuJoCo derives 59 bodies' mass and inertia
    tensor from mesh volume at `density="1100"`/`"300"`. Stating that result
    explicitly is exactly what the compiler would have computed.

⚠ THIS IS A DEVIATION AND IS LABELLED AS ONE. A baked constant that outlives
its justification is the failure mode `point_mass`'s tendon workaround
demonstrated. What keeps it honest is `check_bake`: it diffs the baked model
against the unbaked one over every table, and the ONLY thing it is allowed to
skip is the geom rows of the deleted geoms — matched BY NAME, so a surviving
geom that moved id is a failure, not a silent pass.

The gate therefore chains:

    dog.xml  --check_bake (all non-geom tables exact, geoms name-matched)-->
    baked    --diff_models (the ordinary layer-1 gate)-->  our ported XML

Layer 1 compares our XML against the BAKED reference, which is only a valid
reference because step one holds.
"""

import os
import xml.etree.ElementTree as etree

_SUITE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'references', 'dm_control-main', 'dm_control', 'suite')
_ASSET_DIR = os.path.join(_SUITE, 'dog_assets')

# The three `<include>`s at the top of dog.xml, plus every mesh/texture the
# `<asset>` block names. `meshdir`/`texturedir` are "dog_assets/", so the keys
# MuJoCo looks up are the paths as written in the XML.
def assets():
    out = {}
    for name in ('skybox.xml', 'visual.xml', 'materials.xml'):
        p = os.path.join(_SUITE, 'common', name)
        with open(p, 'rb') as f:
            out['./common/' + name] = f.read()
    for name in os.listdir(_ASSET_DIR):
        p = os.path.join(_ASSET_DIR, name)
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                out['dog_assets/' + name] = f.read()
    return out


def _find_element(root, tag, name):
    for e in root.iter(tag):
        if e.get('name') == name:
            return e
    raise KeyError('no <%s name="%s">' % (tag, name))


def _parent_map(root):
    return {c: p for p in root.iter() for c in p}


def make_model_xml(floor_size, remove_ball):
    """`dog.py::make_model`, copied. Returns the XML as bytes."""
    with open(os.path.join(_SUITE, 'dog.xml'), 'rb') as f:
        xml_string = f.read()
    mjcf = etree.XML(xml_string)
    parents = _parent_map(mjcf)

    # set floor size.
    floor = _find_element(mjcf, 'geom', 'floor')
    floor.attrib['size'] = str(floor_size) + ' ' + str(floor_size) + ' .1'

    if remove_ball:
        # Remove ball, target and walls.
        for tag, name in (('body', 'ball'), ('geom', 'target'),
                          ('camera', 'ball'), ('camera', 'head')):
            e = _find_element(mjcf, tag, name)
            parents[e].remove(e)
        for wall_name in ['px', 'nx', 'py', 'ny']:
            e = _find_element(mjcf, 'geom', 'wall_' + wall_name)
            parents[e].remove(e)

    return etree.tostring(mjcf)


def _compile(xml_bytes):
    import mujoco
    return mujoco.MjModel.from_xml_string(xml_bytes, assets())


def _fmt(x):
    """17 significant digits — a float64 round-trips at 17, not at 16."""
    return repr(float(x))


def bake_xml(xml_bytes):
    """Replace dog's mesh geoms with the inertia they imply. Returns bytes."""
    import mujoco

    m = _compile(xml_bytes)
    mjcf = etree.XML(xml_bytes)
    parents = _parent_map(mjcf)

    # 1. Give every non-world body an explicit <inertial> holding exactly what
    #    the compiler just derived. Written FIRST, so the values come from the
    #    model that still has its meshes.
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        body = _find_element(mjcf, 'body', name)
        for old in body.findall('inertial'):
            body.remove(old)
        el = etree.Element('inertial')
        el.set('pos', ' '.join(_fmt(v) for v in m.body_ipos[b]))
        el.set('quat', ' '.join(_fmt(v) for v in m.body_iquat[b]))
        el.set('mass', _fmt(m.body_mass[b]))
        el.set('diaginertia', ' '.join(_fmt(v) for v in m.body_inertia[b]))
        body.insert(0, el)

    # 2. Delete every mesh geom, the <asset> meshes they name, and the skin
    #    (visual only, and it references the meshes).
    #
    #    ⚠ THE DELETION SET COMES FROM THE COMPILED MODEL, NOT FROM THE XML
    #    TEXT. The obvious text rule — "type='mesh', or a bone-family class" —
    #    is WRONG, and `check_bake` caught it: `iris_L`, `iris_R`, `pupil_L`
    #    and `pupil_R` carry `class="visible_bone"` (a bone class) but override
    #    it with `type="ellipsoid"`/`type="sphere"`. Deleting them removed four
    #    real geoms. Asking the compiler which geoms ARE meshes has no such
    #    failure mode.
    mesh_names = set()
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
            if nm is None:
                raise AssertionError(
                    'mesh geom %d is unnamed; bake_xml matches by name' % i)
            mesh_names.add(nm)
    for geom in list(mjcf.iter('geom')):
        if geom.get('name') in mesh_names:
            parents[geom].remove(geom)
    for asset in mjcf.iter('asset'):
        for child in list(asset):
            if child.tag in ('mesh', 'skin'):
                asset.remove(child)
    for skin in list(mjcf.iter('skin')):
        parents[skin].remove(skin)

    # 3. The bone default classes now have no users. Leaving them is harmless
    #    (MuJoCo ignores unused classes) and keeps the diff to this function
    #    small, so they stay — but the density they carry is now dead, and that
    #    is precisely the fact the deviation note above is about.
    return etree.tostring(mjcf)


def check_bake(floor_size=15, remove_ball=True):
    """Gate the deviation: baked model == unbaked model, minus dead geoms.

    Returns a list of human-readable mismatches; empty means the bake changed
    nothing that can affect physics.
    """
    import numpy as np
    import mujoco
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mjmodel_diff import _TABLES

    raw_xml = make_model_xml(floor_size, remove_ball)
    raw = _compile(raw_xml)
    baked = _compile(bake_xml(raw_xml))

    bad = []

    # (a) The premise: no mesh geom may collide. If one did, deleting it would
    #     change the physics and the whole deviation is invalid.
    live_mesh = [i for i in range(raw.ngeom)
                 if raw.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                 and (raw.geom_contype[i] or raw.geom_conaffinity[i])]
    if live_mesh:
        bad.append('%d MESH geoms COLLIDE - the bake is not valid: %s'
                   % (len(live_mesh), live_mesh[:5]))
    n_mesh = int((np.asarray(raw.geom_type) ==
                  mujoco.mjtGeom.mjGEOM_MESH).sum())
    if n_mesh == 0:
        bad.append('no mesh geoms found - check_bake is not testing anything')
    if raw.ngeom - baked.ngeom != n_mesh:
        bad.append('deleted %d geoms but only %d were meshes'
                   % (raw.ngeom - baked.ngeom, n_mesh))

    # (b) Every table that is not about geoms must be bit-identical.
    #     `body_geomnum`/`body_geomadr` are geom INDEXING tables — deleting 162
    #     geoms necessarily renumbers them, so they are skipped here and the
    #     surviving geoms are checked by name in (c) instead. Nothing else is
    #     exempt; in particular `body_mass`, `body_inertia`, `body_ipos`,
    #     `body_iquat` and `body_invweight0` — the quantities the meshes exist
    #     to produce — are compared at tolerance 0.0.
    for n in _TABLES:
        if n.startswith('geom_') or n in ('body_geomnum', 'body_geomadr'):
            continue
        if not hasattr(raw, n) or not hasattr(baked, n):
            continue
        a = np.asarray(getattr(raw, n), dtype=np.float64)
        b = np.asarray(getattr(baked, n), dtype=np.float64)
        if a.shape != b.shape:
            bad.append('%s: shape %s != %s' % (n, a.shape, b.shape))
            continue
        if a.size == 0:
            continue
        d = np.abs(a - b)
        if d.max() > 0.0:
            i = np.unravel_index(int(np.argmax(d)), d.shape)
            bad.append('%s%s: raw %r != baked %r' % (n, list(i), a[i], b[i]))

    # (c) The surviving geoms, matched BY NAME so an id shift is a failure.
    #     Unnamed geoms are matched by (bodyname, type, position-in-body).
    def key(m, i):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if nm is not None:
            return nm
        b = m.geom_bodyid[i]
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        k = sum(1 for j in range(m.geom_adr if False else i)
                if m.geom_bodyid[j] == b
                and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, j) is None)
        return '%s#%d' % (bn, k)

    raw_keys = {key(raw, i): i for i in range(raw.ngeom)
                if raw.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH}
    baked_keys = {key(baked, i): i for i in range(baked.ngeom)}
    missing = set(raw_keys) - set(baked_keys)
    extra = set(baked_keys) - set(raw_keys)
    if missing:
        bad.append('geoms lost by the bake: %s' % sorted(missing)[:5])
    if extra:
        bad.append('geoms invented by the bake: %s' % sorted(extra)[:5])

    geom_tables = [t for t in _TABLES if t.startswith('geom_')]
    for t in geom_tables:
        a = np.asarray(getattr(raw, t), dtype=np.float64)
        b = np.asarray(getattr(baked, t), dtype=np.float64)
        for k in sorted(set(raw_keys) & set(baked_keys)):
            x, y = a[raw_keys[k]], b[baked_keys[k]]
            if not np.array_equal(x, y):
                bad.append('%s[%s]: raw %r != baked %r' % (t, k, x, y))
                break

    return bad


def port_fragment(floor_size=15, remove_ball=True):
    """The exact text that goes into `dog/dog_xml.mojo`, split at the floor.

    Returns `(head, floor_line, tail)`. `dog_xml.mojo` is GENERATED from this
    rather than transcribed — 69 kB of MJCF is not something to retype, and a
    generator makes the four deviations below auditable in one place instead
    of being invisible edits inside a wall of XML.

    On top of the mesh bake, the port drops four things, none of which any
    mjModel table in `mjmodel_diff._TABLES` records:

      * the three `<include>`s — `merge_mjcf` splices in our own copies of
        skybox / visual / materials instead;
      * `<compiler meshdir texturedir>` — both directories are now empty;
      * the `skin` and `tennis_ball` `<texture>`/`<material>` pairs, which name
        PNG FILES ON DISK. A ported XML has no asset bundle, so MuJoCo could
        not compile our string at all with these present. `skin` is dead after
        the bake; `tennis_ball` only dresses the ball, which stand/walk/trot/
        run delete anyway. ⚠ Phase 5 (fetch) KEEPS the ball and will need a
        builtin material in its place.

    The floor line is split out because it is the only per-task difference:
    `floor_size = move_speed * _DEFAULT_TIME_LIMIT`, so stand and walk share
    15, trot is 45 and run is 135.
    """
    import re
    t = bake_xml(make_model_xml(floor_size, remove_ball)).decode()
    t = re.sub(r'\s*<include file="[^"]*" />', '', t)
    t = re.sub(r'\s*<compiler [^>]*/>', '', t)
    for name in ('skin', 'tennis_ball'):
        t = re.sub(r'\s*<texture name="%s"[^>]*/>' % name, '', t)
        t = re.sub(r'\s*<material name="%s"[^>]*/>' % name, '', t)
    t = re.sub(r'\s*<asset>\s*</asset>', '', t)

    m = re.search(r'[ \t]*<geom name="floor"[^>]*/>\n?', t)
    if m is None:
        raise AssertionError('floor geom not found')
    return t[:m.start()], m.group(), t[m.end():]


def model(floor_size=15, remove_ball=True):
    """The BAKED reference model — what our ported XML is compared against."""
    return _compile(bake_xml(make_model_xml(floor_size, remove_ball)))


def raw_model(floor_size=15, remove_ball=True):
    """The reference model as dm_control builds it, meshes and all."""
    return _compile(make_model_xml(floor_size, remove_ball))


def baked_xml_text(floor_size=15, remove_ball=True):
    return bake_xml(make_model_xml(floor_size, remove_ball)).decode()


def compare_xml_to_reference(xml_string, floor_size=15, remove_ball=True):
    """Layer-1 gate: our XML text vs the baked reference, every table."""
    import mujoco
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mjmodel_diff import diff_models
    ref = model(floor_size, remove_ball)
    got = mujoco.MjModel.from_xml_string(xml_string)
    return diff_models(ref, got)


def n_tables_compared():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mjmodel_diff import n_tables
    return n_tables()


if __name__ == '__main__':
    bad = check_bake()
    print('check_bake: %d mismatches' % len(bad))
    for b in bad[:40]:
        print('   ', b)
    r, k = raw_model(), model()
    print('raw   ngeom %d nmesh %d nbody %d' % (r.ngeom, r.nmesh, r.nbody))
    print('baked ngeom %d nmesh %d nbody %d' % (k.ngeom, k.nmesh, k.nbody))


# =============================================================================
# Task reference — `dog.py`'s Physics accessors, Stand and Move, in numpy.
#
# `from dm_control.suite import dog` is not importable here (no dm_env, no
# lxml), so these are TRANSCRIBED from dog.py against a raw `mujoco` model.
# Same discipline as `make_model_xml` above: a copy that diverges visibly
# beats an import that cannot run.
# =============================================================================

_MIN_UPRIGHT_COSINE = 0.8660254037844387   # cos(deg2rad(30))
_STAND_HEIGHT_FRACTION = 0.9


def _tolerance(x, lower, upper, margin, sigmoid='gaussian',
               value_at_margin=0.1):
    """`dm_control.utils.rewards.tolerance`, the four sigmoids dog uses."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        return np.where(in_bounds, 1.0, 0.0)
    d = np.where(x < lower, lower - x, x - upper) / margin
    if sigmoid == 'gaussian':
        scale = np.sqrt(-2 * np.log(value_at_margin))
        val = np.exp(-0.5 * (d * scale) ** 2)
    elif sigmoid == 'linear':
        scale = 1 - value_at_margin
        scaled = 1 - d * scale
        val = np.where(abs(d) < 1 / scale, scaled, 0.0)
    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_margin - 1
        val = 1 / (d * scale + 1)
    else:
        raise NotImplementedError(sigmoid)
    return np.where(in_bounds, 1.0, val)


def _named_sensor(m, d, name):
    import mujoco
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return d.sensordata[m.sensor_adr[i]:m.sensor_adr[i] + m.sensor_dim[i]]


def _bid(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def stand_height(m):
    """`Stand._stand_height` — measured at qpos0, so a model constant."""
    import numpy as np, mujoco
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    h = np.array([d.xpos[_bid(m, 'torso'), 2], d.xpos[_bid(m, 'pelvis'), 2]])
    return h * _STAND_HEIGHT_FRACTION


def body_weight(m):
    """`Stand._body_weight` = -gravity[2] * body_subtreemass['torso']."""
    return float(-m.opt.gravity[2] * m.body_subtreemass[_bid(m, 'torso')])


def observation(m, d):
    """`Stand.get_observation_components`, concatenated in order (223)."""
    import numpy as np
    hinge = m.jnt_type == 3                       # mjJNT_HINGE
    joint_angles = d.qpos[m.jnt_qposadr[hinge]]
    joint_vels = d.qvel[m.jnt_dofadr[hinge]]
    tp_height = d.xpos[[_bid(m, 'torso'), _bid(m, 'pelvis')], 2]
    zproj = np.vstack((d.xmat[_bid(m, 'skull')].reshape(3, 3)[2],
                       d.xmat[_bid(m, 'torso')].reshape(3, 3)[2],
                       d.xmat[_bid(m, 'pelvis')].reshape(3, 3)[2]))
    torso_frame = d.xmat[_bid(m, 'torso')].reshape(3, 3)
    com_vel = _named_sensor(m, d, 'torso_linvel').dot(torso_frame)
    inertial = np.hstack([_named_sensor(m, d, n) for n in
                          ('accelerometer', 'velocimeter', 'gyro')])
    foot_forces = np.hstack([_named_sensor(m, d, n) for n in
                             ('foot_L', 'foot_R', 'hand_L', 'hand_R')])
    touch = np.hstack([_named_sensor(m, d, n) for n in
                       ('palm_L', 'palm_R', 'sole_L', 'sole_R')])
    return np.hstack((joint_angles, joint_vels, tp_height, zproj.flatten(),
                      com_vel, inertial, foot_forces, touch, d.act.copy()))


def stand_reward_factors(m, d):
    """`Stand.get_reward_factors` — SIX numbers (upright is three)."""
    import numpy as np
    sh = stand_height(m)
    bw = body_weight(m)
    h = d.xpos[[_bid(m, 'torso'), _bid(m, 'pelvis')], 2]
    torso = _tolerance(h[0], sh[0], float('inf'), sh[0])
    pelvis = _tolerance(h[1], sh[1], float('inf'), sh[1])
    upright_vec = np.vstack((d.xmat[_bid(m, 'skull')].reshape(3, 3)[2],
                             d.xmat[_bid(m, 'torso')].reshape(3, 3)[2],
                             d.xmat[_bid(m, 'pelvis')].reshape(3, 3)[2]))[:, 2]
    upright = _tolerance(upright_vec, _MIN_UPRIGHT_COSINE, float('inf'),
                         _MIN_UPRIGHT_COSINE + 1, 'linear', 0.0)
    touch_sum = np.hstack([_named_sensor(m, d, n) for n in
                           ('palm_L', 'palm_R', 'sole_L', 'sole_R')]).sum()
    touch = _tolerance(touch_sum, bw, float('inf'), bw, 'linear', 0.9)
    return np.hstack((torso, pelvis, upright, touch))


def move_reward_factors(m, d, move_speed):
    """`Move.get_reward_factors` — Stand's six times a seventh."""
    import numpy as np
    standing = stand_reward_factors(m, d)
    torso_frame = d.xmat[_bid(m, 'torso')].reshape(3, 3)
    vx = _named_sensor(m, d, 'torso_linvel').dot(torso_frame)[0]
    speed_margin = max(1.0, move_speed)
    forward = _tolerance(vx, move_speed, 2 * move_speed, speed_margin,
                         'linear', 0.0)
    forward = (4 * forward + 1) / 5
    return np.hstack((standing, forward))


# =============================================================================
# fetch (Phase 5) — `Fetch`'s physics helpers, observation and reward factors
# =============================================================================
#
# Transcribed from `suite/dog.py`. Everything here is the REFERENCE side; the
# port is gated against it by `test_dog_fetch_vs_dm_control.mojo`.
#
# ⚠ `object_velocity` RETURNS (linear, angular) while the raw
# `mj_objectVelocity` writes ANGULAR first (res[0:3]) and linear second
# (res[3:6]). dm_control swaps them. Taking res[:3] here would silently give
# the angular velocity, which is the same shape and entirely wrong.


def _obj_lin_vel(m, d, name, objtype):
    """`physics.data.object_velocity(name, type)[0]` — world-frame LINEAR."""
    import mujoco
    import numpy as np
    tid = {'site': mujoco.mjtObj.mjOBJ_SITE, 'geom': mujoco.mjtObj.mjOBJ_GEOM}[objtype]
    oid = mujoco.mj_name2id(m, tid, name)
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, tid, oid, res, 0)   # 0 = world frame
    return res[3:6].copy()


def _head_frame(m, d):
    import numpy as np
    import mujoco
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'head')
    return np.array(d.site_xmat[sid]).reshape(3, 3)


def _site_pos(m, d, name):
    import numpy as np
    import mujoco
    return np.array(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)])


def _geom_pos(m, d, name):
    import numpy as np
    import mujoco
    return np.array(d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)])


def ball_in_head_frame(m, d):
    """`Physics.ball_in_head_frame` — 6 numbers, position THEN velocity."""
    import numpy as np
    head_frame = _head_frame(m, d)
    head_pos = _site_pos(m, d, 'head')
    ball_pos = _geom_pos(m, d, 'ball')
    head_to_ball = ball_pos - head_pos
    head_vel = _obj_lin_vel(m, d, 'head', 'site')
    ball_vel = _obj_lin_vel(m, d, 'ball', 'geom')
    head_to_ball_vel = ball_vel - head_vel
    # `.dot(frame)` is a ROW-vector product, i.e. R^T v — world -> head.
    return np.hstack((head_to_ball.dot(head_frame),
                      head_to_ball_vel.dot(head_frame)))


def target_in_head_frame(m, d):
    """`Physics.target_in_head_frame` — 3 numbers."""
    head_frame = _head_frame(m, d)
    head_pos = _site_pos(m, d, 'head')
    target_pos = _geom_pos(m, d, 'target')
    return (target_pos - head_pos).dot(head_frame)


def ball_to_mouth_distance(m, d):
    """`Physics.ball_to_mouth_distance` — the MEAN of the two bite sites."""
    import numpy as np
    ball_pos = _geom_pos(m, d, 'ball')
    upper = np.linalg.norm(ball_pos - _site_pos(m, d, 'upper_bite'))
    lower = np.linalg.norm(ball_pos - _site_pos(m, d, 'lower_bite'))
    return 0.5 * (upper + lower)


def ball_to_target_distance(m, d):
    """`Physics.ball_to_target_distance`."""
    import numpy as np
    return float(np.linalg.norm(_geom_pos(m, d, 'ball') - _geom_pos(m, d, 'target')))


def fetch_observation(m, d):
    """`Fetch.get_observation_components` — Stand's 223 plus 6 + 3 = 232."""
    import numpy as np
    return np.hstack((observation(m, d),
                      ball_in_head_frame(m, d),
                      target_in_head_frame(m, d)))


def fetch_reward_factors(m, d):
    """`Fetch.get_reward_factors` — Stand's SIX plus reach_ball, fetch_ball."""
    import numpy as np
    import mujoco
    standing = stand_reward_factors(m, d)

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'upper_bite')
    bite_radius = float(m.site_size[sid][0])
    bite_margin = 2
    reach_ball = _tolerance(ball_to_mouth_distance(m, d),
                            0, bite_radius, bite_margin, 'reciprocal')
    reach_ball = (6 * reach_ball + 1) / 7

    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'target')
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    target_radius = float(m.geom_size[gid][0])
    bring_margin = float(m.geom_size[fid][0])
    ball_near_target = _tolerance(ball_to_target_distance(m, d),
                                  0, target_radius, bring_margin, 'reciprocal')
    fetch_ball = (ball_near_target + 1) / 2

    # ⚠ APPLIED AFTER THE RESCALING — `1`, not `(6*1 + 1)/7`.
    if ball_to_target_distance(m, d) < 2 * target_radius:
        reach_ball = 1

    return np.hstack((standing, reach_ball, fetch_ball))
