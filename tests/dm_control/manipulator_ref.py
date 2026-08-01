"""Reference manipulator model builder for `test_manipulator_vs_dm_control.mojo`.

`manipulator.xml` is a static file, but no task uses it as written:
`dm_control/suite/manipulator.py::make_model(use_peg, insert)` DELETES the
prop bodies the chosen task does not need, and which bodies go changes every
body/geom/site index after the arm. So the reference side has to run that same
surgery rather than `from_xml_path`-ing the file.

`from dm_control.suite import manipulator` is not importable in this
environment (`dm_env` is missing and `dm_control.suite.__init__` imports it at
module scope), and `make_model` itself needs `lxml`, which is also absent. So
`make_model_xml` below is `make_model` COPIED with exactly two mechanical
substitutions, the same pair `swimmer_ref.py` needed:

    lxml.etree                      -> xml.etree.ElementTree   (stdlib)
    etree.tostring(pretty_print=1)  -> ElementTree.tostring()

plus a local stand-in for `xml_tools.find_element`, which is four lines and
would otherwise drag in the whole `dm_control` package. The element tree is
identical either way.

Keeping this a copy rather than an import is the point: if the reference
generator ever changes, this diverges visibly instead of silently agreeing
with our port because both were written by the same hand.
"""

import os
import xml.etree.ElementTree as etree

_SUITE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "references",
    "dm_control-main",
    "dm_control",
    "suite",
)

_ASSET_FILENAMES = [
    "./common/materials.xml",
    "./common/skybox.xml",
    "./common/visual.xml",
]

# `manipulator._ALL_PROPS`, verbatim.
_ALL_PROPS = frozenset(["ball", "target_ball", "cup", "peg", "target_peg", "slot"])


def assets():
    """`dm_control.suite.common.ASSETS`, rebuilt from the reference tree."""
    out = {}
    for name in _ASSET_FILENAMES:
        with open(os.path.join(_SUITE_DIR, name), "rb") as f:
            out[name] = f.read()
    return out


def _read_model(model_filename):
    with open(os.path.join(_SUITE_DIR, model_filename), "rb") as f:
        return f.read()


def _find_element(root, tag, name):
    """`dm_control.utils.xml_tools.find_element`, minus the import."""
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    raise ValueError("Element with tag {!r} and name {!r} not found".format(tag, name))


def _parent_map(root):
    """ElementTree has no `getparent()`; build the reverse index once."""
    return {child: parent for parent in root.iter() for child in parent}


def make_model_xml(use_peg, insert):
    """`manipulator.make_model`, returning just the XML string."""
    xml_string = _read_model("manipulator.xml")
    mjcf = etree.fromstring(xml_string)

    # Select the desired prop.
    if use_peg:
        required_props = ["peg", "target_peg"]
        if insert:
            required_props += ["slot"]
    else:
        required_props = ["ball", "target_ball"]
        if insert:
            required_props += ["cup"]

    # Remove unused props
    parents = _parent_map(mjcf)
    for unused_prop in _ALL_PROPS.difference(required_props):
        prop = _find_element(mjcf, "body", unused_prop)
        parents[prop].remove(prop)

    return etree.tostring(mjcf)


def model(use_peg=False, insert=False):
    """The compiled reference `mjModel` for one manipulator task."""
    import mujoco

    return mujoco.MjModel.from_xml_string(make_model_xml(use_peg, insert), assets())


# --- Task-side reference: observation and reward -----------------------------
# Copies of `Bring.get_observation` / `_ball_reward` + `rewards.tolerance`,
# with the same two mechanical substitutions the model builder needed. Kept as
# copies rather than imports so a change upstream diverges VISIBLY instead of
# silently agreeing with our port because both were written by the same hand.

_ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
               'finger', 'fingertip', 'thumb', 'thumbtip']
_TOUCH_SENSORS = ['palm_touch', 'finger_touch', 'thumb_touch',
                  'fingertip_touch', 'thumbtip_touch']
_CLOSE = .01


def _jid(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def _bid(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def _sid(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)


def observation(m, d):
    """`Bring.get_observation(fully_observable=True)`, flattened."""
    import numpy as np
    out = []
    # arm_pos: np.vstack([sin, cos]).T  -> interleaved (sin, cos) per joint
    q = np.array([d.qpos[m.jnt_qposadr[_jid(m, n)]] for n in _ARM_JOINTS])
    out.extend(np.vstack([np.sin(q), np.cos(q)]).T.ravel())
    # arm_vel
    out.extend([d.qvel[m.jnt_dofadr[_jid(m, n)]] for n in _ARM_JOINTS])
    # touch
    out.extend(np.log1p([d.sensordata[m.sensor_adr[
        __import__('mujoco').mj_name2id(
            m, __import__('mujoco').mjtObj.mjOBJ_SENSOR, n)]]
        for n in _TOUCH_SENSORS]))

    def body_2d_pose(name):
        b = _bid(m, name)
        return [d.xpos[b][0], d.xpos[b][2], d.xquat[b][0], d.xquat[b][2]]

    out.extend(body_2d_pose('hand'))
    out.extend(body_2d_pose('ball'))
    out.extend([d.qvel[m.jnt_dofadr[_jid(m, n)]]
                for n in ('ball_x', 'ball_z', 'ball_y')])
    out.extend(body_2d_pose('target_ball'))
    return np.array(out, dtype=np.float64)


def _tolerance(x, lower, upper, margin):
    """`rewards.tolerance` with the default gaussian sigmoid."""
    import numpy as np
    if lower <= x <= upper:
        return 1.0
    d = ((lower - x) if x < lower else (x - upper)) / margin
    scale = np.sqrt(-2 * np.log(0.1))          # value_at_margin = 0.1
    return float(np.exp(-0.5 * (d * scale) ** 2))


def reward(m, d):
    """`Bring._ball_reward`."""
    import numpy as np
    a, b = _sid(m, 'ball'), _sid(m, 'target_ball')
    dist = float(np.linalg.norm(d.site_xpos[a] - d.site_xpos[b]))
    return _tolerance(dist, 0.0, _CLOSE, _CLOSE * 2)
