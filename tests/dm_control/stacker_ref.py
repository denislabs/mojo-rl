"""Reference stacker model builder for `test_stacker_vs_dm_control.mojo`.

`stacker.xml` is a static file, but neither task uses it as written:
`dm_control/suite/stacker.py::make_model(n_boxes)` DELETES `box{n_boxes..3}`,
and since the boxes precede the target, deleting them renumbers the target
body, geom and site. So the reference side has to run that same surgery rather
than `from_xml_path`-ing the file.

`from dm_control.suite import stacker` is not importable in this environment
(`dm_env` is missing and `dm_control.suite.__init__` imports it at module
scope), and `make_model` itself needs `lxml`, which is also absent. So
`make_model_xml` below is `make_model` COPIED with exactly two mechanical
substitutions, the same pair `manipulator_ref.py` and `swimmer_ref.py` needed:

    lxml.etree                      -> xml.etree.ElementTree   (stdlib)
    etree.tostring(pretty_print=1)  -> ElementTree.tostring()

plus a local stand-in for `xml_tools.find_element`, which is four lines and
would otherwise drag in the whole `dm_control` package. The element tree is
identical either way.

Keeping this a copy rather than an import is the point: if the reference
generator ever changes, this diverges visibly instead of silently agreeing with
our port because both were written by the same hand.

⚠ `stacker.xml` contains a stray `>` after `</visual>` on line 15. It is legal
XML — a bare `>` is permitted in character data — and both `lxml` and the
stdlib parser read it as text belonging to `<mujoco>`, so `make_model_xml`
round-trips it unchanged and MuJoCo ignores it. Our port drops the whole
cosmetic `<visual>` block and the stray character with it.
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


def make_model_xml(n_boxes):
    """`stacker.make_model`, returning just the XML string."""
    xml_string = _read_model("stacker.xml")
    mjcf = etree.fromstring(xml_string)

    # Remove unused boxes
    parents = _parent_map(mjcf)
    for b in range(n_boxes, 4):
        box = _find_element(mjcf, "body", "box" + str(b))
        parents[box].remove(box)

    return etree.tostring(mjcf)


def model(n_boxes=2):
    """The compiled reference `mjModel` for one stacker task."""
    import mujoco

    return mujoco.MjModel.from_xml_string(make_model_xml(n_boxes), assets())


# --- Task-side reference: observation and reward -----------------------------
# Copies of `Stack.get_observation` / `Stack.get_reward` + `rewards.tolerance`,
# with the same substitutions. Kept as copies rather than imports so a change
# upstream diverges VISIBLY instead of silently agreeing with our port because
# both were written by the same hand.

_ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
               'finger', 'fingertip', 'thumb', 'thumbtip']
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


def _box_names(n_boxes):
    return ['box' + str(b) for b in range(n_boxes)]


def _box_joint_names(n_boxes):
    """`Stack.__init__`: `for name in box_names: for dim in 'xyz'`.

    ⚠ 'xyz', not the model's x, z, y — so this is a PERMUTATION, and
    `manipulator`'s equivalent ('xzy') is not.
    """
    out = []
    for name in _box_names(n_boxes):
        for dim in 'xyz':
            out.append('_'.join([name, dim]))
    return out


def observation(m, d, n_boxes=2):
    """`Stack.get_observation(fully_observable=True)`, flattened."""
    import numpy as np
    out = []
    # arm_pos: np.vstack([sin, cos]).T  -> interleaved (sin, cos) per joint
    q = np.array([d.qpos[m.jnt_qposadr[_jid(m, n)]] for n in _ARM_JOINTS])
    out.extend(np.vstack([np.sin(q), np.cos(q)]).T.ravel())
    # arm_vel
    out.extend([d.qvel[m.jnt_dofadr[_jid(m, n)]] for n in _ARM_JOINTS])
    # touch: `np.log1p(self.data.sensordata)` — ALL of it, in sensor order.
    out.extend(np.log1p(np.array(d.sensordata, dtype=np.float64)))

    def body_2d_pose(name, orientation=True):
        b = _bid(m, name)
        if orientation:
            return [d.xpos[b][0], d.xpos[b][2], d.xquat[b][0], d.xquat[b][2]]
        return [d.xpos[b][0], d.xpos[b][2]]

    out.extend(body_2d_pose('hand'))
    for name in _box_names(n_boxes):
        out.extend(body_2d_pose(name))
    out.extend([d.qvel[m.jnt_dofadr[_jid(m, n)]]
                for n in _box_joint_names(n_boxes)])
    out.extend(body_2d_pose('target', orientation=False))
    return np.array(out, dtype=np.float64)


def _tolerance(x, lower, upper, margin):
    """`rewards.tolerance` with the default gaussian sigmoid."""
    import numpy as np
    if lower <= x <= upper:
        return 1.0
    d = ((lower - x) if x < lower else (x - upper)) / margin
    scale = np.sqrt(-2 * np.log(0.1))          # value_at_margin = 0.1
    return float(np.exp(-0.5 * (d * scale) ** 2))


def _site_distance(m, d, s1, s2):
    """`Physics.site_distance`."""
    import numpy as np
    return float(np.linalg.norm(d.site_xpos[_sid(m, s1)] - d.site_xpos[_sid(m, s2)]))


def reward(m, d, n_boxes=2):
    """`Stack.get_reward` — `box_is_close * hand_is_far`."""
    box_size = m.geom_size[
        __import__('mujoco').mj_name2id(
            m, __import__('mujoco').mjtObj.mjOBJ_GEOM, 'target')][0]
    min_box_to_target_distance = min(
        _site_distance(m, d, name, 'target') for name in _box_names(n_boxes))
    box_is_close = _tolerance(min_box_to_target_distance, 0.0, 0.0,
                              2 * box_size)
    hand_to_target_distance = _site_distance(m, d, 'grasp', 'target')
    hand_is_far = _tolerance(hand_to_target_distance, .1, float('inf'), _CLOSE)
    return box_is_close * hand_is_far
