"""Reference swimmer model builder for `test_swimmer_vs_dm_control.mojo`.

Every other dm_control domain in this suite is a static `.xml`, so its parity
test just calls `mujoco.MjModel.from_xml_path(...)`. Swimmer is PROCEDURAL:
`dm_control/suite/swimmer.py::_make_model(n_bodies)` parses `swimmer.xml` (head
only) and appends the segment chain, the motors and the sensors. There is no
file on disk to load, so the reference side has to run that same generator.

`from dm_control.suite import swimmer` is not importable in this environment
(`dm_env` is not installed, and `dm_control.suite.__init__` imports it at module
scope), and `_make_model` itself needs `lxml`, which is also absent. So
`make_model_xml` below is `_make_model` COPIED VERBATIM with exactly two
mechanical substitutions:

    lxml.etree                      -> xml.etree.ElementTree   (stdlib)
    etree.tostring(pretty_print=1)  -> ElementTree.tostring()

Both are serialization-only; the element tree they build is identical. The
`common.ASSETS` dict is rebuilt from the same three files by the same keys.

Keeping this a copy rather than an import is the point: if the reference
generator ever changes, this diverges visibly instead of silently agreeing with
our port because both were written by the same hand.
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


def _make_body(body_index):
    """Generates an xml string defining a single physical body."""
    body_name = "segment_{}".format(body_index)
    visual_name = "visual_{}".format(body_index)
    inertial_name = "inertial_{}".format(body_index)
    body = etree.Element("body", name=body_name)
    body.set("pos", "0 .1 0")
    etree.SubElement(body, "geom", {"class": "visual", "name": visual_name})
    etree.SubElement(body, "geom", {"class": "inertial", "name": inertial_name})
    return body


def make_model_xml(n_bodies):
    """Generates an xml string defining a swimmer with `n_bodies` bodies."""
    if n_bodies < 3:
        raise ValueError(
            "At least 3 bodies required. Received {}".format(n_bodies)
        )
    mjcf = etree.fromstring(_read_model("swimmer.xml"))
    head_body = mjcf.find("./worldbody/body")
    actuator = etree.SubElement(mjcf, "actuator")
    sensor = etree.SubElement(mjcf, "sensor")

    parent = head_body
    for body_index in range(n_bodies - 1):
        site_name = "site_{}".format(body_index)
        child = _make_body(body_index=body_index)
        child.append(etree.Element("site", name=site_name))
        joint_name = "joint_{}".format(body_index)
        joint_limit = 360.0 / n_bodies
        joint_range = "{} {}".format(-joint_limit, joint_limit)
        child.append(
            etree.Element("joint", {"name": joint_name, "range": joint_range})
        )
        motor_name = "motor_{}".format(body_index)
        actuator.append(etree.Element("motor", name=motor_name, joint=joint_name))
        velocimeter_name = "velocimeter_{}".format(body_index)
        sensor.append(
            etree.Element("velocimeter", name=velocimeter_name, site=site_name)
        )
        gyro_name = "gyro_{}".format(body_index)
        sensor.append(etree.Element("gyro", name=gyro_name, site=site_name))
        parent.append(child)
        parent = child

    # Move tracking cameras further away from the swimmer according to its
    # length.
    cameras = mjcf.findall("./worldbody/body/camera")
    scale = n_bodies / 6.0
    for cam in cameras:
        if cam.get("mode") == "trackcom":
            old_pos = cam.get("pos").split(" ")
            new_pos = " ".join([str(float(dim) * scale) for dim in old_pos])
            cam.set("pos", new_pos)

    return etree.tostring(mjcf)


def model(n_bodies):
    """The compiled reference `mjModel` for an `n_bodies`-link swimmer."""
    import mujoco

    return mujoco.MjModel.from_xml_string(make_model_xml(n_bodies), assets())
