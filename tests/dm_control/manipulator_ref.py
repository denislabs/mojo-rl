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
