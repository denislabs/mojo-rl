"""Reference quadruped model builder for `test_quadruped_vs_dm_control.mojo`.

`quadruped.xml` is a static file, but no task uses it as written:
`dm_control/suite/quadruped.py::make_model()` (line 55) DELETES the walls, the
ball body, the target site, the terrain geom and every rangefinder SENSOR, and
rewrites the floor geom's size. Walk and run differ only in that floor size
(`_DEFAULT_TIME_LIMIT * speed` = 20*.5 = 10 and 20*5 = 100).

`from dm_control.suite import quadruped` is not importable in this environment
(`dm_env` is missing and `dm_control.suite.__init__` imports it at module
scope), and `make_model` itself needs `lxml`, which is also absent. So
`make_model_xml` below is `make_model` COPIED with the same two mechanical
substitutions `manipulator_ref.py` and `swimmer_ref.py` needed:

    lxml.etree                      -> xml.etree.ElementTree   (stdlib)
    etree.tostring(pretty_print=1)  -> ElementTree.tostring()

plus local stand-ins for `xml_tools.find_element` and lxml's `getparent()`,
which ElementTree does not have. The element tree is identical either way.

Keeping this a copy rather than an import is the point: if the reference
generator ever changes, this diverges visibly instead of silently agreeing
with our port because both were written by the same hand.

⚠ `remove_blank_text=True` on the reference parser has no ElementTree
equivalent. It only affects whitespace between elements, which the MJCF
compiler ignores — but it does mean the XML STRING this returns is not
byte-identical to the reference's. Compare compiled MODELS, never the text.
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

# `quadruped._WALLS`, verbatim.
_WALLS = ["wall_px", "wall_py", "wall_nx", "wall_ny"]

# `quadruped._DEFAULT_TIME_LIMIT`, `_WALK_SPEED`, `_RUN_SPEED`.
_DEFAULT_TIME_LIMIT = 20
_WALK_SPEED = 0.5
_RUN_SPEED = 5


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


def make_model_xml(floor_size=None, terrain=False, rangefinders=False,
                   walls_and_ball=False):
    """`quadruped.make_model`, returning just the XML string."""
    xml_string = _read_model("quadruped.xml")
    mjcf = etree.fromstring(xml_string)

    # Set floor size.
    if floor_size is not None:
        floor_geom = mjcf.find(".//geom[@name='floor']")
        floor_geom.attrib["size"] = f"{floor_size} {floor_size} .5"

    # Remove walls, ball and target.
    if not walls_and_ball:
        parents = _parent_map(mjcf)
        for wall in _WALLS:
            wall_geom = _find_element(mjcf, "geom", wall)
            parents[wall_geom].remove(wall_geom)

        # Remove ball.
        ball_body = _find_element(mjcf, "body", "ball")
        parents[ball_body].remove(ball_body)

        # Remove target.
        target_site = _find_element(mjcf, "site", "target")
        parents[target_site].remove(target_site)

    # Remove terrain.
    if not terrain:
        parents = _parent_map(mjcf)
        terrain_geom = _find_element(mjcf, "geom", "terrain")
        parents[terrain_geom].remove(terrain_geom)

    # Remove rangefinders if they're not used, as range computations can be
    # expensive, especially in a scene with heightfields.
    if not rangefinders:
        parents = _parent_map(mjcf)
        for rf in mjcf.findall(".//rangefinder"):
            parents[rf].remove(rf)

    return etree.tostring(mjcf)


def model(run=False):
    """The compiled reference `mjModel` for `walk` (default) or `run`.

    The ONLY difference is the floor plane's half-extent, which a plane geom
    ignores dynamically — it is here so that a size mismatch shows up as a
    model-constant diff rather than as nothing at all.
    """
    import mujoco

    speed = _RUN_SPEED if run else _WALK_SPEED
    return mujoco.MjModel.from_xml_string(
        make_model_xml(floor_size=_DEFAULT_TIME_LIMIT * speed), assets()
    )


# --- XML-authoring gate -------------------------------------------------------

# Every mjModel table that the ported XML can get wrong. Both sides are
# compiled by MuJoCo, so this compares the XML TEXT and nothing else — our
# parser and our engine are not in the loop. Listed exhaustively on purpose:
# picking a subset is how `jnt_solimp` stayed wrong (the `<freejoint>`
# expansion was inheriting `solimplimit` from `<default class="body">`, which
# MuJoCo's `mjs_addFreeJoint` does not) through five other quadruped gates.
# The table list, the counts, the `<option>` fields and the compare loop moved
# to `mjmodel_diff.py` on 2026-08-03, when humanoid_CMU became the second user
# of this gate and `docs/DM_CONTROL_PORT_PHASE2.md` queued ~40 more ports behind
# it. Behaviour is unchanged except that the shared list also covers `nexclude`
# and `exclude_signature`, which quadruped has none of.


def compare_xml_to_reference(xml_string, run=False):
    """Compile `xml_string` with MuJoCo and diff it against the reference.

    Returns a list of human-readable mismatch strings; empty means the ported
    XML compiles to the reference model exactly.
    """
    import mujoco
    from mjmodel_diff import diff_models

    return diff_models(model(run=run),
                       mujoco.MjModel.from_xml_string(xml_string))


def n_tables_compared():
    """How many mjModel tables `compare_xml_to_reference` sweeps."""
    from mjmodel_diff import n_tables

    return n_tables()
