"""ToddlerBot reaches the end of `parse_xml_full`, with MuJoCo's counts.

`<inertial fullinertia>` was the LAST raise standing between the parser and
the real robot: all three variants spell 45 of their bodies that way, and
`full_parser` refused the file before `<contact>` was ever reached. This is
phase 1a's deliverable — "the model parses at all" — held to MuJoCo's own
element counts rather than to "it did not throw".

⚠ THIS IS A PARSE RESULT AND NOTHING MORE. `init_fields`, the field build and
the first step on real ToddlerBot geometry are still unmeasured. `<keyframe>`
(phase 1b) is parsed by NOBODY and is silent — the reference env resets from
`keyframe("home").qpos`, so a rollout built on this today starts from the
wrong pose without raising. Do not read a green here as "ToddlerBot runs".

⚠ Counts come from loading the model with MuJoCo, never from grep
(`feedback_count_model_elements_with_mujoco_not_grep`).

Run with:
    pixi run mojo run -I . tests/physics3d/test_toddlerbot_parses.mojo
"""

from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.parser.full_parser import parse_xml_full

comptime ROOT = String("references/mujoco_menagerie-main/")


def _read(path: String) raises -> String:
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "r")
    var s = String(f.read())
    _ = f.close()
    return s^


def _check(dirp: String, scene: String, robot: String) raises:
    """Our merged parse against MuJoCo's own load of the same scene."""
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(ROOT + dirp + scene)

    var merged = merge_mjcf(
        _read(ROOT + dirp + scene), _read(ROOT + dirp + robot)
    )
    var fmd = parse_xml_full(merged)

    # `fmd.bodies` excludes the worldbody; `m.nbody` includes it.
    var ours_nbody = len(fmd.bodies) + 1
    var mj_nbody = Int(py=m.nbody)
    var ours_njnt = len(fmd.joints)
    var mj_njnt = Int(py=m.njnt)
    var ours_ngeom = len(fmd.geoms)
    var mj_ngeom = Int(py=m.ngeom)
    var ours_npair = len(fmd.pairs)
    var mj_npair = Int(py=m.npair)
    var ours_neq = len(fmd.equalities)
    var mj_neq = Int(py=m.neq)

    print(
        "  ", dirp,
        " nbody", ours_nbody, "/", mj_nbody,
        " njnt", ours_njnt, "/", mj_njnt,
        " ngeom", ours_ngeom, "/", mj_ngeom,
        " npair", ours_npair, "/", mj_npair,
        " neq", ours_neq, "/", mj_neq,
    )

    # Vacuity: 45 bodies must actually be spelled `fullinertia`, or this file
    # gates the merge and not the decomposition.
    var n_full = 0
    var robot_src = _read(ROOT + dirp + robot)
    var at = robot_src.find("fullinertia")
    while at != -1:
        n_full += 1
        at = robot_src.find("fullinertia", at + 1)
    assert_true(
        n_full >= 40,
        "only " + String(n_full) + " `fullinertia` spellings in " + robot
        + " — this fixture no longer exercises the decomposition",
    )

    assert_true(
        ours_nbody == mj_nbody,
        "nbody " + String(ours_nbody) + " != MuJoCo " + String(mj_nbody),
    )
    assert_true(
        ours_njnt == mj_njnt,
        "njnt " + String(ours_njnt) + " != MuJoCo " + String(mj_njnt),
    )
    assert_true(
        ours_ngeom == mj_ngeom,
        "ngeom " + String(ours_ngeom) + " != MuJoCo " + String(mj_ngeom),
    )
    assert_true(
        ours_npair == mj_npair,
        "npair " + String(ours_npair) + " != MuJoCo " + String(mj_npair),
    )
    assert_true(
        ours_neq == mj_neq,
        "neq " + String(ours_neq) + " != MuJoCo " + String(mj_neq),
    )


def test_toddlerbot_2xm_parses() raises:
    """The walk scene — 91 pairs, the full collision set."""
    print("=== toddlerbot_2xm parses ===")
    _check("toddlerbot_2xm/", "scene.xml", "toddlerbot_2xm.xml")
    print("  PASS")


def test_toddlerbot_2xm_mjx_parses() raises:
    """The TRAINING scene. Structurally the same robot with 8 pairs, so a
    pair-table regression shows up here and not in the walk scene."""
    print("=== toddlerbot_2xm_mjx parses ===")
    _check("toddlerbot_2xm/", "scene_mjx.xml", "toddlerbot_2xm_mjx.xml")
    print("  PASS")


def test_toddlerbot_2xc_parses() raises:
    """The second variant — same counts, different mass and mesh data."""
    print("=== toddlerbot_2xc parses ===")
    _check("toddlerbot_2xc/", "scene.xml", "toddlerbot_2xc.xml")
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
