"""Reference humanoid_CMU model for `test_humanoid_cmu_vs_dm_control.mojo`.

humanoid_CMU is a STATIC `.xml` — there is no `make_model` to mirror, unlike
swimmer / manipulator / stacker / quadruped — so the reference side is just the
file on disk. What this module adds is the one deliberate deviation and its
accounting.

THE PORT DROPS THE `<sensor>` BLOCK. `merge_mjcf` carries no `<sensor>` content
into the parsed model, and none of the three tasks reads a sensor except
`thorax_subtreelinvel`, which we compute from `Data.xvel` via
`sensors.subtree_linvel`. So a naive layer-1 diff would report `nsensor 8 != 0`
and four table SHAPE mismatches, drowning any real finding.

Rather than skip the sensor tables — which would leave the deviation
unmeasured — `model()` returns the reference compiled from the SAME xml with
the SAME block removed, so both sides are sensor-free and every remaining
table is compared at tolerance 0.0. `sensor_block_contents()` then states
exactly what was removed, and the test asserts it, so the day upstream adds a
sensor the tasks DO read, this fails instead of silently continuing to drop it.
"""

import os
import re

_SUITE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "references",
    "dm_control-main",
    "dm_control",
    "suite",
)

_XML_PATH = os.path.join(_SUITE_DIR, "humanoid_CMU.xml")

# Every sensor the reference declares, in declaration order. The port drops all
# of them; the one the tasks actually read is marked.
EXPECTED_SENSORS = [
    ("subtreelinvel", "thorax_subtreelinvel"),   # <- the only one read
    ("velocimeter", "sensor_root_veloc"),
    ("gyro", "sensor_root_gyro"),
    ("accelerometer", "sensor_root_accel"),
    ("touch", "sensor_touch_ltoes"),
    ("touch", "sensor_touch_rtoes"),
    ("touch", "sensor_touch_rfoot"),
    ("touch", "sensor_touch_lfoot"),
]

READ_BY_TASKS = ["thorax_subtreelinvel"]


def raw_xml():
    """The reference file, verbatim."""
    with open(_XML_PATH) as f:
        return f.read()


def sensor_block_contents():
    """(sensor_tag, name) pairs declared in the reference's <sensor> block."""
    src = raw_xml()
    block = re.search(r"<sensor>(.*?)</sensor>", src, re.S)
    if block is None:
        return []
    return re.findall(r'<(\w+)\s+name="([^"]+)"', block.group(1))


def _strip_sensors(xml):
    return re.sub(r"[ \t]*<sensor>.*?</sensor>\n?", "", xml, flags=re.S)


def model(with_sensors=False):
    """Compile the reference. Sensor block removed unless asked for.

    Assets are resolved from the suite directory, so `<include>` of
    `./common/*.xml` and the `material="grid"` lookups work exactly as
    dm_control's loader does.
    """
    import mujoco

    xml = raw_xml() if with_sensors else _strip_sensors(raw_xml())
    return mujoco.MjModel.from_xml_string(xml, _assets())


def _assets():
    assets = {}
    for rel in ["./common/materials.xml", "./common/skybox.xml",
                "./common/visual.xml"]:
        path = os.path.join(_SUITE_DIR, rel.lstrip("./"))
        with open(path, "rb") as f:
            assets[rel] = f.read()
    return assets


def compare_xml_to_reference(xml_string):
    """Diff our merged XML against the sensor-stripped reference.

    Returns a list of mismatch strings; empty means the ported XML compiles to
    the reference model exactly.
    """
    import mujoco
    from mjmodel_diff import diff_models

    return diff_models(model(), mujoco.MjModel.from_xml_string(xml_string))
