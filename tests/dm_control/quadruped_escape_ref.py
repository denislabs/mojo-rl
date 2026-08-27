"""Reference `Escape` observation and reward, for `test_quadruped_escape_*`.

⚠⚠ THE TERRAIN IS NOT GENERATED HERE. `Escape.initialize_episode` builds it
with `scipy.ndimage.zoom(order=3)`, and our port uses bilinear interpolation
instead — a LABELLED deviation (see `quadruped_escape_config.custom_reset_full_cpu`).
So a per-episode terrain cannot be compared. What CAN be compared, and is
everything else, is the observation and the reward GIVEN a terrain: the caller
writes the same grid into both engines and this evaluates dm_control's own
formulas on it.

`dm_control.utils.rewards` is imported for real rather than transcribed — it is
pure numpy and it is the actual reference. `Physics.rangefinder`,
`Physics.origin` and `Escape.get_reward` are copied, because
`dm_control.suite.quadruped` is not importable here (`dm_env` is missing and
`suite/__init__` imports it at module scope) — the same constraint
`quadruped_ref.py` documents.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (
    os.path.join(_REPO, "references", ".dmc_deps"),
    os.path.join(_REPO, "references", "dm_control-main"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from dm_control.utils import rewards  # noqa: E402

ESCAPE_XML = "mojo_rl/envs/dm_control/assets/quadruped_escape.xml"
_HEIGHTFIELD_ID = 0


def load():
    """`(model, data)` for the shipped escape asset."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(os.path.join(_REPO, ESCAPE_XML))
    return m, mujoco.MjData(m)


def set_terrain(m, grid):
    """Write `grid` (a flat list, `nrow*ncol`) into `m.hfield_data`."""
    res = int(m.hfield_nrow[_HEIGHTFIELD_ID])
    adr = int(m.hfield_adr[_HEIGHTFIELD_ID])
    assert len(grid) == res * res, (len(grid), res * res)
    m.hfield_data[adr:adr + res * res] = np.asarray(grid, dtype=np.float32)


def rangefinder(m, d):
    """`Physics.rangefinder` — ⚠ a MISS becomes 1.0, not tanh(-1)."""
    import mujoco

    ids = [
        i for i in range(m.nsensor)
        if int(m.sensor_type[i]) == int(mujoco.mjtSensor.mjSENS_RANGEFINDER)
    ]
    raw = np.array([d.sensordata[int(m.sensor_adr[i])] for i in ids])
    return np.where(raw == -1.0, 1.0, np.tanh(raw))


def origin(m, d):
    """`Physics.origin` — the world origin in the TORSO frame."""
    import mujoco

    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    frame = d.xmat[b].reshape(3, 3)
    return -d.xpos[b].dot(frame)


def origin_distance(m, d):
    import mujoco

    s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "workspace")
    return float(np.linalg.norm(d.site_xpos[s]))


def torso_upright(m, d):
    import mujoco

    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    return float(d.xmat[b].reshape(3, 3)[2, 2])


def escape_reward(m, d):
    """`Escape.get_reward` — `_upright_reward(20) * escape_reward`."""
    terrain_size = float(m.hfield_size[_HEIGHTFIELD_ID, 0])
    esc = rewards.tolerance(
        origin_distance(m, d),
        bounds=(terrain_size, float("inf")),
        margin=terrain_size,
        value_at_margin=0,
        sigmoid="linear",
    )
    deviation = np.cos(np.deg2rad(20))
    upright = rewards.tolerance(
        torso_upright(m, d),
        bounds=(deviation, float("inf")),
        sigmoid="linear",
        margin=1 + deviation,
        value_at_margin=0,
    )
    return float(upright * esc)


def site_ids(m):
    """The indices the port hardcodes, resolved by NAME."""
    import mujoco

    rf = [
        int(m.sensor_objid[i]) for i in range(m.nsensor)
        if int(m.sensor_type[i]) == int(mujoco.mjtSensor.mjSENS_RANGEFINDER)
    ]
    return dict(
        workspace=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "workspace"),
        rf_first=rf[0],
        n_rf=len(rf),
        torso_body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso"),
        terrain_geom=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain"),
    )
