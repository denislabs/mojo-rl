"""physics3d studio — the authoring tool's own pieces.

Deliberately NOT built on `Phyics3dEnv`: an env is the RL contract (obs /
reward / done / action space) and a composed scene is not a task. Forcing the
studio through it would mean inventing an observation and a reward for a table
with two cubes on it — fabricating exactly what the user is meant to author
later. The env belongs to the BAKE phase, where a scene becomes a task.
See `docs/PHYSICS3D_STUDIO_PLAN.md` §5.1.
"""

from .pick import Ray, Hit, ray_through_pixel, pick_geom
from .outline import outline_geom, outline_body, SELECT_COLOR
from .panel import StudioPanel, PanelOut, build_ui, SIDEBAR_W, RIGHT_W
from .scene import SceneDoc, Instance, scene_from_base
from .writer import to_mjcf as export_mjcf, unwritable
from .validate import (
    Diagnostic, validate_document, validate_model, validate_all,
    worst_severity, count_at,
    format_diagnostic, severity_name, SEV_INFO, SEV_WARN, SEV_ERROR,
)
from .structure import (
    EditResult, delete_body, delete_geom, delete_joint, delete_site,
    delete_element, leftover_dangling, find_named,
)
from .remap import remap_state, RemapReport, joint_qpos_adr, joint_dof_adr
from .edit import (
    Edit, EditLog, apply_edit, needs_rebuild, field_name,
    TARGET_GEOM, TARGET_BODY,
)
