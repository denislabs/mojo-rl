"""physics3d studio — the authoring tool's own pieces.

Deliberately NOT built on `Phyics3dEnv`: an env is the RL contract (obs /
reward / done / action space) and a composed scene is not a task. Forcing the
studio through it would mean inventing an observation and a reward for a table
with two cubes on it — fabricating exactly what the user is meant to author
later. The env belongs to the BAKE phase, where a scene becomes a task.
See `docs/PHYSICS3D_STUDIO_PLAN.md` §5.1.
"""

from .pick import Ray, Hit, ray_through_pixel, pick_geom
from .panel import StudioPanel, PanelOut, build_panel
