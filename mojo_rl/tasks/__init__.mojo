"""`mojo_rl.tasks` — declarative tasks over the physics3d scene composer.

See `docs/TASK_LAYER_PLAN.md` (design) and
`docs/TASK_LAYER_IMPLEMENTATION.md` (what is built).

⚠ THE DEPENDENCY IS ONE-WAY, and §7 of the plan says to keep it that way:
`tasks/` CALLS `physics3d/studio`'s composer and never reimplements it, and
`physics3d` never imports `tasks` — or the engine stops being usable without
the task layer.

`tasks/` also produces a VALUE consumed by `Phyics3dEnvConfig`. It does not
implement `Env` and knows nothing about agents.
"""

from .spec import (
    FamilySpec, TaskSpec, SlotSpec, RegionSpec, InitSpec,
    parse_family, parse_task, load_family, load_task,
    validate_task_against_family,
    SLOT_FREE, SLOT_STATIC, slot_kind_name, slot_kind_from_name,
    SCHEMA_VERSION,
)
from .family import (
    compose_family, write_family_scene, scene_path, park_pos,
    SCENE_DIR, BASE_PREFIX, PARK_SPACING,
)
from .predicates import (
    Goal, BoundGoal, GoalTerm, BoundTerm,
    parse_goal, bind_goal, require_tier_a, slot_body_id, site_id,
    op_name, op_arity, op_is_tier_a, op_is_composite,
    MAX_GOAL_TERMS,
)
from .sampler import (
    Placement, RegionFrame, SampleReport,
    sample_placements, MAX_PLACE_ATTEMPTS, PLACEMENT_SALT,
)
from .eval import (
    eval_goal, region_sites,
    pred_in_rect, pred_near, pred_above, pred_upright,
)
