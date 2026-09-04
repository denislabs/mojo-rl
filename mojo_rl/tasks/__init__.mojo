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
    region_rects,
)
from .tape import (
    encode_goal, eval_tape, TAPE_WORDS, TERM_WORDS, MAX_TAPE_TERMS,
)
from .gpu_eval import (
    eval_tape_gpu, region_table_words,
    CUR_IDX_REGION_SITE, CUR_IDX_REGION_X0, CUR_IDX_REGION_Y0,
    CUR_IDX_REGION_X1, CUR_IDX_REGION_Y1, MAX_CURRICULUM_REGIONS,
)
from .reset import (
    SlotAddress, free_slot_addresses, reset_slots,
    write_free_pose, write_free_vel_zero,
)
from .active import (
    active_mask, mask_slots, MASK_SLOT_LIMIT,
)
from .obs import (
    slot_active, write_free_slot_obs, write_free_slot_obs_host,
    FREE_JOINT_NQ, FREE_JOINT_NV,
)
from .init_table import (
    InitTable, write_init_table, load_init_table, append_init_rows,
    family_key, INIT_COLUMN, TASK_COLUMN, MASK_COLUMN, INIT_TIME_WORDS,
)
from .eval_report import SuccessReport
