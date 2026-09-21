"""`libero_goal` on the batched GPU env, driven by OSC_POSE.

    var env = LiberoGoalOscEnv[N](ctx, seed)
    env.set_osc_refs(build_osc_refs(...), ctx)     # from the parsed scene
    env.reset_batch[N](ctx, seed)
    env.step_batch[N](ctx, seed)

The config is `libero_osc_config.LiberoOscConfig` over this family's generated
placement table; everything it does, and why, is documented there. This module
keeps the `libero_goal` names every driver and gate already imports.
"""

from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.libero.osc_config import (
    LiberoOscConfig, LIBERO_FRAME_SKIP, LIBERO_HORIZON, LIBERO_TIMESTEP,
)
from noeira.envs.libero.placement.libero_goal import LiberoGoalPlacement
from noeira.envs.libero.models.libero_goal_xml import LiberoGoalModel


comptime LIBERO_GOAL_FRAME_SKIP: Int = LIBERO_FRAME_SKIP
comptime LIBERO_GOAL_HORIZON: Int = LIBERO_HORIZON
comptime LIBERO_GOAL_TIMESTEP: Float64 = LIBERO_TIMESTEP

comptime LiberoGoalOscConfig = LiberoOscConfig[LiberoGoalPlacement]

comptime LiberoGoalOscEnv = Phyics3dBatchedEnv[
    LiberoGoalModel, LiberoGoalOscConfig, _, CRBA_TREEWALK=True
]
"""The batched env, parameterised on the lane count.

⚠⚠ IT RUNS ON METAL — THE "nv = 37 STACK" EXPLANATION WAS WRONG. Both the
Metal "failed to compile metallib" and NVIDIA's ptxas "Unresolved extern
function 'KGEN_CompilerRT_GetOrCreateGlobal'" were the elliptic Newton branch
building eight `ScratchPool`-backed scratches in the kernel, because
`cap[]` is 0 for a model with no tendons and no equalities
(`newton_solve.mojo`, `EQ_CAP`). With that guarded,
`examples/libero/libero_osc_batched.mojo` builds and steps on an M1 Pro."""
