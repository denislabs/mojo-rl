"""MuZero GPU MCTS — re-export shim.

The actual implementation lives in
``mojo_rl/planners/tree_search/mcts_gpu.mojo``. This file remains as a
source-compatibility shim so existing imports keep working through the
Phase 3b strangler migration:

    from mojo_rl.deep_agents.muzero.gpu_mcts import (
        GPUMCTSState, gpu_mcts_init_root_kernel, ...
    )

See ``docs/PLANNERS_PACKAGE.md`` Phase 3b for the migration plan. When
the MuZero agent's ``select_action_gpu`` is rewired to a generic
``GenericGPUMCTS`` orchestrator (and the analogous EZv2 / AlphaZero
rewires land), this shim can be deleted.
"""

from mojo_rl.planners.tree_search.mcts_gpu import (
    TPB,
    MAX_DEPTH,
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_select_kernel,
    gpu_mcts_expand_kernel,
    gpu_mcts_backup_kernel,
    gpu_mcts_backup_negated_kernel,
    gpu_mcts_extract_actions_kernel,
    gpu_mcts_extract_root_value_kernel,
    gpu_mcts_apply_legal_mask_kernel,
    gpu_mcts_apply_legal_mask_with_noise_kernel,
    gpu_mcts_extract_actions_masked_kernel,
    gpu_mcts_extract_actions_temp_kernel,
    gpu_mcts_copy_parent_state_kernel,
    gpu_mcts_store_child_state_kernel,
    gpu_mcts_copy_root_state_kernel,
    gpu_mcts_expand_alphazero_kernel,
    gpu_mcts_batched_select_and_copy_kernel,
    gpu_mcts_batched_expand_backup_kernel,
    gpu_mcts_batched_expand_backup_masked_kernel,
    gpu_mcts_batched_select_and_build_dyn_kernel,
    gpu_mcts_batched_expand_backup_muzero_kernel,
    gpu_mcts_build_dyn_input_kernel,
    gpu_mcts_copy_pred_input_kernel,
)
