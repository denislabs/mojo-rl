"""EZv2 GPU Gumbel-search MCTS — **re-export shim**.

The actual implementation now lives at
``mojo_rl/planners/tree_search/mcts_gpu_gumbel.mojo``. This module
re-exports the same public surface so callers that imported the kernels
or the ``EZV2GPUMCTSState`` struct from this path keep working — strangler
pattern, mirrors the MuZero GPU MCTS move done in Phase 3b.

New code should prefer the planners-package import:

    from mojo_rl.planners.tree_search import (
        EZV2GPUMCTSState,
        run_gumbel_search_gpu,
    )
"""

from mojo_rl.planners.tree_search.mcts_gpu_gumbel import (
    MAX_DEPTH,
    EZV2GPUMCTSState,
    gz_scatter_root_hidden_kernel,
    gz_init_root_kernel,
    gz_select_kernel,
    gz_copy_pred_input_kernel,
    gz_expand_kernel,
    gz_backup_kernel,
    gz_halve_active_kernel,
    gz_extract_policy_kernel,
    run_gumbel_search_gpu,
)
