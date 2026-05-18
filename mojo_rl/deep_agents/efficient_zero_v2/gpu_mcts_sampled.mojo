"""EZv2 GPU sampled-Gumbel MCTS — **re-export shim**.

The actual implementation now lives at
``mojo_rl/planners/tree_search/mcts_gpu_gumbel_sampled.mojo``. This module
re-exports the same public surface so callers that imported the kernels or
``EZV2GPUSampledMCTSState`` from this path keep working — strangler
pattern, mirrors the discrete move from earlier in Phase 3.

New code should prefer the planners-package import:

    from mojo_rl.planners.tree_search import (
        EZV2GPUSampledMCTSState,
        run_sampled_gumbel_search_gpu,
    )
"""

from mojo_rl.planners.tree_search.mcts_gpu_gumbel_sampled import (
    MAX_DEPTH,
    EZV2GPUSampledMCTSState,
    gs_scatter_root_hidden_kernel,
    gs_init_root_kernel,
    gs_select_kernel,
    gs_copy_pred_input_kernel,
    gs_expand_kernel,
    gs_backup_kernel,
    gs_halve_active_kernel,
    gs_extract_kernel,
    run_sampled_gumbel_search_gpu,
)
