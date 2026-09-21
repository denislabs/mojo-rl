# MuZero on nn + deep_agents (Phase B). Learned model h/g/f + K-step unroll
# BPTT + MCTS planning (learned dynamics) + n-step targets w/ two-player sign
# flips. Builds on the shared `deep_agents/zero/` infrastructure.

from .nets import MZRepNet, MZRepNetCNN, MZRepNetC4Conv, MZDynNet, MZPredNet
from .nets_spatial import (
    MZRepNetC4Spatial,
    MZDynNetC4Spatial,
    MZPredNetC4Spatial,
)
from .blocks import mz_unroll_train_step_cpu, mz_unroll_train_step_gpu
from .config import MuZeroMLPConfig, MuZeroCNNConfig
from .agent import MuZeroAgent
from .batched_agent import MuZeroBatchedAgent
from .selfplay_cpu import run_muzero_selfplay_cpu
from .selfplay_gpu import run_muzero_selfplay_gpu, mz_sync_gpu_to_cpu
from .selfplay_gpu_device import (
    run_muzero_selfplay_gpu_device,
    run_muzero_gumbel_selfplay_gpu,
)
from .selfplay_gpu_batched import (
    run_muzero_gumbel_selfplay_gpu_batched,
    run_muzero_gumbel_selfplay_gpu_batched_devreplay,
)
from .selfplay_2p_cpu import run_muzero_selfplay_2p_cpu
from .selfplay_arena_gumbel_2p import (
    run_muzero_selfplay_arena_gumbel_2p,
    mz_candidate_winrate,
    mz_eval_both_colors,
    MZArenaResult,
    MZArenaRunResult,
)
