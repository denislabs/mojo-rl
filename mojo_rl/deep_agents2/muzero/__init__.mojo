# MuZero on nn2 + deep_agents2 (Phase B). Learned model h/g/f + K-step unroll
# BPTT + MCTS planning (learned dynamics) + n-step targets w/ two-player sign
# flips. Builds on the shared `deep_agents2/zero/` infrastructure.

from .nets import MZRepNet, MZDynNet, MZPredNet
from .blocks import mz_unroll_train_step_cpu, mz_unroll_train_step_gpu
from .config import MuZeroMLPConfig
from .agent import MuZeroAgent
from .selfplay_cpu import run_muzero_selfplay_cpu
from .selfplay_gpu import run_muzero_selfplay_gpu, mz_sync_gpu_to_cpu
from .selfplay_gpu_device import (
    run_muzero_selfplay_gpu_device,
    run_muzero_gumbel_selfplay_gpu,
)
from .selfplay_2p_cpu import run_muzero_selfplay_2p_cpu
