from .state import State
from .action import Action
from .env import Env
from .renderer import Renderer
from .env_traits import (
    DiscreteStateEnv,
    ContinuousStateEnv,
    DiscreteActionEnv,
    ContinuousActionEnv,
    DiscreteEnv,
    BoxDiscreteActionEnv,
    BoxContinuousActionEnv,
    GPUDiscreteEnv,
    GPUContinuousEnv,
)
from .space import Space, DiscreteSpace, BoxSpace
from .agent import Agent
from .tabular_agent import TabularAgent
from .replay_buffer import Transition, ReplayBuffer, PrioritizedReplayBuffer
from .continuous_replay_buffer import (
    ContinuousTransition,
    ContinuousReplayBuffer,
)
from .metrics import (
    EpisodeMetrics,
    TrainingMetrics,
    compute_success_rate,
    compute_convergence_episode,
)
from .tile_coding import TileCoding, TiledWeights
from .linear_fa import (
    LinearWeights,
    PolynomialFeatures,
    RBFFeatures,
    make_grid_rbf_centers,
    make_mountain_car_poly_features,
    FeatureExtractor,
)

from .vec_env import (
    VecStepResult,
    simd_splat_f64,
    simd_splat_i32,
    simd_eq_i32,
    simd_ge_i32,
    simd_lt_f64,
    simd_gt_f64,
    simd_or,
    random_simd,
    random_simd_centered,
    simd_any,
    simd_all,
    simd_count_true,
)

# Shared RL utilities (import from submodules to avoid naming conflicts)
# Usage: from core.utils.softmax import softmax
#        from core.utils.gae import compute_gae
#        from core.utils.normalization import normalize
#        from core.utils.shuffle import shuffle_indices
from . import utils
