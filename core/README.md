# core/ - Core RL Abstractions

Trait definitions, replay buffers, function approximation, and shared utilities for all RL algorithms.

## Module Structure

```
core/
├── state.mojo              # State trait (Copyable, Movable, ImplicitlyCopyable)
├── action.mojo             # Action marker trait
├── env.mojo                # Env trait (step, reset, get_state, render, close)
├── agent.mojo              # Agent trait (select_action, update, reset)
├── obs_state.mojo          # ObsState[N]: generic N-dimensional observation wrapper
├── cont_action.mojo        # ContAction[N]: generic N-dimensional continuous action
├── env_renderer.mojo       # Renderer trait (ref-based safe borrowing)
├── space.mojo              # DiscreteSpace, BoxSpace[dim]
├── tabular_agent.mojo      # TabularAgent trait (Q-table agents)
├── env_traits.mojo         # Environment trait hierarchy
│                            #   DiscreteEnv, BoxDiscreteActionEnv, BoxContinuousActionEnv,
│                            #   GPUDiscreteEnv, GPUContinuousEnv, CurriculumScheduler
├── replay_buffer.mojo      # Transition, ReplayBuffer, PrioritizedReplayBuffer
├── continuous_replay_buffer.mojo # ContinuousTransition, ContinuousReplayBuffer
├── tile_coding.mojo        # TileCoding: multi-tiling function approximation
├── linear_fa.mojo          # LinearWeights, FeatureExtractor, PolynomialFeatures, RBFFeatures
├── metrics.mojo            # EpisodeMetrics, TrainingMetrics
├── vec_env.mojo            # VecStepResult, vectorized environment support
├── sum_tree.mojo           # SumTree for O(log n) priority sampling
└── hyperparam/             # Hyperparameter search infrastructure
    ├── param_space.mojo    # TabularParamSpace, NStepParamSpace, etc.
    ├── search_result.mojo  # TrialResult, SearchResults (CSV export)
    └── agent_factories.mojo # Factory functions for agent creation
```

## Key Traits

| Trait | Purpose |
|-------|---------|
| `State` | Base environment state (equality, copyability) |
| `Action` | Marker trait for environment actions |
| `Env` | Generic environment (step, reset, render) |
| `Agent` | Generic agent (select_action, update) |
| `TabularAgent` | Discrete state/action agent (Q-table lookup) |
| `DiscreteEnv` | Combined discrete state + discrete action |
| `BoxDiscreteActionEnv` | Continuous observations + discrete actions |
| `BoxContinuousActionEnv` | Continuous observations + continuous actions |
| `GPUDiscreteEnv` | GPU-batchable discrete action environment |
| `GPUContinuousEnv` | GPU-batchable continuous action environment |
| `RenderableEnv` | Environment with rendering support |
| `CurriculumScheduler` | Curriculum learning support |

## Function Approximation

- **TileCoding**: Multi-dimensional overlapping tilings with asymmetric offsets. Factories for CartPole, MountainCar, Acrobot.
- **LinearWeights**: Dense weight vectors for arbitrary feature extractors
- **PolynomialFeatures**: Polynomial expansion (x, y, x^2, xy, ...)
- **RBFFeatures**: Radial Basis Functions with configurable centers
