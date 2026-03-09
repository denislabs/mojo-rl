# agents/ - Tabular & Linear RL Algorithms

Classical RL algorithm implementations using Q-tables, tile coding, linear function approximation, and policy gradient methods.

## Algorithms

### TD Methods

| Agent | File | Description |
|-------|------|-------------|
| `QLearningAgent` | `qlearning.mojo` | Off-policy: Q(s,a) += alpha(r + gamma*max Q(s',a') - Q(s,a)) |
| `SARSAAgent` | `sarsa.mojo` | On-policy: Q(s,a) += alpha(r + gamma*Q(s',a') - Q(s,a)) |
| `ExpectedSARSAAgent` | `expected_sarsa.mojo` | Uses E[Q(s',a')] under policy |
| `DoubleQLearningAgent` | `double_qlearning.mojo` | Two Q-tables to reduce overestimation |

### Multi-Step & Eligibility Traces

| Agent | File | Description |
|-------|------|-------------|
| `NStepSARSAAgent` | `nstep_sarsa.mojo` | Configurable n-step returns |
| `SARSALambdaAgent` | `sarsa_lambda.mojo` | Eligibility traces (replacing traces) |
| `MonteCarloAgent` | `monte_carlo.mojo` | First-visit MC with episode returns |

### Model-Based Planning

| Agent | File | Description |
|-------|------|-------------|
| `DynaQAgent` | `dyna_q.mojo` | Q-Learning + simulated experience from learned model |
| `PrioritySweepingAgent` | `priority_sweeping.mojo` | Prioritized updates by TD error magnitude |

### With Experience Replay

| Agent | File | Description |
|-------|------|-------------|
| `QLearningReplayAgent` | `qlearning_replay.mojo` | Off-policy with uniform replay buffer |
| `QLearningPERAgent` | `qlearning_per.mojo` | Off-policy with prioritized replay |

### Function Approximation (Tile Coding)

| Agent | File | Description |
|-------|------|-------------|
| `TiledQLearningAgent` | `tiled_qlearning.mojo` | Q-Learning with tile coding |
| `TiledSARSAAgent` | `tiled_qlearning.mojo` | On-policy SARSA with tile coding |
| `TiledSARSALambdaAgent` | `tiled_qlearning.mojo` | SARSA(lambda) with tile coding |

### Function Approximation (Linear)

| Agent | File | Description |
|-------|------|-------------|
| `LinearQLearningAgent` | `linear_qlearning.mojo` | Q-Learning with polynomial/RBF features |
| `LinearSARSAAgent` | `linear_qlearning.mojo` | On-policy SARSA with arbitrary features |
| `LinearSARSALambdaAgent` | `linear_qlearning.mojo` | SARSA(lambda) with linear FA |

### Policy Gradient Methods

| Agent | File | Description |
|-------|------|-------------|
| `REINFORCEAgent` | `reinforce.mojo` | Monte Carlo policy gradient with optional baseline |
| `REINFORCEWithEntropyAgent` | `reinforce.mojo` | REINFORCE + entropy regularization |
| `ActorCriticAgent` | `actor_critic.mojo` | One-step TD Actor-Critic |
| `ActorCriticLambdaAgent` | `actor_critic.mojo` | Actor-Critic with eligibility traces |
| `A2CAgent` | `actor_critic.mojo` | Advantage Actor-Critic with n-step returns |
| `PPOAgent` | `ppo.mojo` | PPO with clipped surrogate + GAE |
| `PPOAgentWithMinibatch` | `ppo.mojo` | PPO with minibatch sampling |

### Continuous Control (Linear FA)

| Agent | File | Description |
|-------|------|-------------|
| `DDPGAgent` | `ddpg.mojo` | Deterministic actor + Q-critic with target networks |
| `TD3Agent` | `td3.mojo` | Twin critics, delayed policy, target smoothing |
| `SACAgent` | `sac.mojo` | Max entropy RL, stochastic Gaussian policy, auto alpha |

## Common Design Patterns

- All tabular agents implement `TabularAgent` trait and use shared `QTable` structure
- Epsilon-greedy exploration with configurable decay: `epsilon *= epsilon_decay` each episode
- All agents have `train()` and `evaluate()` methods
- Policy gradient agents use tile coding for continuous state spaces
