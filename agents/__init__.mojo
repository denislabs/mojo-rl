from .qlearning import QTable, QLearningAgent
from .sarsa import SARSAAgent
from .expected_sarsa import ExpectedSARSAAgent
from .nstep_sarsa import NStepSARSAAgent
from .sarsa_lambda import SARSALambdaAgent
from .monte_carlo import MonteCarloAgent
from .double_qlearning import DoubleQLearningAgent
from .dyna_q import DynaQAgent
from .priority_sweeping import PrioritySweepingAgent
from .qlearning_replay import QLearningReplayAgent
from .tiled_qlearning import TiledQLearningAgent, TiledSARSAAgent, TiledSARSALambdaAgent
from .linear_qlearning import LinearQLearningAgent, LinearSARSAAgent, LinearSARSALambdaAgent
from .reinforce import REINFORCEAgent, REINFORCEWithEntropyAgent
from .actor_critic import ActorCriticAgent, ActorCriticLambdaAgent, A2CAgent
from .ppo import PPOAgent, PPOAgentWithMinibatch, compute_gae
