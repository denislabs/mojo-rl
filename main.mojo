from core import train_tabular, evaluate_tabular
from envs import GridWorld
from agents import QLearningAgent


fn main():
    print("=== Q-Learning on GridWorld ===")

    var env = GridWorld(width=5, height=5)
    var agent = QLearningAgent(num_states=25, num_actions=4)

    print("Training...")
    _ = train_tabular(env, agent, num_episodes=500, verbose=True)

    print("\nEvaluating...")
    var eval_reward = evaluate_tabular(env, agent, num_episodes=10)
    print("Evaluation avg reward:", eval_reward)
