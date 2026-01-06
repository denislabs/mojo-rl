from envs import GridWorldEnv
from agents import QLearningAgent


fn main():
    print("=== Q-Learning on GridWorld ===")

    var env = GridWorldEnv(width=5, height=5)

    print("Training...")
    var agent = QLearningAgent(env.num_states(), env.num_actions())
    var _ = agent.train(env, num_episodes=500, verbose=True)

    print("\nEvaluating...")
    var eval_reward = agent.evaluate(env, num_episodes=10)
    print("Evaluation avg reward:", eval_reward)
