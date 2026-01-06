from envs import GridWorld
from agents import QLearningAgent

fn main():
    var env = GridWorld(width=5, height=5)
    var agent = QLearningAgent(env.num_states(), env.num_actions())
    print("test")
