from agents.dqn_agent import DQNAgent
from env.turtlebot_env import TurtleBotEnv
import numpy as np

env = TurtleBotEnv()
agent = DQNAgent(state_dim = 36, action_dim = 3)

for episode in range(500):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        # agent.store(state, action, reward, next_state, done)
        # agent.train()
        state = next_state
        total_reward += reward
    print(f"Episode {episode}: Total Reward = {total_reward}")