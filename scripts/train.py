import sys
import os
from agents.dqn_agent import DQNAgent
from env.turtlebot_env import TurtleBotEnv
import numpy as np
import yaml
## training
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
with open(os.path.join(PROJECT_ROOT, "config", "params.yaml"), "r") as f:
    config = yaml.safe_load(f)

env = TurtleBotEnv()
agent = DQNAgent(state_dim = 36, action_dim = 3, gamma = config["agent"]["gamma"], learning_rate = config["agent"]["learning_rate"], batch_size = config["agent"]["batch_size"])

for episode in range(500):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    if episode % 10 == 0:
        agent.update_target()
    print(f"Episode {episode}: Total Reward = {total_reward}")