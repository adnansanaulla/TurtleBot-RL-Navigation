import os
import sys
import numpy as np
import torch
import rospy
from agents.dqn_agent import DQNAgent
from env.turtlebot_env import TurtleBotEnv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
## evaluation
env = TurtleBotEnv()
agent = DQNAgent(state_dim = 36, action_dim = 3)
agent.model.load_state_dict(torch.load("models/checkpoint.pth"))
print("Starting Eval")
num_eval_episodes = 10
for episode in range(num_eval_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done and not rospy.is_shutdown():
        action = agent.act(state, epsilon = 0)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward = total_reward + reward
        step += 1
    print (f"Eval Episode {episode}: Total Reward = {total_reward}, Steps = {step}")