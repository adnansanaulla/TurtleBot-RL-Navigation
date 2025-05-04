import os
import sys
import numpy as np
import rospy
from agents.dqn_agent import DQNAgent
from env.turtlebot_env import TurtleBotEnv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)