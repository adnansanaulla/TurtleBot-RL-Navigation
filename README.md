# TurtleBot-RL-Navigation

Train a TurtleBot3 to navigate using reinforcement learning and PyTorch

## Overview

This project trains a TurtleBot3 robot to navigate through a maze using DQN agent. It uses laser scan data and outputs movement commands to reach its goal.

## Key Technologies

Ros Noetic
Gazebo Sim
TurtleBot3 Burger
PyTorch
Custom Maze World

## Installation

### Prerequisites

Ubuntu 20,04
ROS Noetic
Gazebo 11
Python 3.8+

### Install ROS packages

sudo apt update
sudo apt install-ros-noetic-turtlebot3\* ros-noetic-gazebo-ros-pkgs ros-noetic-xacro

### Clone and build the workspace

cd ~/catkin_ws/src
git clone https://github.com/adnansanaulla/TurtleBot-RL-Navigation.git turtlebot_rl
cd ~/catkin_ws
catkin_make
source devel/setup.bash

### Set TurtleBot3 model

echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc

### Python dependencies

pip3 install -r ~/catkin_ws/src/turtlebot_rl/requirements.txt

## Training the agent

roscore
// in a new terminal:
roslaunch turtlebot_rl train_world.launch
// in another terminal:
python3 ~/catkin_ws/src/turtlebot_rl/scripts/train.py

## Evaluation

python3 ~/catkin_ws/src/turtlebot_rl/scripts/eval.py
