import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import yaml

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma, learning_rate, batch_size):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.memory = []
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.loss_fn = nn.MSELoss()
        self.update_target()
    def act(self, state, epsilon = 0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            return torch.argmax(self.model(state)).item()
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory_pop(0)
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)
        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_model(next.states).max(1, keepdim = True)[0]
            target_q =  rewards + (1 - dones.float()) * self.gamma * max_next_q
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())