import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

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
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)
        self.memory = []
        self.gamma = 0.99
        self.batch_size = 64
    def act(self, state, epsilon = 0.1):
        if random.random() < epsilon:
            return random.randint(0, 2)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            return torch.argmax(self.model(state)).item()
    # add train and update_target functions