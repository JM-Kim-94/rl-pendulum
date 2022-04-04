
import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, q_lr):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, action_dim)

        self.lr = q_lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        q = F.leaky_relu(self.fc_1(x))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q


class DQNAgent:
    def __init__(self, wight_file_path):
        self.state_dim     = 3
        self.action_dim    = 9
        self.lr            = 0.002
        self.trained_model = wight_file_path

        self.Q        = QNetwork(self.state_dim, self.action_dim, self.lr)
        self.Q.load_state_dict(torch.load(self.trained_model))

    def choose_action(self, state):
        with torch.no_grad():
            action = float(torch.argmax(self.Q(state)).numpy())
            real_action = (action - 4) / 2
        return real_action


if __name__ == '__main__':

    log_name = '0404/'
    weight_name = 'DQN_Q_EP410.pt'

    weight_file_path = 'saved_model/' + log_name + weight_name
    agent = DQNAgent(weight_file_path)

    env = gym.make('Pendulum-v1')

    state = env.reset()
    step = 0

    while True:
        env.render()
        real_action = agent.choose_action(torch.FloatTensor(state))

        state_prime, reward, done, _ = env.step([real_action])
        step += 1

        state = state_prime

        if step % 400 == 0:
            state = env.reset()
            print("환경 리셋 , 총 스텝=", step)

