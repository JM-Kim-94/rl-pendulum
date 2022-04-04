
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


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr):
        super(PolicyNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, action_dim)
        self.fc_std = nn.Linear(64, action_dim)

        self.lr = actor_lr

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = 2
        self.min_action = -2
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        # x_t = reparameter.rsample()  # not using stochastic actor for testing
        x_t = mean                     # using deterministic actor for testing
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)

        return action, log_prob


class SAC_Agent:
    def __init__(self, weight_file_path):
        self.trained_model  = weight_file_path
        self.state_dim      = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim     = 1  # [torque] in[-2,2]
        self.lr_pi          = 0.001
        self.DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("사용 장치 : ", self.DEVICE)

        self.PI  = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.DEVICE)
        self.PI.load_state_dict(torch.load(self.trained_model))

    def choose_action(self, s):
        with torch.no_grad():
            action, log_prob = self.PI.sample(s.to(self.DEVICE))
        return action, log_prob


if __name__ == '__main__':

    log_name = '0404/'
    weight_name = 'sac_actor_EP320.pt'

    weight_file_path = 'saved_model/' + log_name + weight_name
    agent = SAC_Agent(weight_file_path)

    env = gym.make('Pendulum-v1')

    state = env.reset()
    step = 0

    while True:
        env.render()
        action, log_prob = agent.choose_action(torch.FloatTensor(state))
        action = action.detach().cpu().numpy()  # GPU에 있는 텐서를 CPU로 옮기고 넘파이로 변환

        state_prime, reward, done, _ = env.step(action)
        step += 1

        state = state_prime

        if step % 200 == 0:
            state = env.reset()
            print("환경 리셋 , 총 스텝=", step)

