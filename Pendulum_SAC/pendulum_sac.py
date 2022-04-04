
import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import namedtuple, deque


# Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])

# ReplayBuffer from https://github.com/seungeunrho/minimalRL
class ReplayBuffer():
    def __init__(self, buffer_limit, DEVICE):
        self.buffer = deque(maxlen=buffer_limit)
        self.dev = DEVICE

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float).to(self.dev)
        a_batch = torch.tensor(a_lst, dtype=torch.float).to(self.dev)
        r_batch = torch.tensor(r_lst, dtype=torch.float).to(self.dev)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float).to(self.dev)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float).to(self.dev)

        # r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)


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
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr):
        super(QNetwork, self).__init__()

        self.fc_s = nn.Linear(state_dim, 32)
        self.fc_a = nn.Linear(action_dim, 32)
        self.fc_1 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, action_dim)

        self.lr = critic_lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):
        h1 = F.leaky_relu(self.fc_s(x))
        h2 = F.leaky_relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=-1)
        q = F.leaky_relu(self.fc_1(cat))
        q = self.fc_out(q)
        return q


class SAC_Agent:
    def __init__(self):
        self.state_dim      = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim     = 1  # [torque] in[-2,2]
        self.lr_pi          = 0.001
        self.lr_q           = 0.001
        self.gamma          = 0.98
        self.batch_size     = 200
        self.buffer_limit   = 100000
        self.tau            = 0.005   # for soft-update of Q using Q-target
        self.init_alpha     = 0.01
        self.target_entropy = -self.action_dim  # == -1
        self.lr_alpha       = 0.005
        self.DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory         = ReplayBuffer(self.buffer_limit, self.DEVICE)
        print("사용 장치 : ", self.DEVICE)

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self.PI  = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.DEVICE)
        self.Q1        = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2        = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def choose_action(self, s):
        with torch.no_grad():
            action, log_prob = self.PI.sample(s.to(self.DEVICE))
        return action, log_prob

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            a_prime, log_prob_prime = self.PI.sample(s_prime)
            entropy = - self.log_alpha.exp() * log_prob_prime
            q1_target, q2_target = self.Q1_target(s_prime, a_prime), self.Q2_target(s_prime, a_prime)
            q_target = torch.min(q1_target, q2_target)
            target = r + self.gamma * done * (q_target + entropy)
        return target

    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch

        td_target = self.calc_target(mini_batch)

        #### Q1 train ####
        q1_loss = F.smooth_l1_loss(self.Q1(s_batch, a_batch), td_target)
        self.Q1.optimizer.zero_grad()
        q1_loss.mean().backward()
        # nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.Q1.optimizer.step()
        #### Q1 train ####

        #### Q2 train ####
        q2_loss = F.smooth_l1_loss(self.Q2(s_batch, a_batch), td_target)
        self.Q2.optimizer.zero_grad()
        q2_loss.mean().backward()
        # nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.Q2.optimizer.step()
        #### Q2 train ####

        #### pi train ####
        a, log_prob = self.PI.sample(s_batch)
        entropy = -self.log_alpha.exp() * log_prob

        q1, q2 = self.Q1(s_batch, a), self.Q2(s_batch, a)
        q = torch.min(q1, q2)

        pi_loss = -(q + entropy)  # for gradient ascent
        self.PI.optimizer.zero_grad()
        pi_loss.mean().backward()
        # nn.utils.clip_grad_norm_(self.pi.parameters(), 2.0)
        self.PI.optimizer.step()
        #### pi train ####

        #### alpha train ####
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        #### alpha train ####

        #### Q1, Q2 soft-update ####
        for param_target, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        #### Q1, Q2 soft-update ####


if __name__ == '__main__':

    ###### logging ######
    log_name = '0404'

    model_save_dir = 'saved_model/' + log_name
    if not os.path.isdir(model_save_dir): os.mkdir(model_save_dir)
    log_save_dir = 'log/' + log_name
    if not os.path.isdir(log_save_dir): os.mkdir(log_save_dir)
    ###### logging ######

    env = gym.make('Pendulum-v1')
    agent = SAC_Agent()

    EPISODE = 500
    print_once = True
    score_list = []

    for EP in range(EPISODE):
        state = env.reset()
        score, done = 0.0, False

        while not done:
            action, log_prob = agent.choose_action(torch.FloatTensor(state))
            action = action.detach().cpu().numpy()  # GPU에 있는 텐서를 CPU로 옮기고 넘파이로 변환

            state_prime, reward, done, _ = env.step(action)

            agent.memory.put((state, action, reward, state_prime, done))

            score += reward

            state = state_prime

            if agent.memory.size() > 1000:  # 1000개의 [s,a,r,s']이 쌓이면 학습 시작
                if print_once: print("학습시작!")
                print_once = False
                agent.train_agent()

        print("EP:{}, Avg_Score:{:.1f}".format(EP, score))
        score_list.append(score)

        if EP % 10 == 0:
            torch.save(agent.PI.state_dict(), model_save_dir + "/sac_actor_EP"+str(EP)+".pt")

    np.savetxt(log_save_dir + '/pendulum_score.txt', score_list)

