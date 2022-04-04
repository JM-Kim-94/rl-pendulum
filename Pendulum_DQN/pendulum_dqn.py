
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
import matplotlib.pyplot as plt


# ReplayBuffer from https://github.com/seungeunrho/minimalRL
class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float)
        a_batch = torch.tensor(a_lst, dtype=torch.float)
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float)

        # r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)


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
    def __init__(self):
        self.state_dim     = 3
        self.action_dim    = 9  # 9개 행동 : -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0
        self.lr            = 0.01
        self.gamma         = 0.98
        self.tau           = 0.01
        self.epsilon       = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min   = 0.001
        self.buffer_size   = 100000
        self.batch_size    = 200
        self.memory        = ReplayBuffer(self.buffer_size)

        self.Q        = QNetwork(self.state_dim, self.action_dim, self.lr)
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

    def choose_action(self, state):
        random_number = np.random.rand()
        maxQ_action_count = 0
        if self.epsilon < random_number:
            with torch.no_grad():
                action = float(torch.argmax(self.Q(state)).numpy())
                # action = float(action.numpy())
                real_action = (action - 4) / 4
                maxQ_action_count = 1
        else:
            action = np.random.choice([n for n in range(9)])
            real_action = (action - 4) / 2  # -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0

        return action, real_action, maxQ_action_count

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            q_target = self.Q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * done * q_target
        return target

    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch
        a_batch = a_batch.type(torch.int64)

        td_target = self.calc_target(mini_batch)

        #### Q train ####
        Q_a = self.Q(s_batch).gather(1, a_batch)
        q_loss = F.smooth_l1_loss(Q_a, td_target)
        self.Q.optimizer.zero_grad()
        q_loss.mean().backward()
        self.Q.optimizer.step()
        #### Q train ####

        #### Q soft-update ####
        for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


if __name__ == '__main__':

    ###### logging ######
    log_name = '0404'

    model_save_dir = 'saved_model/' + log_name
    if not os.path.isdir(model_save_dir): os.mkdir(model_save_dir)
    log_save_dir = 'log/' + log_name
    if not os.path.isdir(log_save_dir): os.mkdir(log_save_dir)
    ###### logging ######

    agent = DQNAgent()

    env = gym.make('Pendulum-v1')

    EPISODE = 500
    print_once = True
    score_list = []  # [-2000]

    for EP in range(EPISODE):
        state = env.reset()
        score, done = 0.0, False
        maxQ_action_count = 0

        while not done:
            action, real_action, count = agent.choose_action(torch.FloatTensor(state))

            state_prime, reward, done, _ = env.step([real_action])

            agent.memory.put((state, action, reward, state_prime, done))

            score += reward
            maxQ_action_count += count

            state = state_prime

            if agent.memory.size() > 1000:  # 1000개의 [s,a,r,s']이 쌓이면 학습 시작
                if print_once: print("학습시작!")
                print_once = False
                agent.train_agent()

        if EP % 10 == 0:
            torch.save(agent.Q.state_dict(), model_save_dir + "/DQN_Q_EP"+str(EP)+".pt")

        # if score > max(score_list):  # 스코어 리스트의 최댓값을 갱신하면 모델 저장
        #     # torch.save(agent.Q.state_dict(), "save_model/1225/DQN_Q_EP" + str(EP) + ".pt")
        #     torch.save(agent.Q.state_dict(), "save_model/DQN_Q_network.pt")
        #     print("...모델 저장...")

        print("EP:{}, Avg_Score:{:.1f}, MaxQ_Action_Count:{}, Epsilon:{:.5f}".format(EP, score, maxQ_action_count, agent.epsilon))
        score_list.append(score)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    # score = [float(s) for s in data]

    plt.plot(score_list)
    plt.show()

    np.savetxt(log_save_dir + '/pendulum_score.txt', score_list)

