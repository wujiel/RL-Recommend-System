import argparse
import pickle
from collections import namedtuple
from itertools import count

import os
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal,MultivariateNormal
from tensorboardX import SummaryWriter


'''
Implementation of soft actor critic, dual Q network version 
Original paper: https://arxiv.org/abs/1801.01290
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()


parser.add_argument("--env_name", default="LunarLanderContinuous-v2")  # OpenAI gym environment name Pendulum-v0
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--epoch', default=1, type=int) # 每次sample batch训练几次

parser.add_argument('--learning_rate', default=3e-4, type=int)
parser.add_argument('--gamma', default=0.99, type=int) # discount gamma
parser.add_argument('--capacity', default=10000, type=int) # replay buffer size
parser.add_argument('--num_episode', default=2000, type=int) #  num of  games
parser.add_argument('--batch_size', default=128, type=int) # mini batch size
parser.add_argument('--max_frame', default=500, type=int) # max frame
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=20, type=int) # 每20episode保存1次模型
parser.add_argument('--load', default=False, type=bool) # load model
args = parser.parse_args()


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device)
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.h_size = hidden_size

        self.fc1 = nn.Linear(state_dim, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.h_size)
        self.mu_head = nn.Linear(self.h_size, action_dim)
        self.log_std_head = nn.Linear(self.h_size, action_dim)
        self.max_action = max_action

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Critic, self).__init__()
        self.h_size = hidden_size
        self.fc1 = nn.Linear(state_dim, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.h_size)
        self.fc3 = nn.Linear(self.h_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Q, self).__init__()
        self.h_size = hidden_size
        self.fc1 = nn.Linear(state_dim + action_dim, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.h_size)
        self.fc3 = nn.Linear(self.h_size, 1)

    def forward(self, s, a):
        s = s.reshape(-1, state_dim)
        a = a.reshape(-1, action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC():
    def __init__(self):
        super(SAC, self).__init__()

        # 两个q网络才是真正的critic
        self.policy_net = Actor(state_dim, action_dim, args.hidden_size).to(device)
        self.value_net = Critic(state_dim, args.hidden_size).to(device)
        self.Target_value_net = Critic(state_dim, args.hidden_size).to(device)
        self.Q_net1 = Q(state_dim, action_dim, args.hidden_size).to(device)
        self.Q_net2 = Q(state_dim, action_dim, args.hidden_size).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=args.learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=args.learning_rate)

        self.replay_buffer = [Transition] * args.capacity
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1
        self.writer = SummaryWriter('./exp-SAC_dual_Q_network')

        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/', exist_ok=True)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()

        action = torch.tanh(z).detach().cpu().numpy()

        return action  # .item() # return a scalar, float32

    def store(self, s, a, r, s_, d):
        index = self.num_transition % args.capacity
        transition = Transition(s, a, r, s_, d)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def evaluate(self, state):
        # 计算动作概率有点问题，动作是多维随机向量，应该将其看做多变量高斯分布，输出一个概率值
        # 即MultivariateNormal(mu,sigma)，mu是向量，sigma是diag矩阵
        # 实际使用Normal时，动作概率是一个和动作同样shape的向量
        # 另外，batch_mu + batch_sigma * z在dist分布下的概率可以用z在noise分布下的概率来近似
        # 所以在202行，才用了一个近似办法，在动作向量的所有维度上求平均，作为联合概率值的近似。
        # 在LunarLanderContinuous-v2游戏上验证了有效性。
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)

        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)  # 标准差=1
        z = noise.sample()
        action_tmp = batch_mu + batch_sigma * z.to(device)
        action = torch.tanh(action_tmp)
        # print('r,',batch_mu + batch_sigma*z,self.normal(action_tmp,batch_mu,batch_sigma.pow(2)),noise.log_prob(z).exp(),dist.log_prob(batch_mu + batch_sigma * z).exp())
        log_prob = dist.log_prob(batch_mu + batch_sigma * z.to(device)).mean(-1) - torch.log(
            1 - action.pow(2) + min_Val).mean(-1)

        return action, log_prob.reshape(-1, 1), z, batch_mu, batch_log_sigma

    def normal(self, x, mu, sigma_sq):  # 计算动作x在policy net定义的高斯分布中的概率值
        a = (-1 * (x - mu).pow(2) / (2 * sigma_sq)).exp()
        b = 1 / (2 * sigma_sq * torch.FloatTensor([np.pi]).expand_as(
            sigma_sq)).sqrt()  # pi.expand_as(sigma_sq)的意义是将标量π扩展为与sigma_sq同样的维度
        return a * b

    def update(self):
        if self.num_training % 500 == 0:
            print("Training ... {} times ".format(self.num_training))
        s = torch.tensor([t.s for t in self.replay_buffer]).float().to(device)
        a = torch.tensor([t.a for t in self.replay_buffer]).to(device)
        r = torch.tensor([t.r for t in self.replay_buffer]).to(device)
        s_ = torch.tensor([t.s_ for t in self.replay_buffer]).float().to(device)
        d = torch.tensor([t.d for t in self.replay_buffer]).float().to(device)

        for _ in range(args.epoch):
            # for index in BatchSampler(SubsetRandomSampler(range(args.capacity)), args.batch_size, False):
            index = np.random.choice(range(args.capacity), args.batch_size, replace=False)
            bn_s = s[index]
            bn_a = a[index].reshape(-1, 1)
            bn_r = r[index].reshape(-1, 1)
            bn_s_ = s_[index]
            bn_d = d[index].reshape(-1, 1)

            target_value = self.Target_value_net(bn_s_)

            # q网络的目标值由v网络输出作为未来奖励（加上立即奖励r）
            next_q_value = bn_r + (1 - bn_d) * args.gamma * target_value

            #Critic网络对计算value
            excepted_value = self.value_net(bn_s)

            excepted_Q1 = self.Q_net1(bn_s, bn_a)
            excepted_Q2 = self.Q_net2(bn_s, bn_a)
            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_s)
            excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))


            # v网络的目标值由q网络的输出和动作的熵决定（ - log_prob）
            next_value = excepted_new_Q - log_prob

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)
            # Critic网络的loss
            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()  # J_V
            # Dual Q net
            Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean()  # J_Q
            Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()

            # 策略网络的优化目标是最大化分数和熵，所以损失函数和分值函数互为相反数
            pi_loss = (log_prob - excepted_new_Q).mean()  # according to original paper

            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/policy_loss', pi_loss, global_step=self.num_training)

            # mini batch gradient descent
            # Critic网络更新
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # update target v net update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - args.tau) + param * args.tau)

            self.num_training += 1

    def save(self):
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.Q_net1.state_dict(), './SAC_model/Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), './SAC_model/Q_net2.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
        self.value_net.load_state_dict(torch.load('./SAC_model/value_net.pth'))
        self.Q_net1.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load('./SAC_model/Q_net2.pth'))
        print("model has been load")

def main():

    agent = SAC()
    if args.load: agent.load()
    if args.render: env.render()
    print("====================================")
    print("Collection Experience...")
    print("====================================")

    ep_r = 0
    for i in range(args.num_episode):
        state = env.reset()
        for t in range(args.max_frame):
            action = agent.select_action(state)
            # print(action)
            next_state, reward, done, info = env.step(action)# np.float32(action)
            ep_r += reward
            if args.render: env.render()
            agent.store(state, action, reward, next_state, done)

            if agent.num_transition >= args.capacity and t%5==0:
                agent.update()

            state = next_state
            if done or t == args.max_frame-1:
                if i % 10 == 0:
                    print("Ep_i {}, the ep_r is {}, the t is {}".format(i, ep_r, t))
                break
        if i % args.log_interval == 0:
            agent.save()
        agent.writer.add_scalar('ep_r', ep_r, global_step=i)
        ep_r = 0


if __name__ == '__main__':
    main()
