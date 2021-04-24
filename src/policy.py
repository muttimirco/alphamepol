import sys, os
sys.path.append(os.getcwd() + '/src')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.dtypes import float_type, int_type, eps

from collections import OrderedDict


class GaussianPolicy(nn.Module):
    """
    Gaussian Policy with state-independent diagonal covariance matrix
    """

    def __init__(self, hidden_sizes, num_features, action_dim, log_std_init=-0.5, activation=nn.ReLU, seed=None):
        super().__init__()

        self.activation = activation

        layers = []
        layers.extend((nn.Linear(num_features, hidden_sizes[0]), self.activation()))
        for i in range(len(hidden_sizes) - 1):
            layers.extend((nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self.activation()))

        self.net = nn.Sequential(*layers)

        self.mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Parameter(log_std_init * torch.ones(action_dim, dtype=float_type))

        # Constants
        self.register_buffer('log_of_pi', torch.tensor(np.log(2*np.pi), dtype=float_type))

        self.initialize_weights(seed)

    def initialize_weights(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.mean.weight)
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)


    def get_log_p(self, states, actions):
        mean, _ = self(states)
        return torch.sum(-0.5 * (self.log_of_pi + 2*self.log_std + ((actions - mean)**2 / (torch.exp(self.log_std) + eps)**2)), dim=1)


    def forward(self, x, deterministic=False):
        mean = self.mean(self.net(x))

        if deterministic:
            output = mean
        else:
            output = mean + torch.randn(mean.size(), dtype=float_type) * torch.exp(self.log_std)

        return mean, output


    def predict(self, s, deterministic=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=float_type).unsqueeze(0)
            return self(s, deterministic=deterministic)[1][0]


class Encoder(nn.Module):
    '''
    Encoder used for the MiniGrid experiments: input image --> cnn features --> feed-forward --> categorical
    '''

    def __init__(self, env, seed=None):
        super().__init__()
        
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        n = env.observation_space.shape[0]
        m = env.observation_space.shape[1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        self.net = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

        self.initialize_weights(seed)

    def initialize_weights(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
        for l in self.image_conv:
            if isinstance(l, nn.Conv2d):
                nn.init.xavier_uniform_(l.weight)
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)


    def get_log_p(self, states, actions):
        dist = self(states)
        actions = torch.reshape(actions, (len(actions),))
        return dist.log_prob(actions)


    def forward(self, obs, deterministic=False):
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.net(x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        return dist


    def predict(self, s, deterministic=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=float_type).unsqueeze(0)
            return self(s, deterministic=deterministic).sample()


class ValueFunction(nn.Module):

    def __init__(self, env, seed=None):
        super().__init__()
        
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        n = env.observation_space.shape[0]
        m = env.observation_space.shape[1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        self.net = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.initialize_weights(seed)

    def initialize_weights(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
        for l in self.image_conv:
            if isinstance(l, nn.Conv2d):
                nn.init.orthogonal_(l.weight)
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.orthogonal_(l.weight)

    def forward(self, obs):
        if len(obs.shape) == 4:
            x = obs.transpose(1, 3).transpose(2, 3)
        else:
            x = obs.unsqueeze(0).transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.net(x)
        #dist = Categorical(logits=F.log_softmax(x, dim=1))
        return x.squeeze(1)


def train_supervised(env, policy, train_steps=100, batch_size=5000):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.00025)
    dict_like_obs = True if type(env.observation_space.sample()) is OrderedDict else False
    #losses = []
    for t_step in range(train_steps):
        #print(t_step)
        optimizer.zero_grad()

        if dict_like_obs:
            states = torch.tensor([env.observation_space.sample()['observation'] for _ in range(batch_size)], dtype=float_type)
        else:
            states = torch.tensor([env.observation_space.sample()[:env.num_features] for _ in range(batch_size)], dtype=float_type)

        actions = policy(states)[0]
        #loss = torch.mean((actions - torch.zeros_like(actions, dtype=float_type)) ** 2)
        x_targets = torch.ones_like(actions[:, 0], dtype=float_type) * -0.2
        y_targets = torch.ones_like(actions[:, 1], dtype=float_type) * -0.2
        loss = torch.mean((actions[:, 0] - x_targets) ** 2) + torch.mean((actions[:, 1] - y_targets) ** 2)
        #losses.append(loss)

        loss.backward()
        optimizer.step()
    
    #fig, ax = plt.subplots()
    #ax.plot([j for j in range(len(losses))], losses)
    #fig.savefig("supervised_loss.png")

    return policy