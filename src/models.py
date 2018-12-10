import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections.abc import Iterable


# RLModel class
class RLModel:
    def __init__(self, random_seed):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def copy_weights_from(self, net, tau=0.001):
        # tau should be a small parameter
        for local_param, ext_param in zip(self.parameters(), net.parameters()):
            local_param.data.copy_((1 - tau) * (local_param.data) + (tau) * ext_param.data)


# Model critic
class CriticArchitecture(nn.Module, RLModel):
    def __init__(self, state_size, action_size, random_seed):
        super(CriticArchitecture, self).__init__()
        if isinstance(state_size, Iterable):
            assert len(state_size) == 1
            state_size = state_size[0]
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128 + action_size)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128 + action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def forward(self, x, actions):
        x = self.bn0(x)
        h = F.relu(self.fc1(x))
        h = torch.cat([h, actions], dim=1)
        h = self.bn1(h)
        h = F.relu(self.fc2(h))
        h = self.bn2(h)
        out = self.fc3(h)
        return out


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc2))


# Model actor
class ActorArchitecture(nn.Module, RLModel):
    def __init__(self, state_size, action_size, random_seed):
        super(ActorArchitecture, self).__init__()
        if isinstance(state_size, Iterable):
            assert len(state_size) == 1
            state_size = state_size[0]
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, action_size)
        self.reset_parameters()

    def forward(self, x):
        x = self.bn0(x)
        h = F.relu(self.fc1(x))
        h = self.bn1(h)
        h = F.relu(self.fc2(h))
        h = self.bn2(h)
        out = F.tanh(self.fc3(h))
        return out

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc2))


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim
