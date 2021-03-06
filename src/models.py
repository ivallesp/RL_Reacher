import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """
        Neural network used to implement the critic function
        :param state_size: size of the state (int)
        :param action_size: size of the action space (int)
        :param random_seed: seed for the random processes (int)
        """
        super(CriticArchitecture, self).__init__()
        torch.manual_seed(random_seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128 + action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def forward(self, x, actions):
        """
        Forward pass of the neural network
        :param x: states (tensor)
        :param actions: actions taken (tensor)
        :return: output of the network (tenso)
        """
        h = F.relu(self.fc1(x))
        h = self.bn1(h)
        h = torch.cat([h, actions], dim=1)
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out


    def reset_parameters(self):
        """
        Neural networks weights initalization
        :return: None
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


# Model actor
class ActorArchitecture(nn.Module, RLModel):
    def __init__(self, state_size, action_size, random_seed):
        """
        Neural network implementing the Actor function
        :param state_size: size of the observation space (int)
        :param action_size: size of the action space (int)
        :param random_seed: seed for the random processes (int)
        """
        super(ActorArchitecture, self).__init__()
        torch.manual_seed(random_seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.reset_parameters()

    def forward(self, x):
        """
        Forward pass of the network
        :param x: states (iterable)
        :return: output value of the network (float)
        """
        h = F.relu(self.fc1(x))
        h = self.bn1(h)
        h = F.relu(self.fc2(h))
        out = F.tanh(self.fc3(h))
        return out

    def reset_parameters(self):
        """
        Initializes the parameters of the networks of tha agent
        :return:
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc2))


def hidden_init(layer):
    """
    Initializer function for weights in Pytorch
    :param layer: number of hidden layers to implement
    :return: None
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim
