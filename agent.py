from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AbstractAgent(nn.Module, ABC):
    def __init__(self, action_space_size, window_size = 11):
        super().__init__()

    #@torch.compile
    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    #@torch.compile
    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class Agent1(AbstractAgent):
    def __init__(self, action_space_size, window_size = 11):
        super().__init__(action_space_size)

        network_depth = 128

        self.network = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(3 * window_size * window_size, network_depth)),
            nn.ReLU(),
            layer_init(nn.Linear(network_depth, network_depth)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(network_depth, action_space_size), std=0.01)
        self.critic = layer_init(nn.Linear(network_depth, 1), std=1)


class Agent2(AbstractAgent):
    def __init__(self, action_space_size, window_size = 11):
        super().__init__(action_space_size)

        network_depth = 256

        self.network = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(3 * window_size * window_size, network_depth)),
            nn.ReLU(),
            layer_init(nn.Linear(network_depth, network_depth)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(network_depth, action_space_size), std=0.01)
        self.critic = layer_init(nn.Linear(network_depth, 1), std=1)

class Agent3(AbstractAgent):
    def __init__(self, action_space_size, window_size = 11):
        super().__init__(action_space_size)

        network_depth = 512

        self.network = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(3 * window_size * window_size, network_depth)),
            nn.ReLU(),
            layer_init(nn.Linear(network_depth, network_depth)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(network_depth, action_space_size), std=0.01)
        self.critic = layer_init(nn.Linear(network_depth, 1), std=1)

class Agent4(AbstractAgent):
    def __init__(self, action_space_size, window_size = 11):
        super().__init__(action_space_size)

        network_depth = 256

        self.network = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(3 * window_size * window_size, network_depth)),
            nn.ReLU(),
            layer_init(nn.Linear(network_depth, network_depth)),
            nn.ReLU(),
            layer_init(nn.Linear(network_depth, network_depth)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(network_depth, action_space_size), std=0.01)
        self.critic = layer_init(nn.Linear(network_depth, 1), std=1)

class Reshape(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class Agent5(AbstractAgent):
    def __init__(self, action_space_size, window_size = 11):
        super().__init__(action_space_size)

        self.network = nn.Sequential(
            Reshape(),
            layer_init(nn.Conv2d(3, 16, 3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.actor = layer_init(nn.Linear(1568, action_space_size), std=0.01)
        self.critic = layer_init(nn.Linear(1568, 1), std=1)

class Agent6(AbstractAgent):
    def __init__(self, action_space_size, window_size = 11):
        super().__init__(action_space_size)

        network_depth = 256

        self.network = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(3 * window_size * window_size, network_depth)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(network_depth, action_space_size), std=0.01)
        self.critic = layer_init(nn.Linear(network_depth, 1), std=1)