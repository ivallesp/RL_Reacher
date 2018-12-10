import pandas as pd
import numpy as np
import random
from collections import deque
import torch

def ewma(x):
    return pd.Series(x).ewm(span=100).mean()


class ExperienceReplay():
    def __init__(self, size=int(1e5)):
        self.size = size
        self.reset()

    @property
    def length(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = deque(maxlen=self.size)

    def append(self, observation):
        self.buffer.append(observation)

    def draw_sample(self, sample_size):
        buffer_sample = random.choices(self.buffer, k=sample_size)
        states, actions, rewards, next_states, dones = zip(*buffer_sample)
        states = torch.from_numpy(np.array(states).squeeze()).float()
        actions = torch.from_numpy(np.array(actions).squeeze()).float()
        rewards = torch.from_numpy(np.expand_dims(np.array(rewards), 1)).float()
        next_states = torch.from_numpy(np.array(next_states).squeeze()).float()
        dones = torch.from_numpy(np.expand_dims(np.array(dones)+0, 1)).float()
        return states, actions, rewards, next_states, dones