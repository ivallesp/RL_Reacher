import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from unityagents import UnityEnvironment

from src.agents import *
from src.rl_utilities import *


env_path = "./envs/Reacher_Windows_x86_64/Reacher.exe"
env = UnityEnvironment(env_path, no_graphics=True)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations
state_size = state.shape[1]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size

replay_buffer = ExperienceReplay(int(1e5))
agent = DDPGAgent(CriticArchitecture, ActorArchitecture, state_size=state_size, action_size=action_size,
                  tau=0.001, epsilon=0.01, gamma=0.99, random_seed=655321)
batch_size = 128
n_episodes = 100000
scores = []

for episode in range(n_episodes):
    env_info = env.reset(train_mode=True)[brain_name]
    agent.reset()
    state = env_info.vector_observations
    score = 0
    done = False
    c = 0
    while not done:
        # take random action
        action = agent.act(state, add_noise=True)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        replay_buffer.append([state, action, reward, next_state, done])
        state = next_state
        score += reward
        c += 1

        if replay_buffer.length > batch_size:
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.draw_sample(128)
            agent.update(states=states_batch,
                         actions=actions_batch,
                         rewards=rewards_batch,
                         next_states=next_states_batch,
                         dones=dones_batch)

    scores.append(score)

