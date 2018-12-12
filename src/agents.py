import copy
from src.models import *
from src.rl_utilities import ExperienceReplay
import random


class DDPGAgent():
    def __init__(self, critic_arch, actor_arch, state_size, action_size, tau, gamma, replay_size, batch_size,
                 n_batches_train, random_seed):
        """
        Agent implementing DDPG algorithm. More info here: https://arxiv.org/abs/1509.02971

        :param critic_arch: pytorch neural network implementing a critic function (s, a -> Q), located in the
        src.models module (pytorch model object)
        :param actor_arch: pytorch neural network implementing a actor function (s -> P(a|s)), located in the
        src.models module (pytorch model object)
        :param state_size: size of the state space (int)
        :param action_size: size of the action space (int)
        :param tau: constant controling the rate of the soft update of the target networks from the local
        networks (float)
        :param gamma: discount factor (float)
        :param replay_size: size of the experience replay buffer (int)
        :param batch_size: size of the batches which are going to be used to train the neural networks (int)
        :param n_batches_train: number of batches to train in each agent step (int)
        :param random_seed: random seed for numpy and pytorch (int)
        """
        np.random.seed(random_seed)
        self.critic_local = critic_arch(state_size, action_size, random_seed)
        self.critic_target = critic_arch(state_size, action_size, random_seed)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=2e-4)

        self.actor_local = actor_arch(state_size, action_size, random_seed)
        self.actor_target = actor_arch(state_size, action_size, random_seed)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=2e-4)


        # Equalize target and local networks
        self._soft_target_update(tau=1)

        # Noise
        self.noise = OUNoise(action_size, random_seed)

        # Experience replay buffer
        self.replay_buffer = ExperienceReplay(int(replay_size))

        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_batches_train = n_batches_train

    def step(self, states, actions, rewards, next_states, dones):
        # Update replay buffer
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.replay_buffer.append([s, a, r, ns, d])

        for _ in range(self.n_batches_train):
            # Sample a batch of experiences
            states_batch, \
            actions_batch, \
            rewards_batch, \
            next_states_batch, \
            dones_batch = self.replay_buffer.draw_sample(self.batch_size)

            # Train
            if self.replay_buffer.length > self.batch_size:
                self.update(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)


    def update(self, states, actions, rewards, next_states, dones):
        self._update_critic(states, actions, rewards, next_states, dones)
        self._update_actor(states)
        self._soft_target_update()

    def reset(self):
        self.noise.reset()

    def act(self, states, epsilon=1):
        states = torch.from_numpy(states).float()
        if states.dim==1:
            states = torch.unsqueeze(states, 0)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local.forward(states).cpu().data.numpy()
        self.actor_local.train()
        actions += self.noise.sample()*epsilon
        actions = np.clip(actions, -1, 1)
        return actions

    def _soft_target_update(self, tau=None):
        if tau is None:
            tau = self.tau
        self.critic_target.copy_weights_from(self.critic_local, tau)
        self.actor_target.copy_weights_from(self.actor_local, tau)

    def _update_critic(self, states, actions, rewards, next_states, dones):
        # Calculate td target
        next_actions = self.actor_target.forward(next_states)
        q_value_next_max = self.critic_target.forward(next_states, next_actions)
        q_value_target = rewards + self.gamma * q_value_next_max * (1 - dones)
        # q_value_target = torch.from_numpy(q_value_target).float()
        # Calculate the loss
        q_value_current = self.critic_local.forward(states, actions)

        loss = F.mse_loss(q_value_current, q_value_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

    def _update_actor(self, states):
        actions_predicted = self.actor_local.forward(states)
        critic_action_values = self.critic_local.forward(states, actions_predicted)
        # Calculate the loss
        loss = -critic_action_values.mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
