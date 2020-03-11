import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


class GaussNoise:
    def __init__(self, sigma):
        super().__init__()

        self.sigma = sigma

    def get_action(self, action):
        noisy_action = np.random.normal(action, self.sigma)
        return noisy_action


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_size,
        init_w=3e-3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, num_actions)
        nn.init.uniform_(self.head.weight, -init_w, init_w)
        nn.init.zeros_(self.head.bias)

    def forward(self, state):
        x = self.net(state)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

    def get_action(self, state):
        state = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        action = self.forward(state)
        action = action.detach().cpu().numpy()[0]
        return action


class Critic(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_size,
        init_w=3e-3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, 1)
        nn.init.uniform_(self.head.weight, -init_w, init_w)
        nn.init.zeros_(self.head.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.net(x)
        x = self.head(x)
        return x

    def get_qvalue(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        action = torch.tensor(action, dtype=torch.float32).to(DEVICE)
        q_value = self.forward(state, action)
        q_value = q_value.detach().cpu().numpy()
        return q_value


class DDPG:
    def __init__(
        self,
        state_dim,
        action_dim,
        noise=None,
        hidden_dim=256,
        tau=1e-3,
        gamma=0.99,
        init_w_actor=3e-3,
        init_w_critic=3e-3,
        critic_lr=1e-3,
        actor_lr=1e-4,
        actor_weight_decay=0.,
        critic_weight_decay=0.,
    ):
        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_dim,
            init_w=init_w_actor
        ).to(DEVICE)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=actor_weight_decay
        )

        self.critic = Critic(
            state_dim,
            action_dim,
            hidden_dim,
            init_w=init_w_critic
        ).to(DEVICE)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=critic_weight_decay
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.noise = noise

        self.tau = tau
        self.gamma = gamma

    def predict_action(self, state, with_noise=False):
        self.actor.eval()
        action = self.actor.get_action(state)
        if self.noise and with_noise:
            action = self.noise.get_action(action)
        self.actor.train()
        return action

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)
        action = torch.tensor(action, dtype=torch.float32).to(DEVICE)
        reward = torch.tensor(
            reward, dtype=torch.float32
        ).unsqueeze(1).to(DEVICE)
        done = torch.tensor(np.float32(done)).unsqueeze(1).to(DEVICE)

        # actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        #  critic loss
        predicted_value = self.critic(state, action)
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        critic_loss = F.mse_loss(predicted_value, expected_value.detach())

        # actor update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # critic update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)
