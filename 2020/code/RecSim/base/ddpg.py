import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import copy

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


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


class GaussNoise:
    def __init__(self, sigma):
        super().__init__()

        self.sigma = sigma

    def get_action(self, action):
        noisy_action = np.random.normal(action, self.sigma)
        return noisy_action


class Critic(nn.Module):
    def __init__(
            self,
            num_inputs,
            num_actions,
            hidden_size,
            init_w=3e-3
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

    def get_q_values(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        q_value = self.forward(state, action)
        q_value = q_value.detach().cpu().numpy()
        return q_value


class Actor(nn.Module):
    def __init__(
            self,
            num_inputs,
            num_actions,
            hidden_size,
            init_w=3e-3
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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.forward(state)
        action = action.detach().cpu().numpy()[0]
        return action


class DDPG:
    def __init__(self, state_dim, action_dim, summary_writer=None, noise=None,
                 buffer_size=10000, hidden_dim=256,  soft_tau=1e-3, batch_size=128,
                 gamma=0.99, init_w_actor=3e-3, init_w_critic=3e-3, critic_lr=1e-3,
                 actor_lr=1e-4, actor_weight_decay=0., critic_weight_decay=0., **kwargs):

        self.actor = Actor(state_dim, action_dim, hidden_dim, init_w=init_w_actor).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_lr,
                                          weight_decay=actor_weight_decay)

        self.critic = Critic(state_dim, action_dim, hidden_dim, init_w=init_w_critic).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr,
                                           weight_decay=critic_weight_decay)

        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.soft_tau = soft_tau
        self.noise = noise
        self.summary_writer = summary_writer
        self.batch_size = batch_size
        self.gamma = gamma

    def predict(self, state, with_noise=True):
        self.actor.eval()
        action = self.actor.get_action(state)
        if self.noise and with_noise:
            action = self.noise.get_action(action)
        self.actor.train()
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
        done = torch.tensor(np.float32(done)).unsqueeze(1).to(device)

        current_q = self.critic(state, action)
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1.0 - done) * self.gamma * target_q).detach()
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self.critic(state, self.actor(state))
        actor_loss = -actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        grad_actor = torch.cat([p.grad.flatten() for p in self.actor.parameters()])
        grad_critic = torch.cat([p.grad.flatten() for p in self.critic.parameters()])

        if self.summary_writer:
            self.summary_writer.add_scalar('loss/Actor loss', actor_loss, self.t)
            self.summary_writer.add_scalar('loss/Critic loss', critic_loss, self.t)
            self.summary_writer.add_scalar('extra/Actor gradient norm', grad_actor.norm(), self.t)
            self.summary_writer.add_scalar('extra/Critic gradient norm', grad_critic.norm(), self.t)

        soft_update(self.critic_target, self.critic, self.soft_tau)
        soft_update(self.actor_target, self.actor, self.soft_tau)