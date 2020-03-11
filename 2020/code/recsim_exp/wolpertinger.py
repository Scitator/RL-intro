import numpy as np
from gym import spaces
from recsim.agent import AbstractEpisodicRecommenderAgent

from .ddpg import DDPG, ReplayBuffer


class Wolpertinger(DDPG):
    def __init__(self, *, action_dim, k_ratio=0.1, **kwargs):
        super().__init__(action_dim=action_dim, **kwargs)
        self.k = max(1, int(action_dim * k_ratio))

    def predict_action(self, state, with_noise=False):

        proto_action = super().predict_action(state, with_noise=with_noise)
        proto_action = proto_action.clip(0, 1)

        actions = np.eye(self.action_dim)
        # first sorting by action probability by `proto_action`
        # second by random :)
        actions_sorting = np.lexsort(
            (np.random.random(self.action_dim), proto_action)
        )
        # take topK proposed actions
        actions = actions[actions_sorting[-self.k:]]
        # make all the state-action pairs for the critic
        states = np.tile(state, [len(actions), 1])
        qvalues = self.critic.get_qvalue(states, actions)
        # find the index of the pair with the maximum value
        max_index = np.argmax(qvalues)
        action, qvalue = actions[max_index], qvalues[max_index]
        return action

    def predict_qvalues(self, state_num=0, action=None, dim=None):
        if dim is None:
            dim = self.action_dim
        if action is None:
            action = np.eye(dim, self.action_dim)
        s = np.zeros((dim, self.action_dim))
        s[:, state_num] = 1
        qvalues = self.critic.get_qvalue(s, action)
        return qvalues


class WolpertingerRecommender(AbstractEpisodicRecommenderAgent):

    def __init__(
        self,
        env,
        state_dim,
        action_dim,
        k_ratio=0.1,
        eps=1e-2,
        train: bool = True,
        batch_size: int = 256,
        buffer_size: int = 10000,
        training_starts: int = None,
        **kwargs,
    ):
        AbstractEpisodicRecommenderAgent.__init__(self, env.action_space)

        self._observation_space = env.observation_space
        self.agent = Wolpertinger(
            state_dim=state_dim,
            action_dim=action_dim,
            k_ratio=k_ratio,
            **kwargs
        )
        self.t = 0
        self.current_episode = {}
        self.train = train
        self.num_actions = env.action_space.nvec[0]

        self.eps = eps
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.training_starts = training_starts or batch_size

    def _extract_state(self, observation):
        user_space = self._observation_space.spaces["user"]
        return spaces.flatten(user_space, observation["user"])

    def _act(self, state):
        if np.random.rand() < self.eps:
            action = np.eye(self.num_actions)[np.random.randint(self.num_actions)]
        else:
            action = self.agent.predict_action(state)
        self.current_episode = {
            "state": state,
            "action": action,
        }
        return np.argmax(action)[np.newaxis]

    def _observe(self, next_state, reward, done):
        if not self.current_episode:
            raise ValueError("Current episode is expected to be non-empty")

        self.current_episode.update({
            "next_state": next_state,
            "reward": reward,
            "done": done
        })

        self.agent.episode = self.current_episode
        if self.train:
            self.replay_buffer.push(**self.current_episode)
            if self.t >= self.training_starts \
                    and len(self.replay_buffer) >= self.batch_size:
                state, action, reward, next_state, done = \
                    self.replay_buffer.sample(self.batch_size)
                self.agent.update(state, action, reward, next_state, done)
        self.current_episode = {}

    def begin_episode(self, observation=None):
        state = self._extract_state(observation)
        return self._act(state)

    def step(self, reward, observation):
        state = self._extract_state(observation)
        self._observe(state, reward, 0)
        self.t += 1
        return self._act(state)

    def end_episode(self, reward, observation=None):
        state = self._extract_state(observation)
        self._observe(state, reward, 1)
