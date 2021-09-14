
from runner import Runner, RunnerParams

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algorithms.ppo import PPO
from torch.distributions import Categorical

class PPORunner(Runner):
    def __init__(self, runner_params):
        super(PPORunner, self).__init__(runner_params)

    def _episode_prepare(self):
        n_state = self._env.observation_space.shape[0]
        n_action = self._env.action_space.n
        self._model = PPO(n_state, n_action)
        self._score = 0.0
        self._print_interval = 20

    def _episode_sim(self):
        s = self._env.reset()
        done = False
        while not done:
            for t in range(self._model.t_horizon):
                prob = self._model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = self._env.step(a)
                
                if self._train:
                    self._model.put_data((s, a, r/self._reward_scale, s_prime, prob[a].item(), done))
                    
                s = s_prime

                self._score += r
                if done:
                    break
            if self._train:
                self._model.train_net()
