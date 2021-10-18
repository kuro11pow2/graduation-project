import sys, os
ppath = lambda x: os.path.dirname(os.path.abspath(x))
sys.path.append(ppath(__file__))

from runner import Runner, RunnerParams

import torch

from reinfoce import Reinforce
from torch.distributions import Categorical

class ReinforceRunner(Runner):
    def __init__(self, env_name, algo_params, runner_params):
        super(ReinforceRunner, self).__init__(env_name, 'Reinforce', algo_params, runner_params)

    def _episode_prepare(self):
        n_state = self._env.observation_space.shape[0]
        n_action = self._env.action_space.n
        self._algo = Reinforce(n_state, n_action, self._algo_params)
        self._score = 0.0
        self._score_sum = 0.0

    def _episode_sim(self, n_epi):
        s = self._env.reset()
        done = False
        self._score = 0.0
        n_step = 0

        while not done:
            prob = self._algo(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = self._step_wrapper(self._env.step(a.item()))

            if self._train:
                self._algo.put_data((r/self._reward_scale, prob[a]))
            if self._save_step_log:
                self._write_step_log(n_step, n_epi, s, a.item(), r, done)

            s = s_prime
            self._score += r
            n_step += 1
            
            if done:
                break 
        
        self._score_sum += self._score 
        if self._train:
            self._algo.train_net()
