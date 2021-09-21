import sys, os, time
ppath = lambda x: os.path.dirname(os.path.abspath(x))
sys.path.append(ppath(__file__))

from runner import Runner, RunnerParams

import torch

from ddqn import DDQN
from torch.distributions import Categorical

class DDQNRunner(Runner):
    def __init__(self, env_name, algo_params, runner_params):
        super(DDQNRunner, self).__init__(env_name, 'DDQN', algo_params, runner_params)

    def _episode_prepare(self):
        n_state = self._env.observation_space.shape[0]
        n_action = self._env.action_space.n
        self._algo = DDQN(n_state, n_action, self._algo_params)
        self._algo.update_net()

        self._score = 0.0

    def _episode_sim(self, n_epi):
        s = self._env.reset()
        done = False

        if self._train:
            self._algo.epsilon = max(0.01, self._algo.start_epsilon - 0.01*(n_epi/200))
        else:
            self._algo.epsilon = 0.0

        while not done:
            a = self._algo.sample_action(torch.from_numpy(s).float())
            s_prime, r, done, info = self._step_wrapper(self._env.step(a))
            
            if self._train:
                self._algo.append_data((s,a,r/self._reward_scale,s_prime, done))
                
            s = s_prime
            self._score += r
            if done:
                break

        if self._train and self._algo.buffer_size() > self._algo.n_train_start:
            self._algo.train_net()

        if n_epi % self._algo.update_interval==0:
            self._algo.update_net()

    def _print_log(self, n_epi):
        super()._print_log(n_epi)
        print(f"n_buffer : {self._algo.buffer_size()}, "\
                    + f"eps : {self._algo.epsilon*100:.1f}%")