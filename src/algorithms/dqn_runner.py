import sys, os, time
ppath = lambda x: os.path.dirname(os.path.abspath(x))
sys.path.append(ppath(__file__))

from runner import Runner, RunnerParams

import torch

from dqn import DQN
from torch.distributions import Categorical

class DQNRunner(Runner):
    def __init__(self, env_name, algo_params, runner_params):
        super(DQNRunner, self).__init__(env_name, 'DQN', algo_params, runner_params)

    def _episode_prepare(self):
        n_state = self._env.observation_space.shape[0]
        n_action = self._env.action_space.n
        self._net = DQN(n_state, n_action, self._algo_params)
        self._net.update_net()

        self._score = 0.0

    def _episode_sim(self, n_epi):
        s = self._env.reset()
        done = False

        if self._train:
            self._net.epsilon = max(0.01, self._net.start_epsilon - 0.01*(n_epi/200))
        else:
            self._net.epsilon = 0.0

        while not done:
            a = self._net.sample_action(torch.from_numpy(s).float())
            s_prime, r, done, info = self._step_wrapper(self._env.step(a))
            
            if self._train:
                self._net.append_data((s,a,r/self._reward_scale,s_prime, done))
                
            s = s_prime
            self._score += r
            if done:
                break

        if self._train and self._net.buffer_size() > self._net.n_train_start:
            self._net.train_net()

        if n_epi % self._net.update_interval==0:
            self._net.update_net()

    def _print_log(self, n_epi):
        super()._print_log(n_epi)
        print(f"n_buffer : {self._net.buffer_size()}, "\
                    + f"eps : {self._net.epsilon*100:.1f}%")