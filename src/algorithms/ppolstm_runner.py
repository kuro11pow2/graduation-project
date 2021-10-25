import sys, os
ppath = lambda x: os.path.dirname(os.path.abspath(x))
sys.path.append(ppath(__file__))

from runner import Runner, RunnerParams

import torch

from ppolstm import PPOlstm
from torch.distributions import Categorical

class PPOlstmRunner(Runner):
    def __init__(self, env_name, algo_params, runner_params):
        super(PPOlstmRunner, self).__init__(env_name, 'PPOlstm', algo_params, runner_params)

    def _before_sim_loop(self):
        n_state = self._env.observation_space.shape[0]
        n_action = self._env.action_space.n
        self._algo = PPOlstm(n_state, n_action, self._algo_params)
        self._score = 0.0

    def _episode_sim(self, n_epi):
        h_out = (torch.zeros([1, 1, self._algo.n_node//2], dtype=torch.float), 
                torch.zeros([1, 1, self._algo.n_node//2], dtype=torch.float))
        s = self._env.reset()
        done = False
        while not done:
            for t in range(self._algo.t_horizon):
                h_in = h_out
                prob, h_out = self._algo.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = self._step_wrapper(self._env.step(a))

                if self._train:
                    self._algo.put_data((s, a, r/self._reward_scale, s_prime, prob[a].item(), h_in, h_out, done))
                
                s = s_prime

                self._score += r
                if done:
                    break
            if self._train:
                self._algo.train_net()
