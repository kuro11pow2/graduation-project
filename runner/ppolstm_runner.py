
from runner import Runner, RunnerParams

import torch

from algorithms.ppolstm import PPOlstm
from torch.distributions import Categorical

class PPOlstmRunner(Runner):
    def __init__(self, env_name, runner_params):
        super(PPOlstmRunner, self).__init__(env_name, 'PPOlstm', runner_params)

    def _episode_prepare(self):
        n_state = self._env.observation_space.shape[0]
        n_action = self._env.action_space.n
        self._model = PPOlstm(n_state, n_action)
        self._score = 0.0
        self._print_interval = 20

    def _episode_sim(self):
        h_out = (torch.zeros([1, 1, self._model.n_node//2], dtype=torch.float), 
                torch.zeros([1, 1, self._model.n_node//2], dtype=torch.float))
        s = self._env.reset()
        done = False
        while not done:
            for t in range(self._model.t_horizon):
                h_in = h_out
                prob, h_out = self._model.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = self._env.step(a)

                if self._train:
                    self._model.put_data((s, a, r/self._reward_scale, s_prime, prob[a].item(), h_in, h_out, done))
                
                s = s_prime

                self._score += r
                if done:
                    break
            if self._train:
                self._model.train_net()
