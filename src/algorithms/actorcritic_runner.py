import sys, os
ppath = lambda x: os.path.dirname(os.path.abspath(x))
sys.path.append(ppath(__file__))

from runner import Runner, RunnerParams

import torch

from actorcritic import ActorCritic
from torch.distributions import Categorical

class ActorCriticRunner(Runner):
    def __init__(self, env_name, algo_params, runner_params):
        super(ActorCriticRunner, self).__init__(env_name, 'ActorCritic', algo_params, runner_params)

    def _before_sim_loop(self):
        n_state = self._env.observation_space.shape[0]
        n_action = self._env.action_space.n
        self._algo = ActorCritic(n_state, n_action, self._algo_params)
        self._score = 0.0
        self._score_sum = 0.0

    def _episode_sim(self, n_epi):
        s = self._env.reset()
        done = False
        self._score = 0.0
        n_step = 0

        while not done:
            prob = self._algo.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = self._step_wrapper(self._env.step(a))

            if self._train:
                self._algo.put_data((s,a,r/self._reward_scale,s_prime,done))
            if self._save_step_log:
                self._write_step_log(n_step, n_epi, s, a, r, done)

            s = s_prime
            self._score += r
            n_step += 1

        self._score_sum += self._score 
           
    def _after_sim(self, n_epi, print_log, cond_check):
        super()._after_sim(n_epi, print_log, cond_check)

        if not self._done and self._train:
            self._algo.train_net()