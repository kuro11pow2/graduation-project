import sys, os
ppath = lambda x: os.path.dirname(os.path.abspath(x))
sys.path.append(ppath(__file__))

from runner import Runner, RunnerParams

import torch

from ppo import PPO
from torch.distributions import Categorical

class PPORunner(Runner):
    def __init__(self, env_name, algo_params, runner_params):
        super(PPORunner, self).__init__(env_name, 'PPO', algo_params, runner_params)

    def _before_sim_loop(self):
        n_state = self._env.observation_space.shape[0]
        n_action = self._env.action_space.n
        self._algo = PPO(n_state, n_action, self._algo_params)
        self._score = 0.0

    def _episode_sim(self, n_epi):
        s = self._env.reset()
        done = False
        while not done:
            for t in range(self._algo.t_horizon):
                prob = self._algo.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = self._step_wrapper(self._env.step(a))
                
                if self._train:
                    self._algo.put_data((s, a, r/self._reward_scale, s_prime, prob[a].item(), done))
                    
                s = s_prime
                self._score += r

                if done:
                    return
                    
            if self._train:
                self._algo.train_net()
