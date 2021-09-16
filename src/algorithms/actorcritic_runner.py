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

    def _episode_prepare(self):
        n_state = self._env.observation_space.shape[0]
        n_action = self._env.action_space.n
        self._net = ActorCritic(n_state, n_action, self._algo_params)
        self._score = 0.0
        self._print_interval = 20

    def _episode_sim(self):
        s = self._env.reset()
        done = False
        while not done:
            for t in range(self._net.n_rollout):
                prob = self._net.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = self._step_wrapper(self._env.step(a))

                if self._train:
                    self._net.put_data((s,a,r/self._reward_scale,s_prime,done))
                
                s = s_prime
                self._score += r
                if done:
                    break                     
            if self._train:
                self._net.train_net()
