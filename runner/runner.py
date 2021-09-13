# import sys, os
# ppath = lambda x: os.path.dirname(os.path.abspath(x))
# file_name = os.getcwd()
# sys.path.append(ppath(file_name))

from abc import *

import gym
from recorder import Recorder
from torch.utils.tensorboard import SummaryWriter

class Runner(metaclass=ABCMeta):
    def __init__(self, env_name, max_episode, print_interval, max_video, record_baseline, reward_scale):
        self.env_name = env_name
        self._max_episode = max_episode
        self._print_interval = print_interval
        self._max_video = max_video
        self._record_baseline = record_baseline
        self._reward_scale = reward_scale
        self._env = None
        self._recorder = None
        self._writer = None
        self._stop = False
        self._set_constant()
    
    def _set_constant(self):
        if not self._record_baseline:
            tmp = None
            if self.env_name == 'LunarLander-v2':
                tmp = 0
            elif self.env_name == 'CartPole-v1':
                tmp = 300
            self._record_baseline = tmp

        if not self._reward_scale:
            tmp = None
            if self.env_name == 'LunarLander-v2':
                tmp = 20.0
            elif self.env_name == 'CartPole-v1':
                tmp = 100.0
            self._reward_scale = tmp
        
    def run(self):
        env = gym.make(self.env_name)
        self._recorder = Recorder(env, False)
        self._env = self._recorder.wrapped_env()
        if not isinstance(self._env.action_space, gym.spaces.discrete.Discrete):
            raise Exception('discrete space만 지원됨.')
        self._writer = SummaryWriter()
        self._episode_loop()
        self._env.close()
        self._writer.flush()
        self._writer.close()

    def _episode_loop(self):
        self._episode_prepare()
        for n_epi in range(self._max_episode):
            self._episode_sim()
            self._record_video(n_epi)
            self._print_log(n_epi)
            if self._stop:
                break

    def _record_video(self, n_epi):
        if n_epi % self._print_interval==0 and n_epi!=0:
            if self._score / self._print_interval > self._record_baseline:
                self._recorder.add_epi([n_epi + 1])
                if (len(self._recorder.recorded_epi()) >= self._max_video):
                    self._env.reset()
                    self._stop = True
    
    def _print_log(self, n_epi):
        if n_epi % self._print_interval==0 and n_epi!=0:
            print("# of episode: {}, avg score: {:.1f}".format(n_epi, self._score/self._print_interval))
            self._writer.add_scalar("score/train", self._score/self._print_interval, n_epi)
            self._score = 0.0

    @abstractmethod
    def _episode_prepare(self):
        pass

    @abstractmethod
    def _episode_sim(self):
        pass
    
