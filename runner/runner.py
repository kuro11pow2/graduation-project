# import sys, os
# ppath = lambda x: os.path.dirname(os.path.abspath(x))
# file_name = os.getcwd()
# sys.path.append(ppath(file_name))

from abc import *

import gym
from recorder import Recorder
from torch.utils.tensorboard import SummaryWriter

class RunnerParams:
    def __init__(self, env_name, *, save_model=False, save_name=None, load_model=False, load_name=None, train=True, 
                    max_episode=10000, print_interval=20, max_video=3, record_baseline=None, reward_scale=None):
        self.env_name = env_name
        self.save_model = save_model
        self.save_name = save_name
        self.load_model = load_model
        self.load_name = load_name
        self.train = train
        self.max_episode = max_episode
        self.print_interval = print_interval
        self.max_video = max_video
        self.record_baseline = record_baseline
        self.reward_scale = reward_scale

class Runner(metaclass=ABCMeta):
    def __init__(self, runner_params):
        self._env_name = runner_params.env_name
        self._save_model = runner_params.save_model
        self._save_name = runner_params.save_name
        self._load_model = runner_params.load_model
        self._load_name = runner_params.load_name
        self._train = runner_params.train
        self._max_episode = runner_params.max_episode
        self._print_interval = runner_params.print_interval
        self._max_video = runner_params.max_video
        self._record_baseline = runner_params.record_baseline
        self._reward_scale = runner_params.reward_scale
        self._env = None
        self._recorder = None
        self._writer = None
        self._stop = False
        self._set_constant()
    
    def _set_constant(self):
        if not self._record_baseline:
            tmp = None
            if self._env_name == 'LunarLander-v2':
                tmp = 100
            elif self._env_name == 'CartPole-v1':
                tmp = 300
            self._record_baseline = tmp

        if not self._reward_scale:
            tmp = None
            if self._env_name == 'LunarLander-v2':
                tmp = 20.0
            elif self._env_name == 'CartPole-v1':
                tmp = 100.0
            self._reward_scale = tmp
        
    def run(self):
        env = gym.make(self._env_name)
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
        if self._load_model:
            self._load()
        for n_epi in range(self._max_episode):
            self._episode_sim()
            self._record_video(n_epi)
            self._print_log(n_epi)
            if self._stop:
                break
        if self._save_model:
            self._save()

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

    @abstractmethod
    def _save(self, path):
        pass

    @abstractmethod
    def _load(self, path, name):
        pass
    
