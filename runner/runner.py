
from abc import *

import gym
from recorder import Recorder
from torch.utils.tensorboard import SummaryWriter
import torch

class RunnerParams:
    def __init__(self, *, save_net=False, save_name=None, load_net=False, load_name=None, 
                    train=True,  max_episode=10000, print_interval=20, 
                    max_video=3, video_record_interval=0,
                    target_score=0, record_baseline=100, 
                    reward_scale=1.0, step_wrapper=lambda x: x):
        self.save_net = save_net
        self.save_name = save_name
        self.load_net = load_net
        self.load_name = load_name
        self.train = train
        self.max_episode = max_episode
        self.print_interval = print_interval
        self.max_video = max_video
        self.video_record_interval = video_record_interval
        self.record_baseline = record_baseline
        self.target_score = target_score
        self.reward_scale = reward_scale
        self.step_wrapper = step_wrapper

class Runner(metaclass=ABCMeta):
    def __init__(self, env_name, algo_name, algo_params, runner_params):
        self._env_name = env_name
        self._algo_name = algo_name
        self._algo_params = algo_params
        self._save_net = runner_params.save_net
        self._save_name = runner_params.save_name
        self._load_net = runner_params.load_net
        self._load_name = runner_params.load_name
        self._train = runner_params.train
        self._max_episode = runner_params.max_episode
        self._print_interval = runner_params.print_interval
        self._max_video = runner_params.max_video
        self._video_record_interval = runner_params.video_record_interval
        self._record_baseline = runner_params.record_baseline
        self._target_score = runner_params.target_score
        self._reward_scale = runner_params.reward_scale
        self._step_wrapper = runner_params.step_wrapper
        self._score = 0.0
        self._env = None
        self._recorder = None
        self._writer = None
        self._stop = False
        
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
        print('초기 설정 시작')
        self._episode_prepare()
        print(f'algorithm: {self._algo_name}')
        print(f'env: {self._env_name}')
        print(f'state space: {self._env.observation_space.shape}')
        print(f'action space: {self._env.action_space}')

        if self._load_net:
            print('네트워크 로딩 시작')
            self._load()

        print('시뮬레이션 시작')
        for n_epi in range(1, self._max_episode+1):
            # 마지막 비디오가 정상적으로 녹화되려면 반드시 다음 episode를 돌려야 함.
            self._episode_sim(n_epi)
            if self._stop:
                break
            self._record_video(n_epi)
            if n_epi % self._print_interval == 0:
                self._print_log(n_epi)
                self._score = 0.0
        print('시뮬레이션 종료')

        if self._save_net:
            print('네트워크 저장 시작')
            self._save()

    def _record_video(self, n_epi):
        cond = self._video_record_interval and (n_epi % self._video_record_interval == 0)
        aver_score = self._score / (n_epi % self._print_interval if n_epi % self._print_interval else self._print_interval)
        cond = cond or aver_score > self._record_baseline
        if cond:
            self._recorder.add_epi([n_epi])
            print(f'{n_epi=} 비디오 저장')
            if (len(self._recorder.recorded_epi()) >= self._max_video):
                self._stop = True
    
    def _print_log(self, n_epi):
        print("# of episode: {}, avg score: {:.1f}".format(n_epi, self._score/self._print_interval))
        self._writer.add_scalar("score/train", self._score/self._print_interval, n_epi)
        if (self._target_score <= self._score / self._print_interval):
            self._stop = True

    def _save(self, path='./weights'):
        import time, os
        name = self._save_name if self._save_name else (str(int(time.time())) + '.pt')
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self._net.state_dict(), path + '/' + self._algo_name + '-' + self._env_name + '-' + name)
        except OSError:
            raise

    def _load(self, path='./weights'):
        self._net.load_state_dict(torch.load(path + '/' + self._load_name))
        self._net.eval()

    @abstractmethod
    def _episode_prepare(self):
        pass

    @abstractmethod
    def _episode_sim(self, n_epi):
        pass
