
from abc import *
import time, os

import gym
from recorder import Recorder
from torch.utils.tensorboard import SummaryWriter
import torch

class RunnerParams:
    def __init__(self, *, save_net=False, name_postfix=None, load_net=False, load_name=None, 
                    train=True,  max_episode=10000, interval=20, 
                    max_video=3, video_record_interval=0,
                    target_score=0, 
                    reward_scale=1.0, step_wrapper=lambda x: x):
        self.save_net = save_net
        self.name_postfix = name_postfix
        self.load_net = load_net
        self.load_name = load_name
        self.train = train
        self.max_episode = max_episode
        self.interval = interval
        self.max_video = max_video
        self.video_record_interval = video_record_interval
        self.target_score = target_score
        self.reward_scale = reward_scale
        self.step_wrapper = step_wrapper
    
    def __str__(self):
        s = ''
        s += f'train={self.train}-'
        s += f'intvl={self.interval}-'
        s += f'rwdscl={self.reward_scale}'
        return s

class Runner(metaclass=ABCMeta):
    def __init__(self, env_name, algo_name, algo_params, runner_params):
        self._env_name = env_name
        self._algo_name = algo_name
        self._algo_params = algo_params
        self._runner_params = runner_params
        self._save_net = runner_params.save_net
        self._name_postfix = runner_params.name_postfix
        self._load_net = runner_params.load_net
        self._load_name = runner_params.load_name
        self._train = runner_params.train
        self._max_episode = runner_params.max_episode
        self._interval = runner_params.interval
        self._max_video = runner_params.max_video
        self._video_record_interval = runner_params.video_record_interval
        self._target_score = runner_params.target_score
        self._reward_scale = runner_params.reward_scale
        self._step_wrapper = runner_params.step_wrapper
        self._score = 0.0
        self._end_score = None
        self._env = None
        self._net = None
        self._recorder = None
        self._writer = None
        
    def run(self):
        env = gym.make(self._env_name)
        self._recorder = Recorder(env, False)
        self._env = self._recorder.wrapped_env()
        if not isinstance(self._env.action_space, gym.spaces.discrete.Discrete):
            raise Exception('discrete space만 지원됨.')
        name = f'runs/{self._algo_name}'
        name += f'-{self._env_name}'
        name += f'-{str(self._runner_params)}'
        if self._name_postfix:
            name += f'-{self._name_postfix}'
        name += f'-{(str(int(time.time())))}'
        self._writer = SummaryWriter(log_dir=name)
        self._episode_loop()
        self._env.close()
        self._writer.flush()
        self._writer.close()

    def _episode_loop(self):
        print('초기 설정')
        self._episode_prepare()
        print(f'algorithm: {self._algo_name}')
        print(f'env: {self._env_name}')
        print(f'state space: {self._env.observation_space.shape}')
        print(f'action space: {self._env.action_space}')

        if self._load_net:
            print('네트워크 로딩')
            self._load()

        print('시뮬레이션 시작')
        for n_epi in range(self._max_episode):
            reset_score = n_epi % self._interval == 0
            if reset_score:
                self._score = 0.0

            record_video = self._video_record_interval and n_epi % self._video_record_interval == 0
            if record_video:
                self._recorder.record_start()
                self._episode_sim(n_epi)
                avg_score = self._score / (n_epi % self._interval + 1)
                print(f'{n_epi=}, {avg_score=} 비디오 저장')
                self._recorder.record_end()
            else:
                self._episode_sim(n_epi)

            avg_score = self._score / (n_epi % self._interval + 1)
            print_log = n_epi % self._interval == 0
            if print_log:
                self._print_log(n_epi)

            if self._is_done(n_epi):
                print(f'종료 조건 만족. 최종 {self._interval}번 평균 점수 {avg_score}')
                self._end_score = avg_score
                break

        print('시뮬레이션 종료')

        if self._save_net:
            print('네트워크 저장')
            self._save()
        
    def _print_log(self, n_epi):
        avg_score = self._score / (n_epi % self._interval + 1)
        print(f"에피소드: {n_epi}, 평균 점수: {avg_score:.1f}")
        self._writer.add_scalar("score/train", avg_score, n_epi)
    
    def _is_done(self, n_epi):
        avg_score = self._score / (n_epi % self._interval + 1)
        if (self._target_score <= avg_score):
            return True
        elif self._recorder.n_recorded >= self._max_video:
            return True
        else:
            return False

    def _save(self, path='./weights'):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            name = f'{path}/'
            name += f'{self._algo_name}'
            name += f'-{self._env_name}'
            name += f'-{self._end_score}'
            name += f'-{str(self._runner_params)}'
            if self._name_postfix:
                name += f'-{self._name_postfix}'
            name += f'-{(str(int(time.time())))}.pt'
            torch.save(self._net.state_dict(), name)
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
