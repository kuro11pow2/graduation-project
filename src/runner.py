
from abc import *
import time, os

import gym
from recorder import Recorder
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
import torch


class RunnerParams:
    def __init__(self, *, save_net=False, name_postfix=None, load_net=False, load_name=None, 
                    train=True,  max_episode=10000, check_interval=20, print_interval=20,
                    max_video=3, video_record_interval=0,
                    target_score=0, 
                    reward_scale=1.0, step_wrapper=lambda x: x,
                    save_step_log=False):
        self.save_net = save_net
        self.name_postfix = name_postfix
        self.load_net = load_net
        self.load_name = load_name
        self.train = train
        self.max_episode = max_episode
        self.check_interval = check_interval
        self.print_interval = print_interval
        self.max_video = max_video
        self.video_record_interval = video_record_interval
        self.target_score = target_score
        self.reward_scale = reward_scale
        self.step_wrapper = step_wrapper
        self.save_step_log = save_step_log
    
    def __str__(self):
        s = ''
        s += f'train={self.train}-'
        s += f'intvl={self.check_interval}-'
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
        self._check_interval = runner_params.check_interval
        self._print_interval = runner_params.print_interval
        self._max_video = runner_params.max_video
        self._video_record_interval = runner_params.video_record_interval
        self._target_score = runner_params.target_score
        self._reward_scale = runner_params.reward_scale
        self._step_wrapper = runner_params.step_wrapper
        self._save_step_log = runner_params.save_step_log
        self._score = 0.0
        self._score_sum = 0.0
        self._end_score = None
        self._env = None
        self._algo = None
        self._recorder = None
        self._writer = None
        self._logger = None
        
    def run(self):
        env = gym.make(self._env_name)
        self._recorder = Recorder(env, False)
        self._env = self._recorder.wrapped_env()
        if not isinstance(self._env.action_space, gym.spaces.discrete.Discrete):
            raise Exception('discrete space만 지원됨.')
        name = f'{self._algo_name}'
        name += f'-{self._env_name}'
        name += f'-{str(self._runner_params)}'
        if self._name_postfix:
            name += f'-{self._name_postfix}'
        name += f'-{(str(int(time.time())))}'
        self._writer = SummaryWriter(log_dir='runs/'+name)
        self._logger = Logger('logs', name)
        self._episode_loop()
        self._env.close()
        if self._save_step_log:
            self._logger.save()
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
            print('네트워크 불러오기')
            self._load()

        print('시뮬레이션 시작')
        for n_epi in range(self._max_episode):
            record_video = self._video_record_interval and (n_epi + 1) % self._video_record_interval == 0
            print_log = self._print_interval and (n_epi + 1) % self._print_interval == 0
            cond_check = self._check_interval and (n_epi + 1) % self._check_interval == 0

            if record_video:
                self._recorder.record_start()
                self._episode_sim(n_epi)
                print(f'{n_epi=}, {self._score=} 비디오 저장')
                self._recorder.record_end()
            else:
                self._episode_sim(n_epi)

            if print_log:
                self._print_log(n_epi, self._score)
            
            if cond_check:
                avg_score = self._score_sum / self._check_interval
                self._write_check_log(n_epi, avg_score)
                
                if self._is_done(n_epi, avg_score):
                    print(f'종료 조건 만족. 최종 {self._check_interval}번 평균 점수 {avg_score}')
                    self._end_score = avg_score
                    break
                
                self._score_sum = 0.0

        print('시뮬레이션 종료')

        if self._save_net:
            print('네트워크 저장')
            self._save()
        
    def _print_log(self, n_epi, score):
        print(f"에피소드: {n_epi}, 점수: {score:.1f}")
    
    def _write_check_log(self, n_epi, avg_score):
        self._writer.add_scalar("episode/avg_score", avg_score, n_epi)

    def _write_step_log(self, step, n_epi, state, action, reward, done):
        state = {f'state{i}':n for i, n in enumerate(state)}
        params = { 'step':step, 'episode':n_epi, 'reward':reward, 'done':done, 'action':action, **state}
        self._logger.append(params)
    
    def _is_done(self, n_epi, avg_score):
        if (self._target_score <= avg_score):
            return True
        elif self._recorder.n_recorded >= self._max_video:
            return True
        else:
            return False

    def _save(self, dir='./weights'):
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
            name = f'{self._algo_name}'
            name += f'-{self._env_name}'
            name += f'-{int(self._end_score)}'
            name += f'-{str(self._runner_params)}'
            if self._name_postfix:
                name += f'-{self._name_postfix}'
            name += f'-{(str(int(time.time())))}.pt'
            self._algo.save_net(dir, name)
        except OSError:
            raise
            
    def _load(self, dir='./weights'):
        self._algo.load_net(dir, self._load_name)
        self._algo.set_eval()

    @abstractmethod
    def _episode_prepare(self):
        pass

    @abstractmethod
    def _episode_sim(self, n_epi):
        pass
