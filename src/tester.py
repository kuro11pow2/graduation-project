from runner import RunnerParams
from env import Env

class RunnerTester:
    def __init__(self, runner, algo_params, envs: list=None):
        self.target_runner = runner
        self.algo_params = algo_params
        self.envs = envs if envs else list(Env)

    def test(self):
        """
        알고리즘 동작 여부를 확인하기 위한 최저 기준
        """
        print(f'runner 테스트 시작')
        for env_name in self.envs:
            if env_name == Env.CARTPOLE:
                runner_params = RunnerParams(train=True, save_net=False,
                                            max_episode=10000, target_score=499, record_baseline=450, 
                                            video_record_interval=0,
                                            reward_scale=100.0, max_video=10)
            elif env_name == Env.MOUNTAINCAR:
                runner_params = RunnerParams(train=True, save_net=False,
                                            max_episode=10000, target_score=200, record_baseline=175, 
                                            video_record_interval=80,
                                            reward_scale=100.0, max_video=10, step_wrapper=lambda x: (x[0], x[1] + abs(x[0][1])/(x[0][0] + 1.3) + ((x[0][0] + 1.2) ** 2), x[2], x[3]))
            elif env_name == Env.LUNARLANDER:
                runner_params = RunnerParams(train=True, save_net=False,
                                            max_episode=10000, target_score=100, record_baseline=0, 
                                            video_record_interval=80,
                                            reward_scale=20.0, max_video=10)
            else:
                continue
            tmp = self.target_runner(env_name.value, self.algo_params, runner_params)
            tmp.run()
        return True
            

class EnvTester:
    def __init__(self, env: Env, runners: list=[]):
        self.env = env
        self.runners = runners

    def test(self):
        print(f'env 테스트 시작')