from runner import Runner, RunnerParams
from env import Env

class RunnerTester:
    def __init__(self, runner: Runner, algo_params, envs: list=[]):
        self.target_runner = runner
        self.algo_params = algo_params
        self.envs = envs if len(envs) else list(Env)

    def test(self):
        """
        알고리즘 동작 여부를 확인하기 위한 최저 기준
        """
        print(f'runner 테스트 시작')
        for env_name in self.envs:
            if env_name == Env.CARTPOLE:
                runner_params = RunnerParams(train=True, save_model=True, \
                    max_episode=2000, record_baseline=300, reward_scale=100.0, max_video=1)
            elif env_name == Env.MOUNTAINCAR:
                runner_params = RunnerParams(train=True, save_model=True, \
                    max_episode=2000, record_baseline=100, reward_scale=20.0, max_video=1)
            elif env_name == Env.LUNARLANDER:
                runner_params = RunnerParams(train=True, save_model=True, \
                    max_episode=2000, record_baseline=100, reward_scale=20.0, max_video=1)
            else:
                continue
            runner = self.target_runner(env_name.value, self.algo_params, runner_params)
            runner.run()
            if runner._stop == False:
                return False
        return True
            

class EnvTester:
    def __init__(self, env: Env, runners: list=[]):
        self.env = env
        self.runners = runners

    def test(self):
        print(f'env 테스트 시작')