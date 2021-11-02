from manifest import Manifest

from env import Env
from runner import RunnerParams
from algorithms.dqn import DQN
from algorithms.ddqn import DDQN
from algorithms.reinforce import Reinforce
from algorithms.actorcritic import ActorCritic

import itertools

class Trainer:
    def __init__(self, check_intervals=None):
        self._testcases = []
        self._check_intervals = check_intervals
        self.allcases = [*itertools.product(Manifest.envs, Manifest.algos)]
    
    def default_hyperparam(self, env, algo):
        algo_param = None

        if env == Env.CARTPOLE:
            if algo == Reinforce:
                algo_param = Manifest.get_param_class(algo)(
                        n_node=32, learning_rate=0.0005, gamma=0.98)
            elif algo == ActorCritic:
                algo_param = Manifest.get_param_class(algo)(
                        n_node=32, learning_rate=0.0005, gamma=0.98)
            elif algo == DQN:
                algo_param = Manifest.get_param_class(algo)(
                        n_node=32, learning_rate=0.0005, gamma=0.98, buffer_limit=50000, 
                        batch_size=32, n_train_start=2000, start_epsilon=0.1, update_interval=10)
            elif algo == DDQN:
                algo_param = Manifest.get_param_class(algo)(
                        n_node=32, learning_rate=0.0005, gamma=0.98, buffer_limit=50000, 
                        batch_size=32, n_train_start=2000, start_epsilon=0.1, update_interval=10)
            else:
                raise Exception(f'algorithm does not exist: {algo}')
        elif env == Env.LUNARLANDER:
            if algo == Reinforce:
                algo_param = Manifest.get_param_class(algo)(
                        n_node=128, learning_rate=0.0025, gamma=0.98)
            elif algo == ActorCritic:
                algo_param = Manifest.get_param_class(algo)(
                        n_node=128, learning_rate=0.0025, gamma=0.98)
            elif algo == DQN:
                algo_param = Manifest.get_param_class(algo)(
                        n_node=128, learning_rate=0.0025, gamma=0.98, buffer_limit=100000, 
                        batch_size=64, n_train_start=10000, start_epsilon=0.2, update_interval=20)
            elif algo == DDQN:
                algo_param = Manifest.get_param_class(algo)(
                        n_node=128, learning_rate=0.0025, gamma=0.98, buffer_limit=100000, 
                        batch_size=64, n_train_start=10000, start_epsilon=0.2, update_interval=20)
            else:
                raise Exception(f'algorithm does not exist: {algo}')
        else:
            raise Exception(f'env does not exist: {env}')

        return algo_param

    def add_case(self, env, algo, algo_param_dic=None):
        if not algo_param_dic:
            algo_param_dic = dict()

        algo_param_class = Manifest.get_param_class(algo)
        algo_param_dic = {**self.default_hyperparam(env, algo).__dict__, **algo_param_dic}
        algo_param = algo_param_class(**algo_param_dic)
        algo_runner = Manifest.get_runner_class(algo)

        self._testcases += [(env, algo_runner, algo_param)]

    def run(self, runner_param_dic=dict(), debug=False):
        for check_interval in self._check_intervals:
            runnerp = None
            if debug:
                runnerp = RunnerParams(save_net=True, max_video=1000, video_record_interval=self._check_intervals, 
                                            print_interval=self._check_intervals)
            else:
                runnerp = RunnerParams(save_net=True, video_record_interval=0, print_interval=0)

            runner_param_tmp = {**runnerp.__dict__, **runner_param_dic}
            runnerp = RunnerParams(**runner_param_tmp)

            runnerp.check_interval = check_interval

            for i, (env, runner, algop) in enumerate(self._testcases):
                runnerp.name_postfix=str(algop)

                if env == Env.CARTPOLE:
                    runnerp.target_score = 500.0
                    runnerp.reward_scale = 100.0
                elif env == Env.LUNARLANDER:
                    runnerp.target_score = 200.0
                    runnerp.reward_scale = 30.0

                runner(env.value, algop, runnerp).run()
