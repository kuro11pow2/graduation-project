from manifest import Manifest

from runner import RunnerParams

class Player:

    def __init__(self, path, model_names):
        self.path = path
        self.model_names = model_names

    def run(self, debug=False):
        cases = []
        for name in self.model_names:
            tokens = name.split('_')
            load_name = name + ".pt"
            algo_name, env_name, last_score = tokens[0:3]
            train, check_interval, reward_scale = tokens[3:6]
            algo_params = dict()
            for token in tokens[6:-1]:
                k, v = token.split('=')
                k = Manifest.get_param_full(k)
                try: algo_params[k] = int(v)
                except: algo_params[k] = float(v)
            algo = Manifest.get_algo_class(algo_name)
            algop = Manifest.get_param_class(algo)(**algo_params)
            runner = Manifest.get_runner_class(algo)
            cases.append((env_name, runner, algop, algo_params, load_name))

        for env, runner, algop, algo_params, load_name in cases:
            print(f'\t[ {env}, {runner} ]\n parameters {algo_params.items()}\n')
            runnerp = None
            if debug:
                runnerp = RunnerParams(train=False, save_net=False, load_net=True, target_score=9999.0,
                                        load_name=load_name, name_postfix=str(algop), 
                                        check_interval=1, max_video=3, save_check_log=False, save_step_log=True,
                                        print_interval=1, video_record_interval=1, max_episode=1000)
            else:
                runnerp = RunnerParams(train=False, save_net=False, load_net=True, target_score=9999.0,
                                        load_name=load_name, name_postfix=str(algop), 
                                        check_interval=1, max_video=0, save_check_log=False, save_step_log=True,
                                        print_interval=0, video_record_interval=0, max_episode=1000)

            runner(env, algop, runnerp).run()
        
        print('모두 종료됨')