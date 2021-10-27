from env import Env
from runner import RunnerParams
from algorithms.dqn import DQN, DQNParams
from algorithms.dqn_runner import DQNRunner
from algorithms.ddqn import DDQN, DDQNParams
from algorithms.ddqn_runner import DDQNRunner
from algorithms.reinforce import Reinforce, ReinforceParams
from algorithms.reinforce_runner import ReinforceRunner
from algorithms.actorcritic import ActorCritic, ActorCriticParams
from algorithms.actorcritic_runner import ActorCriticRunner

class Manifest:
    envs = [Env.CARTPOLE, Env.LUNARLANDER]
    algos = [Reinforce, ActorCritic, DQN, DDQN]
    algo_runners = {Reinforce: ReinforceRunner, ActorCritic: ActorCriticRunner, 
                    DQN: DQNRunner, DDQN: DDQNRunner}
    algo_params = {Reinforce: ReinforceParams, ActorCritic: ActorCriticParams, 
                    DQN: DQNParams, DDQN: DDQNParams}
    algo_name_dic = {'Reinforce': Reinforce, 'ActorCritic': ActorCritic, 'DQN': DQN, 'DDQN': DDQN}

    short_dic = {
        'train': 'train',
        'intvl': 'check_interval',
        'rwdscl': 'reward_scale',
        'node': 'n_node',
        'lRate': 'learning_rate',
        'gma': 'gamma',
        'nRoll': 'n_rollout',
        'nBuf': 'buffer_limit',
        'nBat': 'batch_size',
        'nStrt': 'n_train_start',
        'updIntvl': 'update_interval',
        'lmb': 'lmbda',
        'epsclp': 'eps_clip',
        'k': 'k_epoch',
        't': 't_horizon'
    }
    full_dic = {v: k for k, v in short_dic.items()}

    @classmethod
    def get_algo_class(cls, algo_name):
        return cls.algo_name_dic[algo_name]

    @classmethod
    def get_runner_class(cls, algo_class):
        return cls.algo_runners[algo_class]

    @classmethod
    def get_param_class(cls, algo_class):
        return cls.algo_params[algo_class]
    
    @classmethod
    def get_param_full(cls, short_param):
        return cls.short_dic[short_param]
    
    @classmethod
    def get_param_short(cls, full_param):
        return cls.full_dic[full_param]