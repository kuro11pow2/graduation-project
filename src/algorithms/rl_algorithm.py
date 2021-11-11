from abc import *
from collections import deque

class RlAlgorithm(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        """
        1. 사용할 네트워크 정의
        2. optimizer 정의
        """

    @abstractmethod
    def get_action(self, s):
        """
        현재 상태에서 수행할 액션을 return
        (pi 또는 v를 사용하여 계산)
        """
        
    def pi(self, x, softmax_dim=0):
        """
        policy 계산값 return 
        """
    
    def v(self, x):
        """
        value function 계산값 return
        """
    
    @abstractmethod
    def train_net(self):
        """
        1. 학습할 데이터 얻기
        2. loss 구하기
        3. optimizer의 gradient를 비우기
            optimizer.zero_grad()
        4. gradient 구하기
            loss.backward()
        5. 업데이트하기
            optimizer.step()
        """


class ExpReplay(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.replay_buffer = deque()

    @abstractmethod
    def put_data(self, transition):
        """
        1. transition 데이터를 받아서 저장하기
        """

    @abstractmethod
    def make_batch(self):
        """
        1. 수집한 transition 데이터들을 항목 별로 텐서로 만들어서 돌려주기
        """
