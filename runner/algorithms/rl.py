from abc import *

class RL(nn.Module):
    def __init__(self):
        super(RL, self).__init__()
        """
        1. transition 저장할 메모리 정의
        2. 사용할 네트워크 정의
        3. optimizer 정의
        """
        

    def pi(self, x, softmax_dim=0):
        """
        (policy optimization 계열의 경우)
        policy 계산값 (확률분포) return 
        """
    
    def v(self, x):
        """
        (policy optimization 계열의 경우)
        value function 계산값 return
        """
      
    def put_data(self, transition):
        """
        1. transition 데이터를 받아서 저장하기
        """
        
    def make_batch(self):
        """
        1. 수집한 transition 데이터들을 항목 별로 텐서로 만들어서 돌려주기
        """
        
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