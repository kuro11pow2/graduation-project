import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000 # 버퍼의 최대 크기
batch_size    = 32 # 버퍼에서 샘플링하는 사이즈

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition): # 버퍼에 데이터 뒤에 넣기 (크기가 초과되면 가장 앞 데이터가 제거됨)
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n) # 버퍼에서 n개 샘플링하여 미니 배치 만든다
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        # state, action 다 따로 리스트에 담아서 텐서로 만든다.
        for transition in mini_batch:
            # 현재 state, action, reward, 다음 state, 종료 여부 
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s) # 얘는 원래 list
            a_lst.append([a]) # 얘는 list가 아니라서 감싸줌
            r_lst.append([r]) # 얘도. (타입을 맞춰준 것)
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Q value는 음수일 수도 있는데 relu는 0이상의 값만 리턴하므로 여기엔 activation 안 넣는다.
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        # 입실론 그리디를 위해 액션을 따로 구현함
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    # 에피소드 끝날 때마다 얘가 호출됨. 
    # 그때마다 replay buffer에서 랜덤으로 뽑아서 업데이트하는 걸 10번 진행함.
    # 이러면 버퍼가 작을 때 중복으로 뽑힐 확률이 너무 높아짐. 
    # 그래서 버퍼가 충분히 쌓인 이후에 학습을 진행함.
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        # 여기서 s는 state 32개이다.
        # (sample 메소드가 리턴하는 게 텐서)
        # 즉 s는 shape: [32, 4] 이다. (cartpole의 state가 4개임)
        # 그럼 q_out은? [32, 2]가 된다. (cartpole의 action이 2개임)
        # 이렇게 batch로 처리하면 따로따로 32번 처리하는 것보다 훨씬 빠름
        q_out = q(s)

        # q_out에는 s의 모든 액션에 대해 계산한 값이 담겨있다.
        # a는 내가 실제로 수행한 액션이다.
        # q_a는 q_out에서 내가 실제로 한 액션에 대한 값만 뽑은 것.
        # 당연히 q_a의 shape은 [32, 1]이 됨.
        # (참고로 gather의 첫 번째 인자는 인덱스를 셀 차원의 번호이다.)
        # (cartpole의 경우 q_out의 1번 차원 크기는 shape의 두 번째 항인 2가 됨.)
        # (예를 들어 a = [0, 1]이라고 치면 1번 차원 인덱스를 0, 1로 고정한 값을 수집함)
        q_a = q_out.gather(1,a) 

        # q가 아닌 q_target을 사용하는 것에 주의. Vpi 대신하는 친구
        # q_target의 shape은 [32, 2]가됨. 두 액션 중 큰 값을 내는 친구를 골라야 함.
        # .max(1)은 1번 축 기준 최대를 고른다는 말. 1번 축은 action에 대한 축임.
        # 즉 1번 축 상에서 max라는 말은 특정 state에서 max인 action을 고른다는 것.
        # 따라서 q_target(s_prime).max(1)의 shape은 [32]가 됨.
        # 근데 연산을 하려면 shape을 맞춰줘야하기 때문에 unsqueeze를 해서 [32, 1]로 늘린다.
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # 만약 이 액션으로 에피소드가 끝났다면 done_mask = 0임.
        # 마지막 state에선 r만 적용하기 위해 이렇게 해주는 것.
        target = r + gamma * max_q_prime * done_mask
        # 두 값의 차이가 우리가 줄이고 싶은 것. = loss function
        loss = F.smooth_l1_loss(q_a, target)
        
        # optimizer의 gradient를 비우고
        optimizer.zero_grad()
        # gradient가 back propagation 되면서 구해진다
        loss.backward()
        # gradient를 통해 업데이트됨.
        optimizer.step()
