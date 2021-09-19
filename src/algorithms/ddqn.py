import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DDQNParams:
    def __init__(self, *, n_node=128, learning_rate=0.0005, gamma=0.98, buffer_limit=50000, batch_size=32, n_train_start=2000, start_epsilon=0.08, update_interval=20):
        self.n_node = n_node
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_limit = buffer_limit
        self.batch_size = batch_size
        self.n_train_start = n_train_start
        self.start_epsilon = start_epsilon
        self.update_interval = update_interval

    def __str__(self):
        s = ''
        s += f'node={self.n_node}-'
        s += f'lRate={self.learning_rate}-'
        s += f'gma={self.gamma}-'
        s += f'nBuf={self.buffer_limit}-'
        s += f'nBat={self.batch_size}-'
        s += f'nStrt={self.n_train_start}-'
        s += f'updIntvl={self.update_interval}'
        return s
        

class DDQN(nn.Module):
    def __init__(self, n_state, n_action, params):
        super(DDQN, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_node = params.n_node
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.buffer_limit = params.buffer_limit
        self.batch_size = params.batch_size
        self.n_train_start = params.n_train_start
        self.start_epsilon = params.start_epsilon
        self.update_interval = params.update_interval

        self.buffer = collections.deque(maxlen=self.buffer_limit)

        self.fc1 = nn.Linear(self.n_state, self.n_node)
        self.fc2 = nn.Linear(self.n_node, self.n_node)
        self.fc3 = nn.Linear(self.n_node, self.n_action)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()

    def put(self, transition): # 버퍼에 데이터 뒤에 넣기 (크기가 초과되면 가장 앞 데이터가 제거됨)
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n) # 버퍼에서 n개 샘플링하여 미니 배치 만든다
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s) # 얘는 원래 list
            a_lst.append([a]) # 얘는 list가 아니라서 감싸줌
            r_lst.append([r]) # 얘도. (타입을 맞춰준 것)
            s_prime_lst.append(s_prime)
            done_mask = 0 if done else 1
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def buffer_size(self):
        return len(self.buffer)
                
    def train_net(self, target_net):
        for i in range(10):
            s,a,r,s_prime,done_mask = self.sample(self.batch_size)

            qnet_outs = self(s)
            tnet_outs = target_net(s_prime)

            max_a = torch.max(qnet_outs, 1)[1].unsqueeze(1)
            q_estimated = tnet_outs.gather(1, max_a)
            target = r + self.gamma * q_estimated * done_mask

            q_a = qnet_outs.gather(1, a)
            loss = F.smooth_l1_loss(q_a, target)
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
