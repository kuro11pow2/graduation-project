import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQNParams:
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
        s += f'node={self.n_node}_'
        s += f'lRate={self.learning_rate}_'
        s += f'gma={self.gamma}_'
        s += f'nBuf={self.buffer_limit}_'
        s += f'nBat={self.batch_size}_'
        s += f'nStrt={self.n_train_start}_'
        s += f'updIntvl={self.update_interval}'
        return s

class QNet(nn.Module):
    def __init__(self, n_state, n_action, params):
        super(QNet, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_node = params.n_node
        self.learning_rate = params.learning_rate

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
        if random.random() < epsilon:
            return random.randrange(self.n_action)
        else: 
            return out.argmax().item()

class DQN:
    def __init__(self, n_state, n_action, params):
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.buffer_limit = params.buffer_limit
        self.batch_size = params.batch_size
        self.n_train_start = params.n_train_start
        self.start_epsilon = params.start_epsilon
        self.update_interval = params.update_interval

        self.net = QNet(n_state, n_action, params)
        self.target_net = QNet(n_state, n_action, params)
        self.replay_buffer = collections.deque(maxlen=self.buffer_limit)
    
    def update_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def append_data(self, transition):
        """
        replay buffer의 뒤에 transition 삽입
        buffer_limit을 초과하면 FIFO로 동작
        """
        self.replay_buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.replay_buffer, n) # 버퍼에서 n개 샘플링하여 미니 배치 만든다
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

    def sample_action(self, obs):
        return self.net.sample_action(obs, self.epsilon)
    
    def buffer_size(self):
        return len(self.replay_buffer)
                
    def train_net(self):
        for i in range(10):
            s,a,r,s_prime,done_mask = self.sample(self.batch_size)

            q_out = self.net(s)
            q_a = q_out.gather(1,a) 

            max_q_prime = self.target_net(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            
            self.net.optimizer.zero_grad()
            loss.backward()
            self.net.optimizer.step()
    
    def save_net(self, dir, name):
        torch.save({
            'net': self.net.state_dict(),
            'target_net': self.target_net.state_dict()
        }, dir + '/' + name)

    def load_net(self, dir, name):
        checkpoint = torch.load(dir + '/' + name)
        self.net.load_state_dict(checkpoint['net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
    
    def set_train(self):
        self.net.train()
        self.target_net.train()
    
    def set_eval(self):
        self.net.eval()
        self.target_net.eval()