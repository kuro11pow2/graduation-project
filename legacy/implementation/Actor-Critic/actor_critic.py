import torch    # pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        # 이번엔 REINFORCE와 다르게 네트워크가 3개이다. 
        # fc1에서 두 개로 갈라져서 fc_pi와 fc_v로 간다.

        # 현재 state를 256개 수로 인코딩했다고 이해하면 된다.
        self.fc1 = nn.Linear(4,256) 
        # 256개에서 pi를 나타내는 2개로 
        self.fc_pi = nn.Linear(256,2) 
        # 256개에서 v를 나타내는 1개로
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        # 확률분포이기 때문에 softmax 취함.
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        # batch로 학습하면 learning-rate에 덜 민감해지고, 학습이 잘 된다.
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        tmp = torch.tensor(s_lst, dtype=torch.float), \
                            torch.tensor(a_lst), \
                            torch.tensor(r_lst, dtype=torch.float), \
                            torch.tensor(s_prime_lst, dtype=torch.float), \
                            torch.tensor(done_lst, dtype=torch.float)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = tmp
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        # td target과 delta 식 그 자체임
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        # 확률 분포를 얻어서
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        # detach() 메소드는 원본 tensor에서 gradient 전파를 방지하는 tensor를 생성한다.
        # storage를 공유하기 때문에 원본 tensor가 변하면 같이 바뀐다.
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()