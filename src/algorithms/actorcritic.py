import torch    # pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticParams:
    def __init__(self, *, n_node=256, learning_rate=0.0002, gamma=0.98):
        self.n_node = n_node
        self.learning_rate = learning_rate
        self.gamma = gamma

    def __str__(self):
        s = ''
        s += f'node={self.n_node}_'
        s += f'lRate={self.learning_rate}_'
        s += f'gma={self.gamma}'
        return s

class ActorCritic(nn.Module):
    def __init__(self, n_state, n_action, params):
        super(ActorCritic, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_node = params.n_node
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.data = []

        self.fc1 = nn.Linear(self.n_state,self.n_node) 
        self.fc_pi = nn.Linear(self.n_node,self.n_action) 
        self.fc_v = nn.Linear(self.n_node,1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def pi(self, x, softmax_dim = 0): # forward로 변경?
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
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
        td_target = r + self.gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def save_net(self, dir, name):
        torch.save({
            'net': self.state_dict()
        }, dir + '/' + name)

    def load_net(self, dir, name):
        checkpoint = torch.load(dir + '/' + name)
        self.load_state_dict(checkpoint['net'])
    
    def set_train(self):
        self.train()
    
    def set_eval(self):
        self.eval()