import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPOParams:
    def __init__(self, *, n_node=128, learning_rate=0.0001, gamma=0.98, lmbda=0.95, eps_clip=0.1, k_epoch=3, t_horizon=20):
        self.n_node = n_node
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.k_epoch = k_epoch
        self.t_horizon = t_horizon
    
    def __str__(self):
        s = ''
        s += f'node={self.n_node}-'
        s += f'lRate={self.learning_rate}-'
        s += f'gma={self.gamma}-'
        s += f'lmb={self.lmbda}-'
        s += f'epsclp={self.eps_clip}-'
        s += f'k={self.k_epoch}-'
        s += f't={self.t_horizon}'
        return s

class PPO(nn.Module):
    def __init__(self, n_state, n_action, params):
        super(PPO, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_node = params.n_node
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.lmbda = params.lmbda
        self.eps_clip = params.eps_clip
        self.k_epoch = params.k_epoch
        self.t_horizon = params.t_horizon
        self.data = []
        
        self.fc1   = nn.Linear(self.n_state, self.n_node)
        self.fc_pi = nn.Linear(self.n_node, self.n_action)
        self.fc_v  = nn.Linear(self.n_node, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def pi(self, x, softmax_dim=0):
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
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.k_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

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
