import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPOlstmParams:
    def __init__(self, *, n_node=64, learning_rate=0.0001, gamma=0.98, lmbda=0.95, eps_clip=0.1, k_epoch=3, t_horizon=20):
        self.n_node = n_node
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.k_epoch = k_epoch
        self.t_horizon = t_horizon

    def __str__(self):
        s = ''
        s += f'node={self.n_node}_'
        s += f'lRate={self.learning_rate}_'
        s += f'gma={self.gamma}_'
        s += f'lmb={self.lmbda}_'
        s += f'epsclp={self.eps_clip}_'
        s += f'k={self.k_epoch}_'
        s += f't={self.t_horizon}'
        return s

class PPOlstm(nn.Module):
    def __init__(self, n_state, n_action, params):
        super(PPOlstm, self).__init__()
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
        
        self.fc1   = nn.Linear(n_state, self.n_node)
        self.lstm  = nn.LSTM(self.n_node, self.n_node//2)
        self.fc_pi = nn.Linear(self.n_node//2, n_action)
        self.fc_v  = nn.Linear(self.n_node//2, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, self.n_node)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, self.n_node)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(self.k_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
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
