import torch    # pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReinforceParams:
    def __init__(self, *, n_node=128, learning_rate=0.0005, gamma=0.98):
        self.n_node = n_node
        self.learning_rate = learning_rate
        self.gamma = gamma

    def __str__(self):
        s = ''
        s += f'node={self.n_node}-'
        s += f'lRate={self.learning_rate}-'
        s += f'gma={self.gamma}'
        return s

class Reinforce(nn.Module):
    def __init__(self, n_state, n_action, params):
        super(Reinforce, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_node = params.n_node
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.data = []

        self.fc1 = nn.Linear(self.n_state,self.n_node) 
        self.fc2 = nn.Linear(self.n_node,self.n_action) 
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
    
    def put_data(self, transition):
        self.data.append(transition)
  
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

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