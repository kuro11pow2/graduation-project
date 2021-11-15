import torch    # pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98

# pytorch의 nn.Module 클래스를 상속해서 만든다. 
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # 에피소드 완료되기 전까지 임시로 데이터를 저장할 변수
        self.data = []

        # state가 4개이기 때문에 feature vector는 4차원이다. 은닉층은 128차원으로 잡았다.
        # nn.Linear()는 fully connected이다. 즉, 4차원에서 128차원으로 가는 선형변환이다.
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # 정방향 전달하는 모든 호출에서 수행할 계산을 정의한다.
    def forward(self, x):
        # 4차원 feature vector를 128차원으로 변환하고 ReLU()를 취한 값을 tensor로 return한다.
        x = F.relu(self.fc1(x))
        # 128차원을 2차원으로 바꾸고 softmax()를 취한 값을 tensor로 return한다.
        x = F.softmax(self.fc2(x), dim=0)
        return x

    # 데이터 쌓아두기
    def put_data(self, item):
        self.data.append(item)

    # 네트워크 학습시키기
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        # 거꾸로 보는 이유는 return을 쉽게 계산하기 위해서이다. 
        # 예를 들어, 에피소드가 100스텝이라고 하자. 
        # 99스텝에서 reward를 구하면 이게 return이다.
        # 98스텝에서 reward를 구하면, 99스텝에서 계산한 return에 감마를 곱하고 reward를 더해서 return을 구할 수 있다.
        for r, prob in self.data[::-1]:
            # return을 cumulative하게 계산해 나간다.
            R = r + gamma * R
            # loss는 -log(pi) * R 로 정의한다.
            loss = -torch.log(prob) * R
            # autograd가 backpropagation을 자동으로 처리한다.
            # model의 parameter에 대한 loss의 gradient를 계산한다.
            loss.backward()
        # single optimization step을 수행하여 parameter를 update함.
        self.optimizer.step()
        # 이미 학습한 데이터를 제거한다.
        self.data = []