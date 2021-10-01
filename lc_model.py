import torch
import torch.nn as nn

class SVM(nn.Module):
    def __init__(self,hidden_size):
        super(SVM, self).__init__()
        self.linear1 = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        y = self.sigmoid(self.linear1(x))
        return y.view(-1)

class LinearRegression(nn.Module):
    def __init__(self,hidden_size):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(hidden_size,3)
    def forward(self,x):
        y = self.linear1(x)
        return y.view(-1,3)

class GRU_SVM(nn.Module):
    def __init__(self,hidden_size):
        super(GRU_SVM, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(2,self.hidden_size,batch_first=True)
        self.output = nn.Linear(self.hidden_size,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, h):
        y,h = self.gru(x,h)
        y = self.output(y)
        y = self.sigmoid(y)
        return y

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.layer1 = nn.Linear(14,32)
        self.layer2 = nn.Linear(32,16)
        self.layer3 = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.sigmoid(y)
        return y
