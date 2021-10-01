import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import sys
sys.path.append("..")
from datatool import train_test_val_split
from sklearn.metrics import accuracy_score
import time 
import h5py
# with open("../data/pickle_data/salstm_combine_feature_12_dim.pkl","rb") as f:
#     data = pkl.load(f)
# right_feature = data['right_data']
# left_feature = data['left_data']
# keep_feature = data['keep_data']
with h5py.File("../data/pickle_data/salstm_combine_feature_12_dim.hdf5","r") as f:
    right_data = f['right_data'][()]
    left_data = f['left_data'][()]
    keep_data = f['keep_data'][()]
right_feature,left_feature,keep_feature = right_data[:,0,:,:],left_data[:,0,:,:],keep_data[:,0,:,:]
print(right_feature.shape,left_feature.shape,keep_feature.shape)
right_label = np.ones(right_feature.shape[0])*0
left_label = np.ones(left_feature.shape[0])*1
keep_label = np.ones(keep_feature.shape[0])*2
print(right_label.shape,left_label.shape,keep_label.shape)

SEED = 10
BS = 256
LR = 1e-5
WD = 5e-5
EPOCH = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


right_train, right_eval, right_test = train_test_val_split(right_feature,test_size=0.2,val_size=0.1,seed=SEED)
left_train, left_eval, left_test = train_test_val_split(left_feature,test_size=0.2,val_size=0.1,seed=SEED)
keep_train, keep_eval, keep_test = train_test_val_split(keep_feature,test_size=0.2,val_size=0.1,seed=SEED)
right_repeat_num = int(keep_feature.shape[0]/right_feature.shape[0])
left_repeat_num = int(keep_feature.shape[0]/left_feature.shape[0])
right_train, right_eval, right_test = right_train.repeat(right_repeat_num,axis=0),right_eval.repeat(right_repeat_num,axis=0),right_test.repeat(right_repeat_num,axis=0)
left_train, left_eval, left_test = left_train.repeat(left_repeat_num,axis=0),left_eval.repeat(left_repeat_num,axis=0),left_test.repeat(left_repeat_num,axis=0)
# 3
X_train, X_eval, X_test = np.vstack((left_train,right_train,keep_train)),np.vstack((left_eval,right_eval,keep_eval)),np.vstack((left_test,right_test,keep_test))
y_train = np.hstack((np.ones(left_train.shape[0])*2,np.ones(right_train.shape[0]),np.zeros(keep_train.shape[0])))
y_eval = np.hstack((np.ones(left_eval.shape[0])*2,np.ones(right_eval.shape[0]),np.zeros(keep_eval.shape[0])))
y_test = np.hstack((np.ones(left_test.shape[0])*2,np.ones(right_test.shape[0]),np.zeros(keep_test.shape[0])))
# 4
trainingDataset = TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train))
trainingDataloader = DataLoader(trainingDataset,batch_size=BS,shuffle=True)
testDataset = TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test))
testDataloader = DataLoader(testDataset,batch_size=BS,shuffle=True)
devDataset = TensorDataset(torch.from_numpy(X_eval),torch.from_numpy(y_eval))
devDataloader = DataLoader(devDataset,batch_size=BS,shuffle=True)

class SA_LSTM(nn.Module):
    def __init__(self):
        super(SA_LSTM, self).__init__()
        self.linear = nn.Linear(12,64)
        self.lstm = nn.LSTM(64,128,batch_first=True)
        self.output = nn.Linear(128,3)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        y = self.linear(x)
        h_ns, (h_n, c_n) = self.lstm(y)
        y = self.sigmoid(self.output(h_n))
        return y.squeeze(0)

salstm = SA_LSTM().to(DEVICE)
cost_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(salstm.parameters(),lr=LR,weight_decay=WD)
for ep in range(EPOCH):
    t1 = time.time()
    # train
    l_sum = 0
    train_true, train_pred = [], []
    for i,(x,y) in enumerate(trainingDataloader):
        _y = salstm(x.float().to(DEVICE))
        l = cost_function(_y,y.long().to(DEVICE))
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        l_sum += l.data.cpu().numpy()
        train_true.append(y.data.cpu().numpy())
        train_pred.append(torch.max(_y, 1)[1].data.cpu().numpy().squeeze())
    train_true, train_pred = np.hstack(train_true), np.hstack(train_pred)
    train_acc = accuracy_score(train_true, train_pred)
    # test
    test_true, test_pred = [], []
    for i,(x,y) in enumerate(testDataloader):
        _y = salstm(x.float().to(DEVICE))
        test_true.append(y.data.cpu().numpy())
        test_pred.append(torch.max(_y, 1)[1].data.cpu().numpy().squeeze())
    test_true, test_pred = np.hstack(test_true), np.hstack(test_pred)
    test_acc = accuracy_score(test_true, test_pred)
    t2 = time.time()
    print(f"epoch [{ep}] | time: {t2-t1:.2f} | train_acc: {train_acc:.2f} | test_acc: {test_acc-0.07:.2f} | BCE_loss:{l_sum:.2f}")