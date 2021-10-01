import pickle as pkl
import sys
import os
sys.path.append("..")
from datatool import train_test_val_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from model import Encoder,BertConfig
from lc_model import LinearRegression
from lc_tool import setup_logger,setup_seed,getGrad,adjust_learning_rate
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import argparse
import logging

def get_confusion_matrix(y_true, y_pred, mode="recall"):
    '''Evaluate performance according to different label.
    Parameters:
    ---
    y_true: array
        Labels. 1-dim vector. 0:keep, 1:right, 2:left.
    y_pred: array
        Predictions. 1-dim vector.
    mode: string 
        options including {"count","precision","recall"}.
    
    Output:
    ---
    confusion_matrix:
        3x3 matrix. Elements' meanings are according to the mode.
    '''
    y_true_0,y_pred_0 = y_true==0,y_pred==0
    y_true_1,y_pred_1 = y_true==1,y_pred==1
    y_true_2,y_pred_2 = y_true==2,y_pred==2
    y00 = (y_pred_0*y_true_0).sum()
    y01 = (y_pred_0*y_true_1).sum()
    y02 = (y_pred_0*y_true_2).sum()
    
    y10 = (y_pred_1*y_true_0).sum()
    y11 = (y_pred_1*y_true_1).sum()
    y12 = (y_pred_1*y_true_2).sum()
    
    y20 = (y_pred_2*y_true_0).sum()
    y21 = (y_pred_2*y_true_1).sum()
    y22 = (y_pred_2*y_true_2).sum()
    confusion_matrix = np.array([[y00,y01,y02],[y10,y11,y12],[y20,y21,y22]])
    if mode == "count":
        confusion_matrix = confusion_matrix
    elif mode=="precision":
        confusion_matrix = confusion_matrix/confusion_matrix.sum(axis = 1)
    elif mode=="recall":
        confusion_matrix = confusion_matrix/confusion_matrix.sum(axis = 0)
    return confusion_matrix


def fit(x,encoder,decoder,device,hist_len=6):
    encoderInput = x[:,:,:hist_len,:].float().to(device)
    encoderOutput,lastEncoderHiddenState,attention = encoder(encoderInput)
    decoderInput = lastEncoderHiddenState.squeeze(0)
    decoderOutput = decoder(decoderInput)
    return decoderOutput, attention

def dev(dataloader,encoder,decoder,device,mode="dev"):
    encoder.eval()
    decoder.eval()
    y_true, y_pred = [],[]
    for i,(x,y) in enumerate(dataloader):
        decoderOutput,attention = fit(x,encoder,decoder,device)
        y_true.append(y.data.cpu().numpy())
        y_pred.append(torch.max(decoderOutput, 1)[1].data.cpu().numpy().squeeze())
    y_true, y_pred = np.hstack(y_true), np.hstack(y_pred)
    return y_true, y_pred
    
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BS = 256
SEED = 10
setup_seed(SEED)
conf_json = {
        "hidden_size":32,
        "num_hidden_layers":1,
        "num_attention_heads":1,
        "intermediate_size":32*4,
        "hidden_act":"gelu",
        "hidden_dropout_prob":0,
        "attention_probs_dropout_prob":0
    }
config = BertConfig()
config = config.from_dict(conf_json)
encoder = Encoder(config).to(DEVICE)
decoder = LinearRegression(conf_json['hidden_size']).to(DEVICE)
encoder.load_state_dict(torch.load('model/encoder_sd-10_lr-0.0001_wd-5e-05_maxep-100_bs-256_opt-adam_clip-5.02021122340520.pkl'))
decoder.load_state_dict(torch.load('model/decoder_sd-10_lr-0.0001_wd-5e-05_maxep-100_bs-256_opt-adam_clip-5.02021122340520.pkl'))

# load + split data
# 1
with open("new_data/total.pkl","rb") as f:
    data = pkl.load(f)
right_data, left_data, keep_data = data['right'], data['left'], data['keep']
# 2
right_train, right_eval, right_test = train_test_val_split(right_data,test_size=0.2,val_size=0.1,seed=SEED)
left_train, left_eval, left_test = train_test_val_split(left_data,test_size=0.2,val_size=0.1,seed=SEED)
keep_train, keep_eval, keep_test = train_test_val_split(keep_data,test_size=0.2,val_size=0.1,seed=SEED)
right_repeat_num = int(keep_data.shape[0]/right_data.shape[0])
left_repeat_num = int(keep_data.shape[0]/left_data.shape[0])
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

y_true, y_pred = dev(devDataloader,encoder,decoder,DEVICE,'dev')
# train_acc = dev(trainingDataloader,encoder,decoder,DEVICE,'dev')
# test_acc = dev(testDataloader,encoder,decoder,DEVICE,'dev')


confusion_matrix = get_confusion_matrix(y_true, y_pred)
print(confusion_matrix)
print((y_pred==y_true).sum()/len(y_true))