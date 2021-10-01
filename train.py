'''2 class:{keep, change}'''

import pickle as pkl
import os
from datatool import train_test_val_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from model import Encoder,BertConfig
from lc_model import SVM
from lc_tool import setup_logger,setup_seed,getGrad
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import argparse
import logging

def fit(x,encoder,decoder,device):
    encoderInput = x[:,:,:hist_len,:].float().to(device)
    encoderOutput,lastEncoderHiddenState,attention = encoder(encoderInput)
    decoderInput = lastEncoderHiddenState.squeeze(0)
    decoderOutput = decoder(decoderInput)
    return decoderOutput


def train(trainingDataloader,encoder,decoder,device,cost_function,encoder_optimizer,decoder_optimizer):
    encoder.train()
    decoder.train()
    l_sum = 0
    train_true, train_pred, grads = [], [], []
    for i,(x,y) in enumerate(trainingDataloader):
        decoderOutput = fit(x,encoder,decoder,device)
        l = cost_function(decoderOutput,y.float().to(device))
        l.backward()
        grads.extend(getGrad(encoder).values())
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        l_sum += l.data.cpu().numpy()
        train_true.append(y.data.cpu().numpy())
        train_pred.append((decoderOutput.data.cpu().numpy())>0.5)
    train_true, train_pred = np.hstack(train_true), np.hstack(train_pred)
    train_acc = accuracy_score(train_true, train_pred)
    return train_acc, l_sum, grads

def dev(dataloader,encoder,decoder,device,mode="dev"):
    encoder.eval()
    decoder.eval()
    y_true, y_pred = [],[]
    for i,(x,y) in enumerate(dataloader):
        decoderOutput = fit(x,encoder,decoder,device)
        y_true.append(y.data.cpu().numpy())
        y_pred.append((decoderOutput.data.cpu().numpy())>0.5)
    y_true, y_pred = np.hstack(y_true), np.hstack(y_pred)
    acc = accuracy_score(y_true, y_pred)
    if mode == "dev":
        return acc
    if mode == "test":
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        return acc, prec, recall




if __name__ == "__main__":
    # training args
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=np.float64, default=1e-4)
    parser.add_argument('-ep', '--max_epoch', type=int, default=100)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-5)
    parser.add_argument('-bs', '--batch_size', type=int, default=256)
    parser.add_argument('-opt', '--optimizer', type=str, default='adam')
    parser.add_argument('-sd', '--seed', type=int, default=10)
    args = parser.parse_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BS = args.batch_size
    LR = args.learning_rate
    WD = args.weight_decay
    MAX_EP = args.max_epoch
    OPT = args.optimizer
    SEED = args.seed
    # set seed & logger
    setup_seed(SEED)
    logger = setup_logger("logs",SEED,LR,WD,MAX_EP,BS,OPT)

    conf_json = {
        "hidden_size":32,
        "num_hidden_layers":1,
        "num_attention_heads":1,
        "intermediate_size":32*4,
        "hidden_act":"gelu",
        "hidden_dropout_prob":0,
        "attention_probs_dropout_prob":0
    }
    logger.info(f"seed:{SEED}, lr:{LR}, wd:{WD}, max_ep:{MAX_EP}, bs:{BS}, opt:{OPT}, hidden_size:{conf_json['hidden_size']}")
    config = BertConfig()
    config = config.from_dict(conf_json)
    encoder = Encoder(config).to(DEVICE)
    decoder = SVM(conf_json['hidden_size']).to(DEVICE)
    # cost function & optimizer
    cost_function = nn.BCELoss()
    if OPT == "adam":
        encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=LR,weight_decay=WD)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr=LR,weight_decay=WD)
    elif OPT == "sgd":
        encoder_optimizer = torch.optim.SGD(encoder.parameters(),lr=LR,weight_decay=WD)
        decoder_optimizer = torch.optim.SGD(decoder.parameters(),lr=LR,weight_decay=WD)
    hist_len = 6
    
    with open("pickle_data/23w10v1.pkl","rb") as f:
        data = pkl.load(f)
    pos_data,neg_data = data['pos_data'],data['neg_data']
    pos_train, pos_eval, pos_test = train_test_val_split(pos_data,test_size=0.2,val_size=0.1,seed=0)
    neg_train, neg_eval, neg_test = train_test_val_split(neg_data,test_size=0.2,val_size=0.1,seed=0)
    pos_train, pos_eval, pos_test = pos_train.repeat(10,axis=0),pos_eval.repeat(10,axis=0),pos_test.repeat(10,axis=0)
    X_train, X_eval, X_test = np.vstack((pos_train,neg_train)),np.vstack((pos_eval,neg_eval)),np.vstack((pos_test,neg_test))
    y_train = np.hstack((np.ones(pos_train.shape[0]),np.zeros(neg_train.shape[0])))
    y_eval = np.hstack((np.ones(pos_eval.shape[0]),np.zeros(neg_eval.shape[0])))
    y_test = np.hstack((np.ones(pos_test.shape[0]),np.zeros(neg_test.shape[0])))
    trainingDataset = TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train))
    trainingDataloader = DataLoader(trainingDataset,batch_size=BS,shuffle=True)
    testDataset = TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test))
    testDataloader = DataLoader(testDataset,batch_size=BS,shuffle=True)
    devDataset = TensorDataset(torch.from_numpy(X_eval),torch.from_numpy(y_eval))
    devDataloader = DataLoader(devDataset,batch_size=BS,shuffle=True)

    best = {"epoch":0,"acc":0,"params":(None,None),}
    for ep in range(MAX_EP):
        t1 = time.time()
        train_acc, l_sum, grad = train(trainingDataloader,encoder,decoder,DEVICE,cost_function,encoder_optimizer,decoder_optimizer)
        dev_acc = dev(devDataloader,encoder,decoder,DEVICE,'dev')
        if dev_acc > best["acc"]:
            best['epoch'] = ep
            best['acc'] = dev_acc
            best['params'] = (encoder.state_dict(),decoder.state_dict())
        t2 = time.time()
        logger.info(f"Epoch[{ep:2.0f}] | time : {t2-t1:.2f} | BCE_loss:{l_sum:.2f} | train_acc: {train_acc:.4f} | eval_acc: {dev_acc:.4f} | encoder_grad : {max(grad):.2f}/{min(grad):.2f}/{sum(grad):.2f}")
    logger.info(f'\nFinish training. Best epoch : {best["epoch"]}. Dev Acc : {best["acc"]:.2f}. Reload epoch{best["epoch"]} params.')
    encoder.load_state_dict(best["params"][0])
    decoder.load_state_dict(best["params"][1])
    test_acc, test_prec, test_recall = dev(testDataloader,encoder,decoder,DEVICE,'test')
    logger.info(f"\nTest result: acc : {test_acc:.2f}, prec : {test_prec:.2f}, recall : {test_recall:.2f}")

