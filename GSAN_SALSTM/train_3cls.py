import pickle as pkl
import sys
import os
sys.path.append("..")
from datatool import train_test_val_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from model import BertConfig,Encoder
from lc_model import LinearRegression
from lc_tool import setup_logger,setup_seed,getGrad,adjust_learning_rate
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import argparse
import logging


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

class Mapping_layer(nn.Module):
    def __init__(self):
        super(Mapping_layer, self).__init__()
        self.linear = nn.Linear(32,12)
    def forward(self,x):
        y = self.linear(x)
        return y

def fit(x,encoder,mapping_layer,decoder,device,hist_len=6):
    encoderInput = x[:,:,:hist_len,:].float().to(device)
    encoderOutput,lastEncoderHiddenState,attention = encoder(encoderInput)
    decoderInput = mapping_layer(encoderOutput)
    decoderOutput = decoder(decoderInput)
    return decoderOutput

def train(trainingDataloader,encoder,mapping_layer,decoder,device,cost_function,encoder_optimizer,decoder_optimizer,mapping_optimizer,clip):
    encoder.train()
    decoder.train()
    l_sum = true = total = 0
    for i,(x,y) in enumerate(trainingDataloader):
        y = y.to(device)
        decoderOutput = fit(x,encoder,mapping_layer,decoder,device)
        _y = torch.max(decoderOutput, 1)[1]
        l = cost_function(decoderOutput,y.long().to(device))
        l.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        encoder_optimizer.step()
        decoder_optimizer.step()
        mapping_optimizer.step()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        mapping_optimizer.zero_grad()
        l_sum += l
        true += sum(y==_y)
        total += y.shape[0]
    train_acc = true.data.cpu().numpy()/total
    return train_acc, l_sum.data.cpu().numpy()

def dev(dataloader,encoder,mapping_layer,decoder,device,mode="dev"):
    encoder.eval()
    decoder.eval()
    true = total = 0
    for i,(x,y) in enumerate(dataloader):
        y = y.to(device)
        decoderOutput = fit(x,encoder,mapping_layer,decoder,device)
        _y = torch.max(decoderOutput, 1)[1]
        true += sum(y==_y)
        total += y.shape[0]
    acc = true.data.cpu().numpy()/total
    if mode == "dev":
        return acc
    if mode == "test":
        # prec = precision_score(y_true, y_pred, average='micro')
        # recall = recall_score(y_true, y_pred, average='micro')
        # return acc, prec, recall
        return acc


if __name__ == "__main__":
    # training args
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=np.float64, default=1e-4)
    parser.add_argument('-ep', '--max_epoch', type=int, default=100)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-5)
    parser.add_argument('-bs', '--batch_size', type=int, default=256)
    parser.add_argument('-opt', '--optimizer', type=str, default='adam')
    parser.add_argument('-sd', '--seed', type=int, default=10)
    parser.add_argument('-cl', '--clip', type=float, default=5)
    parser.add_argument('--with-pretrained', dest="pretrained", action='store_true')
    parser.add_argument('--no-pretrained', dest="pretrained", action='store_false')
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BS = args.batch_size
    LR = args.learning_rate
    WD = args.weight_decay
    MAX_EP = args.max_epoch
    OPT = args.optimizer
    SEED = args.seed
    CLIP = args.clip
    PRE_TRAIN = args.pretrained
    # set seed & logger
    setup_seed(SEED)
    exp_root = "../logs/3cls/"
    exp_id = f"sd-{SEED}_lr-{LR}_wd-{WD}_maxep-{MAX_EP}_bs-{BS}_opt-{OPT}_clip-{CLIP}"+"".join([str(x) for x in list(time.localtime(time.time()))])
    logger = setup_logger(exp_root,exp_id)
    conf_json = {
        "hidden_size":32,
        "num_hidden_layers":1,
        "num_attention_heads":1,
        "intermediate_size":32*4,
        "hidden_act":"gelu",
        "hidden_dropout_prob":0,
        "attention_probs_dropout_prob":0
    }
    logger.info(f"seed:{SEED}, lr:{LR}, wd:{WD}, max_ep:{MAX_EP}, bs:{BS}, opt:{OPT}, clip:{CLIP}, hidden_size:{conf_json['hidden_size']}")
    config = BertConfig()
    config = config.from_dict(conf_json)
    encoder = Encoder(config).to(DEVICE)
    mapping_layer = Mapping_layer().to(DEVICE)
    decoder = SA_LSTM().to(DEVICE)
    if PRE_TRAIN:
        encoder.load_state_dict(torch.load('last/e_last.pkl'))
        decoder.load_state_dict(torch.load('last/d_last.pkl'))
        
    # cost function & optimizer
    cost_function = nn.CrossEntropyLoss()
    if OPT == "adam":
        encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=LR,weight_decay=WD)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr=LR,weight_decay=WD)
        mapping_optimizer = torch.optim.Adam(mapping_layer.parameters(),lr=LR,weight_decay=WD)
    elif OPT == "sgd":
        encoder_optimizer = torch.optim.SGD(encoder.parameters(),lr=LR,weight_decay=WD)
        decoder_optimizer = torch.optim.SGD(decoder.parameters(),lr=LR,weight_decay=WD)
        mapping_optimizer = torch.optim.SGD(mapping_layer.parameters(),lr=LR,weight_decay=WD)
    hist_len = 6
    # load + split data
    # 1
    with open("../new_neighbor/new_data/total.pkl","rb") as f:
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
    # Train
    best = {"epoch":0,"acc":0,"params":(None,None),}
    for ep in range(MAX_EP):
        t1 = time.time()
        if ep == 500:
            logger.info(f"learning rate changed from {LR} to {LR/10}")
            adjust_learning_rate(encoder_optimizer,LR/10)
            adjust_learning_rate(decoder_optimizer,LR/10)
        train_acc, l_sum = train(trainingDataloader,encoder,mapping_layer,decoder,DEVICE,cost_function,encoder_optimizer,decoder_optimizer,mapping_optimizer,CLIP)
        dev_acc = dev(devDataloader,encoder,mapping_layer,decoder,DEVICE,'dev')
        # qw = encoder.encoder.layer[0].attention.self.query.weight.data.cpu().numpy().sum()
        # kw = encoder.encoder.layer[0].attention.self.key.weight.data.cpu().numpy().sum()
        # vw = encoder.encoder.layer[0].attention.self.value.weight.data.cpu().numpy().sum()
        # w = np.array([qw,kw,vw])
        if dev_acc > best["acc"]:
            best['epoch'] = ep
            best['acc'] = dev_acc
            best['params'] = (encoder.state_dict(),decoder.state_dict())
        t2 = time.time()
        logger.info(f"Epoch[{ep:2.0f}] | time : {t2-t1:.2f} | BCE_loss:{l_sum:.2f} | train_acc: {train_acc:.4f} | eval_acc: {dev_acc:.4f}")
    logger.info(f'\nFinish training. Best epoch : {best["epoch"]}. Dev Acc : {best["acc"]:.2f}. Reload epoch{best["epoch"]} params.')

    # Test
    encoder.load_state_dict(best["params"][0])
    decoder.load_state_dict(best["params"][1])
    test_acc= dev(testDataloader,encoder,mapping_layer,decoder,DEVICE,'test')
    logger.info(f"\nTest result: acc : {test_acc:.2f}")

    torch.save(encoder.state_dict(), os.path.join(exp_root,f"e{exp_id}.pkl"))
    torch.save(decoder.state_dict(), os.path.join(exp_root,f"d{exp_id}.pkl"))
    
    torch.save(encoder.state_dict(), f'last/e_last.pkl')
    torch.save(decoder.state_dict(), f'last/d_last.pkl')

    logger.info(f"Models are saved in {exp_root}.")
    

