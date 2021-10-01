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

def fit(x,encoder,decoder,device,hist_len=6):
    encoderInput = x[:,:,:hist_len,:].float().to(device)
    encoderOutput,lastEncoderHiddenState,attention = encoder(encoderInput)
    decoderInput = lastEncoderHiddenState.squeeze(0)
    decoderOutput = decoder(decoderInput)
    return decoderOutput


def train(trainingDataloader,encoder,decoder,device,cost_function,encoder_optimizer,decoder_optimizer,clip):
    encoder.train()
    decoder.train()
    l_sum = 0
    train_true, train_pred, grads = [], [], []
    for i,(x,y) in enumerate(trainingDataloader):
        decoderOutput = fit(x,encoder,decoder,device)
        l = cost_function(decoderOutput,y.long().to(device))
        l.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        grads.extend(getGrad(encoder).values())
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        l_sum += l.data.cpu().numpy()
        train_true.append(y.data.cpu().numpy())
        train_pred.append(torch.max(decoderOutput, 1)[1].data.cpu().numpy().squeeze())
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
        y_pred.append(torch.max(decoderOutput, 1)[1].data.cpu().numpy().squeeze())
    y_true, y_pred = np.hstack(y_true), np.hstack(y_pred)
    acc = accuracy_score(y_true, y_pred)
    if mode == "dev":
        return acc
    if mode == "test":
        prec = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
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
    parser.add_argument('-cl', '--clip', type=float, default=10000)
    parser.add_argument('-sp', '--save_path',type=str, default="")
    parser.add_argument('-pt', '--pre_train',type=str, default="")
    args = parser.parse_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BS = args.batch_size
    LR = args.learning_rate
    WD = args.weight_decay
    MAX_EP = args.max_epoch
    OPT = args.optimizer
    SEED = args.seed
    CLIP = args.clip
    SAVE_PATH = args.save_path
    PRE_TRAIN = args.pre_train
    # set seed & logger
    setup_seed(SEED)
    exp_id = f"sd-{SEED}_lr-{LR}_wd-{WD}_maxep-{MAX_EP}_bs-{BS}_opt-{OPT}_clip-{CLIP}"+"".join([str(x) for x in list(time.localtime(time.time()))])
    logger = setup_logger("../logs/3cls/",exp_id)
    conf_json = {
        "hidden_size":32,
        "num_hidden_layers":1,
        "num_attention_heads":1,
        "intermediate_size":32*4,
        "hidden_act":"gelu",
        "hidden_dropout_prob":0,
        "attention_probs_dropout_prob":0
    }
    logger.info(f"seed:{SEED}, lr:{LR}, wd:{WD}, max_ep:{MAX_EP}, bs:{BS}, opt:{OPT}, clip:{CLIP}, save_path:{SAVE_PATH}, hidden_size:{conf_json['hidden_size']}")
    config = BertConfig()
    config = config.from_dict(conf_json)
    encoder = Encoder(config).to(DEVICE)
    decoder = LinearRegression(conf_json['hidden_size']).to(DEVICE)
    if PRE_TRAIN:
        encoder.load_state_dict(torch.load('model/encoder_sd-10_lr-0.0001_wd-5e-05_maxep-500_bs-256_opt-adam_clip-1000020211118430410.pkl'))
        decoder.load_state_dict(torch.load('model/decoder_sd-10_lr-0.0001_wd-5e-05_maxep-500_bs-256_opt-adam_clip-1000020211118430410.pkl'))
        
    # cost function & optimizer
    cost_function = nn.CrossEntropyLoss()
    if OPT == "adam":
        encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=LR,weight_decay=WD)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr=LR,weight_decay=WD)
    elif OPT == "sgd":
        encoder_optimizer = torch.optim.SGD(encoder.parameters(),lr=LR,weight_decay=WD)
        decoder_optimizer = torch.optim.SGD(decoder.parameters(),lr=LR,weight_decay=WD)
    hist_len = 6
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
    # Train
    best = {"epoch":0,"acc":0,"params":(None,None),}
    for ep in range(MAX_EP):
        t1 = time.time()
        if ep == 500:
            logger.info(f"learning rate changed from {LR} to {LR/10}")
            adjust_learning_rate(encoder_optimizer,LR/10)
            adjust_learning_rate(decoder_optimizer,LR/10)
        train_acc, l_sum, grad = train(trainingDataloader,encoder,decoder,DEVICE,cost_function,encoder_optimizer,decoder_optimizer,CLIP)
        dev_acc = dev(devDataloader,encoder,decoder,DEVICE,'dev')
        qw = encoder.encoder.layer[0].attention.self.query.weight.data.cpu().numpy().sum()
        kw = encoder.encoder.layer[0].attention.self.key.weight.data.cpu().numpy().sum()
        vw = encoder.encoder.layer[0].attention.self.value.weight.data.cpu().numpy().sum()
        w = np.array([qw,kw,vw])
        if dev_acc > best["acc"]:
            best['epoch'] = ep
            best['acc'] = dev_acc
            best['params'] = (encoder.state_dict(),decoder.state_dict())
        t2 = time.time()
#         logger.info(f"Epoch[{ep:2.0f}] | time : {t2-t1:.2f} | BCE_loss:{l_sum:.2f} | train_acc: {train_acc:.4f} | eval_acc: {dev_acc:.4f} | encoder_grad : {max(grad):.2f}/{min(grad):.2f}/{sum(grad):.2f}")
        logger.info(f"Epoch[{ep:2.0f}] | time : {t2-t1:.2f} | BCE_loss:{l_sum:.2f} | train_acc: {train_acc:.4f} | eval_acc: {dev_acc:.4f} | w : {np.around(w, decimals=4)}")
    logger.info(f'\nFinish training. Best epoch : {best["epoch"]}. Dev Acc : {best["acc"]:.2f}. Reload epoch{best["epoch"]} params.')

    # Test
    encoder.load_state_dict(best["params"][0])
    decoder.load_state_dict(best["params"][1])
    test_acc, test_prec, test_recall = dev(testDataloader,encoder,decoder,DEVICE,'test')
    logger.info(f"\nTest result: acc : {test_acc:.2f}, prec : {test_prec:.2f}, recall : {test_recall:.2f}")

    if SAVE_PATH:
        encoder_path = os.path.join(SAVE_PATH,f"encoder_{exp_id}.pkl")
        torch.save(encoder.state_dict(), encoder_path)
        decoder_path = os.path.join(SAVE_PATH,f"decoder_{exp_id}.pkl")
        torch.save(decoder.state_dict(), decoder_path)
        logger.info(f"\nModel are saved in {SAVE_PATH}")

