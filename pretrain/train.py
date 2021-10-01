import pickle as pkl
import numpy as np
import torch
import sys
import os
sys.path.extend(["../","../../"])
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader,TensorDataset
from datatool import ArrayDataset,rmse,load_array_data,train_test_val_split
from lc_tool import setup_seed,setup_logger
from model import Encoder, Decoder, BertConfig
import time
import argparse
def pretrain(training_dataloader,encoder,decoder,encoder_optimizer,decoder_optimizer,loss_func,clip,device):
    training_loss = 0
    for i,(x,y) in enumerate(training_dataloader):
        x = x.float().to(device)
        encoderOutput,lastEncoderHiddenState,attention = encoder(x)
        decoderInput = x[:,0,:,:]
        pred_y = decoder(decoderInput, encoderOutput)
        loss = loss_func(pred_y[:,:-1,:],decoderInput[:,1:,:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        training_loss += loss.item()
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    return training_loss/(i+1)

def get_optimizer(lr,wd,opt="sgd"):
    if opt == "sgd":
        encoder_optimizer = torch.optim.SGD(encoder.parameters(),lr=lr,weight_decay=wd)
        decoder_optimizer = torch.optim.SGD(decoder.parameters(),lr=lr,weight_decay=wd)
    elif opt == "adam":
        encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=lr,weight_decay=wd)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr=lr,weight_decay=wd)
    return encoder_optimizer, decoder_optimizer
    
class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(config.hidden_size,2)
    def forward(self,x,encoder_output):
        y = self.linear(encoder_output)
        return y + x

if __name__ == "__main__":    
    
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
    
    BS = args.batch_size
    LR = args.learning_rate
    WD = args.weight_decay
    MAX_EP = args.max_epoch
    OPT = args.optimizer
    SEED = args.seed
    CLIP = args.clip
    # redirect stdout to log file
    exp_root = "../logs/pretrain/"
    exp_id = f"sd-{SEED}_lr-{LR}_wd-{WD}_maxep-{MAX_EP}_bs-{BS}_opt-{OPT}_clip-{CLIP}"+"".join([str(x) for x in list(time.localtime(time.time()))])
    logger = setup_logger(exp_root,exp_id)

    # basic configuration
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
    
    epoch = args.max_epoch
    batch_size = args.batch_size
    lr = args.learning_rate
    wd = args.weight_decay
    seed = args.seed
    opt = args.optimizer
    clip = args.clip
    pre_train = args.pretrained
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    setup_seed(seed)
    # record hyper parameters
    logger.info(f"max epoch : {epoch}, batch size : {batch_size}, lr : {lr}, wd : {wd}, device : {device}, seed : {seed}, opt: {opt}, pretrain: {pre_train}")
    logger.info("\nmodel hyper parameters:")
    for k,v in conf_json.items():
        logger.info(f"{k} : {v}")

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
    # 4
    trainingDataset = TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train))
    trainingDataloader = DataLoader(trainingDataset,batch_size=BS,shuffle=True)

    # init model & optimizer & loss function
    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)
    if pre_train:
        encoder.load_state_dict(torch.load('models/e2021391527441680.pkl',map_location='cpu'),strict=False)
        decoder.load_state_dict(torch.load('models/d2021391527441680.pkl',map_location='cpu'),strict=False)
    
    encoder_optimizer, decoder_optimizer = get_optimizer(lr,wd,opt)
    loss_func = torch.nn.MSELoss()

    # train and val
    best = {"epoch":0,"loss":10000,"params":(None,None),}
    for ep in range(epoch):
        t1 = time.time()
        if ep==30:
            encoder_optimizer, decoder_optimizer = get_optimizer(lr,wd,"sgd")
        training_loss = pretrain(trainingDataloader,encoder,decoder,encoder_optimizer,decoder_optimizer,loss_func,clip,device)
        qw = encoder.encoder.layer[0].attention.self.query.weight.data.cpu().numpy().sum()
        kw = encoder.encoder.layer[0].attention.self.key.weight.data.cpu().numpy().sum()
        vw = encoder.encoder.layer[0].attention.self.value.weight.data.cpu().numpy().sum()
        w = np.array([qw,kw,vw])
        if training_loss < best["loss"]:
            best['epoch'] = ep
            best['loss'] = training_loss
            best['params'] = (encoder.state_dict(),decoder.state_dict())
        t2 = time.time()
        logger.info(f"epoch[{ep:3}] | mse@train : {training_loss:6.2f} | w : {np.around(w, decimals=4)} |time : {t2-t1:5.2f}") 
    
    # save model
    torch.save(encoder.state_dict(), os.path.join(exp_root,f"e{exp_id}.pkl"))
    torch.save(decoder.state_dict(), os.path.join(exp_root,f"d{exp_id}.pkl"))
    
    torch.save(encoder.state_dict(), f'last/e_last.pkl')
    torch.save(decoder.state_dict(), f'last/d_last.pkl')

    logger.info(f"Models are saved in {exp_root}.")