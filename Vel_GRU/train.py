import os
import pickle as pkl
import numpy as np
import torch
import sys
sys.path.extend(["../","../../"])
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader
from datatool import ArrayDataset,rmse,load_array_data
from lc_tool import setup_seed,setup_logger
from model import BertEncoder, BertConfig
import time
import argparse
import logging


class GRU_Linear(nn.Module):
    def __init__(self,hidden_size):
        super(GRU_Linear, self).__init__()
        self.gru = nn.GRU(18, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size,18)
    def forward(self,x,h=None):
        y, h = self.gru(x,h)
        y = self.linear(y)
        y = y + x
        return y ,h

def train(training_dataloader,model,optimizer,loss_func,clip,device):
    training_loss = 0
    for i,x in enumerate(training_dataloader):
        x = x.float().to(device)
        y,h = model(x)
        loss = loss_func(y[:,:-1,:2],x[:,1:,:2])
        loss.backward()
        training_loss += loss.data.cpu().numpy()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad()
    return training_loss

def val(val_dataloader, model, device):
    hist_len,pred_len = 7, 10
    rmse_loss = np.zeros(5)
    pred_track,ture_track = [],[]
    for i,x in enumerate(val_dataloader):
        x_t,h_t = model(x[:,:hist_len,:].float().to(device))
        x_t = x_t[:,-1,:].unsqueeze(1)
        pred_track_t = []        
        for time_step in range(pred_len): # 10
            x_t, h_t = model(x_t.float().to(device),h_t)
            h_t = h_t.data
            pred_track_t.append(x_t[:,:,:2].data.cpu().numpy())
        pred_track.append(np.concatenate(pred_track_t,axis=1))
        ture_track.append(x[:,hist_len:,:2].data.cpu().numpy())
    pred_track = np.concatenate(pred_track,axis=0)
    ture_track = np.concatenate(ture_track,axis=0)
    assert pred_track.shape == ture_track.shape
    for s in range(int(pred_len/2)):
        rmse_loss[s] += rmse(pred_track[:,(s+1)*2-1,:],ture_track[:,(s+1)*2-1,:])
    return np.around(rmse_loss, decimals=4)

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=np.float64, default=1e-4)
    parser.add_argument('-ep', '--max_epoch', type=int, default=100)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-5)
    parser.add_argument('-bs', '--batch_size', type=int, default=4096)
    parser.add_argument('-opt', '--optimizer', type=str, default='adam')
    parser.add_argument('-sd', '--seed', type=int, default=10)
    parser.add_argument('-cl', '--clip', type=float, default=5)
    parser.add_argument('--with-pretrained', dest="pretrained", action='store_true')
    parser.add_argument('--no-pretrained', dest="pretrained", action='store_false')
    parser.add_argument('--hidden_size', type=int, default=320)
    parser.add_argument('--device_id', type=int, default=0)
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()
    BS = args.batch_size
    LR = args.learning_rate
    WD = args.weight_decay
    MAX_EP = args.max_epoch
    OPT = args.optimizer
    SEED = args.seed
    CLIP = args.clip
    PRE_TRAIN = args.pretrained
    hidden_size = args.hidden_size
    device_id = args.device_id
    # set seed & logger
    setup_seed(SEED)
    exp_root = "../logs/3cls/"
    exp_id = f"sd-{SEED}_lr-{LR}_wd-{WD}_maxep-{MAX_EP}_bs-{BS}_opt-{OPT}_clip-{CLIP}"+"".join([str(x) for x in list(time.localtime(time.time()))])
    logger = setup_logger(exp_root,exp_id)
    # basic configuration
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    # record hyper parameters
    logger.info(f"max epoch : {MAX_EP}, batch size : {BS}, lr : {LR}, wd : {WD}, device : {device}, seed : {SEED}, opt: {OPT}, pretrain: {PRE_TRAIN}")

    # load data
    with h5py.File("../new_neighbor_track_pred/data/TrainSet.hdf5","r") as f:
        training_data = f['feature'][()]
    # training_data = np.transpose(training_data,axes=(0,2,1,3))
    training_data = training_data.reshape(*training_data.shape[:-2],-1)
    with h5py.File("../new_neighbor_track_pred/data/TestSet.hdf5","r") as f:
        test_data = f['feature'][()]
    # test_data = np.transpose(test_data,axes=(0,2,1,3))
    test_data = test_data.reshape(*test_data.shape[:-2],-1)
    with h5py.File("../new_neighbor_track_pred/data/ValSet.hdf5","r") as f:
        val_data = f['feature'][()]
    # val_data = np.transpose(val_data,axes=(0,2,1,3))
    val_data = val_data.reshape(*val_data.shape[:-2],-1)
    training_dataset = ArrayDataset(training_data)
    training_dataloader = DataLoader(training_dataset,batch_size=BS,shuffle=True)
    val_dataset = ArrayDataset(val_data)
    val_dataloader = DataLoader(val_dataset,batch_size=BS)
    test_dataset = ArrayDataset(test_data)
    test_dataloader = DataLoader(test_dataset,batch_size=BS,shuffle=True)
    
    logger.info(f"training set size : {training_data.shape}")
    logger.info(f"val set size : {val_data.shape}")
    logger.info(f"test set size : {test_data.shape}")

    # init model & optimizer & loss function
    model = GRU_Linear(hidden_size).to(device)
    logger.info(model)
    if PRE_TRAIN:
        model.load_state_dict(torch.load('last/last.pkl',map_location='cpu'),strict=False)
    
    if OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr=LR,weight_decay=WD)
    elif OPT == "adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)

    loss_func = torch.nn.MSELoss()

    # train and val
    best = {"epoch":0,"rmse":10000,"params":(None,None),}
    for ep in range(MAX_EP):
        t1 = time.time()
        training_loss = train(training_dataloader,model,optimizer,loss_func,CLIP,device)
        val_rmse = val(val_dataloader,model,device)
        if val_rmse[-1] < best["rmse"]:
            best['epoch'] = ep
            best['rmse'] = val_rmse[-1]
            best['params'] = (model.state_dict())
        t2 = time.time()
        logger.info(f"epoch[{ep:3}] | mse@train : {training_loss:6.2f} | rmse@val : {val_rmse} |time : {t2-t1:5.2f}") 
    logger.info(f"\nTraining done. Epoch {best['epoch']} achieved best rmse {best['rmse']}. Load epoch {best['epoch']} parameters for test.")
    # Test
    model.load_state_dict(best["params"])
    test_rmse = val(test_dataloader,model,device)
    logger.info(f"\nTest result: rmse : {test_rmse}")
    
    # save model
    torch.save(model.state_dict(), os.path.join(exp_root,f"{exp_id}.pkl"))
    
    torch.save(model.state_dict(), f'last/last.pkl')
    logger.info(f"model is saved in 'models/{exp_root}/'.")