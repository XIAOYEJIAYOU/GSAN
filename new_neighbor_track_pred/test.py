
import pickle as pkl
import numpy as np
import torch
import sys
sys.path.extend(["../","../../"])
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader
from datatool import ArrayDataset,rmse,Logger,load_array_data
from lc_tool import setup_seed
from model import Encoder, Decoder, BertConfig
import time
import argparse

def val(val_dataloader,encoder,decoder,device):
    hist_len,pred_len = 6,10
    rmse_loss = np.zeros(5)
    for i,x in enumerate(val_dataloader):
        encoder_input = x[:,:,:hist_len,:].float().to(device)
        # encoder 
        encoder_output,last_encoder_hidden_state,attention = encoder(encoder_input)
        
        # preheat
        decoder_hidden_state = torch.zeros(last_encoder_hidden_state.shape).float().to(device)
        for t in range(encoder_input.shape[2]):
            decoder_input = encoder_input[:,0,t,:].unsqueeze(1)
            decoder_output,decoder_hidden_state = decoder(decoder_x=decoder_input,encoder_h=last_encoder_hidden_state,decoder_h=decoder_hidden_state)
        
        # decoder
        decoder_input = x[:,0,hist_len,:].unsqueeze(1).float().to(device)
        y_pred_array_list = []
        for t in range(pred_len):
            decoder_output,decoder_hidden_state = decoder(decoder_x=decoder_input,encoder_h=last_encoder_hidden_state,decoder_h=decoder_hidden_state)
            decoder_input = decoder_output.data
            y_pred_array_list.append(decoder_output.data.cpu().numpy())
        y_pred_array = np.concatenate(y_pred_array_list,axis=1)
        y_true_array = x[:,0,hist_len+1:,:].data.cpu().numpy()
        
        for s in range(int(pred_len/2)):
            rmse_loss[s] += rmse(y_pred_array[:,(s+1)*2-1,:],y_true_array[:,(s+1)*2-1,:])
    return np.around(rmse_loss/(i+1), decimals=4)


if __name__ == "__main__":
    # parameters
    batch_size = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load test data
    with h5py.File("data/ValSet.hdf5","r") as f:
        valid_data = f['feature'][()]
    valid_data = np.transpose(valid_data,axes=(0,2,1,3))
    valid_dataset = ArrayDataset(valid_data)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size)
    
    with h5py.File("data/TestSet.hdf5","r") as f:
        test_data = f['feature'][()]
    test_data = np.transpose(test_data,axes=(0,2,1,3))
    test_dataset = ArrayDataset(test_data)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size)
    # load trained model
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
    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)
    encoder.load_state_dict(torch.load("models/e20211361423630.pkl",map_location='cpu'),strict=False)
    decoder.load_state_dict(torch.load("models/d20211361423630.pkl",map_location='cpu'),strict=False)
    
    test_rmse = val(test_dataloader,encoder,decoder,device)
    valid_rmse = val(valid_dataloader,encoder,decoder,device)
    print(f"test rmse : {test_rmse} | valid rmse : {valid_rmse}")