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

def train(training_dataloader,encoder,decoder,encoder_optimizer,decoder_optimizer,loss_func,clip,device,teacher_force=True):
    hist_len,pred_len = 6,10
    training_loss = 0
    rmse_loss =np.zeros(5)
    for i,x in enumerate(training_dataloader):
        loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_input = x[:,:,:hist_len,:].float().to(device)
        # encoder 
        encoder_output,last_encoder_hidden_state,attention = encoder(encoder_input)
        
        # decoder
        decoder_inputs = x[:,0,:,:].float().to(device)
        decoder_hidden_state = torch.zeros(last_encoder_hidden_state.shape).float().to(device)
        y_pred_array_list = []
        for t in range(decoder_inputs.shape[1]-1):
            decoder_input = decoder_inputs[:,t,:].unsqueeze(1)
            if (teacher_force is False) and (t!=0):
                decoder_input = decoder_output.data
            decoder_output,decoder_hidden_state = decoder(decoder_x=decoder_input,encoder_h=last_encoder_hidden_state,decoder_h=decoder_hidden_state)
            loss += loss_func(decoder_output,decoder_inputs[:,t+1,:].unsqueeze(1))
            y_pred_array_list.append(decoder_output.data.cpu().numpy())
        y_pred_array = np.concatenate(y_pred_array_list,axis=1)
        y_true_array = x[:,0,1:,:].data.cpu().numpy()
        for s in range(int(pred_len/2)):
            rmse_loss[s] += rmse(y_pred_array[:,hist_len+2*s+1,:],y_true_array[:,hist_len+2*s+1,:])
        
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        training_loss += loss.item()
        encoder_optimizer.step()
        decoder_optimizer.step()
    return training_loss,np.around(rmse_loss/(i+1), decimals=4)

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

def get_optimizer(lr,wd,opt="sgd"):
    if opt == "sgd":
        encoder_optimizer = torch.optim.SGD(encoder.parameters(),lr=lr,weight_decay=wd)
        decoder_optimizer = torch.optim.SGD(decoder.parameters(),lr=lr,weight_decay=wd)
    elif opt == "adam":
        encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=lr,weight_decay=wd)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr=lr,weight_decay=wd)
    return encoder_optimizer, decoder_optimizer
    

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=np.float64, default=1e-5)
    parser.add_argument('-ep', '--max_epoch', type=int, default=100)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-5)
    parser.add_argument('-bs', '--batch_size', type=int, default=256)
    parser.add_argument('-opt', '--optimizer', type=str, default='sgd')
    parser.add_argument('-sd', '--seed', type=int, default=10)
    parser.add_argument('-cl', '--clip', type=float, default=5)
    parser.add_argument('--with-pretrained', dest="pretrained", action='store_true')
    parser.add_argument('--no-pretrained', dest="pretrained", action='store_false')
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()
    
 
    
    # redirect stdout to log file
    localtime = time.localtime(time.time())
    name = "".join([str(x) for x in list(localtime)])
    sys.stdout = Logger(f"logs/{name}")

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
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(seed)
    # record hyper parameters
    print(f"max epoch : {epoch}, batch size : {batch_size}, lr : {lr}, wd : {wd}, device : {device}, seed : {seed}, opt: {opt}, pretrain: {pre_train}")
    print("\nmodel hyper parameters:")
    for k,v in conf_json.items():
        print(f"{k} : {v}")

    # load data
    with h5py.File("data/TrainSet.hdf5","r") as f:
        training_data = f['feature'][()]
    # training_data = np.transpose(training_data,axes=(0,2,1,3))
    with h5py.File("data/TestSet.hdf5","r") as f:
        test_data = f['feature'][()]
    # test_data = np.transpose(test_data,axes=(0,2,1,3))
    with h5py.File("data/ValSet.hdf5","r") as f:
        val_data = f['feature'][()]
    # val_data = np.transpose(val_data,axes=(0,2,1,3))
    training_dataset = ArrayDataset(training_data)
    training_dataloader = DataLoader(training_dataset,batch_size=batch_size,shuffle=True)
    val_dataset = ArrayDataset(val_data)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size)
    test_dataset = ArrayDataset(test_data)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    print(f"training set size : {training_data.shape}")
    print(f"val set size : {val_data.shape}")
    print(f"test set size : {test_data.shape}")

    # init model & optimizer & loss function
    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)
    if pre_train:
        encoder.load_state_dict(torch.load('models/e2021391527441680.pkl',map_location='cpu'),strict=False)
        decoder.load_state_dict(torch.load('models/d2021391527441680.pkl',map_location='cpu'),strict=False)
    
    encoder_optimizer, decoder_optimizer = get_optimizer(lr,wd,opt)
    loss_func = torch.nn.MSELoss()

    # train and val
    best = {"epoch":0,"rmse":10000,"params":(None,None),}
    for ep in range(epoch):
        t1 = time.time()
        if ep==30:
            encoder_optimizer, decoder_optimizer = get_optimizer(lr,wd,"sgd")
        training_loss,train_rmse = train(training_dataloader,encoder,decoder,encoder_optimizer,decoder_optimizer,loss_func,clip,device,teacher_force=True)
        val_rmse = val(val_dataloader,encoder,decoder,device)
        qw = encoder.encoder.layer[0].attention.self.query.weight.data.cpu().numpy().sum()
        kw = encoder.encoder.layer[0].attention.self.key.weight.data.cpu().numpy().sum()
        vw = encoder.encoder.layer[0].attention.self.value.weight.data.cpu().numpy().sum()
        w = np.array([qw,kw,vw])
        if val_rmse[-1] < best["rmse"]:
            best['epoch'] = ep
            best['rmse'] = val_rmse[-1]
            best['params'] = (encoder.state_dict(),decoder.state_dict())
        t2 = time.time()
        print(f"epoch[{ep:3}] | mse@train : {training_loss:6.2f} | rmse@train : {train_rmse} | rmse@val : {val_rmse} | w : {np.around(w, decimals=4)} |time : {t2-t1:5.2f}") 
    print(f"\nTraining done. Epoch {best['epoch']} achieved best rmse {best['rmse']}. Load epoch {best['epoch']} parameters for test.")
    # Test
    encoder.load_state_dict(best["params"][0])
    decoder.load_state_dict(best["params"][1])
    test_rmse = val(test_dataloader,encoder,decoder,device)
    print(f"\nTest result: rmse : {test_rmse}")
    
    # save model
    torch.save(encoder.state_dict(), f'models/e{name}.pkl')
    torch.save(decoder.state_dict(), f'models/d{name}.pkl')
    
    torch.save(encoder.state_dict(), f'models/last/e_last.pkl')
    torch.save(decoder.state_dict(), f'models/last/d_last.pkl')

    print(f"encoder is saved in 'models/e{name}.pkl'.")
    print(f"decoder is saved in 'models/d{name}.pkl'.")