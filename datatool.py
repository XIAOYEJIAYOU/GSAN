import pandas as pd
import torch
import random
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from torch.utils.data import Dataset
import time
import pickle as pkl
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

def load_array_data(dataname):
    if dataname == "matlab":
        with h5py.File("temp/matlab/matlabTrainSet.hdf5","r") as f:
            training_data = f["data"][()]
        with h5py.File("temp/matlab/matlabTestSet.hdf5","r") as f:
            test_data = f["data"][()]
        with h5py.File("temp/matlab/matlabValSet.hdf5","r") as f:
            val_data = f["data"][()]
    elif dataname == "mini":
        with open("temp/mini/mini_train.pkl","rb") as f:
            training_data = pkl.load(f)
        with open("temp/mini/mini_test.pkl","rb") as f:
            test_data = pkl.load(f)
        with open("temp/mini/mini_val.pkl","rb") as f:
            val_data = pkl.load(f)
    elif dataname == "mini17":
        with open("temp/mini/mini_train17.pkl","rb") as f:
            training_data = pkl.load(f)
        with open("temp/mini/mini_test17.pkl","rb") as f:
            test_data = pkl.load(f)
        with open("temp/mini/mini_val17.pkl","rb") as f:
            val_data = pkl.load(f)
    elif dataname == "mini17distance":
        with open("temp/mini/mini_train17distance.pkl","rb") as f:
            training_data = pkl.load(f)
        with open("temp/mini/mini_test17distance.pkl","rb") as f:
            test_data = pkl.load(f)
        with open("temp/mini/mini_val17distance.pkl","rb") as f:
            val_data = pkl.load(f)
    elif dataname == "id_intercection":
        with h5py.File("temp/id/train/intercection.hdf5","r") as f:
            training_data = f["data"][()]
        val_data,test_data = None, None
    else:
        raise ValueError(f"Cannot load {dataname}.")
    return training_data,val_data,test_data

'''
def rmse(y_true,y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())
'''

def rmse(y_true,y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean(axis=-1)).mean()

def ade(y_true,y_pred):
    return ((((y_pred - y_true)**2).sum(axis=2))**0.5).mean(axis=1)


def vid_is_unique(data):
    is_unique = []
    for fid,df in data.groupby("Frame_ID"):
        track_num = df.shape[0]
        vehicle_num = df.Vehicle_ID.unique().shape[0]
        is_unique.append(track_num == vehicle_num)
    is_unique = np.array(is_unique)
    return is_unique.all()

def reset_idx(data):
    data = data.sort_values(by=["Vehicle_ID","Frame_ID"])
    data.Vehicle_ID = data.Vehicle_ID-data.Vehicle_ID.min()
    data.Frame_ID = data.Frame_ID-data.Frame_ID.min()
    return data

def foot2meter(data):
    transform_weight = 0.3048
    data = data.sort_values(by=["Vehicle_ID","Frame_ID"])
    data.Local_X = data.Local_X*transform_weight
    data.Local_Y = data.Local_Y*transform_weight
    return data


def vehicle2track(data):
    '''
    Some track with same vehicle_id are discontinuous. Splite discontinuous track to continuous track.
    '''
    new_vid = 0
    new_data_list = []
    for _, df in data.groupby("Vehicle_ID"):
        df = df.sort_values("Frame_ID")
        fxy = df[["Frame_ID","Local_X","Local_Y"]].values
        current_fid = df.Frame_ID.values[:-1]
        next_fid = df.Frame_ID.values[1:]
        interval = next_fid-current_fid

        if set(interval)==set([1]):
            vid = np.repeat(new_vid,df.shape[0])
            vfxy = np.hstack((vid.reshape(-1,1),fxy))
            new_data_list.append(vfxy)
            new_vid += 1
        else:
            idx = [0]
            idx.extend(np.where(interval!=1)[0]+1)
            idx.extend([df.shape[0]])
            for i in range(len(idx)):
                if i+1>len(idx)-1:
                    break
                pfxy = fxy[idx[i]:idx[i+1]]
                vid = np.repeat(new_vid,pfxy.shape[0])
                vfxy = np.hstack((vid.reshape(-1,1),pfxy))
                new_data_list.append(vfxy)
                new_vid += 1
    new_data_array = np.vstack(new_data_list)
    new_dataframe = pd.DataFrame(new_data_array,columns=["Vehicle_ID","Frame_ID","Local_X","Local_Y"])
    return new_dataframe

def get_displacement(data):
    data = data.sort_values(by=["Vehicle_ID","Frame_ID"])
    distance_list = []
    for vid,df in data.groupby("Vehicle_ID"):
        df = df.sort_values("Frame_ID")
        current_xy = df[["Local_X","Local_Y"]].to_numpy()[:-1,:]
        next_xy = df[["Local_X","Local_Y"]].to_numpy()[1:,:]
        distance = (((next_xy - current_xy)**2).sum(axis=1))**0.5
        distance_list.extend([0])
        distance_list.extend(distance)
    data["displacement"] = distance_list
    return data

def getNeighborGraph(data,radius=10):
    x,y,v_id,f_id = data.Local_X,data.Local_Y,data.Vehicle_ID,data.Frame_ID

    vehicle_num, frame_num = v_id.max()+1, f_id.max()+1
    sparse_X = csr_matrix((x, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num))) # i行:车id;j列:时间;元素为i车j时刻的坐标x
    sparse_Y = csr_matrix((y, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num))) # i行:车id;j列:时间;元素为i车j时刻的坐标y
    I_mat = (sparse_X!=0)*1 # i行:车id;j列:时间;元素为i车j时刻是否出现,出现为1,否则为0
    mask = []
    for v in range(I_mat.shape[0]):
        concurrent_mask = I_mat.multiply(I_mat[v]) #同或 [1,0,1,0,0,0,1] & [1,0,1,1,1,0,0] = [1,0,1,0,0,0,0]

        # 邻居xy坐标
        concurrent_X = concurrent_mask.multiply(sparse_X) 
        concurrent_Y = concurrent_mask.multiply(sparse_Y)

        # 自己xy坐标
        self_x = concurrent_mask.multiply(sparse_X[v])
        self_y = concurrent_mask.multiply(sparse_Y[v])

        # 差值
        delta_x = self_x - concurrent_X
        delta_y = self_y - concurrent_Y

        # 邻居x坐标在半径以内的指示矩阵
        x_in_id = np.where((delta_x.data>-radius) & (delta_x.data<radius))
        xc = delta_x.tocoo()
        xrow_in = xc.row[x_in_id]
        xcol_in = xc.col[x_in_id]
        xI_data = np.ones(xrow_in.shape[0])
        xneighbor_in_mat = csr_matrix((xI_data, (xrow_in, xcol_in)), shape=(I_mat.shape[0], I_mat.shape[1]))

        # 邻居y坐标在半径以内的指示矩阵
        y_in_id = np.where((delta_y.data>-radius) & (delta_y.data<radius))
        yc = delta_y.tocoo()
        yrow_in = yc.row[y_in_id]
        ycol_in = yc.col[y_in_id]
        yI_data = np.ones(yrow_in.shape[0])
        yneighbor_in_mat = csr_matrix((yI_data, (yrow_in, ycol_in)), shape=(I_mat.shape[0], I_mat.shape[1]))

        neighbor_in_mat = xneighbor_in_mat.multiply(yneighbor_in_mat).tolil()
        neighbor_in_mat[v] = I_mat[v]
        mask.append(neighbor_in_mat.tocsr())
    return mask

def graph2seq(data,graph_list,seq_length=16,max_vnum=30,down_sample_rate=5,sort_func="distance"):
    x,y,v_id,f_id = data.Local_X,data.Local_Y,data.Vehicle_ID,data.Frame_ID
    vehicle_num, frame_num = v_id.max()+1, f_id.max()+1
    sparse_X = csr_matrix((x, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num))) # i行:车id;j列:时间;元素为i车j时刻的坐标x
    sparse_Y = csr_matrix((y, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num))) # i行:车id;j列:时间;元素为i车j时刻的坐标y
    seq_windows = []
    for v,graph in enumerate(graph_list):
        if graph.data.size==0:
            continue
        row = np.unique(graph.tocoo().row)
        col = np.unique(graph.tocoo().col)
        row_start,row_end = row.min(), row.max()+1
        col_start,col_end = col.min(), col.max()+1
        dense_v = v - row_start
        dense_I = graph[row_start:row_end,col_start:col_end].toarray()
        dense_x = sparse_X[row_start:row_end,col_start:col_end].toarray()
        dense_y = sparse_Y[row_start:row_end,col_start:col_end].toarray()
        dense_xy = np.stack((dense_x,dense_y),axis=2) # (vum,total_seq,2)
        if dense_xy.shape[0]<max_vnum:
            padding_num = max_vnum-dense_xy.shape[0]
            padding_xy = np.zeros((padding_num,dense_xy.shape[1],dense_xy.shape[2]))
            padding_I = np.zeros((padding_num,dense_I.shape[1]))
            dense_xy = np.vstack([dense_xy,padding_xy])
            dense_I = np.vstack([dense_I,padding_I])
        for i in range(dense_xy.shape[1]): # for loop on sequence dim
            if (i+seq_length)*down_sample_rate > dense_xy.shape[1]:
                break
            window = dense_xy[:,i:i+seq_length*down_sample_rate:down_sample_rate,:] # (vum=30,seq=16,2)
            if sort_func == "duration":
                dense_seq_I = dense_I[:,i:(i+seq_length)*down_sample_rate:down_sample_rate]
                related_score = dense_seq_I.sum(axis=1)
                related_score[dense_v] = related_score[dense_v] + 100 # actually 1 is enough
                related_rank = np.argsort(-related_score)
            elif sort_func == "distance":
                related_score = ade(window[:,:6,:],window[dense_v,:6,:])
                related_rank = np.argsort(related_score)
            window = window[related_rank[:max_vnum],:,:]
            seq_windows.append(window.astype(np.float32))
    if len(seq_windows)==0:
        seq_data = None
    else:
        seq_data = np.stack(seq_windows).astype(np.float32) #(n,vum=30,seq=16,2)
    return seq_data

def train_test_val_split(data_mat,test_size=0.2,val_size=0.1,seed=None):
    train_size = 1 - test_size - val_size
    val_idx = int(data_mat.shape[0]*val_size)
    train_idx = int(data_mat.shape[0]*(train_size+val_size))
    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(data_mat)
    split_idx = [val_idx,train_idx]
    val_set,train_set,test_set = np.split(data_mat,split_idx)
    return train_set, val_set, test_set



class ArrayDataset(Dataset):
    def __init__(self, array):
        self.array = array
    def __len__(self): 
        return len(self.array)
    def __getitem__(self, i): 
        return self.array[i]
    
    
def plot_batch(one_batch,figsize=(10,10),alpha=0.3):
    fig = plt.figure(figsize=figsize)
    for i, seq in enumerate(one_batch):
        mask = (seq!=0)
        if mask.sum()==0:
            continue      
        hist_seq,fut_seq = seq[:6],seq[6:]
        if i == 0:
            plt.plot(hist_seq[:,0],hist_seq[:,1],c="orange",label="hist",lw=3)
            plt.plot(fut_seq[:,0],fut_seq[:,1],c="green",label="future",lw=3,ls='--')
            plt.gca().add_patch(patches.Rectangle((seq[5,0]-0.15, seq[5,1]-2), 0.3, 4,edgecolor="black",facecolor="orange"))
            plt.text(x= seq[5,0]-0.5, y = seq[5,1]-2,s=f"{i}",fontsize=12,fontweight="bold", horizontalalignment='center')
        elif i == 1:
            hist_mask,fut_mask = (hist_seq!=0),(fut_seq!=0)
            mhist_seq = hist_seq[hist_mask].reshape(-1,2)
            mfut_seq = fut_seq[fut_mask].reshape(-1,2)  
            # plot hist
            if hist_mask.sum()==0:
                pass
            else:
                plt.plot(mhist_seq[:,0],mhist_seq[:,1],c="blue",label="neighbor hist",alpha=alpha,lw=3)
            # plot future
            if fut_mask.sum()==0:
                pass
            else:
                plt.plot(mfut_seq[:,0],mfut_seq[:,1],c="blue",label="neighbor future",alpha=alpha,lw=3,ls='--')
            #plot current
            if (seq[5]!=0).all():
                plt.gca().add_patch(patches.Rectangle((seq[5,0]-0.15, seq[5,1]-2), 0.3, 4,edgecolor="black",facecolor="blue"))
                plt.text(x= seq[5,0]-0.5, y = seq[5,1]-2,s=f"{i}",fontsize=12,fontweight="bold", horizontalalignment='center')
        else:
            hist_mask,fut_mask = (hist_seq!=0).any(axis=1),(fut_seq!=0).any(axis=1)
            mhist_seq = hist_seq[hist_mask].reshape(-1,2)
            mfut_seq = fut_seq[fut_mask].reshape(-1,2)  
            # plot hist
            if hist_mask.sum()==0:
                pass
            else:
                plt.plot(mhist_seq[:,0],mhist_seq[:,1],c="blue",alpha=alpha,lw=3)
            # plot future
            if fut_mask.sum()==0:
                pass
            else:
                plt.plot(mfut_seq[:,0],mfut_seq[:,1],c="blue",alpha=alpha,lw=3,ls='--')
            #plot current
            if (seq[5]!=0).all():
                plt.gca().add_patch(patches.Rectangle((seq[5,0]-0.15, seq[5,1]-2), 0.3, 4,edgecolor="black",facecolor="blue"))
                plt.text(x= seq[5,0]-0.5, y = seq[5,1]-2,s=f"{i}",fontsize=12,fontweight="bold", horizontalalignment='center')
    plt.legend(loc="best")
    return fig


class Logger(object):
    def __init__(self,log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
    def flush(self):
        pass    
    

def matlab2dataframe(T):
    dataset_list = []
    for dataset in T:
        vid_list,df_list,n = [],[],0
        for vehicle in dataset:
            if vehicle.size==0:
                continue
            else:
                vid_list.append(np.repeat(n,vehicle.shape[1]))
                df_list.append(vehicle)
                n += 1
        vid_array = np.hstack(vid_list)
        df_array = np.hstack(df_list)
        recovery_dataframe = pd.DataFrame(np.vstack([vid_array,df_array]).transpose(),columns=["Vehicle_ID","Frame_ID","Local_X","Local_Y"])
        dataset_list.append(recovery_dataframe)
    return dataset_list

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
