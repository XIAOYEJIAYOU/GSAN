


import pandas as pd
import time
import numpy as np
import scipy.io as scp
from scipy.sparse import coo_matrix, csr_matrix
import h5py
import sys
sys.path.append("..")
from datatool import vid_is_unique,foot2meter,vehicle2track,reset_idx,getNeighborGraph,graph2seq,get_displacement,train_test_val_split,matlab2dataframe

def get_surrounding_id(df,max_distance=30):
    mat = df[["Local_X","Local_Y"]].to_numpy().reshape(-1,1,2)
    mat = np.repeat(mat,repeats=mat.shape[0],axis=1)
    mat_T = np.transpose(mat,(1,0,2))
    assert (mat[0,:,:] == mat_T[:,0,:]).all()
    xy_distance = mat - mat_T
    distance = (((xy_distance)**2).sum(axis=2))**0.5
    # 1. leftPreceding delta_x>0.5, delta_y<-1
    leftpreceding_indicator = xy_distance[:,:,0]>0.5 # x left bias
    leftpreceding_indicator *= xy_distance[:,:,1]<-1 # y preceding bias 
    leftpreceding_distance = leftpreceding_indicator*distance # select proposed rigion
    leftpreceding_distance *= leftpreceding_distance<max_distance # remove too far vehicle
    leftpreceding_distance += (leftpreceding_distance==0)*10000 # give a big bias for 0-value elements
    leftpreceding_id = leftpreceding_distance.argmin(axis=1).reshape(-1,1)
    # 2. Preceding -0.5<=delta_x<=0.5, delta_y<-1
    preceding_indicator = xy_distance[:,:,0]>=-0.5
    preceding_indicator *= xy_distance[:,:,0]<=0.5
    preceding_indicator *= xy_distance[:,:,1]<-1
    preceding_distance = preceding_indicator*distance 
    preceding_distance *= preceding_distance<max_distance 
    preceding_distance += (preceding_distance==0)*10000
    preceding_id = preceding_distance.argmin(axis=1).reshape(-1,1)
    # 3. rightPreceding delta_x<-0.5, delta_y<-1
    rightpreceding_indicator = xy_distance[:,:,0]<-0.5
    rightpreceding_indicator *= xy_distance[:,:,1]<-1
    rightpreceding_distance = rightpreceding_indicator*distance 
    rightpreceding_distance *= rightpreceding_distance<max_distance 
    rightpreceding_distance += (rightpreceding_distance==0)*10000
    rightpreceding_id = rightpreceding_distance.argmin(axis=1).reshape(-1,1)
    # 4. leftAlongside delta_x>0.5, -1<=delta_y<=1
    leftalongside_indicator = xy_distance[:,:,0]>0.5
    leftalongside_indicator *= xy_distance[:,:,1]>=-1
    leftalongside_indicator *= xy_distance[:,:,1]<=1
    leftalongside_distance = leftalongside_indicator*distance 
    leftalongside_distance *= leftalongside_distance<max_distance 
    leftalongside_distance += (leftalongside_distance==0)*10000
    leftalongside_id = leftalongside_distance.argmin(axis=1).reshape(-1,1)
    # 6. rightAlongside delta_x<-0.5, -1<=delta_y<=1
    rightalongside_indicator = xy_distance[:,:,0]<-0.5
    rightalongside_indicator *= xy_distance[:,:,1]>=-1
    rightalongside_indicator *= xy_distance[:,:,1]<=1
    rightalongside_distance = rightalongside_indicator*distance 
    rightalongside_distance *= rightalongside_distance<max_distance 
    rightalongside_distance += (rightalongside_distance==0)*10000
    rightalongside_id = rightalongside_distance.argmin(axis=1).reshape(-1,1)
    # 7. leftfollowing delta_x>0.5, delta_y>1
    leftfollowing_indicator = xy_distance[:,:,0]>0.5
    leftfollowing_indicator *= xy_distance[:,:,1]>1
    leftfollowing_distance = leftfollowing_indicator*distance 
    leftfollowing_distance *= leftfollowing_distance<max_distance 
    leftfollowing_distance += (leftfollowing_distance==0)*10000
    leftfollowing_id = leftfollowing_distance.argmin(axis=1).reshape(-1,1)
    # 8. following -0.5<=delta_x<=0.5, delta_y>1
    following_indicator = xy_distance[:,:,0]>=-0.5
    following_indicator *= xy_distance[:,:,0]<=0.5
    following_indicator *= xy_distance[:,:,1]>1
    following_distance = following_indicator*distance 
    following_distance *= following_distance<max_distance 
    following_distance += (following_distance==0)*10000
    following_id = following_distance.argmin(axis=1).reshape(-1,1)
    # 9. rightfollowing delta_x<-0.5, delta_y>1
    rightfollowing_indicator = xy_distance[:,:,0]<=-0.5
    rightfollowing_indicator *= xy_distance[:,:,1]>1
    rightfollowing_distance = rightfollowing_indicator*distance 
    rightfollowing_distance *= rightfollowing_distance<max_distance 
    rightfollowing_distance += (rightfollowing_distance==0)*10000
    rightfollowing_id = rightfollowing_distance.argmin(axis=1).reshape(-1,1)
    # cat
    frame = df.Frame_ID.to_numpy().reshape(-1,1)
    id = df.Vehicle_ID.to_numpy().reshape(-1,1)
    xy = df[["Local_X","Local_Y"]].to_numpy()
    new_data = np.hstack([frame,id,xy,leftpreceding_id,preceding_id,rightpreceding_id,leftalongside_id,rightalongside_id,leftfollowing_id,following_id,rightfollowing_id])
    return new_data

matlab_dataset_name = [
        'TrainSet',
        'TestSet',
        'ValSet'
    ]
for name in matlab_dataset_name:
    path = f'../data/{name}.mat'
    T = scp.loadmat(path)['tracks']
    dataset_list = matlab2dataframe(T)
    for i, dataset in enumerate(dataset_list):
        t1 = time.time()
        useful_data = reset_idx(dataset)
        useful_data = foot2meter(useful_data)
        surrounding_vids = []
        for fid, df in useful_data.groupby("Frame_ID"):
            if df.shape[0]==1:
                surrounding_vid = np.array([fid,df.Vehicle_ID.to_numpy()[0],df.Local_X.to_numpy()[0],df.Local_Y.to_numpy()[0]])
                surrounding_vid = np.hstack([surrounding_vid,np.zeros(8)]).reshape(1,-1)
            else:
                surrounding_vid = get_surrounding_id(df)
            surrounding_vids.append(surrounding_vid)
        surrounding_vids = np.vstack(surrounding_vids)
        surrounding_vids.shape
        columns = ["frame","id","x","y",'leftPrecedingId','precedingId','rightPrecedingId','leftAlongsideId','rightAlongsideId','rightFollowingId','followingId','leftFollowingId']
        new_dataframe = pd.DataFrame(surrounding_vids,columns=columns)
        new_dataframe.to_csv(f"data/{name}_{i}.csv",index=False)
        t2 = time.time()
        print(f"{name}_{i},{new_dataframe.shape},{t2-t1:.2f}s")