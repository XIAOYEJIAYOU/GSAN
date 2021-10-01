'''
Instruction：
Following process_1_csv2pkl.py, this script:
1. Split data according to lane_keep, lane_change_left, lane_change_right.
2. Select all lane_change data and select lane keep data which the number is 10 times than lane change case.
3. Combine ego vehicle's coordinates and neighbors feature. 

OUTPUT：
1. path : ../pickle_data/balanced_feature_P{i}.pkl
2. element : {"right_data":right_data_array,"left_data":left_data_array,"keep_data":keep_data_array}
3. element shape : (n, 30, 17, 10)
4. feature meaning : ego_x, ego_y, left_lane_exist, right_lane_exist, preceding_distance, following_distance, left_preceding_distance, left_following_distance, right_preceding_distance, right_following_distance 
'''



import pandas as pd
import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import sys
sys.path.append("..")
from datatool import ade,vid_is_unique,foot2meter,vehicle2track,reset_idx,getNeighborGraph,graph2seq,get_displacement,train_test_val_split,matlab2dataframe
from collections import Counter
import time
import pickle as pkl
import h5py

t1 = time.time()
pickle_folder = "../data/pickle_data"
salstm_keep_data_list, salstm_right_data_list,salstm_left_data_list = [], [], []
gsan_keep_data_list, gsan_right_data_list,gsan_left_data_list = [], [], []
i = 0
for i in range(60):
    # SALSTM
    SALSTM_file_path = f"../data/pickle_data/SALSTM_data_{i}.hdf5"
    with h5py.File(SALSTM_file_path,"r") as f:
        label = f['label'][()]
        salstm_data = f['feature'][()]
    salstm_left_number = salstm_right_number = salstm_keep_number = 0
    if (label == 1).any():
        salstm_left_data = salstm_data[label==1]
        salstm_left_data_list.append(salstm_left_data)
        salstm_left_number = salstm_left_data.shape[0]
    if (label == -1).any():
        salstm_right_data = salstm_data[label==-1]
        salstm_right_data_list.append(salstm_right_data)
        salstm_right_number = salstm_right_data.shape[0]
    if (label == 0).any():
        salstm_keep_number = max(salstm_left_number,salstm_right_number)*10
        salstm_keep_data = salstm_data[label==0]
        salstm_keep_data_list.append(salstm_keep_data[:salstm_keep_number])
    print(f"file: {SALSTM_file_path}, keep: {salstm_keep_number}, right: {salstm_right_number}, left: {salstm_left_number}")
    # GSAN
    GSAN_file_path = f"../data/pickle_data/GSAN_data_{i}.pkl"
    with open(GSAN_file_path,"rb") as f:
        _ = pkl.load(f)
    gsan_data = _['data']
    label = _['label']
    gsan_left_number = gsan_right_number = gsan_keep_number = 0
    if (label == 1).any():
        gsan_left_data = gsan_data[label==1]
        gsan_left_data_list.append(gsan_left_data)
        gsan_left_number = gsan_left_data.shape[0]
    if (label == -1).any():
        gsan_right_data = gsan_data[label==-1]
        gsan_right_data_list.append(gsan_right_data)
        gsan_right_number = gsan_right_data.shape[0]
    if (label == 0).any():
        gsan_keep_number = max(gsan_left_number,gsan_right_number)*10
        gsan_keep_data = gsan_data[label==0]
        gsan_keep_data_list.append(gsan_keep_data[:gsan_keep_number])
    print(f"file: {GSAN_file_path}, keep: {gsan_keep_number}, right: {gsan_right_number}, left: {gsan_left_number}")

    if (i+1) % 10 == 0:
        salstm_keep_data_array = np.vstack(salstm_keep_data_list)
        salstm_right_data_array = np.vstack(salstm_right_data_list)
        salstm_left_data_array = np.vstack(salstm_left_data_list)
        print(salstm_keep_data_array.shape,salstm_right_data_array.shape,salstm_left_data_array.shape)
        gsan_keep_data_array = np.vstack(gsan_keep_data_list)
        gsan_right_data_array = np.vstack(gsan_right_data_list)
        gsan_left_data_array = np.vstack(gsan_left_data_list)
        print(gsan_keep_data_array.shape,gsan_right_data_array.shape,gsan_left_data_array.shape)
        right_data_array = np.concatenate((gsan_right_data_array,salstm_right_data_array),axis=3)
        left_data_array = np.concatenate((gsan_left_data_array,salstm_left_data_array),axis=3)
        keep_data_array = np.concatenate((gsan_keep_data_array,salstm_keep_data_array),axis=3)
        print(keep_data_array.shape,right_data_array.shape,left_data_array.shape)
        _ = {
            "right_data":right_data_array,
            "left_data":left_data_array,
            "keep_data":keep_data_array
        }
        with open(f"../data/pickle_data/balanced_feature_P{i}.pkl","wb") as f:
            pkl.dump(_,f)

        # empty
        salstm_keep_data_list, salstm_right_data_list,salstm_left_data_list = [], [], []
        gsan_keep_data_list, gsan_right_data_list,gsan_left_data_list = [], [], []
    i += 1
if len(salstm_keep_data_list)>1:
    salstm_keep_data_array = np.vstack(salstm_keep_data_list)
    salstm_right_data_array = np.vstack(salstm_right_data_list)
    salstm_left_data_array = np.vstack(salstm_left_data_list)
    print(salstm_keep_data_array.shape,salstm_right_data_array.shape,salstm_left_data_array.shape)
    gsan_keep_data_array = np.vstack(gsan_keep_data_list)
    gsan_right_data_array = np.vstack(gsan_right_data_list)
    gsan_left_data_array = np.vstack(gsan_left_data_list)
    print(gsan_keep_data_array.shape,gsan_right_data_array.shape,gsan_left_data_array.shape)
    right_data_array = np.concatenate((gsan_right_data_array,salstm_right_data_array),axis=3)
    left_data_array = np.concatenate((gsan_left_data_array,salstm_left_data_array),axis=3)
    keep_data_array = np.concatenate((gsan_keep_data_array,salstm_keep_data_array),axis=3)
    print(keep_data_array.shape,right_data_array.shape,left_data_array.shape)
    _ = {
        "right_data":right_data_array,
        "left_data":left_data_array,
        "keep_data":keep_data_array
    }
    with open(f"../data/pickle_data/balanced_feature_P{i}.pkl","wb") as f:
        pkl.dump(_,f)
elif len(salstm_keep_data_list)==1:
    salstm_keep_data_array = np.asarray(salstm_keep_data_list)
    salstm_right_data_array = np.asarray(salstm_right_data_list)
    salstm_left_data_array = np.asarray(salstm_left_data_list)
    print(salstm_keep_data_array.shape,salstm_right_data_array.shape,salstm_left_data_array.shape)
    gsan_keep_data_array = np.asarray(gsan_keep_data_list)
    gsan_right_data_array = np.asarray(gsan_right_data_list)
    gsan_left_data_array = np.asarray(gsan_left_data_list)
    print(gsan_keep_data_array.shape,gsan_right_data_array.shape,gsan_left_data_array.shape)
    right_data_array = np.concatenate((gsan_right_data_array,salstm_right_data_array),axis=3)
    left_data_array = np.concatenate((gsan_left_data_array,salstm_left_data_array),axis=3)
    keep_data_array = np.concatenate((gsan_keep_data_array,salstm_keep_data_array),axis=3)
    print(keep_data_array.shape,right_data_array.shape,left_data_array.shape)
    _ = {
        "right_data":right_data_array,
        "left_data":left_data_array,
        "keep_data":keep_data_array
    }
    with open(f"../data/pickle_data/balanced_feature_P{i}.pkl","wb") as f:
        pkl.dump(_,f)
else:
    print("Empty. Nothing left.")
t2 = time.time()
print(f"time:{t2-t1:.2f}")