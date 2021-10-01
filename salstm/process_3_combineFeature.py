'''
Instruction：
Following process_2_pklDataBalance.py, this script:
1. Calculate velocity, accelerate, radian
2. Combine [accelerate, radian] feature with previous 10 features.

OUTPUT：
1. path : ../pickle_data/salstm_combine_feature_12_dim.hdf5
2. element : {"right_data":right_data_array,"left_data":left_data_array,"keep_data":keep_data_array}
3. element shape : (n, 30, 17, 12)
4. feature meaning : ego_x, ego_y, left_lane_exist, right_lane_exist, preceding_distance, following_distance, left_preceding_distance, left_following_distance, right_preceding_distance, right_following_distance, accelerate, radian
'''

import pickle as pkl
import torch
import torch.nn as nn
import numpy as np
import os
import h5py
import time
t1 = time.time()
def getVelocity(mat,slot=0.5):
    # mat:neighborhood (num, seq_len, 2)
    mat_1 = mat[:,1:,:]
    padding = np.expand_dims(mat[:,-1,:],axis=1)
    mat_1 = np.concatenate((mat_1,padding),axis=1)
    distance = ((mat_1-mat)**2).sum(axis=-1)
    velocity = distance / slot
    velocity[:,-1] = velocity[:,-2]
    return velocity

def getAccelerate(mat, slot=0.5):
    # mat:velocity (num, seq_len)
    mat_1 = mat[:,1:]
    padding = np.expand_dims(mat[:,-1],axis=1)
    mat_1 = np.concatenate((mat_1,padding),axis=1)
    accelerate = (mat_1 - mat) / slot
    return accelerate

def getVectorRadian(a,b):
    # useless
    # a, b: vector
    a_norm = np.sqrt(a.dot(a))
    b_norm = np.sqrt(b.dot(b))
    cos = a.dot(b) / (a_norm * b_norm)
    radian = np.arccos(cos)
    return radian

def getRadian(mat):
    # mat:neighborhood (num, seq_len, 2)
    # 1. fomulate road direction
    direction_x = np.ones((mat.shape[0],mat.shape[1],1))
    direction_y = np.zeros((mat.shape[0],mat.shape[1],1)) 
    base_direction = np.concatenate((direction_x,direction_y),axis=2)
    # 2. vehicle direction
    mat_1 = mat[:,1:,:]
    padding = np.expand_dims(mat[:,-1,:],axis=1)
    mat_1 = np.concatenate((mat_1,padding),axis=1)
    vehicle_direction = mat_1 - mat
    # 3. calculate cosin and then calculate arccos get the radian
    inner_dot = (vehicle_direction * base_direction).sum(axis=2)
    vehicle_direction_norm = np.sqrt((vehicle_direction * vehicle_direction).sum(axis=2))
    base_direction_norm = np.sqrt((base_direction * base_direction).sum(axis=2))
    cos = inner_dot / (vehicle_direction_norm * base_direction_norm + 1e-8)
    radian = np.arccos(cos)
    radian[:,0] = radian[:,1]
    return radian


keep_data_list, right_data_list,left_data_list = [], [], []
for path in os.listdir("../data/pickle_data/"):
    if "balanced" not in path:
        continue
    with open(os.path.join("../data/pickle_data/",path),"rb") as f:
        data = pkl.load(f)
    right_data, left_data, keep_data = data['right_data'], data['left_data'], data['keep_data']
    keep_data_list.append(keep_data)
    right_data_list.append(right_data)
    left_data_list.append(left_data)
keep_data = np.vstack(keep_data_list)
right_data = np.vstack(right_data_list)
left_data = np.vstack(left_data_list)
print(left_data.shape,keep_data.shape,right_data.shape)
slot = 0.5 
right_accelerate,left_accelerate,keep_accelerate = [],[],[]
right_radian,left_radian,keep_radian = [],[],[]
for sample in right_data:
    velocity = getVelocity(sample[:,:,:2])
    accelerate = getAccelerate(velocity)
    radian = getRadian(sample[:,:,:2])
    right_accelerate.append(accelerate)
    right_radian.append(radian)
for sample in left_data:
    velocity = getVelocity(sample[:,:,:2])
    accelerate = getAccelerate(velocity)
    radian = getRadian(sample[:,:,:2])
    left_accelerate.append(accelerate)
    left_radian.append(radian)
for sample in keep_data:
    velocity = getVelocity(sample[:,:,:2])
    accelerate = getAccelerate(velocity)
    radian = getRadian(sample[:,:,:2])
    keep_accelerate.append(accelerate)
    keep_radian.append(radian)
right_accelerate = np.expand_dims(np.asarray(right_accelerate),axis=3)
left_accelerate = np.expand_dims(np.asarray(left_accelerate),axis=3)
keep_accelerate = np.expand_dims(np.asarray(keep_accelerate),axis=3)

right_radian = np.expand_dims(np.asarray(right_radian),axis=3)
left_radian = np.expand_dims(np.asarray(left_radian),axis=3)
keep_radian = np.expand_dims(np.asarray(keep_radian),axis=3)

print(right_accelerate.shape,left_accelerate.shape,keep_accelerate.shape)
print(right_radian.shape,left_radian.shape,keep_radian.shape)
# # xy
# right_xy = right_data[:,0,:,:]
# left_xy = left_data[:,0,:,:]
# keep_xy = keep_data[:,0,:,:]
# print(right_xy.shape,left_xy.shape,keep_xy.shape)


# with open("../pickle_data/23w10v1_3cls_salstm_feature.pkl","rb") as f:
#     _ = pkl.load(f)
# # load data
# right_feature = _["right_data"]
# left_feature = _["left_data"]
# keep_feature = _["keep_data"]
# print(right_feature.shape,left_feature.shape,keep_feature.shape)

right_combine_feature = np.concatenate((right_data,right_accelerate,right_radian),axis=3)
left_combine_feature = np.concatenate((left_data,left_accelerate,left_radian),axis=3)
keep_combine_feature = np.concatenate((keep_data,keep_accelerate,keep_radian),axis=3)
print(right_combine_feature.shape,left_combine_feature.shape,keep_combine_feature.shape)

with h5py.File("../data/pickle_data/salstm_combine_feature_12_dim.hdf5","w") as f:
    f.create_dataset("right_data",data=right_combine_feature)
    f.create_dataset("left_data",data=left_combine_feature)
    f.create_dataset("keep_data",data=keep_combine_feature)
t2 = time.time()
print(f"time: {t2-t1:.2f}")