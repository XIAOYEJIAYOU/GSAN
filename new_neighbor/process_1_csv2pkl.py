import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import time
import pickle as pkl
import h5py

def ade(y_true,y_pred):
    return ((((y_pred - y_true)**2).sum(axis=2))**0.5).mean(axis=1)

def get_left_right_lane_exited(laneid):
    id2label = {}
    max_laneid, min_laneid = max(laneid), min(laneid)
    laneid_standard = list(range(min_laneid, max_laneid+1))
    threshold = set(laneid_standard)-set(laneid)
    if len(threshold) == 0:
        threshold = 5
    else:
        threshold = list(threshold)[0]
    laneid = np.array(laneid)
    upper = sorted(laneid[laneid<=threshold])
    lower = sorted(laneid[laneid>threshold])
    # (left lane exit,right lane exit)
    # upper 2-right, 3, 4-left
    # lower 6-left, 7, 8-right
    id2label[upper[0]] = [1,0]
    id2label[upper[-1]] = [0,1]
    for id in upper[1:-1]:
        id2label[id] = [1, 1]
    id2label[lower[0]] = [0,1]
    id2label[lower[-1]] = [1,0]
    for id in lower[1:-1]:
        id2label[id] = [1, 1]
    return id2label

    
def changeLane(df):
    uniLaneId = df.laneId.unique()
    if len(uniLaneId) == 1:
        change = False
    elif len(uniLaneId) > 1:
        change = True
    return change

def getChangeXY(df):
    uniLaneId = df.laneId.unique()
    xy_dict = {}
    for i in range(len(uniLaneId)):
        changeId = df[df.laneId == uniLaneId[i]].index[0]
        x = df.loc[changeId].x
        y = df.loc[changeId].y
        xy_dict[changeId] = (x,y)
    return xy_dict

def rightOrLeft(xy_dict):
    label_dict = {"right":-1,"left":1} # 同左异右
    xy_id = list(xy_dict.keys())
    xy_list = list(xy_dict.values())
    labels = []
    for i in range(len(xy_list)-1):
        dx = xy_list[i+1][0] - xy_list[i][0]
        dy = xy_list[i+1][1] - xy_list[i][1]
        if dx*dy > 0:
            labels.append(label_dict['left'])
        elif dx*dy < 0:
            labels.append(label_dict['right'])
    return list(zip(xy_id[1:],labels))

def get_neighbor_position(data):
    # data : dataframe ["frame","x","y","laneId",'precedingId', 'followingId', 'leftPrecedingId','leftFollowingId', 'rightPrecedingId','rightFollowingId']
    frame = data.frame.to_numpy().reshape(-1,1)
    id_mat = data[['precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId','leftFollowingId', 'rightPrecedingId', 'rightAlongsideId','rightFollowingId']].to_numpy()
    frame = frame.repeat(id_mat.shape[1],axis=1)
    id_mat = id_mat.reshape(id_mat.shape[0],id_mat.shape[1],1)
    frame = frame.reshape(frame.shape[0],frame.shape[1],1)
    id_frame_pair = np.concatenate((id_mat,frame),axis=2)
    xy = data[["x","y"]].to_numpy()
    x,y,v_id,f_id = data.x,data.y,data.id,data.frame
    vehicle_num, frame_num = v_id.max()+1, f_id.max()+1
    sparse_X = csr_matrix((x, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    sparse_Y = csr_matrix((y, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    neighbor_position = []
    for i in range(id_frame_pair.shape[0]):
        row = id_frame_pair[i]
        ego_x,ego_y = xy[i,0],xy[i,1]
        new_row = []
        for pair in row:
            if pair[0] == 0:
                position = [0,0]
            else:
                x = sparse_X[pair[0],pair[1]]
                y = sparse_Y[pair[0],pair[1]]
                position = [x,y]
            new_row.append(position)
        neighbor_position.append(new_row)
    neighbor_position = np.array(neighbor_position)
    return neighbor_position

def rebuild_dataformat(useful_data,seq_length = 16,down_sample_rate=5):
    labels,datas = [],[]
    for vid, df in useful_data.groupby("id"):
        xys_data = df[["x","y",
                     'preceding_position_x', 'preceding_position_y', 
                     'following_position_x', 'following_position_y', 
                     'leftPreceding_position_x', 'leftPreceding_position_y', 
                     'leftAlongside_position_x','leftAlongside_position_y',
                     'leftFollowing_position_x', 'leftFollowing_position_y', 
                     'rightPreceding_position_x', 'rightPreceding_position_y', 
                     'rightAlongside_position_x','rightAlongside_position_y',
                     'rightFollowing_position_x','rightFollowing_position_y']]
        df_l = df.label.to_numpy()
        xys_data = xys_data.to_numpy().reshape(-1,9,2)
        for i in range(xys_data.shape[0]):
            if (i+seq_length)*down_sample_rate > xys_data.shape[0]:
                break
            data_window = xys_data[i:i+seq_length*down_sample_rate:down_sample_rate,:,:] # (seq=16,9,2)
            l_window = df_l[i:i+seq_length*down_sample_rate:down_sample_rate] # (seq=16,9,2)
            if l_window[6:].sum()>0:
                l = 1
            elif l_window[6:].sum()<0:
                l = -1
            elif l_window[6:].sum()==0:
                l = 0
            labels.append(l)
            datas.append(data_window)
    datas = np.stack(datas)
    labels = np.hstack(labels)
    return datas, labels

selected_col = ["frame","id","x","y","laneId",'precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId','leftFollowingId', 'rightPrecedingId', 'rightAlongsideId','rightFollowingId']
for i in range(60):
    t1 = time.time()
    if i<9:
        data_path = f"../data/highD-dataset-v1.0/data/0{i+1}_tracks.csv"
    else:
        data_path = f"../data/highD-dataset-v1.0/data/{i+1}_tracks.csv"
    # tracks  
    data = pd.read_csv(data_path)
    useful_data = data[selected_col]
    useful_data = useful_data.sort_values(by=["id","frame"])

    # if right lane and left lane exited or not
    laneid = useful_data.laneId.unique()
    id2label = get_left_right_lane_exited(laneid)
    left_right_lane_exited = []
    for id in useful_data.laneId:
        left_right_lane_exited.append(id2label[id])
    left_right_lane_exited = np.array(left_right_lane_exited)
    useful_data['left_lane_exited'] = left_right_lane_exited[:,0]
    useful_data['right_lane_exited'] = left_right_lane_exited[:,1]

    neighbor_position_matrix = get_neighbor_position(useful_data)
    new_cols_name = [('preceding_position_x', 'preceding_position_y'), 
                     ('following_position_x', 'following_position_y'), 
                     ('leftPreceding_position_x', 'leftPreceding_position_y'), 
                     ('leftAlongside_position_x','leftAlongside_position_y'),
                     ('leftFollowing_position_x', 'leftFollowing_position_y'), 
                     ('rightPreceding_position_x', 'rightPreceding_position_y'), 
                     ('rightAlongside_position_x','rightAlongside_position_y'),
                     ('rightFollowing_position_x','rightFollowing_position_y')]
    for col_idx, (col_name1,col_name2) in enumerate(new_cols_name):
        useful_data[col_name1] = neighbor_position_matrix[:,col_idx,0]
        useful_data[col_name2] = neighbor_position_matrix[:,col_idx,1]

    labels = np.zeros(useful_data.shape[0])
    useful_data['label'] = labels
    laneChangeLabels = []
    for idx, df in data.groupby("id"):
        unique_laneid = df.laneId.unique()
        if not changeLane(df):
            continue
        xy_dict = getChangeXY(df)
        label = rightOrLeft(xy_dict)
        laneChangeLabels.extend(label)

    BEGIN_LOC_ID, END_LOC_ID = data.index[0], data.index[-1]
    for loc_id , label in laneChangeLabels:
        begin_loc_id, end_loc_id = max(loc_id-10, BEGIN_LOC_ID), min(loc_id+10, END_LOC_ID)
        l_num = end_loc_id - begin_loc_id + 1
        l = np.ones(l_num)*label
        useful_data.loc[begin_loc_id:end_loc_id,"label"] = l
    datas, labels = rebuild_dataformat(useful_data)
    data = {"data":datas,"label":labels}
    with open(f"new_data/new_data_{i}.pkl","wb") as f:
        pkl.dump(data,f)
    t2 = time.time()
    print(f"file_{i}  processed. time: {t2-t1:.2f}. data shape {datas.shape}. label shape {labels.shape}.")