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

def getNeighborGraph(data,radius=10):
    x,y,v_id,f_id = data.x,data.y,data.id,data.frame

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

def get_neighbor_distance(data):
    # data : dataframe ["frame","x","y","laneId",'precedingId', 'followingId', 'leftPrecedingId','leftFollowingId', 'rightPrecedingId','rightFollowingId']
    frame = data.frame.to_numpy().reshape(-1,1)
    id_mat = data[['precedingId', 'followingId', 'leftPrecedingId','leftFollowingId', 'rightPrecedingId','rightFollowingId']].to_numpy()
    frame = frame.repeat(id_mat.shape[1],axis=1)
    id_mat = id_mat.reshape(id_mat.shape[0],id_mat.shape[1],1)
    frame = frame.reshape(frame.shape[0],frame.shape[1],1)
    id_frame_pair = np.concatenate((id_mat,frame),axis=2)
    xy = data[["x","y"]].to_numpy()
    x,y,v_id,f_id = data.x,data.y,data.id,data.frame
    vehicle_num, frame_num = v_id.max()+1, f_id.max()+1
    sparse_X = csr_matrix((x, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    sparse_Y = csr_matrix((y, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    neighbor_distance = []
    for i in range(id_frame_pair.shape[0]):
        row = id_frame_pair[i]
        ego_x,ego_y = xy[i,0],xy[i,1]
        new_row = []
        for pair in row:
            if pair[0] == 0:
                distance = 0
            else:
                x = sparse_X[pair[0],pair[1]]
                y = sparse_Y[pair[0],pair[1]]
                distance = (((ego_x-x)**2)+((ego_y-y)**2))**0.5
            new_row.append(distance)
        neighbor_distance.append(new_row)
    neighbor_distance = np.array(neighbor_distance)
    return neighbor_distance

def graph2seq(data,graph_list,seq_length=16,max_vnum=30,down_sample_rate=5,sort_func="distance"):
    # common data
    x,y,v_id,f_id,l = data.x,data.y,data.id,data.frame,data.label
    # left & right lane exited or not
    left_exited,right_exited = data.left_lane_exited,data.right_lane_exited
    # 6-direction distances among neighbors
    pd,fd,lpd,lfd,rpd,rfd = data.preceding_distance,data.following_distance, data.leftPreceding_distance,data.leftFollowing_distance, data.rightPreceding_distance,data.rightFollowing_distance
    vehicle_num, frame_num = v_id.max()+1, f_id.max()+1
    sparse_X = csr_matrix((x, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num))) # i行:车id;j列:时间;元素为i车j时刻的坐标x
    sparse_Y = csr_matrix((y, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num))) # i行:车id;j列:时间;元素为i车j时刻的坐标y
    sparse_L = csr_matrix((l, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num))) # i行:车id;j列:时间;元素为i车j时刻的下一时刻是否lane change
    # feature matrix
    sparse_left = csr_matrix((left_exited, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num))) # i行:车id;j列:时间;元素为i车j时刻左侧是否有车道
    sparse_right = csr_matrix((right_exited, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num))) # i行:车id;j列:时间;元素为i车j时刻右侧是否有车道
    sparse_pd = csr_matrix((pd, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    sparse_fd = csr_matrix((fd, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    sparse_lpd = csr_matrix((lpd, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    sparse_lfd = csr_matrix((lfd, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    sparse_rpd = csr_matrix((rpd, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    sparse_rfd = csr_matrix((rfd, (v_id, f_id)), shape=(int(vehicle_num), int(frame_num)))
    
    seq_windows,label,seq_feature_windows = [],[],[]
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
        dense_l = sparse_L[row_start:row_end,col_start:col_end].toarray()
        # feature mat
        dense_left = sparse_left[row_start:row_end,col_start:col_end].toarray()
        dense_right = sparse_right[row_start:row_end,col_start:col_end].toarray()
        dense_pd = sparse_pd[row_start:row_end,col_start:col_end].toarray()
        dense_fd = sparse_fd[row_start:row_end,col_start:col_end].toarray()
        dense_lpd = sparse_lpd[row_start:row_end,col_start:col_end].toarray()
        dense_lfd = sparse_lfd[row_start:row_end,col_start:col_end].toarray()
        dense_rpd = sparse_rpd[row_start:row_end,col_start:col_end].toarray()
        dense_rfd = sparse_rfd[row_start:row_end,col_start:col_end].toarray()
        dense_feature = np.stack((dense_left,dense_right,dense_pd,dense_fd,dense_lpd,dense_lfd,dense_rpd,dense_rfd,),axis=2) # (vum,total_seq,8)
        if dense_xy.shape[0]<max_vnum:
            padding_num = max_vnum-dense_xy.shape[0]

            padding_xy = np.zeros((padding_num,dense_xy.shape[1],dense_xy.shape[2]))
            dense_xy = np.vstack([dense_xy,padding_xy])
            
            padding_feature = np.zeros((padding_num,dense_feature.shape[1],dense_feature.shape[2]))
            dense_feature = np.vstack([dense_feature,padding_feature])
            
            padding_I = np.zeros((padding_num,dense_I.shape[1]))
            dense_I = np.vstack([dense_I,padding_I])
            
            dense_l = np.vstack([dense_l,padding_I])
            
        for i in range(dense_xy.shape[1]): # for loop on sequence dim
            if (i+seq_length)*down_sample_rate > dense_xy.shape[1]:
                break
            window = dense_xy[:,i:i+seq_length*down_sample_rate:down_sample_rate,:] # (vum=30,seq=16,2)
            window_l = dense_l[:,i:i+seq_length*down_sample_rate:down_sample_rate] # (vum=30,seq=16)
            window_feature = dense_feature[:,i:i+seq_length*down_sample_rate:down_sample_rate] # (vum=30,seq=16)
            if sort_func == "duration":
                dense_seq_I = dense_I[:,i:(i+seq_length)*down_sample_rate:down_sample_rate]
                related_score = dense_seq_I.sum(axis=1)
                related_score[dense_v] = related_score[dense_v] + 100 # actually 1 is enough
                related_rank = np.argsort(-related_score)
            elif sort_func == "distance":
                related_score = ade(window[:,:6,:],window[dense_v,:6,:])
                related_rank = np.argsort(related_score)
            window = window[related_rank[:max_vnum],:,:]
            window_feature = window_feature[related_rank[:max_vnum],:,:]
            seq_windows.append(window)
            seq_feature_windows.append(window_feature)
            if window_l[0,6:].sum()>0:
                l = 1
            elif window_l[0,6:].sum()<0:
                l = -1
            elif window_l[0,6:].sum()==0:
                l = 0
            label.append(l) # 30,17,2 (0-5),6,(7-16)
    if len(seq_windows)==0:
        seq_data = None
        seq_label = None
        feature = None
    else:
        seq_data = np.stack(seq_windows)#(n,vum=30,seq=16,2)
        seq_label = np.stack(label)
        feature = np.stack(seq_feature_windows)
    return seq_data,seq_label,feature


selected_col = ["frame","id","x","y","laneId",'precedingId', 'followingId', 'leftPrecedingId','leftFollowingId', 'rightPrecedingId','rightFollowingId']
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
    
    # get neighbor distances
    neighbor_distances_matrix = get_neighbor_distance(useful_data)
    new_cols_name = ['preceding_distance', 'following_distance', 'leftPreceding_distance','leftFollowing_distance', 'rightPreceding_distance','rightFollowing_distance']
    for col_idx, col_name in enumerate(new_cols_name):
        useful_data[col_name] = neighbor_distances_matrix[:,col_idx]

    # lane change or not
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
    neighbor_graph = getNeighborGraph(useful_data,radius=50)
    seq_data,seq_label,feature = graph2seq(useful_data,neighbor_graph,seq_length=17)
    GSAN_data = {"data":seq_data,"label":seq_label}
    
    with open(f"../data/pickle_data/GSAN_data_{i}.pkl","wb") as f:
        pkl.dump(GSAN_data,f)
    # SALSTM_data = {"feature":feature[:,:,:,:],"label":seq_label}
    # with open(f"../pickle_data/SALSTM_data_{i}.pkl","wb") as f:
    #     pkl.dump(SALSTM_data,f)
    with h5py.File(f"../data/pickle_data/SALSTM_data_{i}.hdf5","w") as f:
        f.create_dataset("feature",data=feature)
        f.create_dataset("label",data=seq_label)
    t2 = time.time()
    print(f"file_{i}  processed. time: {t2-t1:.2f}. data shape {seq_data.shape}. label shape {seq_label.shape}. feature shape {feature.shape}")
    # left_exist, right_exist, preceding_distance, following_distance, left_preceding_distance, left_following_distance, right_preceding_distance, right_following_distance