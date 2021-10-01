# valid, dataset0, iloc 10000 leftpreceding
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import h5py
import time
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

def rebuild_dataformat(useful_data,seq_length = 17,down_sample_rate=5):
    datas = []
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
        xys_data = xys_data.to_numpy().reshape(-1,9,2)
        for i in range(xys_data.shape[0]):
            if (i+seq_length)*down_sample_rate > xys_data.shape[0]:
                break
            data_window = xys_data[i:i+seq_length*down_sample_rate:down_sample_rate,:,:] # (seq=16,9,2)
            datas.append(data_window)
    datas = np.stack(datas)
    return datas

matlab_dataset_name = [
        'TrainSet',
        'TestSet',
        'ValSet'
    ]
for name in matlab_dataset_name:
    t1 = time.time()
    h5_data = []
    for i in range(6):
        _t1 = time.time()
        data = pd.read_csv(f"data/{name}_{i}.csv")
        neighbor_position_matrix = get_neighbor_position(data)
        new_cols_name = [('preceding_position_x', 'preceding_position_y'), 
                         ('following_position_x', 'following_position_y'), 
                         ('leftPreceding_position_x', 'leftPreceding_position_y'), 
                         ('leftAlongside_position_x','leftAlongside_position_y'),
                         ('leftFollowing_position_x', 'leftFollowing_position_y'), 
                         ('rightPreceding_position_x', 'rightPreceding_position_y'), 
                         ('rightAlongside_position_x','rightAlongside_position_y'),
                         ('rightFollowing_position_x','rightFollowing_position_y')]
        for col_idx, (col_name1,col_name2) in enumerate(new_cols_name):
            data[col_name1] = neighbor_position_matrix[:,col_idx,0]
            data[col_name2] = neighbor_position_matrix[:,col_idx,1]
        mat = rebuild_dataformat(data)
        _t2 = time.time()
        print(name, i, mat.shape,f"{_t2-_t1:.2f}")
        h5_data.append(mat)
    h5_data = np.vstack(h5_data)
    with h5py.File(f"data/{name}.hdf5","w") as f:
        f.create_dataset("feature",data=h5_data)
    t2 = time.time()
    print("\n",name, h5_data.shape,f"{t2-t1:.2f}","\n")