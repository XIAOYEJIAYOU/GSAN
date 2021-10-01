import pandas as pd
import time
import numpy as np
import scipy.io as scp
import h5py
import sys
sys.path.append("..")
from datatool import vid_is_unique,foot2meter,reset_idx,getNeighborGraph,graph2seq,matlab2dataframe

if __name__ == '__main__':
    t1 = time.time()
    matlab_dataset_name = [
        'TrainSet',
        'TestSet',
        'ValSet'
    ]
    for name in matlab_dataset_name:
        path = f'data/{name}.mat'
        T = scp.loadmat(path)['tracks']
        dataset_list = matlab2dataframe(T)
        processed_dataset_list = []
        for i, dataset in enumerate(dataset_list):
            useful_data = reset_idx(dataset)
            useful_data = foot2meter(useful_data)
            neighbor_graph = getNeighborGraph(useful_data,radius=50) 
            # convert the graph into sequence piece with shape (vum=30,seq=17,2)
            seq_data = graph2seq(useful_data,neighbor_graph,seq_length=17)
            print(f"processed data shape : {seq_data.shape}")
            processed_dataset_list.append(seq_data)
        processed_dataset = np.vstack( processed_dataset_list)
        with h5py.File(f"temp/matlab/matlab{name}.hdf5","w") as f:
            f.create_dataset("data",data=processed_dataset)
        print(f"\n{name} pre-processed finished.\n")
    