import pickle as pkl
import numpy as np
import time
t1 = time.time()
gsan_keep_data_list, gsan_right_data_list,gsan_left_data_list = [], [], []
for i in range(60):
    with open(f"new_data/new_data_{i}.pkl","rb") as f:
        _ = pkl.load(f)
    data = _['data']
    label = _['label']
    gsan_left_number = gsan_right_number = gsan_keep_number = 0
    if (label == 1).any():
        gsan_left_data = data[label==1]
        gsan_left_data_list.append(gsan_left_data)
        gsan_left_number = gsan_left_data.shape[0]
    if (label == -1).any():
        gsan_right_data = data[label==-1]
        gsan_right_data_list.append(gsan_right_data)
        gsan_right_number = gsan_right_data.shape[0]
    if (label == 0).any():
        gsan_keep_number = max(gsan_left_number,gsan_right_number)*10
        gsan_keep_data = data[label==0]
        gsan_keep_data_list.append(gsan_keep_data[:gsan_keep_number])
    print(i,gsan_left_data.shape,gsan_right_data.shape,gsan_keep_data.shape)
gsan_keep_data_array = np.vstack(gsan_keep_data_list)
gsan_right_data_array = np.vstack(gsan_right_data_list)
gsan_left_data_array = np.vstack(gsan_left_data_list)
print("Totally: ",gsan_left_data_array.shape,gsan_right_data_array.shape,gsan_keep_data_array.shape)

gsan_left_data_array = np.transpose(gsan_left_data_array,axes=(0,2,1,3))
gsan_right_data_array = np.transpose(gsan_right_data_array,axes=(0,2,1,3))
gsan_keep_data_array = np.transpose(gsan_keep_data_array,axes=(0,2,1,3))
print("Transposed:",gsan_left_data_array.shape,gsan_right_data_array.shape,gsan_keep_data_array.shape)

total = {
    "right":gsan_right_data_array,
    "left":gsan_left_data_array,
    "keep":gsan_keep_data_array
}

with open("new_data/total.pkl","wb") as f:
    pkl.dump(total,f)
# with open("new_data/left.pkl","wb") as f:
#     pkl.dump(gsan_left_data_array,f)
# with open("new_data/right.pkl","wb") as f:
#     pkl.dump(gsan_right_data_array,f)
# with open("new_data/keep.pkl","wb") as f:
#     pkl.dump(gsan_keep_data_array,f)
t2 = time.time()
print(f"time : {t2-t1:.2f}")