import pandas as pd
import numpy as np
import torch
from xgboost import train

def train_test_split(file_path,name,pos):
    data = np.loadtxt(fname=file_path, skiprows=0,dtype=int) # 时序网络原始文件（i,j,t）
    total_time_interval = data[-1][2]-data[0][2]
    split_time = data[0][2] + pos*total_time_interval
    times = data[:,2]
    train_his = data[np.argwhere(times<=split_time)[:,0]]
    nodes = np.unique(np.vstack((data[:, 0], data[:, 1])))
    test_his_ori = data[np.argwhere(times>split_time)[:,0]]
    removed_test = []
    for i in range(len(test_his_ori)):  
        event = test_his_ori[i]
        u = event[0]
        v = event[1]
        if u in nodes and v in nodes:
            removed_test.append(event)
    train_his = np.array(train_his,dtype=int)
    removed_test = np.array(removed_test,dtype=int)
    np.savetxt(f"test_{name}_{pos}.csv",removed_test,fmt="%d")
    np.savetxt(f"train_{name}_{pos}.csv",train_his,fmt="%d")
    # removed_test.to_csv(f"test_{name}_{pos}.csv",index=False,sep=" ",columns=False)
    # train_his.to_csv(f"train_{name}_{pos}.csv",index=False,sep=" ")

train_test_split("D:\\OneDrive/OneDrive - 西南大学/TPL/networks/original/college.csv","college",0.75)