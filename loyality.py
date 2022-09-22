# # 基于节点同一阶邻居的熵率来表示节点的loyalty
import sys
from tracemalloc import Snapshot
from turtle import distance, shape

from sympy import im, radsimp
sys.path.append("../")
from init_net import init
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx
import dynetx as dx
import tqdm
import em_hawkes as hks
from init_net import init
from evalution import Evalution
from scipy.special import entr
import random

train_network = "networks/train/fb-forum0-75.0.csv"
train_g = init(train_network)
begin = train_g.graph['time_stamp_list'][0]
end = train_g.graph["time_stamp_list"][-1]
resolution = 60*60*24*31 #一周的时间为分辨率
sampling_range = np.arange(begin,end,resolution)
node_loyality = pd.Series(0,index=list(train_g.nodes),dtype=float)
nodes = list(train_g.nodes)
random.shuffle(nodes)
for node in nodes:
    n = len(train_g[node])
    l = len(sampling_range)
    M = np.zeros(shape=((n,l)),)
    count = 0
    for nbr in train_g[node]:
        for t in train_g.edges[node,nbr]['time_stamp']:
            M[count][int((t-begin)/resolution)] += 1
        count += 1
    reappear = []   # 重新出现边所占的强度
    new = []    # 新边所占强度
    miss = []   # 缺失边所占强度
    degree = 0
    for i in range(l):
        v = M[:,i]
        new_degree = 0
        while degree+new_degree < n and v[degree+new_degree] > 0:       
            new_degree += 1
        totol_links = np.sum(v[:degree+new_degree])
        new_links = np.sum(v[degree:degree+new_degree])
        miss_links = degree - np.count_nonzero(v[:degree])
        reappear_links = np.sum(v[:degree])
        degree = degree + new_degree
        if totol_links > 0: # 该时段产生联系的情况
            new.append(new_links/totol_links)
            reappear.append(reappear_links/totol_links)
            miss.append(miss_links/degree)
        else:   # 该时段没有产生联系的情况
            new.append(0)
            reappear.append(0)
            miss.append(0)
    node_loyality[node] = np.mean(reappear)
    plt.title(f"{node}")
    plt.plot(reappear)
    plt.show()
print(node_loyality)

