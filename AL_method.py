# activity and loyalty method
import numpy as np
import pandas as pd
import sys

from sympy import sequence
sys.path.append("d:\\OneDrive/OneDrive - 西南大学/TPL/")
from init_net import init
import em_hawkes 
from tqdm import tqdm
import RPP_model as rpp
from tick.hawkes import HawkesSumExpKern
from scipy.optimize import leastsq
import networkx as nx
import os
import re

def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos
# 定义函数
def fun(x, p):
    a, b, c = p
    x = x + c
    return a + b*np.log(x)
def error(p, x, y,T):
    l1 = 0
    a,b,c = p
    # l1 = a + b + c
    # last = fun(x-1,p) - y[-1]
    return (fun(x, p) - y)*np.exp((x/T-1)*2) # 将末尾的注意力变强
def model_fit(x, y, defult=[0.1, 1, 1],):
    T = max(x)
    if len(x) < 3:
        return defult,0
    para = leastsq(error, x0=defult, args=(x, y,T))
    return para
def getNexts(pattern):
    next = np.zeros(len(pattern))
    j = 0
    for i in range(1,len(pattern)):
        while j!=0 and pattern[j]!=pattern[i-1]:
            j = next[j]
        if pattern[j] == pattern[i-1]:
            j += 1
        next[i] = j
    return next
def list_search(pattern,sequence):
    next = getNexts(pattern)
    j = 0
    for i,node in enumerate(sequence):
        while j>0 and sequence[i] != pattern[j]:
            j = next[j]
        if sequence[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            return True
    return False
SAMPING_NUM = 100
class ALTLP():
    def __init__(self,train_network_file,T):
        self.name = train_network_file.split('\\')[-1].split('.')[0]
        self.network_file = train_network_file
        self.g = init(train_network_file) #原始的网络文件
        self.begin = self.g.graph['time_stamp_list'][0]
        self.end = self.g.graph['time_stamp_list'][-1]
        self.norm = (self.end - self.begin)/100
        self.node_all_events,self.node_degree_events = self.get_events()
        self.degree_events = self.get_degree_events()
        self.hks_para,self.node_activity = self.hks_pre(self.node_all_events,T)
        # self.node_new_link = self.hks_pre2(self.node_degree_events,T)
        # self.node_loyalty,self.loyalty_trans = self.get_loyalty(T)
        self.node_loyalty,self.node_event_str = self.get_entropy_loyalty()
        # self.node_loyalty = self.node_new_link
        # self.node_loyalty = 1-self.node_loyalty/max(self.node_activity)
        
    def get_events(self):
        all_events_list = []
        degree_events_list = []
        for node in self.g:
            all_events = []
            degree_events = []
            for nbr in self.g[node]:
                all_events += self.g.edges[node,nbr]['time_stamp']
                degree_events.append(self.g.edges[node,nbr]['time_stamp'][0])
            all_events.sort()
            degree_events.sort()
            all_events = np.array(all_events)
            degree_events = np.array(degree_events)
            all_events = (all_events - self.begin)/self.norm
            degree_events = (degree_events - self.begin)/self.norm
            all_events_list.append((node,all_events))
            degree_events_list.append((node,degree_events))
        node_all_events = pd.Series(dict(all_events_list),dtype=object)
        node_degree_events = pd.Series(dict(degree_events_list),dtype=object)
        return node_all_events,node_degree_events
    def get_degree_events(self):
        node_event_dict = {}
        for node in self.g:
            events = []
            for nbr in self.g[node]:
                stamps = np.array(self.g.edges[node,nbr]['time_stamp'])
                stamps = (stamps - self.begin)/self.norm
                events.append(stamps)
            node_event_dict[node] = events
        return node_event_dict
    "三种方式的强度拟合函数，经过实验证明是最简单的霍克斯过程最好"
    def hks_pre(self,events_sr,T):
        hks_path = "D:\\OneDrive/OneDrive - 西南大学/TPL/link_pre2.0/hks_paras/"+self.name+'.csv'
        activity_path = "D:\\OneDrive/OneDrive - 西南大学/TPL/link_pre2.0/activity/"+self.name+'.csv'
        hks_para = []
        node_pre_score = pd.Series(0,index=events_sr.index,dtype=float)
        if os.path.exists(hks_path):
            hks_para = pd.read_csv(hks_path,index_col=[0])
            # node_pre_score = pd.read_csv(activity_path,index_col=[0])
            # node_pre_score = node_pre_score.iloc[:,0]
            replace_para = hks_para.median(axis=0)/2
            for node in events_sr.index:
                events = events_sr[node]
                para = hks_para.loc[node][:]
                if len(events) <= 2: # 大于2的情况才进行拟合
                    pre_value = em_hawkes.cdf(replace_para,events,T) - em_hawkes.cdf(replace_para,events,100)
                    node_pre_score[node] = pre_value 
                else:
                    node_pre_score[node] = (em_hawkes.cdf(para,events,T) - em_hawkes.cdf(para,events,100))
        else:
            hks_para = pd.DataFrame(index=events_sr.index,columns=['mu','alpha','theta'])
            for node in tqdm(events_sr.index,desc='hks estimating...'):
                events = events_sr[node]
                if len(events) > 2: # 大于2的情况才进行拟合
                    para = em_hawkes.em_fit(events,100) # 拟合霍克斯过程的参数
                    if para[1] > 4: # bad fitted
                        para = hks_para.median(axis=0) #用目前的均值代替
                    hks_para.loc[node][:] = para
                    pre_value = em_hawkes.cdf(para,events,T) - em_hawkes.cdf(para,events,100)
                    node_pre_score[node] = pre_value

            # 先将数目能够进行拟合的分析完再来算
            replace_para = hks_para.median(axis=0)/2 #赋予无法拟合的节点相对较小的值
            for node in events_sr.index:
                events = events_sr[node]
                if len(events) <= 2: # 大于2的情况才进行拟合
                    pre_value = em_hawkes.cdf(replace_para,events,T) - em_hawkes.cdf(replace_para,events,100)
                    node_pre_score[node] = pre_value 
            hks_para.to_csv(hks_path)
            node_pre_score.to_csv(activity_path)
        node_pre_score = node_pre_score/max(node_pre_score)
        return hks_para,node_pre_score
    def hks_pre2(self,events_sr,T): #计算新边的到达速率
        hks_para = pd.DataFrame(index=events_sr.index,columns=['mu','alpha','theta'])
        node_pre_score = pd.Series(0,index=events_sr.index,dtype=float)
        for node in tqdm(events_sr.index,desc='hks estimating...'):
            events = events_sr[node]
            if len(events) > 2: # 大于2的情况才进行拟合
                para = em_hawkes.em_fit(events,100) # 拟合霍克斯过程的参数
                if para[1] > 2: # bad fitted
                    para = hks_para.median(axis=0) #用目前的均值代替
                hks_para.loc[node][:] = para
                pre_value = em_hawkes.cdf(para,events,T) - em_hawkes.cdf(para,events,100)
                node_pre_score[node] = pre_value
            replace_para = hks_para.median(axis=0)
            for node in events_sr.index:
                events = events_sr[node]
                if len(events) <= 2:
                    pre_value = em_hawkes.cdf(replace_para,events,T) - em_hawkes.cdf(replace_para,events,100)
                    node_pre_score[node] = pre_value 
        return node_pre_score   
    def tick_hks_pre(self,events_sr,T): # 多重霍克斯过程
        decays = [0.5, 2, 6]
        learner = HawkesSumExpKern(decays=decays,max_iter=300,penalty='l2')
        node_pre = []
        for node in events_sr.index:
            events = events_sr[node]
            learner.fit([events])
            # para = learner.adjacency[0][0]
            # node_pre.append(em_hawkes.cdf(para,events,T) - em_hawkes.cdf(para,events,100))
            ins = learner.estimated_intensity([events],end_time=T,intensity_track_step=1)
            node_pre.append(np.sum(ins[0][0][-30:]))
        tick_node_activity = pd.Series(node_pre,index=events_sr.index,dtype=float)
        return tick_node_activity

    def rpp_pre(self,events_sr,T):
        rpp_pre = pd.DataFrame(index=events_sr.index,columns=['a','b','c'])
        node_pre_score = pd.Series(0,index=events_sr.index,dtype=float)
        for node in tqdm(events_sr.index,desc='rpp_model estmating...'):
            events = events_sr[node]
            x = []
            y = []
            count = 0
            for point in events:
                if not point in x:
                    count += 1
                    x.append(point)
                    y.append(count) 
                count +=1
            y.append(y[-1]) # 添加最后一个点 来辅助预判
            x.append(100)
            x = (x - min(x)) + 1
            if len(x) > 2: # 大于等于3才能拟合
                para,code = rpp.wds_model_fit(x,y)  # rpp 模型预测
                rpp_pre.loc[node][:] = para
                node_pre_score[node] = rpp.fun(T-events[0],para) - rpp.fun(100-events[0],para)
        # 不能拟合的情况用平均值代替       
        replace_para = rpp_pre.mean(axis=0)
        for node in events_sr.index:
            events = events_sr[node]
            x = []
            y = []
            count = 0
            for point in events:
                if not point in x:
                    count += 1
                    x.append(point)
                    y.append(count) 
                count +=1
            y.append(y[-1]) # 添加最后一个点 来辅助预判
            x.append(100)
            x = (x - min(x)) + 1
            if len(x) < 3: # 大于等于3才能拟合
                node_pre_score[node] = rpp.fun(T-events[0],replace_para) - rpp.fun(100-events[0],replace_para)
        return rpp_pre,node_pre_score
    "基于强度衰减网络的忠诚度计算方法"
    def get_loyalty(self,T):
        replace_para = self.hks_para.median(axis=0)
        loylty_values = pd.Series(0,index=self.degree_events.keys(),dtype=float)
        loyalty_changes = []
        for node in self.degree_events.keys():
            events = self.degree_events[node]
            para = self.hks_para.loc[node][:]
            cos_simi = []
            if  para[0] == np.NAN: # 拟合不好或者是点数不够的情况
                para = replace_para
            nbrs = len(events) #邻居数量
            loylty_trans = []
            flag = False
            for i in range(1,SAMPING_NUM+1,1):
                loc = 0
                tem_vec = np.zeros(nbrs)
                for stamps in events: #各个邻居
                    temp_events = stamps[stamps<=i]
                    if len(temp_events) > 0: #已经建立了联系的情况
                        flag = True
                        tem_vec[loc] = em_hawkes.lambd(para,temp_events,i) #当前的强度
                    loc += 1
                if flag:
                    loylty_trans.append(tem_vec)
            if len(loylty_trans) > 1:
                for j in range(len(loylty_trans)-1):
                    cos = cosine_similarity(loylty_trans[j],loylty_trans[len(loylty_trans)-1])
                    if not np.isnan(cos):
                        cos_simi.append(cos)
                    else:
                        cos_simi.append(1)
            else:
                cos_simi.append(1)
            x = list(np.arange(0,len(cos_simi),1))
            loyalty_changes.append(cos_simi)
            para,code = model_fit(x,cos_simi)
            pre = 0
            if code == 1 or code== 5: # 拟合得还可以
                pre = fun(T,para) - fun(100,para)
            if pre < 0:
                pre = 0        
            loylty_values[node] = pre
        loylty_values = loylty_values/max(loylty_values)
        # loylty_values = loylty_values*0.1
        loylty_values = 1 - loylty_values
        return loylty_values,loyalty_changes
    def get_entropy_loyalty(self):
        loyalty_path = "D:\\OneDrive/OneDrive - 西南大学/TPL/link_pre2.0/loyalty_save/"+self.name+".csv"
        node_event_str = dict(zip(self.node_degree_events.index,[[]for i in range(len(self.g))]))
        entropy_loyalty = pd.Series(0,index=self.node_degree_events.index,dtype=float)
        if os.path.exists(loyalty_path):
            entropy_loyalty = pd.read_csv(loyalty_path,index_col=[0])
            entropy_loyalty = entropy_loyalty.iloc[:,0]
        else:
            for line in open(self.network_file, 'r',encoding='utf-8'):  # 读取CSV网络文件
                str_list = line.split()
                n1 = int(str_list[0])  # 节点1
                n2 = int(str_list[1])  # 节点2
                node_event_str[n1].append(n2)
                node_event_str[n2].append(n1)
            for node in tqdm(self.g):
                sequnce = node_event_str[node]
                sequnce = np.array(sequnce)
                n = len(sequnce)
                N = len(self.g[node])
                entropy = 0
                if N==1:
                    entropy_loyalty[node] = 0
                else:
                    for i in range(1,n,1):
                        hi = i+1
                        pattern = sequnce[i:hi]
                        small = 1
                        while hi <= n and hi <= (i+1)*2:
                            if list_search(pattern,sequnce[0:i+1]):
                                small += 1
                                hi+=1
                            else:
                                break
                        entropy += small/n
                    entropy = (np.log(n)/entropy)
                    entropy_loyalty[node] = entropy
            entropy_loyalty = 1-entropy_loyalty/max(entropy_loyalty)
            entropy_loyalty.to_csv(loyalty_path)    #存储
            print("--saving the loyalty value done--")
        return entropy_loyalty,node_event_str
    def get_mobility_loyalty(self,T): # 基于人类活动属性来预测的loyalty 未完成
        node_event_str = pd.Series('',index=self.node_degree_events.index,dtype=str)
        entropy_loyalty = pd.Series(0,index=self.node_degree_events.index,dtype=float)
        for line in open(self.network_file, 'r',encoding='utf-8'):  # 读取CSV网络文件
            str_list = line.split()
            n1 = int(str_list[0])  # 节点1
            n2 = int(str_list[1])  # 节点2
            node_event_str[n1]+=str(n2)+" "
            node_event_str[n2]+=str(n1)+" "
        for node in tqdm(node_event_str.index):
            event_str = node_event_str[node]
            n = len(event_str)
            entropy = 0
            for i in range(2,n*2,2):
                small = 1
                low = i-2
                while low >= 0:
                    pattern = event_str[low:i]
                    if re.match(pattern,event_str[:i]):
                        small += 1
                        low -=2
                    else:
                        break
                entropy += small/n
            entropy = np.log(n)/entropy
            entropy = entropy*0.93 # 真实熵比这个要小一些
            ramdom_entropy = np.log(len(self.g[node]))/np.log(2)
            entropy = entropy/ramdom_entropy
        entropy_loyalty = 1-entropy_loyalty
        return entropy_loyalty

                




class AM: # 活跃度零模型
    def __init__(self,node_activity):
        self.node_activity = node_activity
    def get_score(self,u,v):
        return self.node_activity[u]*self.node_activity[v]

class LM: # 忠诚性零模型
    def __init__(self,node_loyalty,g):
        self.node_loyalty = node_loyalty
        self.g = g
    def get_score(self,u,v):
        score = 0
        pu = 1/len(self.g[u])+(1-1/len(self.g[u]))*self.node_loyalty[u]
        pv = 1/len(self.g[v])+(1-1/len(self.g[v]))*self.node_loyalty[v]
        if self.g.has_edge(u,v):
            l1 = pu/len(self.g[u])
            l2 = pv/len(self.g[v])
            score = l1*l2
        else:
            l1 = (1-pu)/(len(self.g)-1-len(self.g[u]))
            l2 = (1-pv)/(len(self.g)-1-len(self.g[v]))
            score = l1*l2
        return score

class LM2: # 忠诚性零模型
    def __init__(self,node_loyalty,g):
        self.node_loyalty = node_loyalty
        self.g = g
    def get_score(self,u,v):
        score = 0
        pu = self.node_loyalty[u]
        pv = self.node_loyalty[v]
        if self.g.has_edge(u,v):
            score = pu*pv
            # score = 1
        else:
            score = (1-pu)*(1-pv)*(len(self.g[u])*len(self.g[v]))/((len(self.g)-1-len(self.g[u]))*(len(self.g)-1-len(self.g[v])))
            # score = 1
            
        return score



# class ALM3:
#     def __init__(self,AM,LM):
#         self.AM =AM
#         self.LM = LM    
#     def get_score(self,u,v):
#         return self.AM.get_score(u,v)*self.LM.get_score(u,v) #这个也许返回不是浮点数
class ALM3:
    def __init__(self,node_activity,node_loyalty,g):
        self.node_activity =node_activity
        self.node_loyalty = node_loyalty   
        self.g = g 
    def get_score(self,u,v):
        score = 0
        pu = self.node_loyalty[u]
        pv = self.node_loyalty[v]
        if self.g.has_edge(u,v):
            score = pu*pv*self.node_activity[u]*self.node_activity[v]
            # score = 1
        else:
            score = self.node_activity[u]*self.node_activity[v]*(1-pu)*(1-pv)*(len(self.g[u])*len(self.g[v]))/((len(self.g)-1-len(self.g[u]))*(len(self.g)-1-len(self.g[v])))
            # score = 1
            
        return score

class mult_info: # 互信息的方法，已经放弃
    def __init__(self,node_loyalty,g):
        self.node_loyalty = node_loyalty
        self.g = g
    def get_score(self,u,v):
        event_history = []
        N = len(self.g[u])
        for node in self.g[u]: # u的邻居
            for t in self.g.edges[u,node]['time_stamp']:
                event_history.append([node,t])
        for node in self.g[v]:
            if node != u:
                N += 1
                for t in self.g.edges[v,node]['time_stamp']:
                    event_history.append([node,t])
        event_history.sort(key= lambda x:x[1])
        event_history = np.array(event_history)
        mut_sequnce = event_history[:,0]
        n = len(mut_sequnce)
        entropy = 0
        if N==1:
           pass
        else:
            for i in range(1,n,1):
                hi = i+1
                pattern = mut_sequnce[i:hi]
                small = 1
                while hi <= n and hi <= (i+1)*2:
                    if list_search(pattern,mut_sequnce[0:i+1]):
                        small += 1
                        hi+=1
                    else:
                        break
                entropy += small/n
            entropy = (np.log(n)/entropy)
        return entropy-self.node_loyalty[u]-self.node_loyalty[v]