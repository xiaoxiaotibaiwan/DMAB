#  初始化网络以及基本分析
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time


def init(filename):  # [初始化网络] 返回的G包含时序网络的所有信息，时间戳信息以list形式存储在edge上 时间戳根据CSV文件都是有序的，这里没有处理
    begin = time.time()
    time_stamp_list = []  # 存放时间戳列表 由于CSV文件的都是按时间戳升序，判断是否重复只需要判断最后一位
    G = nx.Graph(networks_name=(filename.split('\\')[-1]).split('.')[0], time_stamp_list=time_stamp_list)  # 以图的形式存储网络
    for line in open(filename, 'r',encoding='utf-8'):  # 读取CSV网络文件
        str_list = line.split()
        n1 = int(str_list[0])  # 节点1
        n2 = int(str_list[1])  # 节点2
        n3 = int(str_list[2])  # 时间戳
        edge = (n1, n2)
        G.add_node(n1)
        G.add_node(n2)
        if len(time_stamp_list) == 0 or time_stamp_list[-1] != n3:  # 技巧 如果该if第二局在前第一次的时候会发生列表越界
            time_stamp_list.append(n3)
        # 以下方法将时间戳信息添加到edge上
        if G.has_edge(n1, n2):  # 两节点有边在此基础追加
            G.edges[n1, n2]['time_stamp'].append(n3)
        else:
            G.add_edges_from([edge], time_stamp=[n3])
    temp = list(G.nodes)
    for i in range(len(G.nodes)):  # 添加节点信息描述
        G.nodes[temp[i]]['state'] = []  # 节点的感染事件堆，里面存放二元组（感染时间，持续时间）
    # 全部初始化为易感节点
    del temp
    end = time.time()
    print((G.graph['networks_name'] + ' 网络生成完毕 time:' + str(int(end - begin)) + 's'))
    return G


# [网络的基础分析]
def network_analyze_basic(G):
    degree = nx.degree_histogram(G)
    # x = range(len(degree))
    # y = [z / float(sum(degree)) for z in degree]
    # plt.title("degree distribution of " + G.graph['networks_name'])
    # plt.loglog(x, y, '.')
    # plt.savefig('G:\\figs\\degree\\' + G.graph['networks_name'])
    # 网络平均度值
    mean = np.mean(degree)
    print('平均度值：' + str(mean))
    # 节点总数
    node_num = len(G.nodes)
    print('节点总数:' + str(node_num))
    # 边总数（静态）
    edge_num = len(G.edges)
    print('边总数:' + str(edge_num))
    # 链接总数
    all_link_num = 0
    for edge in G.edges.data():
        all_link_num += len(edge[2]['time_stamp'])
    print('链接总数：' + str(all_link_num))
    # 网络稀疏度
    sparsity = all_link_num / (edge_num * len(G.graph['time_stamp_list']))
    print('稀疏度:' + str(sparsity))


def network_describer(G, filename):
    """
    统计网络中的联系事件按时间的分布情况
    :type G: nx.Graph
    """
    time_stamp_list = []
    last_time = 1457266493
    count = 0
    for line in open(filename, encoding='UTF-8'):  # 读取CSV网络文件
        str_list = line.split()
        t = int(str_list[2])  # 时间戳
        time_stamp_list.append(t / last_time)
        count += 1
    plt.figure(figsize=(10, 5), dpi=300)
    plt.title('superuser')
    plt.hist(time_stamp_list, bins=100, facecolor="tomato", edgecolor="c", alpha=0.7)  # 用偏红的颜色代替感染规模
    plt.show()
    return
    
def network_describe(network):
    net_file = open(network, encoding='UTF-8')
    lines = net_file.readlines()
    begin = int(lines[0].split()[-1])
    end = int(lines[-1].split()[-1])
    events_num = len(lines)
    net_file.close()
    return begin,end,events_num

def network_nodes_remain_time(g):
    """

    :type g: nx.Graph
    """
    nodes_score_list = []
    for node in g:
        first = 0
        last = 0
        for nbr in g[node]:
            t1 = g.edges[node, nbr]['time_stamp'][0]
            t2 = g.edges[node, nbr]['time_stamp'][-1]
            if t1 < first or first == 0:
                first = t1
            if t2 > last or last == 0:
                last = t2
        nodes_score_list.append([node, last - first])
    np.savetxt("G:\\sentinel\\results\\facebook\\nodes_remain_time.csv", nodes_score_list, fmt="%d", delimiter=',')
    print("file saved!")


# G = init("networks\\original\\college.csv")
