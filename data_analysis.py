import scipy.io as scio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd


def input_mat(name,doc):
    data = scio.loadmat(str(name)+".mat")
    return np.array(data[name]).T


doc = "data1"
print(input_mat("netsa",doc))
weight_list = []
s_point_name = []
a_point_name = []


for i in range(len(input_mat("netsa",doc)[0])):
    a_point_name.append(chr(ord("A")+i))
for i in range(len(input_mat("netsa",doc))):
    s_point_name.append(str(i))


G = nx.Graph()
temp = input_mat("anchor",doc)
sa = input_mat("netsa",doc)
ss = input_mat("netss",doc)
a_point = []
color_map = ["red" for _ in range(len(input_mat("netsa",doc)[0]))]
print(color_map)
for i in range(len(temp)):
    a_point.append(tuple(temp[i]))
print(a_point)
for i in range(len(a_point)):
    G.add_node(a_point_name[i]) # 设置锚点

for i in range(len(sa)):
    for j in range(len(sa[i])):
        if sa[i][j] != 0:
            print(s_point_name[i],a_point_name[j])
            G.add_edge(s_point_name[i],a_point_name[j],weight = sa[i][j]) # 设置sa矩阵
            weight_list.append(sa[i][j])
    color_map.append("blue")
#nx.draw(G, node_color=color_map, with_labels=True)
#plt.show()

for i in range(len(ss)):
    for j in range(len(ss[i])):
        if ss[i][j] != 0 and not G.has_edge(s_point_name[i],s_point_name[j]): # 设置ss矩阵
            G.add_edge(s_point_name[i],s_point_name[j],weight = ss[i][j])
            weight_list.append(ss[i][j])
#nx.draw(G, node_color=color_map, with_labels=True)
#plt.show()

print(G.nodes())
jump_point = nx.shortest_path_length(G)
print(dict(jump_point))
short_path = dict(nx.shortest_path_length(G))  # 最小跳数搜路
k = 0
mean_path = 0
for i in sa:
    for j in i:
        if j != 0:
            k += 1
            mean_path += j
mean_path = mean_path / k
print("meanpath "+str(mean_path))

new_sa = []
for i in range(len(sa)):
    temp = []
    for j in range(len(sa[i])):
        temp.append(short_path[a_point_name[j]][s_point_name[i]]*mean_path)
    new_sa.append(temp)

new_sa = np.array(new_sa)
print(new_sa)
print(sa)

###################接下来用三边测距法##################

a_position = input_mat("anchor",doc)
ans_list = []
epsilon1 = 0.05
epsilon2 = 0.001
k = 0
error_list = []
a = input_mat("netsa",doc)
b = input_mat("netss",doc)
u = 0
s = 0
for i in a:
    for j in i:
        if j != 0:
            s += j
            u += 1
s = s/u
adv_dis = s
for i in range(len(new_sa)):
    #############优化初值条件##############
    s = 0
    u = 0
    aa = 0
    for j in sa[i]:
        if j != 0:
            s += j
            u += 1
    if u != 0:
        s = s/u
    if s<adv_dis/2:
        predict_x = 0
        predict_y = 0
    else:
        predict_x = 0
        predict_y = 0
    k += 1
    error = 99999
    echo = 0
    error_picture = []
    while True:
        r_a_hat = []
        for j in range(len(sa[i])):
            r_a_hat.append(((predict_x - a_position[j][0]) ** 2 + (predict_y - a_position[j][1]) ** 2) ** (1 / 2))
        hp = []
        for j in range(len(sa[i])):
            temp = []
            temp.append(-((a_position[j][0]) - predict_x) / r_a_hat[j])
            temp.append(-((a_position[j][1]) - predict_y) / r_a_hat[j])
            hp.append(temp)
        hp = np.array(hp)
        r_to_list = []
        for j in range(len(new_sa[i])):
            r_to_list.append(new_sa[i][j]-r_a_hat[j])
        r_to_list = np.array(r_to_list)
        xy_list = np.dot(np.dot(np.linalg.inv(np.dot(hp.T,hp)),hp.T),r_to_list)
        h = 0
        dis = 0
        for j in range(len(sa[0])):
            if sa[i][j] != 0:
                dis += abs(((xy_list[0]-a_position[j][0])**2+(xy_list[1]-a_position[j][1])**2)**0.5 - sa[i][j]) ** 2
                h += 1
        if h != 0:
            dis = dis / h
        else:
            error = None
            predict_x = xy_list[0]
            predict_y = xy_list[1]
            ans_list.append([predict_x, predict_y])
            error_list.append(error)
            print(str(k) + " points " + str(echo) + " echos " + "error:" + str(error))
            break
        error_picture.append(dis)
        if abs(dis) < epsilon1 or abs(dis-error)<epsilon2:
            predict_x = xy_list[0]
            predict_y = xy_list[1]
            error = dis
            ans_list.append([predict_x,predict_y])
            print(str(k) + " points " +str(echo) + " echos " + "error:" + str(error))
            error_list.append(error)
            break
        elif echo > 3000:
            predict_x = xy_list[0]
            predict_y = xy_list[1]
            ans_list.append([predict_x, predict_y])
            error_list.append(error)
            print(str(k) + " points "+str(echo) + " echos " + "error:" + str(error)+" max echos!")
            break
        else:
            predict_x = xy_list[0]
            predict_y = xy_list[1]
            error = dis
        echo += 1
    #plt.plot(error_picture)
    #plt.show()
print(ans_list)
print(len(ans_list))

################ 误差分析 ############



a = pd.DataFrame(error_list)
print(a.describe())
plt.plot(a)

################可视化###############


import scipy.io as scio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd


def input_mat(name,doc):
    data = scio.loadmat(str(name)+".mat")
    return np.array(data[name]).T


doc = "data1"
print(input_mat("netsa",doc))
weight_list = []
s_point_name = []
a_point_name = []


for i in range(len(input_mat("netsa",doc)[0])):
    a_point_name.append(chr(ord("A")+i))
for i in range(len(input_mat("netsa",doc))):
    s_point_name.append(str(i))


G = nx.Graph()
temp = input_mat("anchor",doc)
sa = input_mat("netsa",doc)
ss = input_mat("netss",doc)
a_point = []
color_map = ["red" for _ in range(len(input_mat("netsa",doc)[0]))]
print(color_map)
for i in range(len(temp)):
    a_point.append(tuple(temp[i]))
print(a_point)
for i in range(len(a_point)):
    G.add_node(a_point_name[i]) # 设置锚点

for i in range(len(sa)):
    for j in range(len(sa[i])):
        if sa[i][j] != 0:
            print(s_point_name[i],a_point_name[j])
            G.add_edge(s_point_name[i],a_point_name[j],weight = sa[i][j]) # 设置sa矩阵
            weight_list.append(sa[i][j])
    color_map.append("blue")
#nx.draw(G, node_color=color_map, with_labels=True)
#plt.show()

for i in range(len(ss)):
    for j in range(len(ss[i])):
        if ss[i][j] != 0 and not G.has_edge(s_point_name[i],s_point_name[j]): # 设置ss矩阵
            G.add_edge(s_point_name[i],s_point_name[j],weight = ss[i][j])
            weight_list.append(ss[i][j])
#nx.draw(G, node_color=color_map, with_labels=True)
#plt.show()

print(G.nodes())
jump_point = nx.shortest_path_length(G)
print(dict(jump_point))
short_path = dict(nx.shortest_path_length(G))  # 最小跳数搜路
k = 0
mean_path = 0
for i in sa:
    for j in i:
        if j != 0:
            k += 1
            mean_path += j
mean_path = mean_path / k
print("meanpath "+str(mean_path))

new_sa = []
for i in range(len(sa)):
    temp = []
    for j in range(len(sa[i])):
        temp.append(short_path[a_point_name[j]][s_point_name[i]]*mean_path)
    new_sa.append(temp)

new_sa = np.array(new_sa)
print(new_sa)
print(sa)

###################接下来用三边测距法##################

a_position = input_mat("anchor",doc)
ans_list = []
epsilon1 = 0.05
epsilon2 = 0.001
k = 0
error_list = []
a = input_mat("netsa",doc)
b = input_mat("netss",doc)
u = 0
s = 0
for i in a:
    for j in i:
        if j != 0:
            s += j
            u += 1
s = s/u
adv_dis = s
for i in range(len(new_sa)):
    #############优化初值条件##############
    s = 0
    u = 0
    aa = 0
    bb = -1
    for j in sa[i]:
        if j != 0:
            s += j
            u += 1
    if u != 0:
        s = s/u
    if u == 1 and s < adv_dis/1.25:
        for o in range(len(sa[i])):
            if sa[i][o] != 0:
                bb = sa[i][o]
                aa = o
        if bb != -1:
            predict_x = a_position[aa][0]+random.normalvariate(bb, 0.01)
            predict_y = a_position[aa][1]-random.normalvariate(bb, 0.01)
    else:
        predict_x = 0
        predict_y = 0
    k += 1
    error = 99999
    echo = 0
    error_picture = []
    while True:
        r_a_hat = []
        for j in range(len(sa[i])):
            r_a_hat.append(((predict_x - a_position[j][0]) ** 2 + (predict_y - a_position[j][1]) ** 2) ** (1 / 2))
        hp = []
        for j in range(len(sa[i])):
            temp = []
            temp.append(-((a_position[j][0]) - predict_x) / r_a_hat[j])
            temp.append(-((a_position[j][1]) - predict_y) / r_a_hat[j])
            hp.append(temp)
        hp = np.array(hp)
        r_to_list = []
        for j in range(len(new_sa[i])):
            r_to_list.append(new_sa[i][j]-r_a_hat[j])
        r_to_list = np.array(r_to_list)
        xy_list = np.dot(np.dot(np.linalg.inv(np.dot(hp.T,hp)),hp.T),r_to_list)
        h = 0
        dis = 0
        for j in range(len(sa[0])):
            if sa[i][j] != 0:
                dis += abs(((xy_list[0]-a_position[j][0])**2+(xy_list[1]-a_position[j][1])**2)**0.5 - sa[i][j]) ** 2
                h += 1
        if h != 0:
            dis = dis / h
        else:
            error = None
            predict_x = xy_list[0]
            predict_y = xy_list[1]
            ans_list.append([predict_x, predict_y])
            error_list.append(error)
            print(str(k) + " points " + str(echo) + " echos " + "error:" + str(error))
            break
        error_picture.append(dis)
        if abs(dis) < epsilon1 or abs(dis-error)<epsilon2:
            predict_x = xy_list[0]
            predict_y = xy_list[1]
            error = dis
            ans_list.append([predict_x,predict_y])
            print(str(k) + " points " +str(echo) + " echos " + "error:" + str(error))
            error_list.append(error)
            break
        elif echo > 3000:
            predict_x = xy_list[0]
            predict_y = xy_list[1]
            ans_list.append([predict_x, predict_y])
            error_list.append(error)
            print(str(k) + " points "+str(echo) + " echos " + "error:" + str(error)+" max echos!")
            break
        else:
            predict_x = xy_list[0]
            predict_y = xy_list[1]
            error = dis
        echo += 1
    #plt.plot(error_picture)
    #plt.show()
print(ans_list)
print(len(ans_list))

################ 误差分析 ############



a = pd.DataFrame(error_list)
print(a.describe())
plt.plot(a)
plt.legend(["before","after"])
plt.show()
################可视化###############
