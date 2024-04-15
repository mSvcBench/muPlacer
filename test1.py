from offload import offload
from offload2 import offload2
from offload import offload
from offload2 import offload2
from offload3 import offload3

from unoffload import unoffload
from unoffload2 import unoffload2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

for k in range(50):
    print(f'test {k}')
    RTT = 0.0869
    min_delay_delta = 0.03
    app_edge = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    Rs = [1345.95598738420, 1335.07762879323, 1388.14116002795,1343.13896648045, 1354.06387665198, 1364.47552447552, 1315.36496350365, 1374.55390334572, 1440.83333333333]
    lambda_val = 40.1576
    M = 10
    Ne = 1e9

    Rcpu = [1, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0, 1, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0]
    Rmem = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Fcm = np.zeros((M,M))
    for i in range(1,M-1):
        a = np.random.randint(i)
        Fcm[a,i]=np.random.uniform(0,0.2)
    Fcm[M-1,0] = 1

    # G = nx.DiGraph(Fcm)
    # nx.draw_planar(G,with_labels=True)
    # plt.show()

    # Fcm = np.array([[0,0.2130,0.2170,0.2170,0.2060,0,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,0,0.2030,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0.2160,0],
    #                 [0,0,0,0,0,0,0.2020,0.1980,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [1,0,0,0,0,0,0,0,0,0]])


    #result1 = offload(Rcpu, Rmem, Fcm, M, lambda_val, Rs, app_edge, min_delay_delta, RTT, Ne)
    #result1=result1[M:]
    best_S_edge2, best_cost2, best_delta2 = offload2(Rcpu, Rmem, Fcm, M, lambda_val, Rs, app_edge, min_delay_delta, RTT, Ne)
    best_S_edge3, best_cost3, best_delta3, best_delta_cost3 = offload3(Rcpu, Rmem, Fcm, M, lambda_val, Rs, app_edge, min_delay_delta, RTT, Ne)
    #print(result1)

    # if (np.array_equal(best_S_edge1,best_S_edge2)==False):
    #     print("Result Mismatch 1 2")
    if (np.array_equal(best_S_edge2,best_S_edge3)==False):
        print(best_S_edge2)
        print(best_S_edge3)
        print(f"Result Mismatch 2 3, cost2 {best_cost2}, cost3 {best_cost3}, delta2 = {best_delta2}, , delta3 = {best_delta3}")

    

for k in range(200):
    print(f'test {k}')
    RTT = 0.0869
    min_delay_delta = -0.02
    app_edge = [1, 1, 1, 1, 1, 0, 0, 1, 0]
    Rs = [1345.95598738420, 1335.07762879323, 1388.14116002795,1343.13896648045, 1354.06387665198, 1364.47552447552, 1315.36496350365, 1374.55390334572, 1440.83333333333]
    lambda_val = 40.1576
    M = 10
    Ne = 1e9

    Rcpu = [1, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0, 1, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0]
    Rmem = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Fcm = np.zeros((M,M))
    for i in range(1,M-1):
        a = np.random.randint(i)
        Fcm[a,i]=np.random.uniform(0,0.2)
    Fcm[M-1,0] = 1

    # G = nx.DiGraph(Fcm)
    # nx.draw_planar(G,with_labels=True)
    # plt.show()

    # Fcm = np.array([[0,0.2130,0.2170,0.2170,0.2060,0,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,0,0.2030,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0.2160,0],
    #                 [0,0,0,0,0,0,0.2020,0.1980,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0,0,0],
    #                 [1,0,0,0,0,0,0,0,0,0]])


    #result1 = offload(Rcpu, Rmem, Fcm, M, lambda_val, Rs, app_edge, min_delay_delta, RTT, Ne)
    #result1=result1[M:]
    
    best_S_edge3, best_cost3, best_delta3, best_delta_cost3 = offload3(Rcpu, Rmem, Fcm, M, lambda_val, Rs, app_edge, min_delay_delta, RTT, Ne)
    print(best_S_edge3)
    print(f"cost3 {best_cost3}, delta3 = {best_delta3}, best_delta_cost3 = {best_delta_cost3}")
