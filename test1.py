from offload_old1 import offload_old1
from offload_old2 import offload_old2
from offload2 import offload
from mfu_heuristic import mfu_heuristic
from IA_heuristic import IA_heuristic
from unoffload import unoffload
from unoffload2 import unoffload2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from igraph import *
from computeNcMat import computeNcMat
import matplotlib.pyplot as plt

def edges_reversal(graph):
    for edge in graph.get_edgelist():
        graph.delete_edges([(edge[0], edge[1])])
        graph.add_edges([(edge[1], edge[0])])

for k in range(100):
    print(f'\n\ntest {k}')  
    RTT = 0.0869
    M = 30
    delta_mes = 0.1
    app_edge = np.zeros(M-1)

    Rs = np.random.randint(1000,50000,M)
    Rs[-1]=0
    Rs = np.append(Rs, Rs)
    lambda_val = 20
    Ne = 1e9
    # Rcpu_quota = 0.25
    # Rcpu = (np.random.randint(32,size=M)+1) * Rcpu_quota
    # Rcpu[-1]=0
    # Rcpu = np.append(Rcpu, Rcpu)
    # Rcpu = np.array([1, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0, 1, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0])
    # Rmem = np.zeros(2*M)
    # Rcpu[M:2*M-1] = Rcpu[M:2*M-1] * app_edge
    # Rmem[M:2*M-1] = Rmem[M:2*M-1] * app_edge

    Fcm = np.zeros([M,M])
    # Barabasi
    # g = Graph.Barabasi(n=M-1, m=2, power=0.9,
    #                    zero_appeal=0.9, directed=True)
    # edges_reversal(g)
    # Fcm[:M-1,:M-1] = np.matrix(g.get_adjacency()) 

    # Random
    n_parents = 2
    for i in range(1,M-1):
        n_parent=np.random.randint(1,n_parents)
        for j in range(n_parent):
            a = np.random.randint(i)
            Fcm[a,i]=np.random.uniform(0,1)
    
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fcm[i,j]=np.random.uniform(0.1,0.3) if Fcm[i,j]>0 else 0
    
    Fcm[M-1,0] = 1
    Nc = computeNcMat(Fcm, M, 1)
    Rcpu_quota = 0.5
    Rcpu = (np.random.randint(32,size=M)+1) * Rcpu_quota
    # Rcpu = Nc/max(Nc) * 16 
    Rcpu[-1]=0
    Rcpu = np.append(Rcpu, Rcpu)
    Rmem = np.zeros(2*M)
    Rcpu[M:2*M-1] = Rcpu[M:2*M-1] * app_edge
    Rmem[M:2*M-1] = Rmem[M:2*M-1] * app_edge

    # G = nx.DiGraph(Fcm)
    # nx.draw(G,with_labels=True)
    # plt.show()

    ## E_PAMP ##
    best_S_edge, best_cost, best_delta, best_delta_cost, n_rounds = offload(Rcpu.copy(), Rmem.copy(), Fcm, M, lambda_val, Rs, app_edge.copy(), delta_mes, RTT, Ne)
    #print(f"Result E_PAMP in offload:\n {best_S_edge},\n CPU_cost: {best_cost}, delta_delay: = {best_delta}, delta_cost: = {best_delta_cost}")
    
    ## MFU ##
    best_S_edge2, best_cost2, best_delta2, best_delta_cost2, n_rounds2 = mfu_heuristic(Rcpu.tolist(), Rmem.tolist(), Fcm, M, lambda_val, Rs, app_edge.tolist(), delta_mes, RTT, Ne)
    #print(f"Result MFU in offload:\n {best_S_edge2},\n CPU_cost: {best_cost2}, delta_delay: = {best_delta2}, delta_cost: = {best_delta_cost2}")

    ## IA ##
    best_S_edge3, best_cost3, best_delta3, best_delta_cost3, n_rounds3 = IA_heuristic(Rcpu.tolist(), Rmem.tolist(), Fcm, M, lambda_val, Rs, app_edge.tolist(), delta_mes, RTT, Ne)
    #print(f"Result IA in offload:\n {best_S_edge3},\n CPU_cost: {best_cost3}, delta_delay: = {best_delta3}, delta_cost: = {best_delta_cost3}")


    # best_S_edge2, best_cost2, best_delta2 = offload_old2(Rcpu, Rmem, Fcm, M, lambda_val, Rs, app_edge, min_delay_delta, RTT, Ne)
    # best_S_edge3, best_cost3, best_delta3, best_delta_cost3 = offload(Rcpu, Rmem, Fcm, M, lambda_val, Rs, app_edge, min_delay_delta, RTT, Ne)
    # #print(result1)

    # # if (np.array_equal(best_S_edge1,best_S_edge2)==False):
    # #     print("Result Mismatch 1 2")
    if  best_cost != best_cost2 or best_cost != best_cost3 or n_rounds != n_rounds2 or n_rounds != n_rounds3:
        #print(Rcpu[:M])
        ## E_PAMP ##
        print(f"Result E_PAMP in offload:\n {best_S_edge},\n CPU_cost: {best_cost}, delta_delay: = {best_delta}, delta_cost: = {best_delta_cost}, rounds: = {n_rounds}")
    
        ## MFU ##
        print(f"Result MFU in offload:\n {best_S_edge2},\n CPU_cost: {best_cost2}, delta_delay: = {best_delta2}, delta_cost: = {best_delta_cost2}, rounds: = {n_rounds2}")

        ## IA ##
        print(f"Result IA in offload:\n {best_S_edge3},\n CPU_cost: {best_cost3}, delta_delay: = {best_delta3}, delta_cost: = {best_delta_cost3}, rounds: = {n_rounds3}")



# for k in range(50):
#     print(f'test {k}')
#     RTT = 0.0869
#     delta_mes = -0.02
#     app_edge = [1, 1, 1, 1, 1, 0, 0, 1, 0]
#     Rs = [1345.95598738420, 1335.07762879323, 1388.14116002795,1343.13896648045, 1354.06387665198, 1364.47552447552, 1315.36496350365, 1374.55390334572, 1440.83333333333]
#     lambda_val = 40.1576
#     M = 10
#     Ne = 1e9

#     Rcpu = [1, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0, 1, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0]
#     Rmem = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     Fcm = np.zeros((M,M))
#     for i in range(1,M-1):
#         a = np.random.randint(i)
#         Fcm[a,i]=np.random.uniform(0,0.2)
#     Fcm[M-1,0] = 1

#     best_S_edge, best_cost, best_delta, best_delta_cost = offload(Rcpu, Rmem, Fcm, M, lambda_val, Rs, app_edge, delta_mes, RTT, Ne)
#     print(f"Result E_PAMP in unoffload: {best_S_edge}, cost: {best_cost}, delta: = {best_delta}")

#     best_S_edge, best_cost, best_delta, best_delta_cost = mfu_heuristic(Fcm, M, app_edge)
#     print(f"Result mfu in offload: {best_S_edge}, cost: {best_cost}, delta: = {best_delta}")

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

    # best_S_edge3, best_cost3, best_delta3, best_delta_cost3 = offload(Rcpu, Rmem, Fcm, M, lambda_val, Rs, app_edge, min_delay_delta, RTT, Ne)
    # print(best_S_edge3)
    # print(f"cost3 {best_cost3}, delta3 = {best_delta3}, best_delta_cost3 = {best_delta_cost3}")
