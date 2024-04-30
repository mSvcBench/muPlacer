from offload2 import offload
from mfu_heuristic import mfu_heuristic
from IA_heuristic import IA_heuristic
# from unoffload import unoffload
# from unoffload2 import unoffload2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from igraph import *
from computeNc import computeNc
from scipy.io import savemat
import time

def edges_reversal(graph):
    for edge in graph.get_edgelist():
        graph.delete_edges([(edge[0], edge[1])])
        graph.add_edges([(edge[1], edge[0])])

np.random.seed(150273)
res=np.array([])
trials = 30
RTT = 0.0869    # RTT edge-cloud
M = 30 # n. microservices
delta_req = 0.03    # requested delay increase
lambda_val = 20     # request per second
Ne = 1e9    # bitrate cloud-edge
S_edge_b = np.zeros(M)  # initial state. 
S_edge_b[M-1] = 1 # Last value is the user must be set equal to one

graph_algorithm = 'random'

barabasi=dict()
barabasi['n'] = M-1
barabasi['m'] = 2
barabasi['power'] = 0.9
barabasi['zero_appeal'] = 0.9

random=dict()
random['n_parents'] = 3

Fcm_range_min = 0.1 # min value of microservice call frequency 
Fcm_range_max = 0.5 # max value of microservice call frequency 
Rcpu_quota = 0.5    # CPU quota
Rcpu_range_min = 1  # min value of requested CPU quota per instance-set
Rcpu_range_max = 32 # max value of requested CPU quota per instance-set
Rs_range_min = 1000 # min value of response size in bytes
Rs_range_max = 50000   # max of response size in bytes

max_algotithms = 10
best_cost_v = np.empty((1,10)) # vector of costs obtained by different algorithms 
best_delta_v = np.empty((1,10)) # vector of delta obtained by different algorithms  
n_rounds_v = np.empty((1,10)) # vector of rounds obtained by different algorithms
p_time_v = np.empty((1,10)) # vector of processing time obtained by different algorithms

show_graph = False
show_plot = True

for k in range(trials):
    print(f'\n\ntrial {k}')  
    Rs = np.random.randint(Rs_range_min,Rs_range_max,M)  # random response size bytes
    Rs[M-1]=0 # user has no response size
    
    # build dependency graph
    Fcm = np.zeros([M,M])   # microservice call frequency matrix
    if graph_algorithm == 'barabasi':
        g = Graph.Barabasi(n=barabasi['n'],m=barabasi['m'],power=barabasi['power'],zero_appeal=barabasi['zero_appeal'], directed=True)
        edges_reversal(g)
        Fcm[:M-1,:M-1] = np.matrix(g.get_adjacency()) 
    elif graph_algorithm == 'random':
        n_parents = 3
        for i in range(1,M-1):
            n_parent=np.random.randint(1,random['n_parents'])
            for j in range(n_parent):
                a = np.random.randint(i)
                Fcm[a,i]=1
        
    # set random values for microservice call frequency matrix
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fcm[i,j]=np.random.uniform(0.1,0.5) if Fcm[i,j]>0 else 0
    Fcm[M-1,0] = 1  # user call microservice 0 (the ingress microservice)

    # set random values for CPU and memory requests
    Rcpu = (np.random.randint(32,size=M)+1) * Rcpu_quota
    Rcpu[M-1]=0   # user has no CPU request
    Rcpu = np.append(Rcpu, Rcpu)
    Rmem = np.zeros(2*M)
    Rcpu[M:] = Rcpu[M:] * S_edge_b # set to zero the CPU requests of the instances not at the edge
    Rmem[M:] = Rmem[M:] * S_edge_b # set to zero the memory requests of the instances not at the edge

    if show_graph:
        G = nx.DiGraph(Fcm)
        nx.draw(G,with_labels=True)
        plt.show()

    
    best_cost_row = np.zeros((1,max_algotithms)) # vector of costs obtained by different algorithms in a trial
    best_delta_row = np.zeros((1,max_algotithms)) # vector of delta obtained by different algorithms in a trial  
    n_rounds_row = np.zeros((1,max_algotithms)) # vector of rounds obtained by different algorithms in a trial
    p_time_row = np.zeros((1,max_algotithms)) # vector of processing time obtained by different algorithms in a trial
    alg_type = [""] * max_algotithms # vector of strings describing algorithms used in a trial
    a=-1
    
    ## E_PAMP ##
    a+=1
    alg_type[a] = "E_PAMP no upgrade limit"
    best_cost = -1
    u_limit = M
    tic = time.time()
    best_S_edge, best_cost, best_delta, best_delta_cost, n_rounds = offload(Rcpu.copy(), Rmem.copy(), Fcm, M, lambda_val, Rs, S_edge_b.copy(), delta_req, RTT, Ne, u_limit)
    toc = time.time()
    print(f'processing time E-PAMP {(toc-tic)} sec')
    print(f"Result {alg_type[a]} for offload:\n {np.argwhere(best_S_edge==1).squeeze()}, Cost: {best_cost}, delta_delay: = {best_delta}, delta_cost: = {best_delta_cost}, rounds: = {n_rounds}")
    best_cost_row[0,a] = best_cost
    best_delta_row[0,a] = best_delta
    n_rounds_row[0,a] = n_rounds
    
    a+=1
    alg_type[a] = "E_PAMP upgrade limit 2"
    best_cost = -1
    tic = time.time()
    best_S_edge, best_cost, best_delta, best_delta_cost, n_rounds = offload(Rcpu.copy(), Rmem.copy(), Fcm, M, lambda_val, Rs, S_edge_b.copy(), delta_req, RTT, Ne, u_limit)
    toc = time.time()
    print(f'processing time E-PAMP {(toc-tic)} sec')
    print(f"Result {alg_type[a]} for offload:\n {np.argwhere(best_S_edge==1).squeeze()}, Cost: {best_cost}, delta_delay: = {best_delta}, delta_cost: = {best_delta_cost}, rounds: = {n_rounds}")
    best_cost_row[0,a] = best_cost
    best_delta_row[0,a] = best_delta
    n_rounds_row[0,a] = n_rounds
    
    
    # # MFU ##
    a+=1
    alg_type[a] = "MFU"
    best_cost = -1
    tic = time.time()
    best_S_edge, best_cost, best_delta, best_delta_cost, n_rounds = mfu_heuristic(Rcpu.copy(), Rmem.copy(), Fcm, M, lambda_val, Rs, S_edge_b.copy(), delta_req, RTT, Ne)
    toc = time.time()
    print(f'processing time E-PAMP {(toc-tic)} sec')
    print(f"Result {alg_type[a]} for offload:\n {np.argwhere(best_S_edge==1).squeeze()}, Cost: {best_cost}, delta_delay: = {best_delta}, delta_cost: = {best_delta_cost}, rounds: = {n_rounds}")
    best_cost_row[0,a] = best_cost
    best_delta_row[0,a] = best_delta
    n_rounds_row[0,a] = n_rounds
    
    ## IA ##
    a+=1
    alg_type[a] = "IA"
    best_cost = -1
    tic = time.time()
    best_S_edge, best_cost, best_delta, best_delta_cost, n_rounds = IA_heuristic(Rcpu.copy(), Rmem.copy(), Fcm, M, lambda_val, Rs, S_edge_b.copy(), delta_req, RTT, Ne)
    toc = time.time()
    print(f'processing time E-PAMP {(toc-tic)} sec')
    print(f"Result {alg_type[a]} for offload:\n {np.argwhere(best_S_edge==1).squeeze()}, Cost: {best_cost}, delta_delay: = {best_delta}, delta_cost: = {best_delta_cost}, rounds: = {n_rounds}")
    best_cost_row[0,a] = best_cost
    best_delta_row[0,a] = best_delta
    n_rounds_row[0,a] = n_rounds

    best_cost_v = np.vstack((best_cost_v,best_cost_row))
    best_delta_v = np.vstack((best_delta_v,best_delta_row))
    n_rounds_v = np.vstack((n_rounds_v,n_rounds_row))
    p_time_v = np.vstack((p_time_v,p_time_row))
           
# Matlab save
mdic = {"best_cost_v": best_cost_v, "best_delta_v": best_delta_v, "n_rounds_v": n_rounds_v, "p_time_v": p_time_v}
savemat("res.mat", mdic)


if show_plot:
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', 'x', '+']
    for i in range(a+1):
        line, = plt.plot(best_cost_v[:,0], best_cost_v[:,i], label=alg_type[i], linestyle='none', marker=markers[i])
    plt.ylabel('cost')
    plt.xlabel(f'cost of {alg_type[i]}')
    plt.legend()
    plt.show()

