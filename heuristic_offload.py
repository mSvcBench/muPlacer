import numpy as np
import networkx as nx
from S2id import S2id
from delayMat import delayMat 
from id2S import id2S


#   RTT : edge-cloud Round Trip Time 
#   Ne : edge-cloud bit rate
#   lambd : average user request frequency
#   Rs : response size in bytes of each istance-set
#   Fcm : call frequency matrix
#   Nc : average number of time the istance-set is involved per user request 
#   M : number of microservices
#   Rcpu_req : CPU seconds for internal functions
#   Ce, Me : CPU and Memory of the edge
#   Rcpu, Rmem : CPU and Mem requested by microservices
#   db = 0 if no db
#   e : number of datacenters
#   app_edge: microservices already in the edge cluster
#   min_delay_delta: minimum delay reduction

# _b : binary encoding of a set of nodes/services 
# _id : identifier of a set of nodes come out from binary encoding
# _n : identifiers of the nodes of a set

def heuristic_offload(Fcm, RTT, Rcpu_req, Rcpu, Rmem, Ce, Me, Ne, lambd, Rs, M, db, horizon, e, app_edge, min_delay_delta):
    # SEARCH ALL PATHS FROM USER TO INSTANCES
    G = nx.DiGraph(Fcm) # Create the graph of the mesh with probabilities
    user = G.number_of_nodes() # User is the last microservice istance-set (root in the graph)
    last_n = user # User is the root of the graph
    paths_n = [] # Define the variable for dependency paths


    # FIND ALL DEPENDENCY PATHS WITH ALL THEIR INSTANCES ONLY IN THE CLOUD CLUSTER ("valid" path)
    for s in range(1, last_n):
        # Find all paths in the graph
        paths = list(nx.all_simple_paths(G, source=user-1, target=s-1)) 
        # define a variable for "valid" path
        valid_path = True
        # Check if the path is "valid"
        for path in paths:
            app_edge_values = app_edge[path]
            # If all microservices in the path have app_edge_values == 1, this path is not "valid"
            if all(value == 1 for value in app_edge_values):
                valid_path = False
                break
        # Add the path in paths_n if it's "valid" 
        if valid_path:
            paths_n.extend(paths)
    

    # CREATE THE LIST OF POSSIBLE ID SUBGRAPHS
    subgraphs_id_origin = [S2id(app_edge)] # The origin subgraph is the actual running configuration 
    # Next cycle will increase it adding possible subgraphs
    for i in range(len(paths_n)):
        SG = paths_n[i]  # SG instances combination of i-th path
        subgraph_b = np.zeros(user)  # Inizialize the subgraph_b vector
        subgraph_b[SG] = 1  # Assign value 1 in subgraph_b vector facing subgraph_n
        subgraph_id = S2id(subgraph_b)  # Convert the subgraph in subgraph_id
        # Check if there is already the current subgraph_id in the list
        if subgraph_id not in subgraphs_id_origin:
            subgraphs_id_origin.append(subgraph_id)  # Add the current subgraph in the id list


    # GREEDY ADDITION OF NODES (SERVICE) OF SUBGRAPHS TO AN EDGE (PAMP ALGORITHM)
    best_edge_Sid = []
    for h in range(2, e+1):
        # repeat the algorithm for every edge data center, considering in each iteration a subproblem made by only a cloud and an edge
        subgraphs_id = subgraphs_id_origin.copy()
        Scur_edge_origin_id = subgraphs_id[0]  # Current state id
        H = []  # History vector
        Rcpu_edge = np.array(Rcpu[(h - 1) * M:h * M]) # CPU requested by instances in the edge
        Rmem_edge = np.array(Rmem[(h - 1) * M:h * M]) # Memory requested by instances in the edge
        Rcpu_origin = np.sum(app_edge * Rcpu_edge) # Total CPU requested by instances in the edge
        Rmem_origin = np.sum(app_edge * Rmem_edge) # Total Memory requested by instances in the edge
        
        # Build the state vector of starting configuration
        Sorigin = np.zeros(2 * M)
        Sorigin[:M - 1] = 1
        Sorigin[M:] = app_edge

        dorigin = delayMat(Sorigin, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2)[0]
        subgraphs_id = subgraphs_id[1:]
        Scur_edge_id = Scur_edge_origin_id
        d_prev = dorigin
        while True:
            nsg = len(subgraphs_id)
            subgraphs_weigths = -np.inf * np.ones(nsg)
            subgraphs_costs = -np.inf * np.ones(nsg)
            subgraphs_r = np.inf * np.ones(nsg)
            Scur_edge_b = id2S(Scur_edge_id, 2 ** M)

            for i in range(nsg):
                sg_id = subgraphs_id[i]
                Snew_edge_id = np.bitwise_or(sg_id - 1, Scur_edge_id - 1) + 1  
                Snew_edge_b = np.array(id2S(Snew_edge_id, 2 ** M))
                Rcpu_new = np.sum(Snew_edge_b * Rcpu_edge)
                Rmem_new = np.sum(Snew_edge_b * Rmem_edge)
                if Rcpu_new > Ce or Rmem_new > Me:
                    subgraphs_weigths[i] = -np.inf
                    subgraphs_r[i] = 0
                    continue
                cost_cpu = Rcpu_new - Rcpu_origin
                cost_mem = Rmem_new - Rmem_origin
                cost = cost_cpu
                Snew = np.zeros(2 * M)
                Snew[:M - 1] = 1
                Snew[M:] = Snew_edge_b
                dnew = delayMat(Snew, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2)[0]
                if dnew == np.inf:
                    subgraphs_weigths[i] = -np.inf
                    subgraphs_r[i] = 0
                    continue
                r = dorigin - dnew
                subgraphs_weigths[i] = min(r, min_delay_delta) / cost
                subgraphs_costs[i] = cost
                subgraphs_r[i] = r
            I = np.argmax(subgraphs_weigths)

            best_sg = subgraphs_id[I]
            Scur_edge_id = np.bitwise_or(best_sg - 1, Scur_edge_id - 1) + 1  # Update edge status inserting the nodes of the best subgraph
            H.append([Scur_edge_id, subgraphs_r[I], subgraphs_costs[I], subgraphs_weigths[I]])
            if subgraphs_r[I] >= min_delay_delta:
                break
            # Prune not considered subgraphs whose nodes are already contained in the edge
            PR = []
            for pr in range(nsg):
                if np.bitwise_and(subgraphs_id[pr] - 1, Scur_edge_id - 1) + 1 == subgraphs_id[pr]:
                    PR.append(pr)
            subgraphs_id = [subgraphs_id[pr] for pr in range(nsg) if pr not in PR]
            if len(subgraphs_id) == 0:
                # All subgraphs considered
                break

        # Resource exaustion
        if len(H) == 0:
            # No capacity for edge computing
            best_edge_Sid.append(Scur_edge_origin_id)
        else:
            H = np.array(H)
            I = np.where(H[:, 1] >= min(min_delay_delta, dorigin - 1e-3))
            I2 = np.argmin(H[I, 2])
            best_edge_Sid.append(H[I[I2], 0])  

    # BUILD THE BEST_S VECTOR

    # Build best_Sid from best_edge_Sid
    best_S = np.zeros(e * M)
    best_S[:M - 1] = 1
    best_S[M-1] = 0
    for h in range(2, e+1):
        best_S[(h - 1) * M:h * M] = id2S(int(best_edge_Sid[h-2]), 2 ** M)
    best_dw, Dn, Tnce, Di, Nc = delayMat(best_S, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, e)
    if len(H) == 0:
        delta = 0
    else:
        delta = H[I[I2], 1]
    return best_S, best_dw, Dn, Tnce, delta